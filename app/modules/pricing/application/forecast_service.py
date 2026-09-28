import hashlib
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from time import monotonic

from fastapi import HTTPException

from app.modules.catalog.application.utils import derive_material_key
from app.modules.catalog.domain.repositories import MaterialRepository
from app.modules.catalog.infrastructure.models import Material
from app.modules.pricing.application.backtesting import construir_folds_temporales
from app.modules.pricing.application.forecast_cache import ForecastCacheEntry, ForecastCacheKey
from app.modules.pricing.application.forecasting import (
    BEST_PROPHET_CONFIG,
    construir_dataset_prophet,
    generar_fechas_mensuales,
    inicio_mes_siguiente,
)
from app.modules.pricing.application.model_selector import ForecastModelSelection, resolve_model_selection
from app.modules.pricing.application.series import PrecioSerieInput, PuntoSeriePrecio, construir_serie_mensual
from app.modules.pricing.domain.economic_price import economic_price_rule_for
from app.modules.pricing.domain.exceptions import (
    ExternalRegressorUnavailableError,
    ForecastSnapshotRequired,
    InsufficientDataException,
)
from app.modules.pricing.domain.repositories import PricingRepository
from app.modules.pricing.infrastructure.forecast_runtime import configurar_cmdstan, importar_dependencias_forecast
from app.modules.pricing.infrastructure.forecast_snapshots import cargar_forecast_snapshot, guardar_forecast_snapshot
from app.modules.pricing.infrastructure.regressors import (
    cargar_regresores_mensuales,
    preparar_regresores_para_fold,
    proyectar_regresores_futuros,
)
from app.modules.pricing.interfaces.schemas import ForecastMetricasRead, ForecastPuntoRead, ForecastSelectionRead
from app.shared.config.settings import settings

FORECAST_DATASET_START = date(2022, 1, 1)
FORECAST_MODEL_NAME = "prophet_oficial_ipc_mayorista"
FORECAST_REGRESSOR_NOTE = "Escenario base: dolar oficial, dolar mayorista e IPC futuros proyectados con crecimiento compuesto segun la tendencia de los ultimos 12 meses."
FORECAST_SELECTOR_DISABLED_SIGNATURE = "selector-off"
FORECAST_SELECTOR_ENABLED_SIGNATURE = "selector-on"
FORECAST_SELECTOR_FALLBACK_ORIGIN = "fallback_regresores"


@dataclass(frozen=True)
class ForecastMaterialResult:
    dataset: list["ProphetRow"]
    metricas: ForecastMetricasRead
    forecast: list[ForecastPuntoRead]
    modelo: str = FORECAST_MODEL_NAME
    supuesto_regresores: str = FORECAST_REGRESSOR_NOTE
    seleccion_modelo: ForecastSelectionRead | None = None
    serie_mensual: list[PuntoSeriePrecio] | None = None
    resolution_source: str = "computed"
    economic_scale_applied: bool = False


@dataclass(frozen=True)
class ProphetRow:
    ds: date | None
    y: float


@dataclass(frozen=True)
class ProphetExecutionResult:
    forecast: list[ForecastPuntoRead]
    dataset: list[ProphetRow]
    mape: float
    mae: float
    modelo_config: dict


@dataclass(frozen=True)
class ForecastExecutionPlan:
    modelo: str
    regresores: tuple[str, ...]
    regresores_df: object | None
    supuesto_regresores: str
    seleccion_modelo: ForecastSelectionRead | None
    cache_signature: str


_forecast_cache: dict[ForecastCacheKey, ForecastCacheEntry] = {}


def _a_dataframe(pd, filas):
    return pd.DataFrame([{"ds": pd.to_datetime(fila.ds), "y": fila.y} for fila in filas])


def construir_firma_dataset(dataset: list) -> str:
    digest = hashlib.sha256()
    for fila in dataset:
        digest.update(f"{fila.ds.isoformat()}|{fila.y:.8f}".encode())
    return digest.hexdigest()


def _descripcion_regresores(regresores: tuple[str, ...]) -> str:
    if not regresores:
        return "Escenario base sin regresores externos."
    return f"Regresores resueltos: {', '.join(regresores)}."


def _selection_to_metadata(selection: ForecastModelSelection, *, advertencia: str | None = None) -> ForecastSelectionRead:
    return ForecastSelectionRead(
        material_key=selection.material_key,
        modelo_resuelto=selection.modelo,
        regresores_resueltos=list(selection.regresores),
        mape_referencia=selection.mape,
        mae_referencia=selection.mae,
        folds=selection.folds,
        confiabilidad=selection.confiabilidad,
        origen_decision=selection.origen_decision,
        justificacion=selection.justificacion,
        no_calibrado=selection.no_calibrado,
        advertencia=advertencia,
    )


def _fallback_selection_for_missing_regressors(
    selection: ForecastModelSelection,
    detalle_error: str,
) -> ForecastSelectionRead:
    return ForecastSelectionRead(
        material_key=selection.material_key,
        modelo_resuelto="prophet_base",
        regresores_resueltos=[],
        mape_referencia=None,
        mae_referencia=None,
        folds=None,
        confiabilidad="no_calibrada",
        origen_decision=FORECAST_SELECTOR_FALLBACK_ORIGIN,
        justificacion=(
            f"Se degrada a prophet_base porque faltan regresores requeridos para "
            f"{selection.modelo}: {detalle_error}"
        ),
        no_calibrado=True,
        advertencia=detalle_error,
    )


def _resolver_plan_legacy(pd) -> ForecastExecutionPlan:
    regresores = ("dolar_oficial", "dolar_mayorista", "ipc")
    return ForecastExecutionPlan(
        modelo=FORECAST_MODEL_NAME,
        regresores=regresores,
        regresores_df=cargar_regresores_mensuales(pd, regresores),
        supuesto_regresores=FORECAST_REGRESSOR_NOTE,
        seleccion_modelo=None,
        cache_signature=f"{FORECAST_SELECTOR_DISABLED_SIGNATURE}:{FORECAST_MODEL_NAME}",
    )


def _resolver_plan_selector(material_key: str, horizonte_meses: int, pd) -> ForecastExecutionPlan:
    selection = resolve_model_selection(material_key, horizonte_meses)
    metadata = _selection_to_metadata(selection)
    if not selection.regresores:
        return ForecastExecutionPlan(
            modelo=selection.modelo,
            regresores=selection.regresores,
            regresores_df=None,
            supuesto_regresores=_descripcion_regresores(selection.regresores),
            seleccion_modelo=metadata,
            cache_signature=f"{FORECAST_SELECTOR_ENABLED_SIGNATURE}:{selection.modelo}",
        )

    try:
        regresores_df = cargar_regresores_mensuales(pd, selection.regresores)
    except ExternalRegressorUnavailableError as exc:
        fallback_metadata = _fallback_selection_for_missing_regressors(selection, str(exc))
        return ForecastExecutionPlan(
            modelo="prophet_base",
            regresores=(),
            regresores_df=None,
            supuesto_regresores="Fallback controlado a prophet_base por regresores no disponibles.",
            seleccion_modelo=fallback_metadata,
            cache_signature=f"{FORECAST_SELECTOR_ENABLED_SIGNATURE}:prophet_base:fallback_regresores",
        )

    return ForecastExecutionPlan(
        modelo=selection.modelo,
        regresores=selection.regresores,
        regresores_df=regresores_df,
        supuesto_regresores=_descripcion_regresores(selection.regresores),
        seleccion_modelo=metadata,
        cache_signature=f"{FORECAST_SELECTOR_ENABLED_SIGNATURE}:{selection.modelo}",
    )


def _resolver_plan_ejecucion(material_key: str, horizonte_meses: int, usar_selector_modelo: bool, pd) -> ForecastExecutionPlan:
    if not usar_selector_modelo:
        return _resolver_plan_legacy(pd)
    return _resolver_plan_selector(material_key, horizonte_meses, pd)


def obtener_forecast_cacheado(material_id: int, horizonte_meses: int, dataset_signature: str) -> ForecastMaterialResult | None:
    cache_key = ForecastCacheKey(
        material_id=material_id,
        horizonte_meses=horizonte_meses,
        dataset_signature=dataset_signature,
    )
    entry = _forecast_cache.get(cache_key)
    ahora = monotonic()
    if entry is None:
        return None
    if entry.expires_at <= ahora:
        _forecast_cache.pop(cache_key, None)
        return None
    return deepcopy(entry.result)


def guardar_forecast_cacheado(
    material_id: int,
    horizonte_meses: int,
    dataset_signature: str,
    result: ForecastMaterialResult,
) -> ForecastMaterialResult:
    cache_key = ForecastCacheKey(
        material_id=material_id,
        horizonte_meses=horizonte_meses,
        dataset_signature=dataset_signature,
    )
    _forecast_cache[cache_key] = ForecastCacheEntry(
        result=deepcopy(result),
        expires_at=monotonic() + settings.forecast_cache_ttl_seconds,
    )
    guardar_forecast_snapshot(cache_key, result)
    return deepcopy(result)


def limpiar_forecast_cache() -> None:
    _forecast_cache.clear()


def _scale_forecast_result_for_economic_price(
    result: ForecastMaterialResult,
    material_key: str,
) -> ForecastMaterialResult:
    rule = economic_price_rule_for(material_key)
    if rule is None or result.economic_scale_applied:
        return result

    def scale(value: Decimal | None) -> Decimal | None:
        return rule.net_price(value) if value is not None else None

    scaled_points = [
        point.model_copy(
            update={
                "precio_lista_proyectado": point.precio_proyectado,
                "precio_proyectado": scale(point.precio_proyectado),
                "precio_optimista": scale(point.precio_optimista),
                "precio_pesimista": scale(point.precio_pesimista),
                "precio_equivalente_25kg": scale(point.precio_equivalente_25kg),
                "precio_equivalente_50kg": scale(point.precio_equivalente_50kg),
            }
        )
        for point in result.forecast
    ]
    scaled_series = None
    if result.serie_mensual is not None:
        monetary_fields = (
            "precio_promedio_normalizado",
            "precio_equivalente_25kg",
            "precio_equivalente_50kg",
            "precio_esperado_anomalia",
            "rango_esperado_min_anomalia",
            "rango_esperado_max_anomalia",
            "precio_original",
        )
        scaled_series = [
            replace(point, **{field: scale(getattr(point, field)) for field in monetary_fields})
            if isinstance(point, PuntoSeriePrecio)
            else point
            for point in result.serie_mensual
        ]

    return replace(
        result,
        dataset=[ProphetRow(ds=row.ds, y=float(rule.net_price(Decimal(str(row.y))))) for row in result.dataset],
        metricas=result.metricas.model_copy(update={"mae": scale(result.metricas.mae)}),
        forecast=scaled_points,
        serie_mensual=scaled_series,
        economic_scale_applied=True,
    )


def _cargar_forecast_cacheado_o_snapshot(
    *,
    material_id: int,
    horizonte_meses: int,
    dataset_signature: str,
) -> ForecastMaterialResult | None:
    forecast_cacheado = obtener_forecast_cacheado(material_id, horizonte_meses, dataset_signature)
    if forecast_cacheado is not None:
        return replace(forecast_cacheado, resolution_source="memory_cache")

    cache_key = ForecastCacheKey(
        material_id=material_id,
        horizonte_meses=horizonte_meses,
        dataset_signature=dataset_signature,
    )
    forecast_persistido = cargar_forecast_snapshot(cache_key)
    if forecast_persistido is None:
        return None

    _forecast_cache[cache_key] = ForecastCacheEntry(
        result=deepcopy(forecast_persistido),
        expires_at=monotonic() + settings.forecast_cache_ttl_seconds,
    )
    return replace(deepcopy(forecast_persistido), resolution_source="snapshot")


def _require_synchronous_compute_allowed(material: Material, horizonte_meses: int) -> None:
    if settings.forecast_allow_synchronous_compute:
        return
    raise ForecastSnapshotRequired(
        f"No hay un snapshot vigente para {material.nombre} a {horizonte_meses} meses; "
        "el calculo sincrono esta deshabilitado."
    )


def serie_mensual_material(material: Material, pricing_repo: PricingRepository):
    registros = [
        PrecioSerieInput(
            fecha=precio.fecha,
            precio_normalizado=precio.precio_normalizado,
            unidad_base=material.unidad_base,
            fuente=precio.fuente.nombre if precio.fuente else None,
            numero_comprobante=precio.numero_comprobante,
        )
        for precio in pricing_repo.get_historical_prices(material.id, FORECAST_DATASET_START)
        if precio.fecha <= date.today()
    ]
    return construir_serie_mensual(registros)


def backtesting_forecast(pd, Prophet, dataset, regresores_df, horizonte_meses: int, regresores: tuple[str, ...]) -> ForecastMetricasRead:
    folds = construir_folds_temporales(dataset, min_train_size=24, test_size=horizonte_meses, step_size=horizonte_meses)
    if not folds:
        raise HTTPException(status_code=422, detail="No hay suficientes datos para evaluar ese horizonte de forecast.")

    abs_errors: list[float] = []
    apes: list[float] = []
    for fold in folds:
        train_df = _a_dataframe(pd, fold.train)
        test_df = _a_dataframe(pd, fold.test)
        modelo = Prophet(stan_backend="CMDSTANPY", **BEST_PROPHET_CONFIG)
        if regresores:
            train_reg, futuro = preparar_regresores_para_fold(
                pd, train_df, test_df, regresores_df, regresores
            )
            for columna in regresores:
                modelo.add_regressor(columna)
            modelo.fit(train_reg)
            forecast = modelo.predict(futuro)[["ds", "yhat"]]
        else:
            modelo.fit(train_df)
            futuro = modelo.make_future_dataframe(periods=len(test_df), freq="MS")
            forecast = modelo.predict(futuro)[["ds", "yhat"]]

        evaluacion = test_df.merge(forecast, on="ds", how="left")
        evaluacion["abs_error"] = (evaluacion["y"] - evaluacion["yhat"]).abs()
        evaluacion["ape"] = evaluacion["abs_error"] / evaluacion["y"] * 100
        abs_errors.extend(evaluacion["abs_error"].tolist())
        apes.extend(evaluacion["ape"].tolist())

    mae = sum(abs_errors) / len(abs_errors)
    mape = sum(apes) / len(apes)
    return ForecastMetricasRead(
        folds=len(folds),
        mae=Decimal(f"{mae:.2f}"),
        mape=Decimal(f"{mape:.2f}"),
        efectividad_informal=Decimal(f"{100 - mape:.2f}"),
    )


def pronosticar_futuro(
    pd,
    Prophet,
    dataset,
    regresores_df,
    regresores: tuple[str, ...],
    horizonte_meses: int,
    unidad_base: str,
    material_nombre: str,
) -> list[ForecastPuntoRead]:
    dataset_df = _a_dataframe(pd, dataset)
    modelo = Prophet(stan_backend="CMDSTANPY", **BEST_PROPHET_CONFIG)
    ultima_fecha = dataset[-1].ds
    if regresores:
        dataset_reg = dataset_df.merge(regresores_df, on="ds", how="left")
        for columna in regresores:
            dataset_reg[columna] = dataset_reg[columna].ffill().bfill()
            modelo.add_regressor(columna)
        modelo.fit(dataset_reg[["ds", "y", *regresores]].copy())

        fechas_futuras = generar_fechas_mensuales(inicio_mes_siguiente(ultima_fecha), horizonte_meses)
        regresores_hasta_ultima_fecha = regresores_df[regresores_df["ds"] <= pd.to_datetime(ultima_fecha)].sort_values("ds")
        if regresores_hasta_ultima_fecha.empty:
            raise HTTPException(status_code=422, detail="No hay regresores externos alineados con la ultima fecha observada.")
        futuro_reg = proyectar_regresores_futuros(pd, regresores_hasta_ultima_fecha, fechas_futuras, regresores)
        futuro = pd.concat([dataset_reg[["ds", *regresores]], futuro_reg], ignore_index=True)
        forecast = modelo.predict(futuro)[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizonte_meses)
    else:
        modelo.fit(dataset_df)
        futuro = modelo.make_future_dataframe(periods=horizonte_meses, freq="MS")
        forecast = modelo.predict(futuro)[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizonte_meses)

    puntos: list[ForecastPuntoRead] = []
    usa_equivalencias = unidad_base == "kg" and material_nombre == "Cemento Portland"
    for _, fila in forecast.iterrows():
        precio = float(fila["yhat"])
        precio_optimista = float(fila["yhat_lower"])
        precio_pesimista = float(fila["yhat_upper"])
        equivalencia_25 = Decimal(f"{precio * 25:.2f}") if usa_equivalencias else None
        equivalencia_50 = Decimal(f"{precio * 50:.2f}") if usa_equivalencias else None
        puntos.append(
            ForecastPuntoRead(
                fecha=fila["ds"].date(),
                precio_proyectado=Decimal(f"{precio:.2f}"),
                precio_optimista=Decimal(f"{precio_optimista:.2f}"),
                precio_pesimista=Decimal(f"{precio_pesimista:.2f}"),
                precio_equivalente_25kg=equivalencia_25,
                precio_equivalente_50kg=equivalencia_50,
            )
        )
    return puntos


def _forecast_material(
    material: Material,
    horizonte_meses: int,
    dataset: list,
    pd,
    Prophet,
    plan: ForecastExecutionPlan,
) -> ForecastMaterialResult:
    metricas = backtesting_forecast(pd, Prophet, dataset, plan.regresores_df, horizonte_meses, plan.regresores)
    forecast = pronosticar_futuro(
        pd,
        Prophet,
        dataset,
        plan.regresores_df,
        plan.regresores,
        horizonte_meses,
        material.unidad_base,
        material.nombre,
    )
    return ForecastMaterialResult(
        dataset=dataset,
        metricas=metricas,
        forecast=forecast,
        modelo=plan.modelo,
        supuesto_regresores=plan.supuesto_regresores,
        seleccion_modelo=plan.seleccion_modelo,
    )


def forecast_material(
    material: Material,
    horizonte_meses: int,
    pricing_repo: PricingRepository,
    usar_selector_modelo: bool = False,
) -> ForecastMaterialResult:
    material_key = derive_material_key(material.nombre)
    puntos = serie_mensual_material(material, pricing_repo)
    dataset = construir_dataset_prophet(puntos, objetivo="precio_promedio_normalizado")
    if len(dataset) < 24 + horizonte_meses:
        raise InsufficientDataException(f"No hay suficientes puntos mensuales para generar el forecast de {material.nombre}")

    signature_base = construir_firma_dataset(dataset)
    if not usar_selector_modelo:
        dataset_signature = f"{signature_base}:{FORECAST_SELECTOR_DISABLED_SIGNATURE}:{FORECAST_MODEL_NAME}"
        forecast_cacheado = _cargar_forecast_cacheado_o_snapshot(
            material_id=material.id,
            horizonte_meses=horizonte_meses,
            dataset_signature=dataset_signature,
        )
        if forecast_cacheado is not None:
            return _scale_forecast_result_for_economic_price(forecast_cacheado, material_key)

        _require_synchronous_compute_allowed(material, horizonte_meses)

        cmdstanpy, pd, Prophet, CmdStanPyBackend, IStanBackend = importar_dependencias_forecast()
        configurar_cmdstan(cmdstanpy, CmdStanPyBackend, IStanBackend)
        plan = _resolver_plan_legacy(pd)
        forecast_result = _forecast_material(material, horizonte_meses, dataset, pd, Prophet, plan)
        forecast_result = ForecastMaterialResult(
            dataset=forecast_result.dataset,
            metricas=forecast_result.metricas,
            forecast=forecast_result.forecast,
            modelo=forecast_result.modelo,
            supuesto_regresores=forecast_result.supuesto_regresores,
            seleccion_modelo=forecast_result.seleccion_modelo,
            serie_mensual=puntos,
        )
        cached_result = guardar_forecast_cacheado(material.id, horizonte_meses, dataset_signature, forecast_result)
        return _scale_forecast_result_for_economic_price(cached_result, material_key)

    selection = resolve_model_selection(material_key, horizonte_meses)
    dataset_signature = f"{signature_base}:{FORECAST_SELECTOR_ENABLED_SIGNATURE}:{selection.modelo}"
    forecast_cacheado = _cargar_forecast_cacheado_o_snapshot(
        material_id=material.id,
        horizonte_meses=horizonte_meses,
        dataset_signature=dataset_signature,
    )
    if forecast_cacheado is not None:
        return _scale_forecast_result_for_economic_price(forecast_cacheado, material_key)

    cmdstanpy, pd, Prophet, CmdStanPyBackend, IStanBackend = importar_dependencias_forecast()
    plan = _resolver_plan_ejecucion(material_key, horizonte_meses, usar_selector_modelo, pd)

    dataset_signature = f"{signature_base}:{plan.cache_signature}"
    forecast_cacheado = _cargar_forecast_cacheado_o_snapshot(
        material_id=material.id,
        horizonte_meses=horizonte_meses,
        dataset_signature=dataset_signature,
    )
    if forecast_cacheado is not None:
        return _scale_forecast_result_for_economic_price(forecast_cacheado, material_key)

    _require_synchronous_compute_allowed(material, horizonte_meses)
    configurar_cmdstan(cmdstanpy, CmdStanPyBackend, IStanBackend)

    forecast_result = _forecast_material(material, horizonte_meses, dataset, pd, Prophet, plan)
    forecast_result = ForecastMaterialResult(
        dataset=forecast_result.dataset,
        metricas=forecast_result.metricas,
        forecast=forecast_result.forecast,
        modelo=forecast_result.modelo,
        supuesto_regresores=forecast_result.supuesto_regresores,
        seleccion_modelo=forecast_result.seleccion_modelo,
        serie_mensual=puntos,
    )
    cached_result = guardar_forecast_cacheado(material.id, horizonte_meses, dataset_signature, forecast_result)
    return _scale_forecast_result_for_economic_price(cached_result, material_key)


def precomputar_forecasts_materiales(
    material_repo: MaterialRepository,
    pricing_repo: PricingRepository,
    horizontes: tuple[int, ...] = (3, 6, 12),
    usar_selector_modelo: bool = True,
) -> list[tuple[int, int]]:
    materiales = material_repo.list_active()
    completados: list[tuple[int, int]] = []
    for material in materiales:
        for horizonte in horizontes:
            forecast_material(
                material,
                horizonte,
                pricing_repo,
                usar_selector_modelo=usar_selector_modelo,
            )
            completados.append((material.id, horizonte))
    return completados
