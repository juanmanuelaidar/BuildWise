from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

from fastapi import HTTPException
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value

from app.modules.catalog.application.utils import derive_material_key
from app.modules.catalog.domain.repositories import MaterialRepository
from app.modules.pricing.application.forecast_service import forecast_material
from app.modules.pricing.application.purchase_recommendations import (
    CONFIANZA_BAJA,
    CONFIANZA_NO_CALIBRADA,
    CONFIANZA_NO_DISPONIBLE,
    _resolver_confiabilidad,
    evaluar_recomendacion_compra,
)
from app.modules.pricing.application.purchase_strategies import evaluar_estrategias_compra
from app.modules.pricing.domain.exceptions import MaterialNotFoundException
from app.modules.pricing.domain.repositories import PricingRepository
from app.modules.pricing.domain.rules import calcular_variacion_esperada_porcentual

PESO_CRITICIDAD = {
    "alta": Decimal("3.00"),
    "media": Decimal("2.00"),
    "baja": Decimal("1.00"),
}

ESTADO_OPTIMAL = "OPTIMAL"
QUANTITY_STEP = Decimal("0.0001")


@dataclass(frozen=True)
class PurchaseOptimizationInputItem:
    material_id: int
    cantidad_objetivo: Decimal
    criticidad: str
    porcentaje_minimo_compra_inmediata: Decimal | None = None


@dataclass(frozen=True)
class OptimizationCandidate:
    material_id: int
    material_key: str
    cantidad_objetivo: Decimal
    precio_actual: Decimal
    precio_proyectado_horizonte: Decimal
    ahorro_unitario_estimado: Decimal
    criticidad: str
    peso_criticidad: Decimal
    confiabilidad: str
    no_calibrado: bool
    mape: Decimal | None = None
    porcentaje_minimo_compra_inmediata: Decimal | None = None
    fecha_base_observada: date | None = None


@dataclass(frozen=True)
class PurchaseOptimizationItemResult:
    material_id: int
    material_key: str
    cantidad_objetivo: Decimal
    cantidad_recomendada_comprar_ahora: Decimal
    cantidad_recomendada_postergar: Decimal
    precio_actual: Decimal
    precio_proyectado_horizonte: Decimal
    costo_compra_ahora: Decimal
    costo_futuro_estimado: Decimal
    ahorro_unitario_estimado: Decimal
    ahorro_total_estimado: Decimal
    impacto_economico_pct: Decimal
    accion_recomendada: str
    criticidad: str
    peso_criticidad: Decimal
    confiabilidad: str
    mape: Decimal | None
    no_calibrado: bool
    fecha_base_observada: date | None = None


@dataclass(frozen=True)
class PurchaseOptimizationResult:
    presupuesto_total: Decimal
    presupuesto_utilizado: Decimal
    presupuesto_restante: Decimal
    horizonte_meses: int
    estado_optimizacion: str
    items: tuple[PurchaseOptimizationItemResult, ...]
    ahorro_total_estimado: Decimal
    justificacion: str
    advertencias: tuple[str, ...]
    fecha_base_calculo: date | None = None


@dataclass(frozen=True)
class OperationalPurchaseRecommendationItem:
    material_id: int
    material_key: str
    accion_recomendada: str
    cantidad_comprar_ahora: Decimal
    cantidad_postergar: Decimal
    impacto_economico_estimado: Decimal
    impacto_economico_pct: Decimal
    confianza: str
    criticidad: str
    recomendacion_simple: str
    mejor_estrategia: str
    ventaja_estrategia_significativa: bool
    explicacion: str
    fecha_base_observada: date | None = None


@dataclass(frozen=True)
class OperationalPurchaseRecommendationResult:
    fecha_calculo: date
    horizonte_meses: int
    presupuesto_total: Decimal
    presupuesto_utilizado: Decimal
    presupuesto_restante: Decimal
    ahorro_total_estimado: Decimal
    decision_resumen: str
    items: tuple[OperationalPurchaseRecommendationItem, ...]
    supuestos: tuple[str, ...]
    advertencias: tuple[str, ...]


def _quantize_amount(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _quantize_quantity(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _quantize_pct(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _es_confiabilidad_baja(confiabilidad: str, no_calibrado: bool) -> bool:
    return no_calibrado or confiabilidad in {
        CONFIANZA_BAJA,
        CONFIANZA_NO_CALIBRADA,
        CONFIANZA_NO_DISPONIBLE,
    }


def _normalizar_estado_optimizacion(status_code: int) -> str:
    return LpStatus[status_code].replace(" ", "_").upper()


def _resolver_accion(cantidad_recomendada: Decimal, cantidad_objetivo: Decimal) -> str:
    if cantidad_recomendada <= 0:
        return "POSTERGAR"
    if cantidad_recomendada >= cantidad_objetivo:
        return "COMPRAR_AHORA"
    return "COMPRA_PARCIAL"


def _calcular_impacto_pct(ahorro_unitario: Decimal, precio_actual: Decimal) -> Decimal:
    if precio_actual <= 0:
        return Decimal("0.0000")
    return _quantize_pct((ahorro_unitario / precio_actual) * Decimal("100"))


def _build_candidate(item: PurchaseOptimizationInputItem, material, forecast_result) -> OptimizationCandidate:
    selection = getattr(forecast_result, "seleccion_modelo", None)
    material_key = selection.material_key if selection is not None and getattr(selection, "material_key", None) else derive_material_key(material.nombre)
    precio_actual = Decimal(f"{forecast_result.dataset[-1].y:.2f}")
    fecha_base_observada = forecast_result.dataset[-1].ds
    precio_proyectado_horizonte = forecast_result.forecast[-1].precio_proyectado
    ahorro_unitario = max(precio_proyectado_horizonte - precio_actual, Decimal("0"))
    confiabilidad = _resolver_confiabilidad(forecast_result)
    no_calibrado = bool(getattr(selection, "no_calibrado", False))
    mape = getattr(forecast_result.metricas, "mape", None)

    return OptimizationCandidate(
        material_id=material.id,
        material_key=material_key,
        cantidad_objetivo=item.cantidad_objetivo,
        precio_actual=precio_actual,
        precio_proyectado_horizonte=precio_proyectado_horizonte,
        ahorro_unitario_estimado=_quantize_amount(ahorro_unitario),
        criticidad=item.criticidad,
        peso_criticidad=PESO_CRITICIDAD[item.criticidad],
        confiabilidad=confiabilidad,
        no_calibrado=no_calibrado,
        mape=mape,
        porcentaje_minimo_compra_inmediata=item.porcentaje_minimo_compra_inmediata,
        fecha_base_observada=fecha_base_observada,
    )


def optimizar_compra_items(
    *,
    presupuesto_total: Decimal,
    horizonte_meses: int,
    candidates: list[OptimizationCandidate],
    advertencias: list[str] | tuple[str, ...] | None = None,
) -> PurchaseOptimizationResult:
    if not candidates:
        raise HTTPException(status_code=422, detail="No hay materiales validos para optimizar la compra.")

    problem = LpProblem("purchase_budget_optimization", LpMinimize)
    variables_ahora: dict[int, LpVariable] = {}
    variables_futuro: dict[int, LpVariable] = {}

    for candidate in candidates:
        variables_ahora[candidate.material_id] = LpVariable(
            f"x_ahora_{candidate.material_id}",
            lowBound=0,
            upBound=float(candidate.cantidad_objetivo / QUANTITY_STEP),
            cat="Integer",
        )
        variables_futuro[candidate.material_id] = LpVariable(
            f"x_futuro_{candidate.material_id}",
            lowBound=0,
            upBound=float(candidate.cantidad_objetivo),
        )

    # Maximize avoided future cost weighted by operational criticality while
    # keeping the immediate purchase within budget. The small spending
    # penalty resolves zero-benefit ties in favor of postponing purchase.
    problem += -lpSum(
        (
            float(candidate.ahorro_unitario_estimado * candidate.peso_criticidad) * 1_000_000
            - float(candidate.precio_actual)
        )
        * float(QUANTITY_STEP) * variables_ahora[candidate.material_id]
        for candidate in candidates
    )
    problem += lpSum(
        float(candidate.precio_actual * QUANTITY_STEP) * variables_ahora[candidate.material_id] for candidate in candidates
    ) <= float(presupuesto_total)

    for candidate in candidates:
        problem += (
            float(QUANTITY_STEP) * variables_ahora[candidate.material_id] + variables_futuro[candidate.material_id]
            == float(candidate.cantidad_objetivo)
        )
        if candidate.porcentaje_minimo_compra_inmediata is not None:
            problem += variables_ahora[candidate.material_id] >= float(
                candidate.cantidad_objetivo * candidate.porcentaje_minimo_compra_inmediata / QUANTITY_STEP
            )

    status_code = problem.solve(PULP_CBC_CMD(msg=False))
    estado_optimizacion = _normalizar_estado_optimizacion(status_code)
    if estado_optimizacion != ESTADO_OPTIMAL:
        raise HTTPException(status_code=422, detail="No fue posible resolver la optimizacion de compra.")

    resultados: list[PurchaseOptimizationItemResult] = []
    presupuesto_utilizado = Decimal("0")
    ahorro_total_estimado = Decimal("0")
    for candidate in candidates:
        valor_ahora = value(variables_ahora[candidate.material_id])
        restante = presupuesto_total - presupuesto_utilizado
        cantidad_recomendada = min(
            Decimal(str(valor_ahora or 0)) * QUANTITY_STEP,
            candidate.cantidad_objetivo,
            restante / candidate.precio_actual,
        ).quantize(Decimal("0.0001"), rounding=ROUND_FLOOR)
        cantidad_recomendada = max(cantidad_recomendada, Decimal("0"))
        costo_compra_ahora = _quantize_amount(candidate.precio_actual * cantidad_recomendada)
        # CBC uses floating point; the published Decimal plan must remain feasible
        # after quantity and per-line currency rounding as well.
        while costo_compra_ahora > restante and cantidad_recomendada > 0:
            cantidad_recomendada -= Decimal("0.0001")
            costo_compra_ahora = _quantize_amount(candidate.precio_actual * cantidad_recomendada)
        minimo = candidate.cantidad_objetivo * (candidate.porcentaje_minimo_compra_inmediata or Decimal("0"))
        if cantidad_recomendada < minimo:
            raise HTTPException(status_code=422, detail="El presupuesto no permite representar la compra minima con la precision de cantidades admitida.")
        cantidad_postergada = _quantize_quantity(candidate.cantidad_objetivo - cantidad_recomendada)
        costo_futuro_estimado = _quantize_amount(candidate.precio_proyectado_horizonte * cantidad_postergada)
        ahorro_total = _quantize_amount(candidate.ahorro_unitario_estimado * cantidad_recomendada)
        presupuesto_utilizado += costo_compra_ahora
        ahorro_total_estimado += ahorro_total
        resultados.append(
            PurchaseOptimizationItemResult(
                material_id=candidate.material_id,
                material_key=candidate.material_key,
                cantidad_objetivo=candidate.cantidad_objetivo,
                cantidad_recomendada_comprar_ahora=cantidad_recomendada,
                cantidad_recomendada_postergar=cantidad_postergada,
                precio_actual=candidate.precio_actual,
                precio_proyectado_horizonte=candidate.precio_proyectado_horizonte,
                costo_compra_ahora=costo_compra_ahora,
                costo_futuro_estimado=costo_futuro_estimado,
                ahorro_unitario_estimado=candidate.ahorro_unitario_estimado,
                ahorro_total_estimado=ahorro_total,
                impacto_economico_pct=_calcular_impacto_pct(
                    candidate.ahorro_unitario_estimado,
                    candidate.precio_actual,
                ),
                accion_recomendada=_resolver_accion(cantidad_recomendada, candidate.cantidad_objetivo),
                criticidad=candidate.criticidad,
                peso_criticidad=candidate.peso_criticidad,
                confiabilidad=candidate.confiabilidad,
                mape=candidate.mape,
                no_calibrado=candidate.no_calibrado,
                fecha_base_observada=candidate.fecha_base_observada,
            )
        )

    advertencias_resultado = list(advertencias or [])
    presupuesto_utilizado = _quantize_amount(presupuesto_utilizado)
    ahorro_total_estimado = _quantize_amount(ahorro_total_estimado)
    presupuesto_restante = _quantize_amount(presupuesto_total - presupuesto_utilizado)
    fechas_base = [candidate.fecha_base_observada for candidate in candidates if candidate.fecha_base_observada is not None]

    return PurchaseOptimizationResult(
        presupuesto_total=_quantize_amount(presupuesto_total),
        presupuesto_utilizado=presupuesto_utilizado,
        presupuesto_restante=presupuesto_restante,
        horizonte_meses=horizonte_meses,
        estado_optimizacion=estado_optimizacion,
        items=tuple(resultados),
        ahorro_total_estimado=ahorro_total_estimado,
        justificacion=(
            "La optimizacion prioriza el ahorro futuro evitado ponderado por criticidad, con variables "
            "explicitas de compra inmediata y postergada, respetando el presupuesto disponible."
        ),
        advertencias=tuple(advertencias_resultado),
        fecha_base_calculo=max(fechas_base) if fechas_base else None,
    )


def optimizar_compra_con_presupuesto(
    *,
    presupuesto_total: Decimal,
    horizonte_meses: int,
    materiales: list[PurchaseOptimizationInputItem],
    material_repo: MaterialRepository,
    pricing_repo: PricingRepository,
    usar_selector_modelo: bool = False,
) -> PurchaseOptimizationResult:
    candidates: list[OptimizationCandidate] = []
    advertencias: list[str] = []

    for item in materiales:
        material = material_repo.get_by_id(item.material_id)
        if material is None:
            raise MaterialNotFoundException(item.material_id)

        try:
            forecast_result = forecast_material(
                material,
                horizonte_meses,
                pricing_repo,
                usar_selector_modelo=usar_selector_modelo,
            )
        except Exception as exc:
            advertencias.append(f"Se excluye material_id={item.material_id} de la optimizacion: {exc}.")
            continue

        if not forecast_result.forecast or not getattr(forecast_result, "dataset", None):
            advertencias.append(
                f"Se excluye material_id={item.material_id} de la optimizacion por falta de forecast util."
            )
            continue

        candidate = _build_candidate(item, material, forecast_result)
        selection = getattr(forecast_result, "seleccion_modelo", None)
        if selection is not None and getattr(selection, "advertencia", None):
            advertencias.append(f"{candidate.material_key}: {selection.advertencia}")
        if _es_confiabilidad_baja(candidate.confiabilidad, candidate.no_calibrado):
            advertencias.append(
                f"{candidate.material_key}: el material se optimiza con cautela por confiabilidad {candidate.confiabilidad}"
                f"{' y modelo no calibrado' if candidate.no_calibrado else ''}."
            )

        candidates.append(candidate)

    return optimizar_compra_items(
        presupuesto_total=presupuesto_total,
        horizonte_meses=horizonte_meses,
        candidates=candidates,
        advertencias=advertencias,
    )


def generar_recomendacion_operativa_compra(
    *,
    presupuesto_total: Decimal,
    horizonte_meses: int,
    materiales: list[PurchaseOptimizationInputItem],
    material_repo: MaterialRepository,
    pricing_repo: PricingRepository,
    usar_selector_modelo: bool = False,
) -> OperationalPurchaseRecommendationResult:
    optimizacion = optimizar_compra_con_presupuesto(
        presupuesto_total=presupuesto_total,
        horizonte_meses=horizonte_meses,
        materiales=materiales,
        material_repo=material_repo,
        pricing_repo=pricing_repo,
        usar_selector_modelo=usar_selector_modelo,
    )

    items_list: list[OperationalPurchaseRecommendationItem] = []
    for item in optimizacion.items:
        variacion_esperada_pct = calcular_variacion_esperada_porcentual(
            item.precio_actual,
            item.precio_proyectado_horizonte,
        )
        recommendation = evaluar_recomendacion_compra(
            material_id=item.material_id,
            material_key=item.material_key,
            horizonte_meses=optimizacion.horizonte_meses,
            cantidad_objetivo=item.cantidad_objetivo,
            variacion_esperada_pct=variacion_esperada_pct,
            precio_actual=item.precio_actual,
            precio_proyectado_horizonte=item.precio_proyectado_horizonte,
            mape=item.mape,
            confiabilidad=item.confiabilidad,
            criticidad=item.criticidad,
            no_calibrado=item.no_calibrado,
            advertencias=(),
        )
        strategy_comparison = evaluar_estrategias_compra(
            material_id=item.material_id,
            material_key=item.material_key,
            horizonte_meses=optimizacion.horizonte_meses,
            cantidad_objetivo=item.cantidad_objetivo,
            precio_actual=item.precio_actual,
            precio_proyectado_horizonte=item.precio_proyectado_horizonte,
            confiabilidad=item.confiabilidad,
            no_calibrado=item.no_calibrado,
            advertencias=(),
        )
        items_list.append(
            OperationalPurchaseRecommendationItem(
                material_id=item.material_id,
                material_key=item.material_key,
                accion_recomendada=item.accion_recomendada,
                cantidad_comprar_ahora=item.cantidad_recomendada_comprar_ahora,
                cantidad_postergar=item.cantidad_recomendada_postergar,
                impacto_economico_estimado=item.ahorro_total_estimado,
                impacto_economico_pct=item.impacto_economico_pct,
                confianza=item.confiabilidad,
                criticidad=item.criticidad,
                recomendacion_simple=recommendation.decision,
                mejor_estrategia=strategy_comparison.mejor_estrategia,
                ventaja_estrategia_significativa=strategy_comparison.ventaja_significativa,
                explicacion=(
                    f"{item.accion_recomendada}: recomendacion simple {recommendation.decision}, "
                    f"mejor estrategia {strategy_comparison.mejor_estrategia}, ahorro unitario estimado "
                    f"{item.ahorro_unitario_estimado}, criticidad {item.criticidad}, confianza {item.confiabilidad}."
                ),
                fecha_base_observada=item.fecha_base_observada,
            )
        )
    items = tuple(items_list)

    comprar_ahora = sum(1 for item in items if item.accion_recomendada == "COMPRAR_AHORA")
    compra_parcial = sum(1 for item in items if item.accion_recomendada == "COMPRA_PARCIAL")
    postergar = sum(1 for item in items if item.accion_recomendada == "POSTERGAR")

    return OperationalPurchaseRecommendationResult(
        fecha_calculo=optimizacion.fecha_base_calculo or date.today(),
        horizonte_meses=optimizacion.horizonte_meses,
        presupuesto_total=optimizacion.presupuesto_total,
        presupuesto_utilizado=optimizacion.presupuesto_utilizado,
        presupuesto_restante=optimizacion.presupuesto_restante,
        ahorro_total_estimado=optimizacion.ahorro_total_estimado,
        decision_resumen=(
            f"Comprar ahora {comprar_ahora} materiales, comprar parcialmente {compra_parcial} "
            f"y postergar {postergar}, respetando el presupuesto disponible."
        ),
        items=items,
        supuestos=(
            "Los precios futuros provienen del forecast del material y horizonte solicitados.",
            "El impacto economico positivo representa ahorro estimado por comprar ahora frente a postergar.",
            "La asignacion minimiza costo esperado respetando presupuesto, cantidades requeridas y no negatividad.",
            "Cada item incorpora la recomendacion simple y la mejor estrategia comparativa como trazabilidad de HU21 y HU22.",
        ),
        advertencias=optimizacion.advertencias,
    )
