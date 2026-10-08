import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from app.modules.pricing.application.forecast_cache import ForecastCacheKey
from app.modules.pricing.application.forecasting import ProphetRow
from app.modules.pricing.interfaces.schemas import ForecastMetricasRead, ForecastPuntoRead, ForecastSelectionRead
from app.shared.config.settings import settings


class ForecastMaterialResultProtocol:
    dataset: list
    serie_mensual: list | None
    metricas: ForecastMetricasRead
    forecast: list[ForecastPuntoRead]


def _snapshot_path() -> Path:
    return Path(settings.forecast_snapshot_path)


def _leer_snapshots() -> dict:
    path = _snapshot_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _escribir_snapshots(data: dict) -> None:
    path = _snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _serializar_key(cache_key: ForecastCacheKey) -> str:
    return f"{cache_key.material_id}:{cache_key.horizonte_meses}:{cache_key.dataset_signature}"


def _serializar_punto_mensual(punto) -> dict:
    return {
        "fecha": punto.fecha.isoformat(),
        "precio_promedio_normalizado": str(punto.precio_promedio_normalizado),
        "unidad_base": punto.unidad_base,
        "precio_equivalente_25kg": None if punto.precio_equivalente_25kg is None else str(punto.precio_equivalente_25kg),
        "precio_equivalente_50kg": None if punto.precio_equivalente_50kg is None else str(punto.precio_equivalente_50kg),
        "cantidad_registros": punto.cantidad_registros,
        "cantidad_facturas": punto.cantidad_facturas,
        "fuentes": list(punto.fuentes),
        "origenes_dato": list(getattr(punto, "origenes_dato", None) or []),
        "variacion_porcentual_anterior": None if punto.variacion_porcentual_anterior is None else str(punto.variacion_porcentual_anterior),
        "es_anomalia": punto.es_anomalia,
        "severidad_anomalia": getattr(punto, "severidad_anomalia", None),
        "score_anomalia": getattr(punto, "score_anomalia", None),
        "confianza_anomalia": None if getattr(punto, "confianza_anomalia", None) is None else str(punto.confianza_anomalia),
        "motivo_anomalia": punto.motivo_anomalia,
        "precio_esperado_anomalia": None if getattr(punto, "precio_esperado_anomalia", None) is None else str(punto.precio_esperado_anomalia),
        "residuo_anomalia_pct": None if getattr(punto, "residuo_anomalia_pct", None) is None else str(punto.residuo_anomalia_pct),
        "limite_residuo_anomalia_pct": None if getattr(punto, "limite_residuo_anomalia_pct", None) is None else str(punto.limite_residuo_anomalia_pct),
        "rango_esperado_min_anomalia": None if getattr(punto, "rango_esperado_min_anomalia", None) is None else str(punto.rango_esperado_min_anomalia),
        "rango_esperado_max_anomalia": None if getattr(punto, "rango_esperado_max_anomalia", None) is None else str(punto.rango_esperado_max_anomalia),
        "tipo_anomalia": getattr(punto, "tipo_anomalia", None),
        "explicacion_anomalia": getattr(punto, "explicacion_anomalia", None),
        "variables_relevantes_anomalia": getattr(punto, "variables_relevantes_anomalia", None),
    }


def _serializar_result(result: ForecastMaterialResultProtocol) -> dict:
    return {
        "dataset": [{"ds": fila.ds.isoformat(), "y": fila.y} for fila in result.dataset],
        "serie_mensual": [_serializar_punto_mensual(punto) for punto in getattr(result, "serie_mensual", None) or []],
        "metricas": result.metricas.model_dump(mode="json"),
        "forecast": [punto.model_dump(mode="json") for punto in result.forecast],
        "modelo": getattr(result, "modelo", None),
        "supuesto_regresores": getattr(result, "supuesto_regresores", None),
        "seleccion_modelo": result.seleccion_modelo.model_dump(mode="json") if getattr(result, "seleccion_modelo", None) else None,
    }


def _deserializar_result(data: dict):
    from app.modules.pricing.application.forecast_service import (
        FORECAST_MODEL_NAME,
        FORECAST_REGRESSOR_NOTE,
        ForecastMaterialResult,
    )
    from app.modules.pricing.application.series import PuntoSeriePrecio

    return ForecastMaterialResult(
        dataset=[ProphetRow(ds=date.fromisoformat(fila["ds"]), y=float(fila["y"])) for fila in data["dataset"]],
        serie_mensual=[
            PuntoSeriePrecio(
                fecha=date.fromisoformat(punto["fecha"]),
                precio_promedio_normalizado=Decimal(punto["precio_promedio_normalizado"]),
                unidad_base=punto["unidad_base"],
                precio_equivalente_25kg=None if punto.get("precio_equivalente_25kg") is None else Decimal(punto["precio_equivalente_25kg"]),
                precio_equivalente_50kg=None if punto.get("precio_equivalente_50kg") is None else Decimal(punto["precio_equivalente_50kg"]),
                cantidad_registros=int(punto["cantidad_registros"]),
                cantidad_facturas=int(punto["cantidad_facturas"]),
                fuentes=list(punto["fuentes"]),
                origenes_dato=list(punto.get("origenes_dato", [])),
                variacion_porcentual_anterior=None if punto.get("variacion_porcentual_anterior") is None else Decimal(punto["variacion_porcentual_anterior"]),
                es_anomalia=bool(punto.get("es_anomalia", False)),
                severidad_anomalia=punto.get("severidad_anomalia"),
                score_anomalia=int(punto["score_anomalia"]) if punto.get("score_anomalia") is not None else None,
                confianza_anomalia=None if punto.get("confianza_anomalia") is None else Decimal(punto["confianza_anomalia"]),
                motivo_anomalia=punto.get("motivo_anomalia"),
                precio_esperado_anomalia=None if punto.get("precio_esperado_anomalia") is None else Decimal(punto["precio_esperado_anomalia"]),
                residuo_anomalia_pct=None if punto.get("residuo_anomalia_pct") is None else Decimal(punto["residuo_anomalia_pct"]),
                limite_residuo_anomalia_pct=None if punto.get("limite_residuo_anomalia_pct") is None else Decimal(punto["limite_residuo_anomalia_pct"]),
                rango_esperado_min_anomalia=None if punto.get("rango_esperado_min_anomalia") is None else Decimal(punto["rango_esperado_min_anomalia"]),
                rango_esperado_max_anomalia=None if punto.get("rango_esperado_max_anomalia") is None else Decimal(punto["rango_esperado_max_anomalia"]),
                tipo_anomalia=punto.get("tipo_anomalia"),
                explicacion_anomalia=punto.get("explicacion_anomalia"),
                variables_relevantes_anomalia=punto.get("variables_relevantes_anomalia"),
            )
            for punto in data.get("serie_mensual", [])
        ],
        metricas=ForecastMetricasRead.model_validate(data["metricas"]),
        forecast=[ForecastPuntoRead.model_validate(punto) for punto in data["forecast"]],
        modelo=data.get("modelo") or FORECAST_MODEL_NAME,
        supuesto_regresores=data.get("supuesto_regresores") or FORECAST_REGRESSOR_NOTE,
        seleccion_modelo=ForecastSelectionRead.model_validate(data["seleccion_modelo"]) if data.get("seleccion_modelo") else None,
    )


def cargar_forecast_snapshot(cache_key: ForecastCacheKey):
    snapshots = _leer_snapshots()
    payload = snapshots.get(_serializar_key(cache_key))
    if payload is None:
        return None
    return _deserializar_result(payload)


def guardar_forecast_snapshot(cache_key: ForecastCacheKey, result: ForecastMaterialResultProtocol) -> None:
    snapshots = _leer_snapshots()
    snapshots[_serializar_key(cache_key)] = _serializar_result(result)
    _escribir_snapshots(snapshots)
