"""Regenerate selector snapshots for bootstrap material IDs 1, 2, 3 from versioned sources.

For an existing database with other IDs or prices, use precompute_forecasts.
"""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.experiments.pricing.mvp_reassessment import MATERIAL_NAMES, frozen_prices, frozen_regressors
from app.modules.pricing.application.forecast_service import (
    ForecastExecutionPlan,
    _descripcion_regresores,
    _forecast_material,
    _selection_to_metadata,
    _validar_referencia_dataset,
    construir_firma_dataset,
)
from app.modules.pricing.application.forecasting import construir_dataset_prophet
from app.modules.pricing.application.model_selector import resolve_model_selection
from app.modules.pricing.application.series import PrecioSerieInput, construir_serie_mensual
from app.modules.pricing.infrastructure.forecast_runtime import configurar_cmdstan, importar_dependencias_forecast
from app.modules.pricing.infrastructure.forecast_snapshots import _serializar_result
from app.operations.bootstrap.import_membrana_megaflex import build_prices


def run():
    cmdstanpy, pd, Prophet, backend, base_backend = importar_dependencias_forecast()
    configurar_cmdstan(cmdstanpy, backend, base_backend)
    np.random.seed(42)
    prices = frozen_prices()
    prices["membrana-megaflex"] = [
        PrecioSerieInput(
            r.fecha, r.precio_normalizado, "kg", r.empresa, r.numero_comprobante, origen_dato=r.origen.upper()
        )
        for r in build_prices()
        if date(2022, 1, 1) <= r.fecha <= date.today()
    ]
    regressors = frozen_regressors()
    # Use the restored historical projection and conditional evaluation policy.
    snapshots = {}
    for material_id, (key, records) in enumerate(prices.items(), 1):
        material = SimpleNamespace(id=material_id, nombre=MATERIAL_NAMES[key], unidad_base="kg")
        series = construir_serie_mensual(records, material_nombre=material.nombre)
        dataset = construir_dataset_prophet(series, "precio_promedio_normalizado")
        signature = construir_firma_dataset(dataset)
        for horizon in range(1, 13):
            selection = resolve_model_selection(key, horizon)
            plan = ForecastExecutionPlan(
                selection.modelo,
                selection.regresores,
                regressors if selection.regresores else None,
                _descripcion_regresores(selection.regresores),
                _selection_to_metadata(selection),
                f"selector-on:{selection.modelo}",
            )
            result = replace(_forecast_material(material, horizon, dataset, pd, Prophet, plan), serie_mensual=series)
            result = _validar_referencia_dataset(result, key)
            snapshots[f"{material_id}:{horizon}:{signature}:{plan.cache_signature}"] = _serializar_result(result)
            print(f"Snapshot {key} {horizon} meses MAPE {result.metricas.mape}", flush=True)
    Path("tmp/forecast_snapshots.json").write_text(json.dumps(snapshots, ensure_ascii=True, indent=2) + "\n")


if __name__ == "__main__":
    run()
