"""Reevaluate the frozen MVP from versioned sources without a live database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import subprocess
from dataclasses import asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
from prophet import Prophet

from app.modules.pricing.application.backtesting import construir_folds_temporales
from app.modules.pricing.application.forecast_service import (
    FORECAST_EVALUATION_VERSION,
    construir_firma_dataset_ex_ante as construir_firma_dataset,
)
from app.modules.pricing.application.forecasting import BEST_PROPHET_CONFIG, construir_dataset_prophet
from app.modules.pricing.application.model_selector import MODEL_REGRESSORS
from app.modules.pricing.application.purchase_optimization import OptimizationCandidate, optimizar_compra_items
from app.modules.pricing.application.series import PrecioSerieInput, construir_serie_mensual
from app.modules.pricing.infrastructure.forecast_runtime import configurar_cmdstan, importar_dependencias_forecast
from app.modules.pricing.infrastructure.regressors import preparar_regresores_para_fold
from app.operations.bootstrap.import_cemento_canonico import read_canonical_csv, validate_rows
from app.operations.bootstrap.import_membrana_megaflex import build_prices as membrana_prices
from app.operations.bootstrap.import_pastina import grouped_prices as pastina_prices

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = date(2026, 3, 31)
START = date(2022, 1, 1)
OUTPUT = ROOT / "db/benchmarks/mvp_2026_03"
SOURCES = (
    "db/bootstrap/cemento_portland_historico.csv",
    "db/bootstrap/ipim_nivel_general_historico.csv",
    "tmp/experiments/icc_historico.csv",
    "tmp/experiments/cac_historico.csv",
    "app/operations/bootstrap/import_pastina.py",
    "app/operations/bootstrap/import_membrana_megaflex.py",
)
IMPLEMENTATION = (
    "app/modules/pricing/application/series.py",
    "app/modules/pricing/application/forecast_service.py",
    "app/modules/pricing/infrastructure/regressors.py",
    "app/experiments/pricing/mvp_reassessment.py",
    "app/experiments/pricing/mvp_snapshots.py",
)
MATERIAL_NAMES = {
    "cemento-portland": "Cemento Portland",
    "pastina": "Pastina",
    "membrana-megaflex": "Membrana Megaflex",
}
CANDIDATES = {
    "cemento-portland": ("prophet_base", "prophet_ipim_nivel_general", "prophet_ipim_icc_var_materials"),
    "pastina": ("prophet_base", "prophet_ipim_cac_labour_force", "prophet_ipim_cac_var_materials"),
    "membrana-megaflex": ("prophet_base", "prophet_ipim_icc_var_general", "prophet_ipim_icc_var_materials"),
}


def source_hashes() -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}


def frozen_prices() -> dict[str, list[PrecioSerieInput]]:
    cement = read_canonical_csv(ROOT / SOURCES[0])
    validate_rows(cement)
    rows = {
        "cemento-portland": [
            PrecioSerieInput(
                r.fecha, r.precio_normalizado, "kg", r.empresa, r.numero_comprobante, origen_dato=r.origen_dato
            )
            for r in cement
        ],
        "pastina": [
            PrecioSerieInput(
                r.fecha, r.precio_normalizado, "kg", r.empresa, r.numero_comprobante, origen_dato=r.origen.upper()
            )
            for r in pastina_prices()[0]
        ],
        "membrana-megaflex": [
            PrecioSerieInput(
                r.fecha, r.precio_normalizado, "kg", r.empresa, r.numero_comprobante, origen_dato=r.origen.upper()
            )
            for r in membrana_prices()
        ],
    }
    return {key: [r for r in values if START <= r.fecha <= CUTOFF] for key, values in rows.items()}


def frozen_regressors() -> pd.DataFrame:
    ipim = pd.read_csv(ROOT / SOURCES[1]).rename(columns={"date": "ds", "value": "ipim_nivel_general"})
    icc = pd.read_csv(ROOT / SOURCES[2]).rename(
        columns={"period": "ds", "var_general": "icc_var_general", "var_materials": "icc_var_materials"}
    )
    cac = pd.read_csv(ROOT / SOURCES[3]).rename(
        columns={"period": "ds", "labour_force": "cac_labour_force", "var_materials": "cac_var_materials"}
    )
    frame = (
        ipim[["ds", "ipim_nivel_general"]]
        .merge(icc[["ds", "icc_var_general", "icc_var_materials"]], on="ds")
        .merge(cac[["ds", "cac_labour_force", "cac_var_materials"]], on="ds")
    )
    frame["ds"] = pd.to_datetime(frame["ds"])
    return frame[frame["ds"].between(pd.Timestamp(START), pd.Timestamp(CUTOFF))].sort_values("ds")


def evaluate(dataset, regressors, horizon: int, model_name: str) -> tuple[dict, list[dict]]:
    columns = MODEL_REGRESSORS.get(model_name, ())
    fold_metrics = []
    predictions = []
    for fold in construir_folds_temporales(dataset, 24, horizon, horizon):
        train = pd.DataFrame({"ds": pd.to_datetime([r.ds for r in fold.train]), "y": [r.y for r in fold.train]})
        test = pd.DataFrame({"ds": pd.to_datetime([r.ds for r in fold.test]), "y": [r.y for r in fold.test]})
        if model_name == "naive_last":
            values = [float(train["y"].iloc[-1])] * len(test)
        else:
            model = Prophet(stan_backend="CMDSTANPY", **BEST_PROPHET_CONFIG, uncertainty_samples=0)
            if columns:
                train, future = preparar_regresores_para_fold(pd, train, test, regressors, columns)
                for column in columns:
                    model.add_regressor(column)
            else:
                future = test[["ds"]]
            model.fit(train, seed=42)
            predicted = model.predict(future)[["ds", "yhat"]]
            values = test[["ds"]].merge(predicted, on="ds", validate="one_to_one")["yhat"].tolist()
        errors = []
        apes = []
        for actual, predicted in zip(test.itertuples(index=False), values, strict=True):
            if actual.y <= 0 or not pd.notna(predicted):
                raise ValueError("Se requieren precios reales positivos y predicciones finitas.")
            error = abs(actual.y - predicted)
            ape = error / actual.y * 100
            errors.append(error)
            apes.append(ape)
            predictions.append(
                {
                    "fold": fold.indice,
                    "cutoff": fold.train[-1].ds.isoformat(),
                    "fecha": actual.ds.date().isoformat(),
                    "real": actual.y,
                    "prediccion": predicted,
                    "abs_error": error,
                    "ape": ape,
                }
            )
        fold_metrics.append({"mae": mean(errors), "mape": mean(apes)})
    return {
        "MAE": mean(r["mae"] for r in fold_metrics),
        "MAPE": mean(r["mape"] for r in fold_metrics),
        "folds": len(fold_metrics),
        "std_mape": pstdev(r["mape"] for r in fold_metrics),
    }, predictions


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cmdstanpy, _, _, backend, base_backend = importar_dependencias_forecast()
    configurar_cmdstan(cmdstanpy, backend, base_backend)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    prices = frozen_prices()
    regressors = frozen_regressors()
    hashes = source_hashes()
    summaries, predictions, anomalies = [], [], {}
    datasets = {}
    manifest = {
        "cutoff": CUTOFF.isoformat(),
        "start": START.isoformat(),
        "evaluation_protocol": FORECAST_EVALUATION_VERSION,
        "sources": hashes,
        "implementation": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in IMPLEMENTATION},
        "prophet_config": BEST_PROPHET_CONFIG,
        "seed": 42,
        "materials": {},
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("prophet", "cmdstanpy", "numpy", "pandas", "scikit-learn", "pulp")
        },
        "limitations": [
            "Los snapshots no conservan fechas de publicacion de indices: esta evaluacion elimina valores del test, pero no acredita disponibilidad publica point-in-time.",
            "La seleccion usa las mismas particiones evaluadas y es exploratoria, sin un test externo independiente.",
            "Pastina y Membrana contienen precios estimados; sus metricas validan el pipeline, no exactitud sobre precios reales independientes.",
            "Las pruebas del optimizador demuestran factibilidad, no ahorro economico realizado.",
        ],
    }
    manifest["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest["source_worktree_dirty"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    for key, records in prices.items():
        series = construir_serie_mensual(records, material_nombre=MATERIAL_NAMES[key])
        dataset = construir_dataset_prophet(series, "precio_promedio_normalizado")
        datasets[key] = dataset
        manifest["materials"][key] = {
            "records": len(records),
            "real": sum(r.origen_dato == "REAL" for r in records),
            "estimated": sum(r.origen_dato != "REAL" for r in records),
            "months": len(dataset),
            "dataset_signature": construir_firma_dataset(dataset),
        }
        anomalies[key] = [
            {
                "fecha": p.fecha.isoformat(),
                "precio": str(p.precio_promedio_normalizado),
                "esperado": str(p.precio_esperado_anomalia),
                "severidad": p.severidad_anomalia,
                "score": p.score_anomalia,
                "residuo_pct": str(p.residuo_anomalia_pct),
            }
            for p in series
            if p.es_anomalia
        ]
        for horizon in (3, 6, 12):
            for model_name in ("naive_last", *CANDIDATES[key]):
                metrics, rows = evaluate(dataset, regressors, horizon, model_name)
                common = {"material_key": key, "horizonte_meses": horizon, "nombre_modelo": model_name}
                summaries.append(
                    {
                        **common,
                        **metrics,
                        "evaluation_protocol": FORECAST_EVALUATION_VERSION,
                        "dataset_signature": construir_firma_dataset(dataset),
                        "cutoff": CUTOFF.isoformat(),
                        "real_only": key == "cemento-portland",
                    }
                )
                predictions.extend({**common, **row} for row in rows)
                print(
                    f"{key} h={horizon} {model_name}: MAPE={metrics['MAPE']:.2f}% folds={metrics['folds']}", flush=True
                )
    write_csv(output / "benchmarks.csv", summaries)
    write_csv(output / "fold_predictions.csv", predictions)
    manifest["benchmark_sha256"] = hashlib.sha256((output / "benchmarks.csv").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "anomalies.json").write_text(json.dumps(anomalies, indent=2, ensure_ascii=False) + "\n")
    # Controlled solver examples with explicit, illustrative inputs.
    cases = []
    for budget in (Decimal("15.00"), Decimal("120.00"), Decimal("1000.00")):
        candidate = OptimizationCandidate(
            1,
            "ejemplo_controlado",
            Decimal("1.0000"),
            Decimal("100000.00"),
            Decimal("120000.00"),
            Decimal("20000.00"),
            "alta",
            Decimal("3"),
            "no_calibrada",
            True,
        )
        result = optimizar_compra_items(presupuesto_total=budget, horizonte_meses=3, candidates=[candidate])
        assert result.presupuesto_utilizado <= budget
        assert all(
            p.cantidad_recomendada_comprar_ahora + p.cantidad_recomendada_postergar == p.cantidad_objetivo
            for p in result.items
        )
        cases.append(asdict(result))
    (output / "optimization_cases.json").write_text(json.dumps(cases, indent=2, default=str) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
