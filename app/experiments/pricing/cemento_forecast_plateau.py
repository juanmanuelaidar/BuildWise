from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from app.experiments.pricing.common import (
    DEFAULT_DATASET_START,
    cargar_indice_externo_mensual,
    configurar_cmdstan,
    construir_dataset_material,
    importar_dependencias_prophet,
)
from app.modules.pricing.application.backtesting import TimeSeriesFold, construir_folds_temporales
from app.modules.pricing.application.forecasting import BEST_PROPHET_CONFIG, ProphetRow


DEFAULT_MATERIAL_NAME = "Cemento Portland"
TARGET_NAME = "precio_promedio_normalizado"
IPIM_NIVEL_GENERAL_SERIES_ID = "448.1_NIVEL_GENERAL_0_0_13_46"
BASELINE_NAME = "prophet_ipim_nivel_general"
DEFAULT_IPIM_SNAPSHOT_CSV = Path("db/bootstrap/ipim_nivel_general_historico.csv")
DEFAULT_CAC_CSV = Path("tmp/experiments/cac_historico.csv")
DEFAULT_ICC_CSV = Path("tmp/experiments/icc_historico.csv")

LAG_COLUMNS = ("lag_1m", "lag_3m", "lag_6m")
MOVING_AVERAGE_COLUMNS = ("ma_3m", "ma_6m")
VARIATION_COLUMNS = ("var_1m_pct", "var_3m_pct", "var_6m_pct")
XGBOOST_TEMPORAL_COLUMNS = (
    "ipim_nivel_general",
    *LAG_COLUMNS,
    *MOVING_AVERAGE_COLUMNS,
    *VARIATION_COLUMNS,
    "month_sin",
    "month_cos",
    "time_idx",
)


@dataclass(frozen=True)
class FoldMetric:
    fold_indice: int
    mae: float
    mape: float


@dataclass(frozen=True)
class FoldPrediction:
    fold_indice: int
    ds: object
    actual: float
    yhat: float


@dataclass(frozen=True)
class ModelRunResult:
    nombre_modelo: str
    regresores_features: str
    horizonte_meses: int
    mae: float | None
    mape: float | None
    folds: int
    observaciones: str
    mejora_vs_baseline: str
    fold_metrics: tuple[FoldMetric, ...]
    predictions: tuple[FoldPrediction, ...]
    skipped: bool = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bloque experimental de amesetamiento de forecasting por material.",
    )
    parser.add_argument(
        "--material",
        default=DEFAULT_MATERIAL_NAME,
        help="Nombre exacto del material a evaluar.",
    )
    parser.add_argument(
        "--horizontes",
        nargs="+",
        type=int,
        default=[3],
        help="Horizontes del backtesting temporal. No mezclar resultados como equivalentes.",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=24,
        help="Tamaño minimo del tramo de entrenamiento para construir folds temporales.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Ruta donde se exportara la tabla comparativa en CSV.",
    )
    parser.add_argument(
        "--modelos",
        nargs="*",
        default=None,
        help="Subset opcional de modelos a ejecutar. Si se omite, corre todos los experimentos.",
    )
    parser.add_argument(
        "--sin-ensemble",
        action="store_true",
        help="No construir ensemble entre los dos mejores modelos disponibles.",
    )
    parser.add_argument(
        "--cac-csv",
        type=Path,
        default=None,
        help=(
            "CSV local opcional de CAC con columnas period,general,materials,labour_force "
            "y/o sus variaciones. Si se informa, se habilitan experimentos con CAC."
        ),
    )
    parser.add_argument(
        "--icc-csv",
        type=Path,
        default=None,
        help=(
            "CSV local opcional de ICC con columnas period,var_general,var_materials,var_labour "
            "y/o niveles. Si se informa, se habilitan experimentos con ICC."
        ),
    )
    return parser


def _importar_dependencias():
    pd, Prophet, CmdStanPyBackend, IStanBackend = importar_dependencias_prophet(
        "Faltan dependencias para correr el experimento experimental de amesetamiento.",
    )
    configurar_cmdstan(
        CmdStanPyBackend,
        IStanBackend,
        "No se encontro CmdStan en ~/.cmdstan/cmdstan-2.38.0.",
    )
    return pd, Prophet


def _material_slug(material_name: str) -> str:
    slug = material_name.lower()
    for source, target in (
        (" ", "_"),
        ("/", "_"),
        ("-", "_"),
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
    ):
        slug = slug.replace(source, target)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _default_output_csv_for_material(material_name: str) -> Path:
    return Path(f"tmp/experiments/{_material_slug(material_name)}_forecast_plateau.csv")


def _load_base_dataframe(pd, material_name: str):
    dataset = construir_dataset_material(
        material_name,
        frecuencia="mensual",
        objetivo=TARGET_NAME,
        dataset_start=DEFAULT_DATASET_START,
    )
    y_df = pd.DataFrame([{"ds": pd.to_datetime(row.ds), "y": row.y} for row in dataset])
    try:
        ipim_df = cargar_indice_externo_mensual(
            pd,
            series_id=IPIM_NIVEL_GENERAL_SERIES_ID,
            columna_salida="ipim_nivel_general",
            dataset_start=DEFAULT_DATASET_START,
        )
    except RuntimeError:
        ipim_df = _load_ipim_snapshot(pd)
    base_df = y_df.merge(ipim_df, on="ds", how="inner").sort_values("ds").reset_index(drop=True)
    if base_df.empty:
        raise RuntimeError(f"No hay dataset mensual disponible para {material_name} con IPIM.")
    return base_df


def _load_ipim_snapshot(pd):
    csv_path = DEFAULT_IPIM_SNAPSHOT_CSV
    if not csv_path.exists():
        raise RuntimeError(
            "No hay valores de IPIM en la base ni snapshot local disponible en "
            f"{csv_path}."
        )

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["date"] >= pd.to_datetime(DEFAULT_DATASET_START)].copy()
    return df.rename(columns={"date": "ds", "value": "ipim_nivel_general"})[["ds", "ipim_nivel_general"]]


def _load_cac_dataframe(pd, csv_path: Path):
    if not csv_path.exists():
        raise RuntimeError(f"No existe el CSV de CAC en {csv_path}.")

    df = pd.read_csv(csv_path)
    required_one_of = {"period", "date", "ds"}
    if not any(column in df.columns for column in required_one_of):
        raise RuntimeError("El CSV de CAC debe incluir una columna temporal: period, date o ds.")

    if "period" in df.columns:
        df["ds"] = pd.to_datetime(df["period"])
    elif "date" in df.columns:
        df["ds"] = pd.to_datetime(df["date"])
    else:
        df["ds"] = pd.to_datetime(df["ds"])

    supported_columns = {
        "general",
        "materials",
        "labour_force",
        "var_general",
        "var_materials",
        "var_labour",
    }
    available = [column for column in df.columns if column in supported_columns]
    if not available:
        raise RuntimeError("El CSV de CAC no contiene columnas soportadas para experimento.")

    for column in available:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df["ds"] >= pd.to_datetime(DEFAULT_DATASET_START)].copy()
    return df[["ds", *available]].sort_values("ds").reset_index(drop=True)


def _load_icc_dataframe(pd, csv_path: Path):
    if not csv_path.exists():
        raise RuntimeError(f"No existe el CSV de ICC en {csv_path}.")

    df = pd.read_csv(csv_path)
    required_one_of = {"period", "date", "ds"}
    if not any(column in df.columns for column in required_one_of):
        raise RuntimeError("El CSV de ICC debe incluir una columna temporal: period, date o ds.")

    if "period" in df.columns:
        df["ds"] = pd.to_datetime(df["period"])
    elif "date" in df.columns:
        df["ds"] = pd.to_datetime(df["date"])
    else:
        df["ds"] = pd.to_datetime(df["ds"])

    supported_columns = {
        "general": "icc_general",
        "materials": "icc_materials",
        "labour_force": "icc_labour_force",
        "var_general": "icc_var_general",
        "var_materials": "icc_var_materials",
        "var_labour": "icc_var_labour",
    }
    rename_map = {column: alias for column, alias in supported_columns.items() if column in df.columns}
    if not rename_map:
        raise RuntimeError("El CSV de ICC no contiene columnas soportadas para experimento.")

    for column in rename_map:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.rename(columns=rename_map)
    df = df[df["ds"] >= pd.to_datetime(DEFAULT_DATASET_START)].copy()
    return df[["ds", *rename_map.values()]].sort_values("ds").reset_index(drop=True)


def _to_dataset_rows(base_df) -> list[ProphetRow]:
    return [ProphetRow(ds=row.ds.date(), y=float(row.y)) for row in base_df.itertuples(index=False)]


def _fold_to_frames(pd, base_df, fold: TimeSeriesFold):
    train_dates = {pd.to_datetime(item.ds) for item in fold.train}
    test_dates = {pd.to_datetime(item.ds) for item in fold.test}
    train_df = base_df[base_df["ds"].isin(train_dates)].copy().sort_values("ds").reset_index(drop=True)
    test_df = base_df[base_df["ds"].isin(test_dates)].copy().sort_values("ds").reset_index(drop=True)
    return train_df, test_df


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _build_feature_snapshot(history: list[float]) -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {
        "lag_1m": history[-1] if len(history) >= 1 else None,
        "lag_3m": history[-3] if len(history) >= 3 else None,
        "lag_6m": history[-6] if len(history) >= 6 else None,
        "ma_3m": mean(history[-3:]) if len(history) >= 3 else None,
        "ma_6m": mean(history[-6:]) if len(history) >= 6 else None,
        "var_1m_pct": None,
        "var_3m_pct": None,
        "var_6m_pct": None,
    }

    var_1 = _safe_div(history[-1], history[-2]) if len(history) >= 2 else None
    var_3 = _safe_div(history[-1], history[-4]) if len(history) >= 4 else None
    var_6 = _safe_div(history[-1], history[-7]) if len(history) >= 7 else None
    snapshot["var_1m_pct"] = (var_1 - 1) * 100 if var_1 is not None else None
    snapshot["var_3m_pct"] = (var_3 - 1) * 100 if var_3 is not None else None
    snapshot["var_6m_pct"] = (var_6 - 1) * 100 if var_6 is not None else None
    return snapshot


def _build_training_feature_frame(pd, train_df, feature_columns: tuple[str, ...]):
    rows: list[dict[str, float | object]] = []
    history: list[float] = []
    for row in train_df.itertuples(index=False):
        snapshot = _build_feature_snapshot(history)
        payload = {
            "ds": row.ds,
            "y": float(row.y),
            "ipim_nivel_general": float(row.ipim_nivel_general),
        }
        payload.update(snapshot)
        rows.append(payload)
        history.append(float(row.y))

    frame = pd.DataFrame(rows)
    if feature_columns:
        frame = frame.dropna(subset=list(feature_columns)).reset_index(drop=True)
    return frame


def _month_features(timestamp) -> tuple[float, float]:
    angle = (timestamp.month - 1) / 12 * 2 * math.pi
    return math.sin(angle), math.cos(angle)


def _predict_prophet_baseline(pd, Prophet, train_df, test_df):
    model = Prophet(stan_backend="CMDSTANPY", **BEST_PROPHET_CONFIG)
    model.add_regressor("ipim_nivel_general")
    train_reg = train_df[["ds", "y", "ipim_nivel_general"]].copy()
    future = pd.concat(
        [
            train_df[["ds", "ipim_nivel_general"]],
            test_df[["ds", "ipim_nivel_general"]],
        ],
        ignore_index=True,
    )
    model.fit(train_reg)
    forecast = model.predict(future)[["ds", "yhat"]]
    return test_df.merge(forecast, on="ds", how="left")


def _predict_prophet_with_exogenous_columns(pd, Prophet, train_df, test_df, regressors: tuple[str, ...]):
    model = Prophet(stan_backend="CMDSTANPY", **BEST_PROPHET_CONFIG)
    for regressor in regressors:
        model.add_regressor(regressor)

    train_reg = train_df[["ds", "y", *regressors]].copy()
    future = pd.concat(
        [
            train_df[["ds", *regressors]],
            test_df[["ds", *regressors]],
        ],
        ignore_index=True,
    )
    model.fit(train_reg)
    forecast = model.predict(future)[["ds", "yhat"]]
    return test_df.merge(forecast, on="ds", how="left")


def _predict_prophet_with_recursive_features(pd, Prophet, train_df, test_df, feature_columns: tuple[str, ...]):
    train_features = _build_training_feature_frame(pd, train_df, feature_columns)
    if train_features.empty:
        raise RuntimeError("No quedaron filas suficientes para entrenar Prophet con features autoregresivas.")

    model = Prophet(stan_backend="CMDSTANPY", **BEST_PROPHET_CONFIG)
    regressors = ("ipim_nivel_general", *feature_columns)
    for regressor in regressors:
        model.add_regressor(regressor)

    model.fit(train_features[["ds", "y", *regressors]].copy())

    history = [float(value) for value in train_df["y"].tolist()]
    predictions: list[dict[str, float | object]] = []
    for offset, row in enumerate(test_df.itertuples(index=False), start=1):
        snapshot = _build_feature_snapshot(history)
        payload = {
            "ds": row.ds,
            "ipim_nivel_general": float(row.ipim_nivel_general),
        }
        for feature in feature_columns:
            payload[feature] = snapshot[feature]
        future_row = pd.DataFrame([payload])
        if future_row[list(regressors)].isna().to_numpy().any():
            raise RuntimeError("No fue posible construir features autoregresivas para el paso futuro.")
        yhat = float(model.predict(future_row)[["yhat"]].iloc[0, 0])
        predictions.append({"ds": row.ds, "yhat": yhat})
        history.append(yhat)

    forecast = pd.DataFrame(predictions)
    return test_df.merge(forecast, on="ds", how="left")


def _import_statsmodels():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        return None
    return SARIMAX


def _predict_sarimax_with_ipim(pd, train_df, test_df):
    SARIMAX = _import_statsmodels()
    if SARIMAX is None:
        raise ModuleNotFoundError("statsmodels no esta instalado")

    candidate_orders = ((1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 1))
    best_result = None
    best_order = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in candidate_orders:
            try:
                fitted = SARIMAX(
                    train_df["y"].astype(float),
                    exog=train_df[["ipim_nivel_general"]].astype(float),
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            except Exception:
                continue
            if best_result is None or fitted.aic < best_result.aic:
                best_result = fitted
                best_order = order

    if best_result is None or best_order is None:
        raise RuntimeError("SARIMAX no encontro una configuracion estable en la grilla experimental.")

    forecast_values = best_result.get_forecast(
        steps=len(test_df),
        exog=test_df[["ipim_nivel_general"]].astype(float),
    ).predicted_mean
    forecast = pd.DataFrame({"ds": test_df["ds"], "yhat": forecast_values})
    return test_df.merge(forecast, on="ds", how="left"), best_order


def _import_xgboost():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor


def _build_xgboost_frame(pd, source_df):
    rows: list[dict[str, float | object]] = []
    history: list[float] = []
    for idx, row in enumerate(source_df.itertuples(index=False)):
        snapshot = _build_feature_snapshot(history)
        month_sin, month_cos = _month_features(row.ds)
        payload = {
            "ds": row.ds,
            "y": float(row.y),
            "ipim_nivel_general": float(row.ipim_nivel_general),
            "month_sin": month_sin,
            "month_cos": month_cos,
            "time_idx": float(idx),
        }
        payload.update(snapshot)
        rows.append(payload)
        history.append(float(row.y))

    frame = pd.DataFrame(rows)
    return frame.dropna(subset=list(XGBOOST_TEMPORAL_COLUMNS)).reset_index(drop=True)


def _predict_xgboost_temporal(pd, train_df, test_df):
    XGBRegressor = _import_xgboost()
    if XGBRegressor is None:
        raise ModuleNotFoundError("xgboost no esta instalado")

    train_frame = _build_xgboost_frame(pd, train_df)
    if train_frame.empty:
        raise RuntimeError("No quedaron filas suficientes para entrenar XGBoost temporal.")

    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(train_frame[list(XGBOOST_TEMPORAL_COLUMNS)], train_frame["y"])

    history = [float(value) for value in train_df["y"].tolist()]
    predictions: list[dict[str, float | object]] = []
    for offset, row in enumerate(test_df.itertuples(index=False), start=1):
        snapshot = _build_feature_snapshot(history)
        month_sin, month_cos = _month_features(row.ds)
        payload = {
            "ipim_nivel_general": float(row.ipim_nivel_general),
            "month_sin": month_sin,
            "month_cos": month_cos,
            "time_idx": float(len(train_df) + offset - 1),
        }
        payload.update(snapshot)
        feature_row = pd.DataFrame([payload])
        if feature_row[list(XGBOOST_TEMPORAL_COLUMNS)].isna().to_numpy().any():
            raise RuntimeError("No fue posible construir features temporales para XGBoost.")
        yhat = float(model.predict(feature_row[list(XGBOOST_TEMPORAL_COLUMNS)])[0])
        predictions.append({"ds": row.ds, "yhat": yhat})
        history.append(yhat)

    forecast = pd.DataFrame(predictions)
    return test_df.merge(forecast, on="ds", how="left")


def _evaluate_predictions(evaluation_df, fold_indice: int):
    evaluation = evaluation_df.copy()
    evaluation["abs_error"] = (evaluation["y"] - evaluation["yhat"]).abs()
    evaluation["ape"] = evaluation["abs_error"] / evaluation["y"] * 100
    fold_metric = FoldMetric(
        fold_indice=fold_indice,
        mae=float(evaluation["abs_error"].mean()),
        mape=float(evaluation["ape"].mean()),
    )
    predictions = tuple(
        FoldPrediction(
            fold_indice=fold_indice,
            ds=row.ds,
            actual=float(row.y),
            yhat=float(row.yhat),
        )
        for row in evaluation.itertuples(index=False)
    )
    return fold_metric, predictions


def _summarize_model_run(
    nombre_modelo: str,
    regresores_features: str,
    horizonte_meses: int,
    fold_metrics: list[FoldMetric],
    predictions: list[FoldPrediction],
    *,
    baseline_result: ModelRunResult | None,
    extra_observations: str = "",
) -> ModelRunResult:
    mae = mean(metric.mae for metric in fold_metrics)
    mape = mean(metric.mape for metric in fold_metrics)
    fold_mape_std = pstdev(metric.mape for metric in fold_metrics) if len(fold_metrics) > 1 else 0.0
    fold_mae_std = pstdev(metric.mae for metric in fold_metrics) if len(fold_metrics) > 1 else 0.0

    observations = [
        f"std_mape_folds={fold_mape_std:.2f}",
        f"std_mae_folds={fold_mae_std:.2f}",
    ]
    improvement = "n/a"
    if baseline_result is not None and baseline_result.mape is not None:
        improvement_value = baseline_result.mape - mape
        improvement = f"{improvement_value:+.2f} pp"
        baseline_std = pstdev(metric.mape for metric in baseline_result.fold_metrics) if len(baseline_result.fold_metrics) > 1 else 0.0
        if improvement_value > 0 and baseline_std > 0 and fold_mape_std > baseline_std * 1.5:
            observations.append(
                f"mejora_en_mape_pero_peor_estabilidad_vs_baseline({fold_mape_std:.2f}>{baseline_std * 1.5:.2f})"
            )

    if extra_observations:
        observations.append(extra_observations)

    return ModelRunResult(
        nombre_modelo=nombre_modelo,
        regresores_features=regresores_features,
        horizonte_meses=horizonte_meses,
        mae=mae,
        mape=mape,
        folds=len(fold_metrics),
        observaciones="; ".join(observations),
        mejora_vs_baseline=improvement,
        fold_metrics=tuple(fold_metrics),
        predictions=tuple(predictions),
    )


def _skip_result(
    nombre_modelo: str,
    regresores_features: str,
    horizonte_meses: int,
    observaciones: str,
) -> ModelRunResult:
    return ModelRunResult(
        nombre_modelo=nombre_modelo,
        regresores_features=regresores_features,
        horizonte_meses=horizonte_meses,
        mae=None,
        mape=None,
        folds=0,
        observaciones=observaciones,
        mejora_vs_baseline="skip",
        fold_metrics=(),
        predictions=(),
        skipped=True,
    )


def _run_model_over_folds(
    pd,
    Prophet,
    base_df,
    folds: list[TimeSeriesFold],
    *,
    nombre_modelo: str,
    regresores_features: str,
    horizonte_meses: int,
    predictor: Callable,
    baseline_result: ModelRunResult | None = None,
) -> ModelRunResult:
    fold_metrics: list[FoldMetric] = []
    predictions: list[FoldPrediction] = []
    extra_notes: list[str] = []

    for fold in folds:
        train_df, test_df = _fold_to_frames(pd, base_df, fold)
        output = predictor(pd, Prophet, train_df, test_df)
        fold_note = ""
        if isinstance(output, tuple):
            evaluation_df, metadata = output
            fold_note = f"fold_{fold.indice}_meta={metadata}"
        else:
            evaluation_df = output
        metric, fold_predictions = _evaluate_predictions(evaluation_df, fold.indice)
        fold_metrics.append(metric)
        predictions.extend(fold_predictions)
        if fold_note:
            extra_notes.append(fold_note)

    return _summarize_model_run(
        nombre_modelo,
        regresores_features,
        horizonte_meses,
        fold_metrics,
        predictions,
        baseline_result=baseline_result,
        extra_observations=" | ".join(extra_notes),
    )


def _build_ensemble_result(
    nombre_modelo: str,
    horizonte_meses: int,
    source_a: ModelRunResult,
    source_b: ModelRunResult,
    baseline_result: ModelRunResult,
):
    map_a = {(item.fold_indice, item.ds): item for item in source_a.predictions}
    map_b = {(item.fold_indice, item.ds): item for item in source_b.predictions}
    common_keys = sorted(set(map_a).intersection(map_b), key=lambda item: (item[0], item[1]))
    if not common_keys:
        return _skip_result(
            nombre_modelo,
            f"promedio({source_a.nombre_modelo}, {source_b.nombre_modelo})",
            horizonte_meses,
            "No hubo predicciones alineables para construir ensemble.",
        )

    fold_metrics: list[FoldMetric] = []
    predictions: list[FoldPrediction] = []
    keys_by_fold: dict[int, list[tuple[int, object]]] = {}
    for key in common_keys:
        keys_by_fold.setdefault(key[0], []).append(key)

    for fold_indice, fold_keys in keys_by_fold.items():
        rows = []
        for key in fold_keys:
            pred_a = map_a[key]
            pred_b = map_b[key]
            yhat = (pred_a.yhat + pred_b.yhat) / 2
            rows.append({"ds": pred_a.ds, "y": pred_a.actual, "yhat": yhat})
        metric, fold_predictions = _evaluate_predictions(__import__("pandas").DataFrame(rows), fold_indice)
        fold_metrics.append(metric)
        predictions.extend(fold_predictions)

    return _summarize_model_run(
        nombre_modelo,
        f"promedio({source_a.nombre_modelo}, {source_b.nombre_modelo})",
        horizonte_meses,
        fold_metrics,
        predictions,
        baseline_result=baseline_result,
        extra_observations=f"ensemble_simple_top2={source_a.nombre_modelo}+{source_b.nombre_modelo}",
    )


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _write_report_csv(rows: list[ModelRunResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nombre_modelo",
                "regresores_features",
                "horizonte_meses",
                "MAE",
                "MAPE",
                "folds",
                "observaciones",
                "mejora_vs_baseline",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.nombre_modelo,
                    row.regresores_features,
                    row.horizonte_meses,
                    _format_metric(row.mae),
                    _format_metric(row.mape),
                    row.folds,
                    row.observaciones,
                    row.mejora_vs_baseline,
                ]
            )


def _print_table(rows: list[ModelRunResult]) -> None:
    headers = [
        "nombre_modelo",
        "regresores/features",
        "horizonte_meses",
        "MAE",
        "MAPE",
        "folds",
        "observaciones",
        "mejora_vs_baseline",
    ]
    data = [
        [
            row.nombre_modelo,
            row.regresores_features,
            str(row.horizonte_meses),
            _format_metric(row.mae),
            _format_metric(row.mape),
            str(row.folds),
            row.observaciones,
            row.mejora_vs_baseline,
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for line in data:
        for idx, value in enumerate(line):
            widths[idx] = max(widths[idx], len(value))

    def render_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    print(render_row(headers))
    print(separator)
    for line in data:
        print(render_row(line))


def _available_models(cac_df, icc_df):
    registry = {
        BASELINE_NAME: {
            "regresores_features": "ipim_nivel_general",
            "runner": lambda pd, Prophet, train_df, test_df: _predict_prophet_baseline(pd, Prophet, train_df, test_df),
        },
        "prophet_ipim_nivel_general_lags": {
            "regresores_features": "ipim_nivel_general + lag_1m + lag_3m + lag_6m",
            "runner": lambda pd, Prophet, train_df, test_df: _predict_prophet_with_recursive_features(
                pd,
                Prophet,
                train_df,
                test_df,
                LAG_COLUMNS,
            ),
        },
        "prophet_ipim_nivel_general_medias_moviles": {
            "regresores_features": "ipim_nivel_general + ma_3m + ma_6m",
            "runner": lambda pd, Prophet, train_df, test_df: _predict_prophet_with_recursive_features(
                pd,
                Prophet,
                train_df,
                test_df,
                MOVING_AVERAGE_COLUMNS,
            ),
        },
        "prophet_ipim_nivel_general_variaciones": {
            "regresores_features": "ipim_nivel_general + var_1m_pct + var_3m_pct + var_6m_pct",
            "runner": lambda pd, Prophet, train_df, test_df: _predict_prophet_with_recursive_features(
                pd,
                Prophet,
                train_df,
                test_df,
                VARIATION_COLUMNS,
            ),
        },
        "sarimax_ipim_nivel_general": {
            "regresores_features": "ipim_nivel_general",
            "runner": lambda pd, Prophet, train_df, test_df: _predict_sarimax_with_ipim(pd, train_df, test_df),
        },
        "xgboost_temporal_ipim": {
            "regresores_features": "ipim_nivel_general + lags + medias_moviles + variaciones + month_sin/cos + time_idx",
            "runner": lambda pd, Prophet, train_df, test_df: _predict_xgboost_temporal(pd, train_df, test_df),
        },
    }
    if cac_df is not None:
        cac_columns = [
            column
            for column in ("general", "materials", "labour_force", "var_general", "var_materials", "var_labour")
            if column in cac_df.columns
        ]
        for column in cac_columns:
            registry[f"prophet_cac_{column}"] = {
                "regresores_features": column,
                "runner": lambda pd, Prophet, train_df, test_df, col=column: _predict_prophet_with_exogenous_columns(
                    pd,
                    Prophet,
                    train_df,
                    test_df,
                    (col,),
                ),
            }
            registry[f"prophet_ipim_cac_{column}"] = {
                "regresores_features": f"ipim_nivel_general + {column}",
                "runner": lambda pd, Prophet, train_df, test_df, col=column: _predict_prophet_with_exogenous_columns(
                    pd,
                    Prophet,
                    train_df,
                    test_df,
                    ("ipim_nivel_general", col),
                ),
            }

    if icc_df is not None:
        icc_columns = [
            column
            for column in (
                "icc_general",
                "icc_materials",
                "icc_labour_force",
                "icc_var_general",
                "icc_var_materials",
                "icc_var_labour",
            )
            if column in icc_df.columns
        ]
        for column in icc_columns:
            registry[f"prophet_{column}"] = {
                "regresores_features": column,
                "runner": lambda pd, Prophet, train_df, test_df, col=column: _predict_prophet_with_exogenous_columns(
                    pd,
                    Prophet,
                    train_df,
                    test_df,
                    (col,),
                ),
            }
            registry[f"prophet_ipim_{column}"] = {
                "regresores_features": f"ipim_nivel_general + {column}",
                "runner": lambda pd, Prophet, train_df, test_df, col=column: _predict_prophet_with_exogenous_columns(
                    pd,
                    Prophet,
                    train_df,
                    test_df,
                    ("ipim_nivel_general", col),
                ),
            }

    return registry


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    pd, Prophet = _importar_dependencias()
    material_name = args.material
    output_csv = args.output_csv or _default_output_csv_for_material(material_name)
    base_df = _load_base_dataframe(pd, material_name)
    cac_path = args.cac_csv or (DEFAULT_CAC_CSV if DEFAULT_CAC_CSV.exists() else None)
    cac_df = _load_cac_dataframe(pd, cac_path) if cac_path is not None else None
    icc_path = args.icc_csv or (DEFAULT_ICC_CSV if DEFAULT_ICC_CSV.exists() else None)
    icc_df = _load_icc_dataframe(pd, icc_path) if icc_path is not None else None
    if cac_df is not None:
        base_df = base_df.merge(cac_df, on="ds", how="inner").sort_values("ds").reset_index(drop=True)
    if icc_df is not None:
        base_df = base_df.merge(icc_df, on="ds", how="inner").sort_values("ds").reset_index(drop=True)
    dataset_rows = _to_dataset_rows(base_df)
    model_registry = _available_models(cac_df, icc_df)
    selected_models = set(args.modelos) if args.modelos else set(model_registry)
    unknown_models = sorted(selected_models.difference(model_registry))
    if unknown_models:
        raise SystemExit(f"Modelos no reconocidos: {', '.join(unknown_models)}")

    all_rows: list[ModelRunResult] = []
    for horizonte in args.horizontes:
        folds = construir_folds_temporales(
            dataset_rows,
            min_train_size=args.min_train_size,
            test_size=horizonte,
            step_size=horizonte,
        )
        horizon_rows: list[ModelRunResult] = []
        baseline_result: ModelRunResult | None = None

        for model_name, definition in model_registry.items():
            if model_name not in selected_models:
                continue
            try:
                result = _run_model_over_folds(
                    pd,
                    Prophet,
                    base_df,
                    folds,
                    nombre_modelo=model_name,
                    regresores_features=definition["regresores_features"],
                    horizonte_meses=horizonte,
                    predictor=definition["runner"],
                    baseline_result=baseline_result,
                )
            except ModuleNotFoundError as exc:
                result = _skip_result(
                    model_name,
                    definition["regresores_features"],
                    horizonte,
                    f"Experimento opcional omitido: {exc}.",
                )
            except Exception as exc:
                result = _skip_result(
                    model_name,
                    definition["regresores_features"],
                    horizonte,
                    f"Experimento omitido por error experimental: {type(exc).__name__}: {exc}",
                )

            if model_name == BASELINE_NAME and not result.skipped:
                baseline_note = "baseline_obligatorio"
                if material_name == DEFAULT_MATERIAL_NAME and horizonte == 3:
                    baseline_note = f"{baseline_note}; referencia_documentada_3m≈4.98"
                baseline_result = result
                horizon_rows.append(
                    ModelRunResult(
                        nombre_modelo=result.nombre_modelo,
                        regresores_features=result.regresores_features,
                        horizonte_meses=result.horizonte_meses,
                        mae=result.mae,
                        mape=result.mape,
                        folds=result.folds,
                        observaciones=f"{result.observaciones}; {baseline_note}",
                        mejora_vs_baseline="0.00 pp",
                        fold_metrics=result.fold_metrics,
                        predictions=result.predictions,
                        skipped=False,
                    )
                )
                continue

            horizon_rows.append(result)

        completed_rows = [row for row in horizon_rows if not row.skipped and row.mape is not None]
        completed_rows.sort(key=lambda item: item.mape)
        if not args.sin_ensemble and len(completed_rows) >= 2 and baseline_result is not None:
            ensemble = _build_ensemble_result(
                "ensemble_simple_top2",
                horizonte,
                completed_rows[0],
                completed_rows[1],
                baseline_result,
            )
            horizon_rows.append(ensemble)

        horizon_rows.sort(
            key=lambda item: (
                item.horizonte_meses,
                item.mape if item.mape is not None else float("inf"),
                item.nombre_modelo,
            )
        )
        print("")
        print("")
        print(f"Material: {material_name}")
        print(f"### Horizonte {horizonte} meses")
        _print_table(horizon_rows)
        all_rows.extend(horizon_rows)

    _write_report_csv(all_rows, output_csv)
    print("")
    print(f"Reporte CSV exportado en: {output_csv}")


if __name__ == "__main__":
    main()
