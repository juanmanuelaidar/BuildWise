from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from math import isfinite
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_DIR = PROJECT_ROOT / "db/benchmarks/mvp_2026_03"
BENCHMARK_SOURCE_FILES = {
    key: (BENCHMARK_DIR / "benchmarks.csv",)
    for key in ("cemento-portland", "pastina", "membrana-megaflex")
}
EVALUATION_PROTOCOL = "ex-ante-v2"

MODEL_REGRESSORS: dict[str, tuple[str, ...]] = {
    "prophet_base": (),
    "prophet_blue": ("dolar_blue",),
    "prophet_oficial": ("dolar_oficial",),
    "prophet_mayorista": ("dolar_mayorista",),
    "prophet_ipc": ("ipc",),
    "prophet_blue_ipc": ("dolar_blue", "ipc"),
    "prophet_oficial_ipc": ("dolar_oficial", "ipc"),
    "prophet_mayorista_ipc": ("dolar_mayorista", "ipc"),
    "prophet_oficial_blue": ("dolar_oficial", "dolar_blue"),
    "prophet_oficial_mayorista": ("dolar_oficial", "dolar_mayorista"),
    "prophet_oficial_ipc_blue": ("dolar_oficial", "ipc", "dolar_blue"),
    "prophet_oficial_ipc_mayorista": ("dolar_oficial", "ipc", "dolar_mayorista"),
    "prophet_icc_materiales": ("icc_materials",),
    "prophet_icc_materiales_ipc": ("icc_materials", "ipc"),
    "prophet_icc_materiales_oficial": ("icc_materials", "dolar_oficial"),
    "prophet_icc_materiales_mayorista": ("icc_materials", "dolar_mayorista"),
    "prophet_icc_nivel_general": ("icc_nivel_general",),
    "prophet_icc_var_general": ("icc_var_general",),
    "prophet_icc_var_materials": ("icc_var_materials",),
    "prophet_icc_var_labour": ("icc_var_labour",),
    "prophet_ipim_nivel_general": ("ipim_nivel_general",),
    "prophet_ipim_icc_var_general": ("ipim_nivel_general", "icc_var_general"),
    "prophet_ipim_icc_var_materials": ("ipim_nivel_general", "icc_var_materials"),
    "prophet_ipim_icc_var_labour": ("ipim_nivel_general", "icc_var_labour"),
    "prophet_ipim_cac_general": ("ipim_nivel_general", "cac_general"),
    "prophet_ipim_cac_materials": ("ipim_nivel_general", "cac_materials"),
    "prophet_ipim_cac_labour_force": ("ipim_nivel_general", "cac_labour_force"),
    "prophet_ipim_cac_var_general": ("ipim_nivel_general", "cac_var_general"),
    "prophet_ipim_cac_var_materials": ("ipim_nivel_general", "cac_var_materials"),
    "prophet_ipim_cac_var_labour": ("ipim_nivel_general", "cac_var_labour"),
    "prophet_cac_general": ("cac_general",),
    "prophet_cac_materials": ("cac_materials",),
    "prophet_cac_labour_force": ("cac_labour_force",),
    "prophet_cac_var_general": ("cac_var_general",),
    "prophet_cac_var_materials": ("cac_var_materials",),
    "prophet_cac_var_labour": ("cac_var_labour",),
}

SUPPORTED_BENCHMARK_MODELS = set(MODEL_REGRESSORS)
UNSUPPORTED_MODEL_PATTERNS = ("lags", "medias_moviles", "variaciones", "ensemble_simple_top2")


MATERIAL_KEY_CEMENTO_PORTLAND = "cemento-portland"
MATERIAL_KEY_PASTINA = "pastina"
MATERIAL_KEY_MEMBRANA_MEGAFLEX = "membrana-megaflex"

ORIGEN_DECISION_MATERIAL_HORIZONTE = "material_horizonte"
ORIGEN_DECISION_MATERIAL_DEFAULT = "material_default"
ORIGEN_DECISION_GLOBAL_FALLBACK = "global_fallback"

CONFIABILIDAD_ALTA = "alta"
CONFIABILIDAD_MEDIA = "media"
CONFIABILIDAD_MEDIA_BAJA = "media-baja"
CONFIABILIDAD_NO_CALIBRADA = "no_calibrada"


@dataclass(frozen=True)
class ForecastModelSelection:
    material_key: str
    horizonte_meses: int
    modelo: str
    regresores: tuple[str, ...]
    mape: Decimal | None
    mae: Decimal | None
    folds: int | None
    confiabilidad: str
    origen_decision: str
    justificacion: str
    no_calibrado: bool


def _parse_float(raw: str | None) -> float | None:
    if raw in {None, "", "-", "skip"}:
        return None
    try:
        value = float(raw)
        return value if isfinite(value) else None
    except ValueError:
        return None


def _parse_int(raw: str | None) -> int | None:
    if raw in {None, "", "-", "skip"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _benchmark_is_supported(model_name: str) -> bool:
    return model_name in SUPPORTED_BENCHMARK_MODELS and not any(pattern in model_name for pattern in UNSUPPORTED_MODEL_PATTERNS)


def _build_selection(
    *,
    material_key: str,
    horizonte_meses: int,
    modelo: str,
    mape: float,
    mae: float,
    folds: int,
) -> ForecastModelSelection:
    # Reference errors describe this frozen experiment. Estimated target prices,
    # few folds, and failure to beat persistence cannot support strong decisions.
    limitations = ["Seleccion exploratoria, sin test externo independiente."]
    if material_key != MATERIAL_KEY_CEMENTO_PORTLAND:
        limitations.append("La serie incluye precios estimados; el error no valida precios reales independientes.")
    if folds < 3:
        limitations.append("Solo hay dos particiones para este horizonte.")
    return ForecastModelSelection(
        material_key=material_key,
        horizonte_meses=horizonte_meses,
        modelo=modelo,
        regresores=MODEL_REGRESSORS[modelo],
        mape=Decimal(f"{mape:.2f}"),
        mae=Decimal(f"{mae:.2f}"),
        folds=folds,
        confiabilidad=CONFIABILIDAD_NO_CALIBRADA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=" ".join(limitations),
        no_calibrado=True,
    )


def _load_manifest() -> dict:
    try:
        manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("evaluation_protocol") != EVALUATION_PROTOCOL:
            return {}
        for relative, expected in manifest["sources"].items():
            if hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() != expected:
                return {}
        if hashlib.sha256((BENCHMARK_DIR / "benchmarks.csv").read_bytes()).hexdigest() != manifest["benchmark_sha256"]:
            return {}
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return {}


_BENCHMARK_MANIFEST = _load_manifest()
BENCHMARK_DATASET_SIGNATURES = {
    key: value["dataset_signature"]
    for key, value in _BENCHMARK_MANIFEST.get("materials", {}).items()
}


def _load_benchmark_selections() -> tuple[dict[tuple[str, int], ForecastModelSelection], dict[str, ForecastModelSelection]]:
    exactas: dict[tuple[str, int], ForecastModelSelection] = {}
    por_material: dict[str, ForecastModelSelection] = {}
    if not _BENCHMARK_MANIFEST:
        return exactas, por_material
    for material_key, paths in BENCHMARK_SOURCE_FILES.items():
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            baselines = {
                _parse_int(row.get("horizonte_meses")): _parse_float(row.get("MAPE"))
                for row in rows if row.get("material_key") == material_key and row.get("nombre_modelo") == "naive_last"
            }
            for raw in rows:
                if raw.get("material_key") != material_key or raw.get("evaluation_protocol") != EVALUATION_PROTOCOL:
                    continue
                if raw.get("dataset_signature") != BENCHMARK_DATASET_SIGNATURES.get(material_key):
                    continue
                modelo = raw.get("nombre_modelo", "").strip()
                if not _benchmark_is_supported(modelo):
                    continue
                horizonte = _parse_int(raw.get("horizonte_meses"))
                mape, mae = _parse_float(raw.get("MAPE")), _parse_float(raw.get("MAE"))
                folds = _parse_int(raw.get("folds"))
                if horizonte is None or mape is None or mae is None or folds is None or folds < 1 or min(mape, mae) < 0:
                    continue
                selection = _build_selection(material_key=material_key, horizonte_meses=horizonte, modelo=modelo, mape=mape, mae=mae, folds=folds)
                baseline = baselines.get(horizonte)
                if baseline is not None and mape >= baseline:
                    selection = replace(selection, justificacion=selection.justificacion + " No mejora el baseline de ultimo precio observado.")
                key = (material_key, horizonte)
                current = exactas.get(key)
                if current is None or (selection.mape, selection.mae) < (current.mape, current.mae):
                    exactas[key] = selection
    for (material_key, _horizonte), selection in exactas.items():
        current = por_material.get(material_key)
        if current is None or (selection.mape, selection.mae) < (current.mape, current.mae):
            por_material[material_key] = selection
    return exactas, por_material


_SELECCIONES_EXACTAS, _SELECCIONES_POR_MATERIAL = _load_benchmark_selections()


_FALLBACK_GLOBAL = ForecastModelSelection(
    material_key="unknown",
    horizonte_meses=0,
    modelo="prophet_base",
    regresores=(),
    mape=None,
    mae=None,
    folds=None,
    confiabilidad=CONFIABILIDAD_NO_CALIBRADA,
    origen_decision=ORIGEN_DECISION_GLOBAL_FALLBACK,
    justificacion=(
        "Se aplica prophet_base sin regresores como fallback operativo porque no existe "
        "una calibracion documentada para el material solicitado."
    ),
    no_calibrado=True,
)


def resolve_model_selection(material_key: str, horizonte_meses: int) -> ForecastModelSelection:
    exacta = _SELECCIONES_EXACTAS.get((material_key, horizonte_meses))
    if exacta is not None:
        return exacta

    por_material = _SELECCIONES_POR_MATERIAL.get(material_key)
    if por_material is not None:
        return replace(
            por_material,
            material_key=material_key,
            horizonte_meses=horizonte_meses,
            mape=None,
            mae=None,
            folds=None,
            confiabilidad=CONFIABILIDAD_NO_CALIBRADA,
            origen_decision=ORIGEN_DECISION_MATERIAL_DEFAULT,
            justificacion=(
                "Se reutiliza la mejor configuracion documentada para este material porque "
                f"no existe una calibracion exacta para horizonte de {horizonte_meses} meses."
            ),
            no_calibrado=True,
        )

    return replace(
        _FALLBACK_GLOBAL,
        material_key=material_key,
        horizonte_meses=horizonte_meses,
    )
