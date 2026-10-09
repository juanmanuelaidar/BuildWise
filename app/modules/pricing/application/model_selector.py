from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_SOURCE_FILES: dict[str, tuple[Path, ...]] = {
    "cemento-portland": (PROJECT_ROOT / "tmp/experiments/cemento_forecast_benchmark_master.csv",),
    "pastina": (PROJECT_ROOT / "tmp/experiments/pastina_forecast_plateau.csv",),
    "membrana-megaflex": (PROJECT_ROOT / "tmp/experiments/membrana_megaflex_forecast_plateau.csv",),
}

MATERIAL_CONFIDENCE = {
    "cemento-portland": "alta",
    "pastina": "media",
    "membrana-megaflex": "media-baja",
}

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
        return float(raw)
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
    regresores = MODEL_REGRESSORS[modelo]
    return ForecastModelSelection(
        material_key=material_key,
        horizonte_meses=horizonte_meses,
        modelo=modelo,
        regresores=regresores,
        mape=Decimal(f"{mape:.2f}"),
        mae=Decimal(f"{mae:.2f}"),
        folds=folds,
        confiabilidad=MATERIAL_CONFIDENCE[material_key],
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada segun benchmark consolidado y "
            "seleccionada por menor MAPE dentro de las variantes ejecutables por el runtime."
        ),
        no_calibrado=False,
    )


def _load_benchmark_selections() -> tuple[dict[tuple[str, int], ForecastModelSelection], dict[str, ForecastModelSelection]]:
    exactas: dict[tuple[str, int], ForecastModelSelection] = {}
    por_material: dict[str, ForecastModelSelection] = {}

    for material_key, paths in BENCHMARK_SOURCE_FILES.items():
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            for raw in rows:
                modelo = raw.get("nombre_modelo", "").strip()
                if not _benchmark_is_supported(modelo):
                    continue

                horizonte = _parse_int(raw.get("horizonte_meses"))
                mape = _parse_float(raw.get("MAPE"))
                mae = _parse_float(raw.get("MAE"))
                folds = _parse_int(raw.get("folds"))
                if horizonte is None or mape is None or mae is None or folds is None:
                    continue

                selection = _build_selection(
                    material_key=material_key,
                    horizonte_meses=horizonte,
                    modelo=modelo,
                    mape=mape,
                    mae=mae,
                    folds=folds,
                )
                key = (material_key, horizonte)
                current = exactas.get(key)
                if current is None or (selection.mape, selection.mae) < (current.mape, current.mae):
                    exactas[key] = selection

    for (material_key, _horizonte), selection in exactas.items():
        current = por_material.get(material_key)
        if current is None or (selection.mape, selection.mae) < (current.mape, current.mae):
            por_material[material_key] = selection

    return exactas, por_material


_BENCHMARK_SELECCIONES_EXACTAS, _BENCHMARK_SELECCIONES_POR_MATERIAL = _load_benchmark_selections()


_LEGACY_SELECCIONES_EXACTAS: dict[tuple[str, int], ForecastModelSelection] = {
    (MATERIAL_KEY_CEMENTO_PORTLAND, 3): ForecastModelSelection(
        material_key=MATERIAL_KEY_CEMENTO_PORTLAND,
        horizonte_meses=3,
        modelo="prophet_ipim_icc_var_materials",
        regresores=("ipim_nivel_general", "icc_var_materials"),
        mape=Decimal("4.22"),
        mae=Decimal("5.82"),
        folds=9,
        confiabilidad=CONFIABILIDAD_ALTA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Cemento Portland a 3 meses segun benchmark "
            "documentado, con mejor promedio observado y mejora consistente frente al baseline."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_CEMENTO_PORTLAND, 6): ForecastModelSelection(
        material_key=MATERIAL_KEY_CEMENTO_PORTLAND,
        horizonte_meses=6,
        modelo="prophet_ipim_icc_var_materials",
        regresores=("ipim_nivel_general", "icc_var_materials"),
        mape=Decimal("5.52"),
        mae=Decimal("7.58"),
        folds=9,
        confiabilidad=CONFIABILIDAD_ALTA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Cemento Portland a 6 meses segun benchmark "
            "documentado, con mejor promedio observado y estabilidad aceptable."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_CEMENTO_PORTLAND, 12): ForecastModelSelection(
        material_key=MATERIAL_KEY_CEMENTO_PORTLAND,
        horizonte_meses=12,
        modelo="prophet_ipim_icc_var_materials",
        regresores=("ipim_nivel_general", "icc_var_materials"),
        mape=Decimal("4.51"),
        mae=Decimal("6.36"),
        folds=9,
        confiabilidad=CONFIABILIDAD_ALTA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Cemento Portland a 12 meses segun benchmark "
            "documentado, con mejor promedio observado y mejora consistente frente al baseline."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_PASTINA, 3): ForecastModelSelection(
        material_key=MATERIAL_KEY_PASTINA,
        horizonte_meses=3,
        modelo="prophet_ipim_cac_labour_force",
        regresores=("ipim_nivel_general", "cac_labour_force"),
        mape=Decimal("4.27"),
        mae=Decimal("97.97"),
        folds=9,
        confiabilidad=CONFIABILIDAD_MEDIA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Pastina a 3 meses segun benchmark documentado, "
            "con mejor promedio observado entre las variantes runtime soportadas."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_PASTINA, 6): ForecastModelSelection(
        material_key=MATERIAL_KEY_PASTINA,
        horizonte_meses=6,
        modelo="prophet_ipim_cac_labour_force",
        regresores=("ipim_nivel_general", "cac_labour_force"),
        mape=Decimal("5.26"),
        mae=Decimal("115.97"),
        folds=9,
        confiabilidad=CONFIABILIDAD_MEDIA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Pastina a 6 meses segun benchmark documentado, "
            "con mejor promedio observado entre las variantes runtime soportadas."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_PASTINA, 12): ForecastModelSelection(
        material_key=MATERIAL_KEY_PASTINA,
        horizonte_meses=12,
        modelo="prophet_ipim_cac_labour_force",
        regresores=("ipim_nivel_general", "cac_labour_force"),
        mape=Decimal("6.98"),
        mae=Decimal("154.11"),
        folds=9,
        confiabilidad=CONFIABILIDAD_MEDIA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Pastina a 12 meses segun benchmark documentado, "
            "con mejor promedio observado entre las variantes runtime soportadas."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_MEMBRANA_MEGAFLEX, 3): ForecastModelSelection(
        material_key=MATERIAL_KEY_MEMBRANA_MEGAFLEX,
        horizonte_meses=3,
        modelo="prophet_ipim_icc_var_materials",
        regresores=("ipim_nivel_general", "icc_var_materials"),
        mape=Decimal("8.08"),
        mae=Decimal("656.08"),
        folds=9,
        confiabilidad=CONFIABILIDAD_MEDIA_BAJA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Membrana Megaflex a 3 meses segun benchmark "
            "documentado, con mejor promedio observado entre las variantes runtime soportadas."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_MEMBRANA_MEGAFLEX, 6): ForecastModelSelection(
        material_key=MATERIAL_KEY_MEMBRANA_MEGAFLEX,
        horizonte_meses=6,
        modelo="prophet_ipim_icc_var_materials",
        regresores=("ipim_nivel_general", "icc_var_materials"),
        mape=Decimal("9.97"),
        mae=Decimal("777.44"),
        folds=9,
        confiabilidad=CONFIABILIDAD_MEDIA_BAJA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Membrana Megaflex a 6 meses segun benchmark "
            "documentado, con mejor promedio observado entre las variantes runtime soportadas."
        ),
        no_calibrado=False,
    ),
    (MATERIAL_KEY_MEMBRANA_MEGAFLEX, 12): ForecastModelSelection(
        material_key=MATERIAL_KEY_MEMBRANA_MEGAFLEX,
        horizonte_meses=12,
        modelo="prophet_ipim_icc_var_materials",
        regresores=("ipim_nivel_general", "icc_var_materials"),
        mape=Decimal("13.57"),
        mae=Decimal("1080.42"),
        folds=9,
        confiabilidad=CONFIABILIDAD_MEDIA_BAJA,
        origen_decision=ORIGEN_DECISION_MATERIAL_HORIZONTE,
        justificacion=(
            "Configuracion recomendada para Membrana Megaflex a 12 meses segun benchmark "
            "documentado, con mejor promedio observado entre las variantes runtime soportadas."
        ),
        no_calibrado=False,
    ),
}

_SELECCIONES_EXACTAS = _BENCHMARK_SELECCIONES_EXACTAS or _LEGACY_SELECCIONES_EXACTAS
_SELECCIONES_POR_MATERIAL = _BENCHMARK_SELECCIONES_POR_MATERIAL or {
    material_key: selection for (material_key, _horizonte), selection in _SELECCIONES_EXACTAS.items()
}

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
