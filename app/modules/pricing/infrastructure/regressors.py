from datetime import date
from math import isfinite
from pathlib import Path

from sqlalchemy import select

from app.modules.pricing.domain.exceptions import ExternalRegressorError, ExternalRegressorUnavailableError
from app.modules.pricing.infrastructure.models import ExternalIndexValue
from app.shared.database.session import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FORECAST_DATASET_START = date(2022, 1, 1)
OFICIAL_CSV = PROJECT_ROOT / "tmp" / "dolares_2022" / "dolar_oficial_historico.csv"
MAYORISTA_CSV = PROJECT_ROOT / "tmp" / "dolares_2022" / "dolar_mayorista_historico.csv"
BLUE_CSV = PROJECT_ROOT / "tmp" / "dolares_2022" / "dolar_blue_historico.csv"
IPC_CSV = PROJECT_ROOT / "tmp" / "ipc_2022" / "ipc_nacional.csv"
IPIM_NIVEL_GENERAL_SERIES_ID = "448.1_NIVEL_GENERAL_0_0_13_46"
ICC_GENERAL_SERIES_ID = "ICC_NIVEL_GENERAL"
ICC_MATERIALS_SERIES_ID = "ICC_MATERIALES"
ICC_LABOUR_FORCE_SERIES_ID = "ICC_MANO_DE_OBRA"
ICC_VAR_GENERAL_SERIES_ID = "ICC_VAR_GENERAL"
ICC_VAR_MATERIALS_SERIES_ID = "ICC_VAR_MATERIALES"
ICC_VAR_LABOUR_SERIES_ID = "ICC_VAR_MANO_DE_OBRA"
CAC_GENERAL_SERIES_ID = "CAC_GENERAL"
CAC_MATERIALS_SERIES_ID = "CAC_MATERIALES"
CAC_LABOUR_FORCE_SERIES_ID = "CAC_MANO_DE_OBRA"
CAC_VAR_GENERAL_SERIES_ID = "CAC_VAR_GENERAL"
CAC_VAR_MATERIALS_SERIES_ID = "CAC_VAR_MATERIALES"
CAC_VAR_LABOUR_SERIES_ID = "CAC_VAR_MANO_DE_OBRA"
REGRESSOR_TREND_WINDOW_MONTHS = 12


def _cargar_dolar_mensual(pd, path: Path, columna: str):
    if not path.exists():
        raise ExternalRegressorUnavailableError(f"No se encontro el CSV del regresor {columna}.")

    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["venta"] = pd.to_numeric(df["venta"], errors="coerce")
    df = df[df["fecha"] >= pd.to_datetime(FORECAST_DATASET_START)].copy()
    df["ds"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    return df.groupby("ds", as_index=False)["venta"].mean().rename(columns={"venta": columna})


def _cargar_ipc_mensual(pd, path: Path):
    if not path.exists():
        raise ExternalRegressorUnavailableError("No se encontro el CSV del regresor ipc.")

    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ipc"] = pd.to_numeric(df["ipc"], errors="coerce")
    df = df[df["fecha"] >= pd.to_datetime(FORECAST_DATASET_START)].copy()
    return df.rename(columns={"fecha": "ds"})[["ds", "ipc"]]


def _cargar_indice_externo_mensual(pd, series_id: str, columna: str):
    with SessionLocal() as db:
        stmt = (
            select(ExternalIndexValue)
            .where(
                ExternalIndexValue.series_id == series_id,
                ExternalIndexValue.date >= FORECAST_DATASET_START,
            )
            .order_by(ExternalIndexValue.date.asc(), ExternalIndexValue.id.asc())
        )
        rows = list(db.scalars(stmt))

    if not rows:
        raise ExternalRegressorUnavailableError(f"No hay valores cargados para el regresor {columna}.")

    return pd.DataFrame(
        [
            {
                "ds": pd.to_datetime(value.date),
                columna: float(value.value),
            }
            for value in rows
        ]
    )


def cargar_regresores_mensuales(pd, columnas: tuple[str, ...] = ("dolar_oficial", "dolar_mayorista", "ipc")):
    if not columnas:
        return None

    loaders = {
        "dolar_oficial": lambda: _cargar_dolar_mensual(pd, OFICIAL_CSV, "dolar_oficial"),
        "dolar_mayorista": lambda: _cargar_dolar_mensual(pd, MAYORISTA_CSV, "dolar_mayorista"),
        "dolar_blue": lambda: _cargar_dolar_mensual(pd, BLUE_CSV, "dolar_blue"),
        "ipc": lambda: _cargar_ipc_mensual(pd, IPC_CSV),
        "ipim_nivel_general": lambda: _cargar_indice_externo_mensual(pd, IPIM_NIVEL_GENERAL_SERIES_ID, "ipim_nivel_general"),
        "icc_nivel_general": lambda: _cargar_indice_externo_mensual(pd, ICC_GENERAL_SERIES_ID, "icc_nivel_general"),
        "icc_materials": lambda: _cargar_indice_externo_mensual(pd, ICC_MATERIALS_SERIES_ID, "icc_materials"),
        "icc_labour_force": lambda: _cargar_indice_externo_mensual(pd, ICC_LABOUR_FORCE_SERIES_ID, "icc_labour_force"),
        "icc_var_general": lambda: _cargar_indice_externo_mensual(pd, ICC_VAR_GENERAL_SERIES_ID, "icc_var_general"),
        "icc_var_materials": lambda: _cargar_indice_externo_mensual(pd, ICC_VAR_MATERIALS_SERIES_ID, "icc_var_materials"),
        "icc_var_labour": lambda: _cargar_indice_externo_mensual(pd, ICC_VAR_LABOUR_SERIES_ID, "icc_var_labour"),
        "cac_general": lambda: _cargar_indice_externo_mensual(pd, CAC_GENERAL_SERIES_ID, "cac_general"),
        "cac_materials": lambda: _cargar_indice_externo_mensual(pd, CAC_MATERIALS_SERIES_ID, "cac_materials"),
        "cac_labour_force": lambda: _cargar_indice_externo_mensual(pd, CAC_LABOUR_FORCE_SERIES_ID, "cac_labour_force"),
        "cac_var_general": lambda: _cargar_indice_externo_mensual(pd, CAC_VAR_GENERAL_SERIES_ID, "cac_var_general"),
        "cac_var_materials": lambda: _cargar_indice_externo_mensual(pd, CAC_VAR_MATERIALS_SERIES_ID, "cac_var_materials"),
        "cac_var_labour": lambda: _cargar_indice_externo_mensual(pd, CAC_VAR_LABOUR_SERIES_ID, "cac_var_labour"),
    }

    faltantes = [columna for columna in columnas if columna not in loaders]
    if faltantes:
        raise ExternalRegressorUnavailableError(f"Regresores no soportados: {', '.join(faltantes)}.")

    combinado = loaders[columnas[0]]()
    for columna in columnas[1:]:
        combinado = combinado.merge(loaders[columna](), on="ds", how="inner")
    return combinado


def proyectar_regresores_futuros(pd, regresores_historicos, fechas_futuras, columnas: tuple[str, ...]):
    historial = regresores_historicos.sort_values("ds").tail(REGRESSOR_TREND_WINDOW_MONTHS).copy()
    if historial.empty:
        raise ExternalRegressorError("No hay historial suficiente de regresores externos para proyectar el forecast.")

    futuros = {"ds": pd.to_datetime(fechas_futuras)}
    periodos = max(len(historial) - 1, 1)

    for columna in columnas:
        valores = historial[columna].dropna().tolist()
        if not valores:
            raise ExternalRegressorError(f"No hay datos del regresor {columna} para proyectar el forecast.")

        ultimo_valor = float(valores[-1])
        primer_valor = float(valores[0])
        tasa_mensual = 0.0 if primer_valor <= 0 else (ultimo_valor / primer_valor) ** (1 / periodos) - 1
        futuros[columna] = [
            ultimo_valor * ((1 + tasa_mensual) ** paso)
            for paso in range(1, len(fechas_futuras) + 1)
        ]

    return pd.DataFrame(futuros)


def proyectar_regresores_futuros_ex_ante(pd, regresores_historicos, fechas_futuras, columnas: tuple[str, ...]):
    historial = regresores_historicos.sort_values("ds").tail(REGRESSOR_TREND_WINDOW_MONTHS).copy()
    if historial.empty:
        raise ExternalRegressorError("No hay historial suficiente de regresores externos para proyectar el forecast.")

    futuros = {"ds": pd.to_datetime(fechas_futuras)}
    for columna in columnas:
        observados = historial[["ds", columna]].dropna()
        if observados.empty:
            raise ExternalRegressorError(f"No hay datos del regresor {columna} para proyectar el forecast.")
        valores = observados[columna].tolist()
        ultimo_valor = float(valores[-1])
        primer_valor = float(valores[0])
        if not all(isfinite(float(valor)) for valor in valores):
            raise ExternalRegressorError(f"El regresor {columna} contiene valores no finitos.")
        ultima_fecha = pd.to_datetime(observados["ds"].iloc[-1])
        primera_fecha = pd.to_datetime(observados["ds"].iloc[0])
        periodos = max((ultima_fecha.year - primera_fecha.year) * 12 + ultima_fecha.month - primera_fecha.month, 1)
        pasos = [(fecha.year - ultima_fecha.year) * 12 + fecha.month - ultima_fecha.month for fecha in pd.to_datetime(fechas_futuras)]
        if any(paso <= 0 for paso in pasos):
            raise ExternalRegressorError("Las fechas proyectadas deben ser posteriores al ultimo regresor observado.")
        # Monthly percentage changes are rates, not price/index levels. Persist
        # their last observed rate; compounding a signed rate is not meaningful.
        if "var_" in columna or any(float(valor) <= 0 for valor in valores):
            futuros[columna] = [ultimo_valor] * len(pasos)
            continue
        tasa_mensual = 0.0 if primer_valor <= 0 else (ultimo_valor / primer_valor) ** (1 / periodos) - 1
        futuros[columna] = [
            ultimo_valor * ((1 + tasa_mensual) ** paso)
            for paso in pasos
        ]

    return pd.DataFrame(futuros)



def preparar_regresores_para_fold(pd, train_df, test_df, regresores_df, columnas: tuple[str, ...]):
    """Reconstruye regresores ex ante: nunca lee los valores reales del test."""
    if regresores_df is None or not columnas:
        raise ExternalRegressorError("Se requieren regresores historicos y columnas para evaluar este fold.")
    fecha_corte = pd.to_datetime(train_df["ds"]).max()
    historicos = regresores_df[regresores_df["ds"] <= fecha_corte][["ds", *columnas]].copy().sort_values("ds")
    if historicos.empty:
        raise ExternalRegressorError("No hay regresores disponibles hasta el corte del fold.")

    train_reg = train_df[["ds", "y"]].merge(historicos, on="ds", how="left").sort_values("ds")
    for columna in columnas:
        # Solo se imputan valores de entrenamiento, nunca se rellena con el test.
        train_reg[columna] = train_reg[columna].ffill()
        if train_reg[columna].isna().any():
            raise ExternalRegressorError(f"Faltan valores de entrenamiento del regresor {columna}.")

    # Si el ultimo regresor disponible es anterior al corte, proyectar tambien
    # los meses intermedios para no desplazar artificialmente el horizonte.
    ultimo_regresor = pd.to_datetime(historicos["ds"]).max()
    fechas_a_proyectar = pd.date_range(
        start=ultimo_regresor + pd.DateOffset(months=1),
        end=pd.to_datetime(test_df["ds"]).max(),
        freq="MS",
    )
    estimados = proyectar_regresores_futuros_ex_ante(pd, historicos, fechas_a_proyectar, columnas)
    futuro_test = test_df[["ds"]].merge(estimados, on="ds", how="left")
    if futuro_test[list(columnas)].isna().any().any():
        raise ExternalRegressorError("No fue posible proyectar todos los regresores del fold.")

    futuro = pd.concat(
        [train_reg[["ds", *columnas]], futuro_test[["ds", *columnas]]],
        ignore_index=True,
    )
    return train_reg[["ds", "y", *columnas]], futuro
