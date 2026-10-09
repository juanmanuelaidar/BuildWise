from dataclasses import dataclass

from app.experiments.pricing.common import (
    DEFAULT_DATASET_START,
    cargar_dolar_mensual,
    cargar_ipc_mensual,
    configurar_cmdstan,
    construir_dataset_material,
    importar_dependencias_prophet,
)
from app.modules.pricing.application.backtesting import construir_folds_temporales
from app.modules.pricing.application.forecasting import ProphetRow


DATASET_START = DEFAULT_DATASET_START
BLUE_CSV = "tmp/dolares_2022/dolar_blue_historico.csv"
OFICIAL_CSV = "tmp/dolares_2022/dolar_oficial_historico.csv"
MAYORISTA_CSV = "tmp/dolares_2022/dolar_mayorista_historico.csv"
IPC_CSV = "tmp/ipc_2022/ipc_nacional.csv"
BEST_PROPHET_CONFIG = {
    "daily_seasonality": False,
    "weekly_seasonality": False,
    "yearly_seasonality": False,
    "changepoint_prior_scale": 0.01,
    "seasonality_prior_scale": 1.0,
    "seasonality_mode": "additive",
}


@dataclass(frozen=True)
class ResultadoModelo:
    nombre: str
    mae: float
    mape: float


def _importar_dependencias():
    return importar_dependencias_prophet("Faltan dependencias para correr el experimento con regresores.")


def _configurar_cmdstan(CmdStanPyBackend, IStanBackend) -> None:
    configurar_cmdstan(CmdStanPyBackend, IStanBackend, "No se encontro CmdStan en ~/.cmdstan/cmdstan-2.38.0.")


def _obtener_dataset_cemento(pd) -> list[ProphetRow]:
    return construir_dataset_material("Cemento Portland", frecuencia="mensual", objetivo="precio_promedio_normalizado", dataset_start=DATASET_START)


def _a_dataframe(pd, filas: list[ProphetRow]):
    return pd.DataFrame([{"ds": pd.to_datetime(fila.ds), "y": fila.y} for fila in filas])


def _cargar_dolar_mensual(pd, path_csv: str, columna_salida: str):
    return cargar_dolar_mensual(pd, path_csv, columna_salida, dataset_start=DATASET_START)


def _cargar_ipc_mensual(pd):
    return cargar_ipc_mensual(pd, IPC_CSV, dataset_start=DATASET_START)


def _combinar_regresores(base_df, otros_df):
    combinado = base_df.copy()
    for df in otros_df:
        combinado = combinado.merge(df, on="ds", how="inner")
    return combinado


def _calcular_metricas(evaluacion_df) -> tuple[float, float]:
    mae = float(evaluacion_df["abs_error"].mean())
    mape = float(evaluacion_df["ape"].mean())
    return mae, mape


def _evaluar_prophet(pd, Prophet, train: list[ProphetRow], test: list[ProphetRow], regresores_df=None, columnas_regresoras=None, nombre="prophet"):
    train_df = _a_dataframe(pd, train)
    test_df = _a_dataframe(pd, test)
    modelo = Prophet(stan_backend="CMDSTANPY", **BEST_PROPHET_CONFIG)

    if regresores_df is None:
        modelo.fit(train_df)
        futuro = modelo.make_future_dataframe(periods=len(test_df), freq="MS")
        forecast = modelo.predict(futuro)[["ds", "yhat"]]
    else:
        columnas_regresoras = columnas_regresoras or []
        full_df = pd.concat([train_df[["ds", "y"]], test_df[["ds", "y"]]], ignore_index=True)
        full_df = full_df.merge(regresores_df, on="ds", how="left")
        for columna in columnas_regresoras:
            full_df[columna] = full_df[columna].ffill().bfill()
            modelo.add_regressor(columna)

        train_reg = full_df.iloc[: len(train_df)][["ds", "y", *columnas_regresoras]].copy()
        futuro = full_df[["ds", *columnas_regresoras]].copy()
        modelo.fit(train_reg)
        forecast = modelo.predict(futuro)[["ds", "yhat"]]

    evaluacion = test_df.merge(forecast, on="ds", how="left")
    evaluacion["abs_error"] = (evaluacion["y"] - evaluacion["yhat"]).abs()
    evaluacion["ape"] = evaluacion["abs_error"] / evaluacion["y"] * 100
    mae, mape = _calcular_metricas(evaluacion)
    return ResultadoModelo(nombre, mae, mape)


def main() -> None:
    pd, Prophet, CmdStanPyBackend, IStanBackend = _importar_dependencias()
    _configurar_cmdstan(CmdStanPyBackend, IStanBackend)

    dataset = _obtener_dataset_cemento(pd)
    blue = _cargar_dolar_mensual(pd, BLUE_CSV, "dolar_blue")
    oficial = _cargar_dolar_mensual(pd, OFICIAL_CSV, "dolar_oficial")
    mayorista = _cargar_dolar_mensual(pd, MAYORISTA_CSV, "dolar_mayorista")
    ipc = _cargar_ipc_mensual(pd)
    folds = construir_folds_temporales(dataset, min_train_size=24, test_size=3, step_size=3)

    configuraciones = {
        "prophet_base": {"regresores_df": None, "columnas": []},
        "prophet_blue": {"regresores_df": blue, "columnas": ["dolar_blue"]},
        "prophet_oficial": {"regresores_df": oficial, "columnas": ["dolar_oficial"]},
        "prophet_mayorista": {"regresores_df": mayorista, "columnas": ["dolar_mayorista"]},
        "prophet_ipc": {"regresores_df": ipc, "columnas": ["ipc"]},
        "prophet_blue_ipc": {
            "regresores_df": _combinar_regresores(blue, [ipc]),
            "columnas": ["dolar_blue", "ipc"],
        },
        "prophet_oficial_ipc": {
            "regresores_df": _combinar_regresores(oficial, [ipc]),
            "columnas": ["dolar_oficial", "ipc"],
        },
        "prophet_mayorista_ipc": {
            "regresores_df": _combinar_regresores(mayorista, [ipc]),
            "columnas": ["dolar_mayorista", "ipc"],
        },
        "prophet_oficial_blue": {
            "regresores_df": _combinar_regresores(oficial, [blue]),
            "columnas": ["dolar_oficial", "dolar_blue"],
        },
        "prophet_oficial_mayorista": {
            "regresores_df": _combinar_regresores(oficial, [mayorista]),
            "columnas": ["dolar_oficial", "dolar_mayorista"],
        },
        "prophet_oficial_ipc_blue": {
            "regresores_df": _combinar_regresores(oficial, [ipc, blue]),
            "columnas": ["dolar_oficial", "ipc", "dolar_blue"],
        },
        "prophet_oficial_ipc_mayorista": {
            "regresores_df": _combinar_regresores(oficial, [ipc, mayorista]),
            "columnas": ["dolar_oficial", "ipc", "dolar_mayorista"],
        },
    }
    acumulado = {nombre: [] for nombre in configuraciones}

    for fold in folds:
        for nombre, config in configuraciones.items():
            acumulado[nombre].append(
                _evaluar_prophet(
                    pd,
                    Prophet,
                    fold.train,
                    fold.test,
                    regresores_df=config["regresores_df"],
                    columnas_regresoras=config["columnas"],
                    nombre=nombre,
                )
            )

    print(f"Dataset cemento desde {DATASET_START}: {len(dataset)} puntos mensuales")
    print(f"Folds: {len(folds)}")
    print("")
    ranking = []
    for nombre, resultados in acumulado.items():
        mae = sum(item.mae for item in resultados) / len(resultados)
        mape = sum(item.mape for item in resultados) / len(resultados)
        ranking.append((nombre, mae, mape))
    for nombre, mae, mape in sorted(ranking, key=lambda item: item[1]):
        print(f"{nombre}: MAE={mae:.2f} | MAPE={mape:.2f}%")


if __name__ == "__main__":
    main()
