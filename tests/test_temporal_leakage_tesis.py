"""Regresiones metodologicas de la evaluacion de tesis: informacion ex ante."""
from dataclasses import replace
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from app.modules.pricing.application.forecast_service import ProphetRow, backtesting_forecast
from app.modules.pricing.application.series import (
    PuntoSeriePrecio,
    _features_anomalia_mensual,
    _intervalo_reentrenamiento_anomalias,
)
from app.modules.pricing.infrastructure.regressors import preparar_regresores_para_fold


def _precio(fecha: date, valor: int = 100) -> PuntoSeriePrecio:
    return PuntoSeriePrecio(
        fecha=fecha,
        precio_promedio_normalizado=Decimal(valor),
        unidad_base="kg",
        precio_equivalente_25kg=None,
        precio_equivalente_50kg=None,
        cantidad_registros=1,
        cantidad_facturas=1,
        fuentes=["Factura compra"],
        variacion_porcentual_anterior=Decimal("1"),
    )


def test_regresores_del_test_no_cambian_predictores_ex_ante() -> None:
    fechas = pd.date_range("2022-01-01", periods=27, freq="MS")
    train = pd.DataFrame({"ds": fechas[:24], "y": [100.0] * 24})
    test = pd.DataFrame({"ds": fechas[24:], "y": [200.0] * 3})
    regresores = pd.DataFrame({"ds": fechas, "ipim": [100.0 + i for i in range(27)]})
    train_a, futuro_a = preparar_regresores_para_fold(pd, train, test, regresores, ("ipim",))

    modificado = regresores.copy()
    modificado.loc[modificado["ds"].isin(fechas[24:]), "ipim"] = 1_000_000.0
    train_b, futuro_b = preparar_regresores_para_fold(pd, train, test, modificado, ("ipim",))

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(futuro_a, futuro_b)
    assert (futuro_a.tail(3)["ipim"] < 1000).all()


def test_backtesting_no_consume_regresor_real_futuro(monkeypatch: pytest.MonkeyPatch) -> None:
    fechas = pd.date_range("2022-01-01", periods=27, freq="MS")
    dataset = [ProphetRow(ds=ds.date(), y=100.0) for ds in fechas]
    regresores = pd.DataFrame({
        "ds": fechas,
        "ipim": [100.0 + i for i in range(24)] + [1_000_000.0] * 3,
    })
    futuros_enviados = []

    class FakeProphet:
        def __init__(self, **_kwargs):
            pass

        def add_regressor(self, _nombre: str) -> None:
            pass

        def fit(self, _df):
            pass

        def predict(self, frame):
            futuros_enviados.extend(frame.tail(3)["ipim"].to_list())
            return pd.DataFrame({"ds": frame["ds"], "yhat": [100.0] * len(frame)})

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_folds_temporales",
        lambda *_args, **_kwargs: [SimpleNamespace(train=dataset[:24], test=dataset[24:])],
    )
    resultado = backtesting_forecast(pd, FakeProphet, dataset, regresores, 3, ("ipim",))
    assert resultado.folds == 1
    assert len(futuros_enviados) == 3
    assert max(futuros_enviados) < 1000


def test_features_del_random_forest_no_contienen_precio_objetivo() -> None:
    puntos = [
        _precio(date(2024 + i // 12, i % 12 + 1, 1), valor=100 + i)
        for i in range(18)
    ]
    antes = _features_anomalia_mensual(puntos, 12, puntos[0])
    modificados = puntos.copy()
    modificados[12] = replace(
        puntos[12],
        precio_promedio_normalizado=Decimal("999999"),
        variacion_porcentual_anterior=Decimal("999"),
        cantidad_registros=999,
    )
    despues = _features_anomalia_mensual(modificados, 12, modificados[0])
    assert antes == despues


def test_random_forest_reentrena_serie_mensual() -> None:
    mensual = [
        _precio(date(2024 + i // 12, i % 12 + 1, 1))
        for i in range(24)
    ]
    densa = [_precio(date(2024, mes, dia)) for mes in (1, 2) for dia in range(1, 16)]
    assert _intervalo_reentrenamiento_anomalias(mensual) == 6
    assert _intervalo_reentrenamiento_anomalias(densa) == 30
