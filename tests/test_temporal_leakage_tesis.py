"""Regresiones metodologicas de la evaluacion de tesis: informacion ex ante."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from app.modules.pricing.application.forecasting import ProphetRow
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


def test_backtesting_historico_es_condicional_a_regresores_observados(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.modules.pricing.application.forecast_service import backtesting_forecast

    fechas = pd.date_range("2022-01-01", periods=27, freq="MS")
    dataset = [ProphetRow(ds=ds.date(), y=100.0) for ds in fechas]
    regresores = pd.DataFrame(
        {
            "ds": fechas,
            "ipim": [100.0 + i for i in range(24)] + [1_000_000.0] * 3,
        }
    )
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
    assert futuros_enviados == [1_000_000.0] * 3


def test_features_del_random_forest_no_contienen_precio_objetivo() -> None:
    puntos = [_precio(date(2024 + i // 12, i % 12 + 1, 1), valor=100 + i) for i in range(18)]
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
    mensual = [_precio(date(2024 + i // 12, i % 12 + 1, 1)) for i in range(24)]
    densa = [_precio(date(2024, mes, dia)) for mes in (1, 2) for dia in range(1, 16)]
    assert _intervalo_reentrenamiento_anomalias(mensual) == 6
    assert _intervalo_reentrenamiento_anomalias(densa) == 30


@pytest.mark.parametrize("last_rate", [-2.5, 0.0, 3.0])
def test_proyeccion_persiste_tasas_mensuales_incluso_negativas(last_rate):
    from app.modules.pricing.infrastructure.regressors import (
        proyectar_regresores_futuros_ex_ante as proyectar_regresores_futuros,
    )

    history = pd.DataFrame({"ds": pd.to_datetime(["2025-01-01", "2025-02-01"]), "icc_var_materials": [4.0, last_rate]})
    future = proyectar_regresores_futuros(
        pd, history, pd.to_datetime(["2025-03-01", "2025-05-01"]), ("icc_var_materials",)
    )
    assert future["icc_var_materials"].tolist() == [last_rate, last_rate]


def test_proyeccion_respeta_meses_transcurridos_con_huecos():
    from app.modules.pricing.infrastructure.regressors import (
        proyectar_regresores_futuros_ex_ante as proyectar_regresores_futuros,
    )

    history = pd.DataFrame({"ds": pd.to_datetime(["2025-01-01", "2025-03-01"]), "ipc": [100.0, 121.0]})
    future = proyectar_regresores_futuros(pd, history, pd.to_datetime(["2025-05-01"]), ("ipc",))
    assert future["ipc"].iloc[0] == pytest.approx(146.41)


def test_proyeccion_rechaza_infinito():
    from app.modules.pricing.domain.exceptions import ExternalRegressorError
    from app.modules.pricing.infrastructure.regressors import (
        proyectar_regresores_futuros_ex_ante as proyectar_regresores_futuros,
    )

    history = pd.DataFrame({"ds": pd.to_datetime(["2025-01-01"]), "ipc": [float("inf")]})
    with pytest.raises(ExternalRegressorError, match="no finitos"):
        proyectar_regresores_futuros(pd, history, [date(2025, 2, 1)], ("ipc",))


def test_backtesting_rechaza_prediccion_faltante(monkeypatch):
    from fastapi import HTTPException

    from app.modules.pricing.application.forecast_service import backtesting_forecast

    dates = pd.date_range("2022-01-01", periods=27, freq="MS")
    dataset = [ProphetRow(ds=d.date(), y=100.0) for d in dates]

    class IncompleteProphet:
        def __init__(self, **kwargs):
            pass

        def fit(self, frame):
            pass

        def make_future_dataframe(self, **kwargs):
            return pd.DataFrame({"ds": dates})

        def predict(self, frame):
            return pd.DataFrame({"ds": dates[:-1], "yhat": [100.0] * 26})

    with pytest.raises(HTTPException, match="completas y finitas"):
        backtesting_forecast(pd, IncompleteProphet, dataset, None, 3, ())


def test_snapshots_versionados_coinciden_con_dataset_y_selector():
    import json
    from pathlib import Path

    from app.experiments.pricing.mvp_reassessment import MATERIAL_NAMES, frozen_prices
    from app.modules.pricing.application.forecast_service import construir_firma_dataset
    from app.modules.pricing.application.forecasting import construir_dataset_prophet
    from app.modules.pricing.application.model_selector import resolve_model_selection
    from app.modules.pricing.application.series import PrecioSerieInput, construir_serie_mensual
    from app.operations.bootstrap.import_membrana_megaflex import build_prices

    snapshots = json.loads(Path("tmp/forecast_snapshots.json").read_text())
    prices = frozen_prices()
    prices["membrana-megaflex"] = [
        PrecioSerieInput(
            r.fecha, r.precio_normalizado, "kg", r.empresa, r.numero_comprobante, origen_dato=r.origen.upper()
        )
        for r in build_prices()
        if date(2022, 1, 1) <= r.fecha <= date.today()
    ]
    for material_id, (key, records) in enumerate(prices.items(), 1):
        dataset = construir_dataset_prophet(
            construir_serie_mensual(records, material_nombre=MATERIAL_NAMES[key]), "precio_promedio_normalizado"
        )
        signature = construir_firma_dataset(dataset)
        for horizon in range(1, 13):
            selection = resolve_model_selection(key, horizon)
            snapshot = snapshots[f"{material_id}:{horizon}:{signature}:selector-on:{selection.modelo}"]
            assert len(snapshot["forecast"]) == horizon
            assert snapshot["metricas"]["folds"] >= 2
            assert snapshot["seleccion_modelo"]["modelo_resuelto"] == selection.modelo
            assert snapshot["seleccion_modelo"]["no_calibrado"]
            if horizon not in (3, 6, 12) or key == "membrana-megaflex":
                assert snapshot["seleccion_modelo"]["mape_referencia"] is None
            else:
                assert snapshot["seleccion_modelo"]["mape_referencia"] is not None
            assert snapshot["seleccion_modelo"]["advertencia"]
            assert all(p["origenes_dato"] for p in snapshot["serie_mensual"])
            assert snapshot["forecast"][0]["fecha"] > snapshot["dataset"][-1]["ds"]


def test_anotacion_anomalia_conserva_proveniencia_del_registro(monkeypatch):
    from app.modules.pricing.application import series

    punto = replace(
        _precio(date(2025, 1, 1)),
        origen_dato="ESTIMADO",
        origenes_dato=["ESTIMADO"],
        metodo_estimacion="IPC",
        numero_comprobante="ESTIMADO-2025-01-01",
        precio_original=Decimal("121"),
    )
    anomaly = series.AnomalyDetectionMetadata(
        motivo="prueba",
        severidad="leve",
        score=1,
        confianza=Decimal("50"),
        precio_esperado=Decimal("90"),
        residuo_pct=Decimal("10"),
        limite_residuo_pct=Decimal("5"),
        rango_esperado_min=Decimal("80"),
        rango_esperado_max=Decimal("100"),
        tipo="salto_puntual",
        explicacion="prueba",
        variables_relevantes=[],
    )
    monkeypatch.setattr(series, "_detectar_anomalias_random_forest", lambda *_args, **_kwargs: {0: anomaly})
    anotado = series._aplicar_anomalias_random_forest([punto])[0]
    assert anotado.es_anomalia
    assert anotado.origen_dato == punto.origen_dato
    assert anotado.origenes_dato == punto.origenes_dato
    assert anotado.metodo_estimacion == punto.metodo_estimacion
    assert anotado.precio_original == punto.precio_original
    assert anotado.numero_comprobante == punto.numero_comprobante
