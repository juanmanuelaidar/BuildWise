from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.modules.catalog.application.utils import derive_material_key
from app.modules.pricing.application.forecast_service import (
    FORECAST_MODEL_NAME,
    ForecastMaterialResult,
    _resolver_plan_ejecucion,
    construir_firma_dataset,
    forecast_material,
    limpiar_forecast_cache,
)
from app.modules.pricing.application.forecasting import ProphetRow
from app.modules.pricing.application.model_selector import resolve_model_selection as resolve_selector_model
from app.modules.pricing.interfaces.routes import obtener_forecast_material
from app.modules.pricing.interfaces.schemas import ForecastMetricasRead, ForecastPuntoRead, ForecastSelectionRead


def _forecast_result(
    valor: str,
    *,
    modelo: str = FORECAST_MODEL_NAME,
    seleccion_modelo: ForecastSelectionRead | None = None,
    supuesto_regresores: str = "nota",
) -> ForecastMaterialResult:
    return ForecastMaterialResult(
        dataset=[ProphetRow(ds=date(2024, 1, 1), y=float(valor))],
        metricas=ForecastMetricasRead(
            folds=1,
            mae=Decimal("1.00"),
            mape=Decimal("2.00"),
            efectividad_informal=Decimal("98.00"),
        ),
        forecast=[
            ForecastPuntoRead(
                fecha=date(2024, 2, 1),
                precio_proyectado=Decimal(valor),
                precio_equivalente_25kg=Decimal("2500.00"),
                precio_equivalente_50kg=Decimal("5000.00"),
            )
        ],
        modelo=modelo,
        supuesto_regresores=supuesto_regresores,
        seleccion_modelo=seleccion_modelo,
    )


def test_construir_firma_dataset_cambia_si_cambia_la_serie() -> None:
    firma_a = construir_firma_dataset([ProphetRow(ds=date(2024, 1, 1), y=100.0)])
    firma_b = construir_firma_dataset([ProphetRow(ds=date(2024, 1, 1), y=101.0)])
    assert firma_a != firma_b


def test_forecast_material_reutiliza_cache_para_la_misma_serie(monkeypatch) -> None:
    limpiar_forecast_cache()
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30
    llamadas = {"cantidad": 0}

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda material, db: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_dataset_prophet",
        lambda puntos, objetivo: dataset,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.importar_dependencias_forecast",
        lambda: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda cache_key: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.guardar_forecast_snapshot",
        lambda cache_key, result: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.configurar_cmdstan",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_regresores_mensuales",
        lambda _pd, columnas: {"columnas": columnas},
    )

    def fake_forecast(material, horizonte_meses, dataset, pd, prophet, plan):
        llamadas["cantidad"] += 1
        return _forecast_result("110.00", modelo=plan.modelo, supuesto_regresores=plan.supuesto_regresores)

    monkeypatch.setattr("app.modules.pricing.application.forecast_service._forecast_material", fake_forecast)

    primer = forecast_material(material, 3, object())
    segundo = forecast_material(material, 3, object())

    assert llamadas["cantidad"] == 1
    assert primer.forecast[0].precio_proyectado == Decimal("110.00")
    assert segundo.forecast[0].precio_proyectado == Decimal("110.00")


def test_forecast_material_invalida_cache_si_cambia_el_dataset(monkeypatch) -> None:
    limpiar_forecast_cache()
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    datasets = [
        [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30,
        [ProphetRow(ds=date(2024, 1, 1), y=101.0)] * 30,
    ]
    resultados = iter([_forecast_result("110.00"), _forecast_result("115.00")])
    llamadas = {"cantidad": 0}

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda material, db: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.importar_dependencias_forecast",
        lambda: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda cache_key: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.guardar_forecast_snapshot",
        lambda cache_key, result: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.configurar_cmdstan",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_regresores_mensuales",
        lambda _pd, columnas: {"columnas": columnas},
    )

    def fake_dataset(puntos, objetivo):
        return datasets.pop(0)

    def fake_forecast(material, horizonte_meses, dataset, pd, prophet, plan):
        llamadas["cantidad"] += 1
        result = next(resultados)
        return _forecast_result(
            str(result.forecast[0].precio_proyectado),
            modelo=plan.modelo,
            supuesto_regresores=plan.supuesto_regresores,
        )

    monkeypatch.setattr("app.modules.pricing.application.forecast_service.construir_dataset_prophet", fake_dataset)
    monkeypatch.setattr("app.modules.pricing.application.forecast_service._forecast_material", fake_forecast)

    primer = forecast_material(material, 3, object())
    segundo = forecast_material(material, 3, object())

    assert llamadas["cantidad"] == 2
    assert primer.forecast[0].precio_proyectado == Decimal("110.00")
    assert segundo.forecast[0].precio_proyectado == Decimal("115.00")


def test_forecast_con_selector_desactivado_mantiene_comportamiento_actual(monkeypatch: pytest.MonkeyPatch) -> None:
    limpiar_forecast_cache()
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda _material, _repo: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda _cache_key: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.guardar_forecast_snapshot",
        lambda _cache_key, result: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_dataset_prophet",
        lambda puntos, objetivo: dataset,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.importar_dependencias_forecast",
        lambda: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.configurar_cmdstan",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_regresores_mensuales",
        lambda _pd, columnas: {"columnas": columnas},
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("El selector no deberia ejecutarse cuando esta desactivado.")

    monkeypatch.setattr("app.modules.pricing.application.forecast_service.resolve_model_selection", fail_if_called)

    def fake_forecast(_material, _horizonte, _dataset, _pd, _prophet, plan):
        return _forecast_result("110.00", modelo=plan.modelo, supuesto_regresores=plan.supuesto_regresores)

    monkeypatch.setattr("app.modules.pricing.application.forecast_service._forecast_material", fake_forecast)

    result = forecast_material(material, 3, object(), usar_selector_modelo=False)

    assert result.modelo == FORECAST_MODEL_NAME
    assert result.seleccion_modelo is None


@pytest.mark.parametrize(
    ("material_id", "nombre", "modelo", "regresores"),
    [
        (1, "Cemento Portland", "prophet_ipim_icc_var_materials", ("ipim_nivel_general", "icc_var_materials")),
        (4, "Pastina", "prophet_ipim_cac_labour_force", ("ipim_nivel_general", "cac_labour_force")),
        (10, "Membrana Megaflex", "prophet_ipim_icc_var_general", ("ipim_nivel_general", "icc_var_general")),
    ],
)
def test_selector_activado_usa_modelo_recomendado(monkeypatch: pytest.MonkeyPatch, material_id: int, nombre: str, modelo: str, regresores: tuple[str, ...]) -> None:
    limpiar_forecast_cache()
    material = SimpleNamespace(id=material_id, nombre=nombre, unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30
    material_key_esperado = derive_material_key(nombre)
    llamadas: list[tuple[str, int]] = []

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda _material, _repo: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda _cache_key: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.guardar_forecast_snapshot",
        lambda _cache_key, result: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_dataset_prophet",
        lambda puntos, objetivo: dataset,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.importar_dependencias_forecast",
        lambda: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.configurar_cmdstan",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_regresores_mensuales",
        lambda _pd, columnas: {"columnas": columnas},
    )
    def fake_resolve_model_selection(material_key: str, horizonte_meses: int):
        llamadas.append((material_key, horizonte_meses))
        return resolve_selector_model(material_key, horizonte_meses)

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.resolve_model_selection",
        fake_resolve_model_selection,
    )

    def fake_forecast(_material, _horizonte, _dataset, _pd, _prophet, plan):
        return _forecast_result(
            "110.00",
            modelo=plan.modelo,
            supuesto_regresores=plan.supuesto_regresores,
            seleccion_modelo=plan.seleccion_modelo,
        )

    monkeypatch.setattr("app.modules.pricing.application.forecast_service._forecast_material", fake_forecast)

    result = forecast_material(material, 3, object(), usar_selector_modelo=True)

    assert result.modelo == modelo
    assert llamadas == [(material_key_esperado, 3), (material_key_esperado, 3)]
    assert result.seleccion_modelo is not None
    assert result.seleccion_modelo.material_key == material_key_esperado
    assert result.seleccion_modelo.modelo_resuelto == modelo
    assert tuple(result.seleccion_modelo.regresores_resueltos) == regresores
    assert result.seleccion_modelo.no_calibrado is False


def test_material_no_calibrado_cae_a_prophet_base(monkeypatch: pytest.MonkeyPatch) -> None:
    limpiar_forecast_cache()
    material = SimpleNamespace(id=999, nombre="Material sin calibrar", unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda _material, _repo: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda _cache_key: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.guardar_forecast_snapshot",
        lambda _cache_key, result: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_dataset_prophet",
        lambda puntos, objetivo: dataset,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.importar_dependencias_forecast",
        lambda: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.configurar_cmdstan",
        lambda *args: None,
    )

    def fake_forecast(_material, _horizonte, _dataset, _pd, _prophet, plan):
        return _forecast_result(
            "110.00",
            modelo=plan.modelo,
            supuesto_regresores=plan.supuesto_regresores,
            seleccion_modelo=plan.seleccion_modelo,
        )

    monkeypatch.setattr("app.modules.pricing.application.forecast_service._forecast_material", fake_forecast)

    result = forecast_material(material, 3, object(), usar_selector_modelo=True)

    assert result.modelo == "prophet_base"
    assert result.seleccion_modelo is not None
    assert result.seleccion_modelo.material_key == "material-sin-calibrar"
    assert result.seleccion_modelo.no_calibrado is True


def test_selector_activado_hace_fallback_por_material_si_no_hay_horizonte_exacto(monkeypatch: pytest.MonkeyPatch) -> None:
    limpiar_forecast_cache()
    material = SimpleNamespace(id=4, nombre="Pastina", unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda _material, _repo: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda _cache_key: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.guardar_forecast_snapshot",
        lambda _cache_key, result: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_dataset_prophet",
        lambda puntos, objetivo: dataset,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.importar_dependencias_forecast",
        lambda: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.configurar_cmdstan",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_regresores_mensuales",
        lambda _pd, columnas: {"columnas": columnas},
    )

    def fake_forecast(_material, _horizonte, _dataset, _pd, _prophet, plan):
        return _forecast_result(
            "110.00",
            modelo=plan.modelo,
            supuesto_regresores=plan.supuesto_regresores,
            seleccion_modelo=plan.seleccion_modelo,
        )

    monkeypatch.setattr("app.modules.pricing.application.forecast_service._forecast_material", fake_forecast)

    result = forecast_material(material, 5, object(), usar_selector_modelo=True)

    assert result.modelo == "prophet_ipim_cac_labour_force"
    assert result.seleccion_modelo is not None
    assert result.seleccion_modelo.material_key == "pastina"
    assert result.seleccion_modelo.origen_decision == "material_default"
    assert result.seleccion_modelo.no_calibrado is True


def test_regresor_faltante_no_rompe_el_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.modules.pricing.domain.exceptions import ExternalRegressorUnavailableError

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_regresores_mensuales",
        lambda _pd, _columnas: (_ for _ in ()).throw(ExternalRegressorUnavailableError("Regresor faltante")),
    )

    selection = _resolver_plan_ejecucion(material_key="cemento-portland", horizonte_meses=3, usar_selector_modelo=True, pd=object())

    assert selection.modelo == "prophet_base"
    assert selection.seleccion_modelo is not None
    assert selection.seleccion_modelo.material_key == "cemento-portland"
    assert selection.seleccion_modelo.no_calibrado is True
    assert selection.seleccion_modelo.advertencia is not None


def test_regresor_faltante_no_rompe_el_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")

    monkeypatch.setattr("app.modules.pricing.interfaces.routes.USAR_SELECTOR_MODELO_FORECAST", True)
    monkeypatch.setattr(
        "app.modules.pricing.interfaces.routes.forecast_material",
        lambda _material, _horizonte, _repo, usar_selector_modelo=False: _forecast_result(
            "110.00",
            modelo="prophet_base",
            supuesto_regresores="Fallback controlado a prophet_base por regresores no disponibles.",
            seleccion_modelo=ForecastSelectionRead(
                modelo_resuelto="prophet_base",
                regresores_resueltos=[],
                mape_referencia=None,
                mae_referencia=None,
                folds=None,
                confiabilidad="no_calibrada",
                origen_decision="fallback_regresores",
                justificacion="Se degrada a prophet_base por faltante operativo de regresores.",
                no_calibrado=True,
                advertencia="Regresor faltante",
                material_key="cemento-portland",
            ),
        ),
    )

    response = obtener_forecast_material(
        1,
        3,
        material_repo=SimpleNamespace(get_by_id=lambda _material_id: material),
        pricing_repo=object(),
    )

    assert response.modelo == "prophet_base"
    assert response.seleccion_modelo is not None
    assert response.seleccion_modelo.advertencia == "Regresor faltante"
    assert response.seleccion_modelo.material_key == "cemento-portland"


def test_respuesta_expone_metadatos_de_seleccion_cuando_corresponde(monkeypatch: pytest.MonkeyPatch) -> None:
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    selection = ForecastSelectionRead(
        modelo_resuelto="prophet_ipim_nivel_general",
        regresores_resueltos=["ipim_nivel_general"],
        mape_referencia=Decimal("4.98"),
        mae_referencia=Decimal("6.76"),
        folds=9,
        confiabilidad="alta",
        origen_decision="material_horizonte",
        justificacion="Configuracion recomendada.",
        no_calibrado=False,
        material_key="cemento-portland",
    )

    monkeypatch.setattr("app.modules.pricing.interfaces.routes.USAR_SELECTOR_MODELO_FORECAST", True)
    monkeypatch.setattr(
        "app.modules.pricing.interfaces.routes.forecast_material",
        lambda _material, _horizonte, _repo, usar_selector_modelo=False: _forecast_result(
            "110.00",
            modelo="prophet_ipim_nivel_general",
            supuesto_regresores="Regresores resueltos: ipim_nivel_general.",
            seleccion_modelo=selection,
        ),
    )

    response = obtener_forecast_material(
        1,
        3,
        material_repo=SimpleNamespace(get_by_id=lambda _material_id: material),
        pricing_repo=object(),
    )

    assert response.modelo == "prophet_ipim_nivel_general"
    assert response.seleccion_modelo is not None
    assert response.seleccion_modelo.modelo_resuelto == "prophet_ipim_nivel_general"
    assert response.seleccion_modelo.mape_referencia == Decimal("4.98")
    assert response.seleccion_modelo.material_key == "cemento-portland"


def test_normalizacion_por_kg_no_cambia_con_selector_activado(monkeypatch: pytest.MonkeyPatch) -> None:
    limpiar_forecast_cache()
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30
    objetivo_usado = {"valor": None}

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda _material, _repo: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda _cache_key: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.guardar_forecast_snapshot",
        lambda _cache_key, result: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.importar_dependencias_forecast",
        lambda: (object(), object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.configurar_cmdstan",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_regresores_mensuales",
        lambda _pd, columnas: {"columnas": columnas},
    )

    def fake_dataset(_puntos, objetivo):
        objetivo_usado["valor"] = objetivo
        return dataset

    def fake_forecast(_material, _horizonte, _dataset, _pd, _prophet, plan):
        return _forecast_result(
            "110.00",
            modelo=plan.modelo,
            supuesto_regresores=plan.supuesto_regresores,
            seleccion_modelo=plan.seleccion_modelo,
        )

    monkeypatch.setattr("app.modules.pricing.application.forecast_service.construir_dataset_prophet", fake_dataset)
    monkeypatch.setattr("app.modules.pricing.application.forecast_service._forecast_material", fake_forecast)

    forecast_material(material, 3, object(), usar_selector_modelo=True)

    assert objetivo_usado["valor"] == "precio_promedio_normalizado"


def test_obtener_forecast_cacheado_success(monkeypatch):
    from time import monotonic

    from app.modules.pricing.application.forecast_service import (
        ForecastCacheEntry,
        ForecastCacheKey,
        _forecast_cache,
        obtener_forecast_cacheado,
    )
    limpiar_forecast_cache()
    res = _forecast_result("100.00")
    cache_key = ForecastCacheKey(material_id=1, horizonte_meses=3, dataset_signature="sig")
    _forecast_cache[cache_key] = ForecastCacheEntry(result=res, expires_at=monotonic() + 100)
    
    assert obtener_forecast_cacheado(1, 3, "sig") == res


def test_precomputar_forecasts_materiales_multiple(monkeypatch):
    material = SimpleNamespace(id=1, nombre="Cemento", unidad_base="kg")
    mock_repo = MagicMock()
    mock_repo.list_active.return_value = [material]
    
    llamadas = []
    def mock_forecast(m, h, r, **_kwargs):
        llamadas.append((m.id, h))
        return
    
    monkeypatch.setattr("app.modules.pricing.application.forecast_service.forecast_material", mock_forecast)
    
    from app.modules.pricing.application.forecast_service import precomputar_forecasts_materiales
    completados = precomputar_forecasts_materiales(mock_repo, object(), horizontes=(3, 6))
    
    assert completados == [(1, 3), (1, 6)]
    assert llamadas == [(1, 3), (1, 6)]

def test_backtesting_forecast_insufficient_data(monkeypatch):
    from app.modules.pricing.application.forecast_service import backtesting_forecast
    monkeypatch.setattr("app.modules.pricing.application.forecast_service.construir_folds_temporales", lambda *args, **kwargs: [])
    
    with pytest.raises(HTTPException) as exc:
        backtesting_forecast(None, None, [], None, 3, [])
    assert exc.value.status_code == 422

def test_pronosticar_futuro_missing_regressors(monkeypatch):
    import pandas as pd

    from app.modules.pricing.application.forecast_service import pronosticar_futuro
    
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)]
    # Provide regressor ONLY after the last date in dataset
    regresores_df = pd.DataFrame({"ds": [pd.to_datetime("2024-02-01")], "r1": [1.0]})
    
    with pytest.raises(HTTPException) as exc:
        pronosticar_futuro(pd, MagicMock(), dataset, regresores_df, ("r1",), 3, "un", "mat")
    assert "No hay regresores externos alineados" in exc.value.detail

def test_forecast_material_loads_snapshot(monkeypatch):
    limpiar_forecast_cache()
    material = SimpleNamespace(id=1, nombre="Cemento", unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30
    
    monkeypatch.setattr("app.modules.pricing.application.forecast_service.serie_mensual_material", lambda *args: ["s"])
    monkeypatch.setattr("app.modules.pricing.application.forecast_service.construir_dataset_prophet", lambda pontos, objetivo: dataset)
    
    res = _forecast_result("100.00")
    monkeypatch.setattr("app.modules.pricing.application.forecast_service.cargar_forecast_snapshot", lambda *args: res)
    
    result = forecast_material(material, 3, object())
    assert result.resolution_source == "snapshot"
    assert result.forecast[0].precio_proyectado == Decimal("100.00")


def test_forecast_material_requiere_snapshot_si_calculo_sincrono_esta_deshabilitado(monkeypatch):
    from app.modules.pricing.domain.exceptions import ForecastSnapshotRequired

    limpiar_forecast_cache()
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    dataset = [ProphetRow(ds=date(2024, 1, 1), y=100.0)] * 30
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.serie_mensual_material",
        lambda *_args: ["serie"],
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_dataset_prophet",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.cargar_forecast_snapshot",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.settings.forecast_allow_synchronous_compute",
        False,
    )

    with pytest.raises(ForecastSnapshotRequired, match="calculo sincrono esta deshabilitado"):
        forecast_material(material, 3, object())

def test_precomputar_forecasts_materiales(monkeypatch):
    material = SimpleNamespace(id=1, nombre="Cemento", unidad_base="kg")
    mock_repo = MagicMock()
    mock_repo.list_active.return_value = [material]
    
    llamadas = []
    def mock_forecast(m, h, r, **_kwargs):
        llamadas.append((m.id, h))
    monkeypatch.setattr("app.modules.pricing.application.forecast_service.forecast_material", mock_forecast)
    
    from app.modules.pricing.application.forecast_service import precomputar_forecasts_materiales
    completados = precomputar_forecasts_materiales(mock_repo, object(), horizontes=(3,))
    
    assert completados == [(1, 3)]
    assert llamadas == [(1, 3)]

def test_backtesting_forecast_with_regressors(monkeypatch):
    import pandas as pd

    from app.modules.pricing.application.forecast_service import backtesting_forecast
    
    dataset = [ProphetRow(ds=date(2024, (i % 12) + 1, 1), y=100.0) for i in range(36)]
    regresores_df = pd.DataFrame({
        "ds": [pd.to_datetime(f"2024-{(i % 12) + 1:02d}-01") for i in range(36)],
        "r1": [1.0] * 36
    })
    
    mock_prophet_instance = MagicMock()
    mock_prophet_instance.predict.return_value = pd.DataFrame({"ds": regresores_df["ds"], "yhat": [100.0] * 36})
    mock_prophet_class = MagicMock(return_value=mock_prophet_instance)
    
    # Mock construir_folds_temporales to return one fold
    from app.modules.pricing.application.backtesting import TimeSeriesFold
    fold = TimeSeriesFold(indice=1, train=dataset[:30], test=dataset[30:])
    monkeypatch.setattr("app.modules.pricing.application.forecast_service.construir_folds_temporales", lambda *args, **kwargs: [fold])
    
    metrics = backtesting_forecast(pd, mock_prophet_class, dataset, regresores_df, 3, ("r1",))
    assert metrics.mape == Decimal("0.00")
    assert mock_prophet_instance.add_regressor.called
    assert mock_prophet_instance.fit.called


def test_backtesting_con_regresores_no_usa_valores_reales_del_periodo_test(monkeypatch):
    import pandas as pd

    from app.modules.pricing.application.backtesting import TimeSeriesFold
    from app.modules.pricing.application.forecast_service import backtesting_forecast

    def month(year: int, month_index: int) -> date:
        zero_based = month_index - 1
        return date(year + zero_based // 12, zero_based % 12 + 1, 1)

    dataset = [
        ProphetRow(ds=month(2022, i), y=100.0)
        for i in range(1, 28)
    ]
    fold = TimeSeriesFold(indice=1, train=dataset[:24], test=dataset[24:27])
    regresores_df = pd.DataFrame(
        {
            "ds": [pd.to_datetime(row.ds) for row in dataset],
            "r1": [float(i) for i in range(1, 25)] + [9999.0, 9999.0, 9999.0],
        }
    )

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.construir_folds_temporales",
        lambda *_args, **_kwargs: [fold],
    )

    proyectados = pd.DataFrame(
        {
            "ds": [pd.to_datetime(row.ds) for row in fold.test],
            "r1": [25.0, 26.0, 27.0],
        }
    )
    llamada_proyeccion = {}

    def fake_proyectar(_pd, historicos, fechas_futuras, columnas):
        llamada_proyeccion["max_ds"] = historicos["ds"].max()
        llamada_proyeccion["fechas"] = list(fechas_futuras)
        llamada_proyeccion["columnas"] = columnas
        assert 9999.0 not in historicos["r1"].tolist()
        return proyectados.copy()

    monkeypatch.setattr(
        "app.modules.pricing.application.forecast_service.proyectar_regresores_futuros",
        fake_proyectar,
    )

    class FakeProphet:
        def __init__(self, **_kwargs):
            self.future_seen = None

        def add_regressor(self, _name):
            return None

        def fit(self, _df):
            return self

        def predict(self, future):
            self.future_seen = future.copy()
            assert future["r1"].tolist() == [25.0, 26.0, 27.0]
            assert 9999.0 not in future["r1"].tolist()
            return pd.DataFrame(
                {
                    "ds": future["ds"],
                    "yhat": [100.0] * len(future),
                }
            )

    metrics = backtesting_forecast(
        pd,
        FakeProphet,
        dataset,
        regresores_df,
        3,
        ("r1",),
    )

    assert llamada_proyeccion["max_ds"] == pd.Timestamp("2023-12-01")
    assert llamada_proyeccion["fechas"] == [row.ds for row in fold.test]
    assert llamada_proyeccion["columnas"] == ("r1",)
    assert metrics.mape == Decimal("0.00")
