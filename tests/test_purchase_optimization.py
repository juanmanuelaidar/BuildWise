from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.modules.auth.interfaces.dependencies import get_current_user
from app.modules.catalog.interfaces.dependencies import get_material_repository
from app.modules.pricing.application import purchase_optimization as purchase_optimization_module
from app.modules.pricing.application.purchase_optimization import (
    OptimizationCandidate,
    PurchaseOptimizationInputItem,
    generar_recomendacion_operativa_compra,
    optimizar_compra_con_presupuesto,
    optimizar_compra_items,
)
from app.modules.pricing.interfaces.dependencies import get_pricing_repository
from app.modules.pricing.interfaces.schemas import ForecastMetricasRead, ForecastPuntoRead, ForecastSelectionRead


def _candidate(
    *,
    material_id: int,
    material_key: str,
    cantidad_objetivo: str,
    precio_actual: str,
    precio_proyectado: str,
    criticidad: str,
    peso_criticidad: str,
    confiabilidad: str = "alta",
    no_calibrado: bool = False,
    porcentaje_minimo_compra_inmediata: Decimal | None = None,
) -> OptimizationCandidate:
    return OptimizationCandidate(
        material_id=material_id,
        material_key=material_key,
        cantidad_objetivo=Decimal(cantidad_objetivo),
        precio_actual=Decimal(precio_actual),
        precio_proyectado_horizonte=Decimal(precio_proyectado),
        ahorro_unitario_estimado=max(Decimal(precio_proyectado) - Decimal(precio_actual), Decimal("0")).quantize(
            Decimal("0.01")
        ),
        criticidad=criticidad,
        peso_criticidad=Decimal(peso_criticidad),
        confiabilidad=confiabilidad,
        no_calibrado=no_calibrado,
        porcentaje_minimo_compra_inmediata=porcentaje_minimo_compra_inmediata,
    )


def _fake_forecast_result(
    *,
    actual: str,
    proyectado: str,
    confiabilidad: str = "alta",
    no_calibrado: bool = False,
):
    return SimpleNamespace(
        dataset=[SimpleNamespace(ds=date(2024, 1, 1), y=float(actual))],
        forecast=[ForecastPuntoRead(fecha=date(2024, 4, 1), precio_proyectado=Decimal(proyectado))],
        metricas=ForecastMetricasRead(
            folds=9,
            mae=Decimal("6.76"),
            mape=Decimal("4.98"),
            efectividad_informal=Decimal("95.02"),
        ),
        seleccion_modelo=ForecastSelectionRead(
            material_key="material-key",
            modelo_resuelto="prophet_base",
            regresores_resueltos=[],
            mape_referencia=Decimal("4.98"),
            mae_referencia=Decimal("6.76"),
            folds=9,
            confiabilidad=confiabilidad,
            origen_decision="material_horizonte",
            justificacion="Configuracion recomendada.",
            no_calibrado=no_calibrado,
            advertencia=None,
        ),
    )


def test_respeta_el_presupuesto_total() -> None:
    result = optimizar_compra_items(
        presupuesto_total=Decimal("120.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="150.00",
                criticidad="alta",
                peso_criticidad="3.00",
            ),
            _candidate(
                material_id=2,
                material_key="arena",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="120.00",
                criticidad="media",
                peso_criticidad="2.00",
            ),
        ],
    )

    assert result.presupuesto_utilizado <= Decimal("120.00")


def test_no_compra_mas_que_la_cantidad_objetivo() -> None:
    result = optimizar_compra_items(
        presupuesto_total=Decimal("1000.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="5.0000",
                precio_actual="100.00",
                precio_proyectado="120.00",
                criticidad="alta",
                peso_criticidad="3.00",
            )
        ],
    )

    assert result.items[0].cantidad_recomendada_comprar_ahora <= Decimal("5.0000")
    assert result.items[0].cantidad_recomendada_postergar >= Decimal("0.0000")


def test_prioriza_ahorro_ponderado_por_criticidad_bajo_presupuesto_restrictivo() -> None:
    result = optimizar_compra_items(
        presupuesto_total=Decimal("100.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="130.00",
                criticidad="alta",
                peso_criticidad="3.00",
            ),
            _candidate(
                material_id=2,
                material_key="pastina",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="140.00",
                criticidad="baja",
                peso_criticidad="1.00",
            ),
        ],
    )

    cantidades = {item.material_id: item.cantidad_recomendada_comprar_ahora for item in result.items}
    assert cantidades[1] == Decimal("1.0000")
    assert cantidades[2] == Decimal("0.0000")


def test_material_con_baja_no_se_prioriza_si_el_beneficio_esperado_es_cero() -> None:
    result = optimizar_compra_items(
        presupuesto_total=Decimal("1000.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="10.0000",
                precio_actual="100.00",
                precio_proyectado="90.00",
                criticidad="alta",
                peso_criticidad="3.00",
            )
        ],
    )

    assert result.items[0].cantidad_recomendada_comprar_ahora == Decimal("0.0000")


def test_con_presupuesto_suficiente_compra_toda_la_cantidad_objetivo_con_beneficio_positivo() -> None:
    result = optimizar_compra_items(
        presupuesto_total=Decimal("1000.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="5.0000",
                precio_actual="100.00",
                precio_proyectado="120.00",
                criticidad="alta",
                peso_criticidad="3.00",
            ),
            _candidate(
                material_id=2,
                material_key="pastina",
                cantidad_objetivo="3.0000",
                precio_actual="50.00",
                precio_proyectado="60.00",
                criticidad="media",
                peso_criticidad="2.00",
            ),
        ],
    )

    cantidades = {item.material_id: item.cantidad_recomendada_comprar_ahora for item in result.items}
    assert cantidades[1] == Decimal("5.0000")
    assert cantidades[2] == Decimal("3.0000")


def test_con_presupuesto_limitado_asigna_primero_al_mayor_ahorro_ponderado() -> None:
    result = optimizar_compra_items(
        presupuesto_total=Decimal("100.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="120.00",
                criticidad="alta",
                peso_criticidad="3.00",
            ),
            _candidate(
                material_id=2,
                material_key="pastina",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="125.00",
                criticidad="media",
                peso_criticidad="2.00",
            ),
        ],
    )

    cantidades = {item.material_id: item.cantidad_recomendada_comprar_ahora for item in result.items}
    assert cantidades[1] == Decimal("1.0000")
    assert cantidades[2] == Decimal("0.0000")


def test_respeta_minimo_de_compra_inmediata_por_criticidad_si_se_informa() -> None:
    result = optimizar_compra_items(
        presupuesto_total=Decimal("100.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="110.00",
                criticidad="alta",
                peso_criticidad="3.00",
            ),
            _candidate(
                material_id=2,
                material_key="pastina",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="140.00",
                criticidad="baja",
                peso_criticidad="1.00",
            ),
        ],
    )
    with_minimum = optimizar_compra_items(
        presupuesto_total=Decimal("100.00"),
        horizonte_meses=3,
        candidates=[
            _candidate(
                material_id=1,
                material_key="cemento-portland",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="110.00",
                criticidad="alta",
                peso_criticidad="3.00",
                porcentaje_minimo_compra_inmediata=Decimal("1.0000"),
            ),
            _candidate(
                material_id=2,
                material_key="pastina",
                cantidad_objetivo="1.0000",
                precio_actual="100.00",
                precio_proyectado="140.00",
                criticidad="baja",
                peso_criticidad="1.00",
            ),
        ],
    )

    cantidades_sin_minimo = {item.material_id: item.cantidad_recomendada_comprar_ahora for item in result.items}
    cantidades_con_minimo = {item.material_id: item.cantidad_recomendada_comprar_ahora for item in with_minimum.items}
    assert cantidades_sin_minimo[1] == Decimal("0.0000")
    assert cantidades_con_minimo[1] == Decimal("1.0000")


def test_agrega_advertencia_si_hay_baja_confiabilidad(monkeypatch) -> None:
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    monkeypatch.setattr(
        purchase_optimization_module,
        "forecast_material",
        lambda *args, **kwargs: _fake_forecast_result(actual="100.00", proyectado="120.00", confiabilidad="baja"),
    )

    class FakeMaterialRepo:
        def get_by_id(self, material_id: int):
            return material if material_id == 1 else None

    result = optimizar_compra_con_presupuesto(
        presupuesto_total=Decimal("1000.00"),
        horizonte_meses=3,
        materiales=[PurchaseOptimizationInputItem(material_id=1, cantidad_objetivo=Decimal("5"), criticidad="alta")],
        material_repo=FakeMaterialRepo(),
        pricing_repo=object(),
    )

    assert result.advertencias
    assert any("confiabilidad baja" in advertencia.lower() for advertencia in result.advertencias)


def test_agrega_advertencia_si_el_modelo_viene_no_calibrado(monkeypatch) -> None:
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    monkeypatch.setattr(
        purchase_optimization_module,
        "forecast_material",
        lambda *args, **kwargs: _fake_forecast_result(actual="100.00", proyectado="120.00", no_calibrado=True),
    )

    class FakeMaterialRepo:
        def get_by_id(self, material_id: int):
            return material if material_id == 1 else None

    result = optimizar_compra_con_presupuesto(
        presupuesto_total=Decimal("1000.00"),
        horizonte_meses=3,
        materiales=[PurchaseOptimizationInputItem(material_id=1, cantidad_objetivo=Decimal("5"), criticidad="alta")],
        material_repo=FakeMaterialRepo(),
        pricing_repo=object(),
    )

    assert result.advertencias
    assert any("no calibrado" in advertencia.lower() for advertencia in result.advertencias)


def test_excluye_material_sin_forecast_suficiente_y_no_rompe_toda_la_optimizacion(monkeypatch) -> None:
    materiales = {
        1: SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg"),
        2: SimpleNamespace(id=2, nombre="Pastina", unidad_base="kg"),
    }

    def fake_forecast(material, *_args, **_kwargs):
        if material.id == 1:
            raise RuntimeError("sin datos")
        return _fake_forecast_result(actual="100.00", proyectado="120.00")

    monkeypatch.setattr(purchase_optimization_module, "forecast_material", fake_forecast)

    class FakeMaterialRepo:
        def get_by_id(self, material_id: int):
            return materiales.get(material_id)

    result = optimizar_compra_con_presupuesto(
        presupuesto_total=Decimal("1000.00"),
        horizonte_meses=3,
        materiales=[
            PurchaseOptimizationInputItem(material_id=1, cantidad_objetivo=Decimal("5"), criticidad="alta"),
            PurchaseOptimizationInputItem(material_id=2, cantidad_objetivo=Decimal("5"), criticidad="media"),
        ],
        material_repo=FakeMaterialRepo(),
        pricing_repo=object(),
    )

    assert [item.material_id for item in result.items] == [2]
    assert any("se excluye material_id=1" in advertencia.lower() for advertencia in result.advertencias)


def test_endpoint_responde_con_contrato_esperado(monkeypatch) -> None:
    materiales = {
        1: SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg"),
        2: SimpleNamespace(id=2, nombre="Pastina", unidad_base="kg"),
    }

    def fake_forecast(material, *_args, **_kwargs):
        if material.id == 1:
            return _fake_forecast_result(actual="1000.00", proyectado="1084.00", confiabilidad="alta")
        return _fake_forecast_result(actual="2000.00", proyectado="1950.00", confiabilidad="alta")

    monkeypatch.setattr(purchase_optimization_module, "forecast_material", fake_forecast)

    class FakeMaterialRepo:
        def get_by_id(self, material_id: int):
            return materiales.get(material_id)

    class FakePricingRepo:
        pass

    app.dependency_overrides[get_material_repository] = lambda: FakeMaterialRepo()
    app.dependency_overrides[get_pricing_repository] = lambda: FakePricingRepo()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, rol="admin")

    try:
        client = TestClient(app)
        response = client.post(
            "/compras/optimizar-presupuesto",
            json={
                "presupuesto_total": 500000,
                "horizonte_meses": 3,
                "materiales": [
                    {"material_id": 1, "cantidad_objetivo": 100, "criticidad": "alta"},
                    {"material_id": 2, "cantidad_objetivo": 40, "criticidad": "media"},
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["presupuesto_total"] == "500000.00"
    assert body["estado_optimizacion"] == "OPTIMAL"
    assert body["items"][0]["material_id"] == 1
    assert body["items"][0]["cantidad_recomendada_comprar_ahora"] == "100.0000"
    assert body["items"][0]["cantidad_recomendada_postergar"] == "0.0000"
    assert body["items"][0]["accion_recomendada"] == "COMPRAR_AHORA"
    assert body["items"][0]["impacto_economico_pct"] == "8.4000"
    assert body["items"][1]["cantidad_recomendada_comprar_ahora"] == "0.0000"
    assert body["items"][1]["cantidad_recomendada_postergar"] == "40.0000"
    assert body["items"][1]["accion_recomendada"] == "POSTERGAR"
    assert body["fecha_base_calculo"] == "2024-01-01"
    assert body["items"][0]["fecha_base_observada"] == "2024-01-01"


def test_recomendacion_operativa_usa_ultima_fecha_observada_como_fecha_calculo(monkeypatch) -> None:
    material = SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg")
    fecha_base = date(2025, 3, 1)

    forecast = _fake_forecast_result(actual="1000.00", proyectado="1300.00", confiabilidad="alta")
    forecast.dataset = [SimpleNamespace(ds=fecha_base, y=1000.00)]
    monkeypatch.setattr(purchase_optimization_module, "forecast_material", lambda *args, **kwargs: forecast)

    class FakeMaterialRepo:
        def get_by_id(self, material_id: int):
            return material if material_id == 1 else None

    result = generar_recomendacion_operativa_compra(
        presupuesto_total=Decimal("200000.00"),
        horizonte_meses=6,
        materiales=[PurchaseOptimizationInputItem(material_id=1, cantidad_objetivo=Decimal("30"), criticidad="alta")],
        material_repo=FakeMaterialRepo(),
        pricing_repo=object(),
    )

    assert result.fecha_calculo == fecha_base
    assert result.items[0].fecha_base_observada == fecha_base


def test_endpoint_recomendacion_operativa_consolida_decision_trazable(monkeypatch) -> None:
    materiales = {
        1: SimpleNamespace(id=1, nombre="Cemento Portland", unidad_base="kg"),
        2: SimpleNamespace(id=2, nombre="Pastina", unidad_base="kg"),
        3: SimpleNamespace(id=3, nombre="Membrana Megaflex", unidad_base="m2"),
    }

    def fake_forecast(material, *_args, **_kwargs):
        proyectados = {
            1: ("1000.00", "1084.00"),
            2: ("2000.00", "2100.00"),
            3: ("1500.00", "1480.00"),
        }
        actual, proyectado = proyectados[material.id]
        return _fake_forecast_result(actual=actual, proyectado=proyectado, confiabilidad="alta")

    monkeypatch.setattr(purchase_optimization_module, "forecast_material", fake_forecast)

    class FakeMaterialRepo:
        def get_by_id(self, material_id: int):
            return materiales.get(material_id)

    class FakePricingRepo:
        pass

    app.dependency_overrides[get_material_repository] = lambda: FakeMaterialRepo()
    app.dependency_overrides[get_pricing_repository] = lambda: FakePricingRepo()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, rol="admin")

    try:
        client = TestClient(app)
        response = client.post(
            "/compras/recomendacion-operativa",
            json={
                "presupuesto_total": 180000,
                "horizonte_meses": 3,
                "materiales": [
                    {"material_id": 1, "cantidad_objetivo": 100, "criticidad": "alta"},
                    {"material_id": 2, "cantidad_objetivo": 40, "criticidad": "media"},
                    {"material_id": 3, "cantidad_objetivo": 20, "criticidad": "baja"},
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["fecha_calculo"] == "2024-01-01"
    assert body["horizonte_meses"] == 3
    assert body["presupuesto_total"] == "180000.00"
    assert body["ahorro_total_estimado"] != "0.00"
    assert body["decision_resumen"]
    assert len(body["items"]) == 3
    assert body["supuestos"]
    assert all("impacto_economico_pct" in item for item in body["items"])
    assert all("recomendacion_simple" in item for item in body["items"])
    assert all("mejor_estrategia" in item for item in body["items"])
    assert all("ventaja_estrategia_significativa" in item for item in body["items"])
    assert all(item["fecha_base_observada"] == "2024-01-01" for item in body["items"])
    assert all(item["accion_recomendada"] in {"COMPRAR_AHORA", "POSTERGAR", "COMPRA_PARCIAL"} for item in body["items"])


def test_hu23_no_usa_ortools() -> None:
    source = inspect.getsource(purchase_optimization_module).lower()
    assert "ortools" not in source


def test_precision_de_cantidades_no_excede_presupuesto_con_precio_alto():
    candidate = _candidate(
        material_id=1,
        material_key="ejemplo",
        cantidad_objetivo="1.0000",
        precio_actual="100000.00",
        precio_proyectado="120000.00",
        criticidad="alta",
        peso_criticidad="3",
    )
    result = optimizar_compra_items(presupuesto_total=Decimal("15.00"), horizonte_meses=3, candidates=[candidate])
    item = result.items[0]
    assert item.cantidad_recomendada_comprar_ahora == Decimal("0.0001")
    assert result.presupuesto_utilizado == Decimal("10.00")
    assert item.cantidad_recomendada_comprar_ahora + item.cantidad_recomendada_postergar == item.cantidad_objetivo


def test_precision_de_cantidades_respeta_minimo_no_representable():
    candidate = _candidate(
        material_id=1,
        material_key="ejemplo",
        cantidad_objetivo="1.0000",
        precio_actual="100.00",
        precio_proyectado="120.00",
        criticidad="alta",
        peso_criticidad="3",
        porcentaje_minimo_compra_inmediata=Decimal("0.333333"),
    )
    result = optimizar_compra_items(presupuesto_total=Decimal("33.34"), horizonte_meses=3, candidates=[candidate])
    assert result.items[0].cantidad_recomendada_comprar_ahora == Decimal("0.3334")
    assert result.presupuesto_utilizado <= result.presupuesto_total
