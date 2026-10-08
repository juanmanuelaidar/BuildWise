from decimal import Decimal

from app.modules.pricing.application.model_selector import (
    MATERIAL_KEY_CEMENTO_PORTLAND,
    MATERIAL_KEY_MEMBRANA_MEGAFLEX,
    MATERIAL_KEY_PASTINA,
    ORIGEN_DECISION_GLOBAL_FALLBACK,
    ORIGEN_DECISION_MATERIAL_DEFAULT,
    ORIGEN_DECISION_MATERIAL_HORIZONTE,
    resolve_model_selection,
)


def test_resuelve_seleccion_exacta_por_material_y_horizonte() -> None:
    selection = resolve_model_selection(MATERIAL_KEY_CEMENTO_PORTLAND, 3)

    assert selection.material_key == MATERIAL_KEY_CEMENTO_PORTLAND
    assert selection.horizonte_meses == 3
    assert selection.modelo == "prophet_ipim_nivel_general"
    assert selection.regresores == ("ipim_nivel_general",)
    assert selection.mae == Decimal("14.17")
    assert selection.mape == Decimal("10.35")
    assert selection.folds == 9
    assert selection.confiabilidad == "no_calibrada"
    assert selection.origen_decision == ORIGEN_DECISION_MATERIAL_HORIZONTE
    assert selection.no_calibrado is True


def test_resuelve_fallback_por_material_si_no_hay_horizonte_exacto() -> None:
    selection = resolve_model_selection(MATERIAL_KEY_PASTINA, 5)

    assert selection.material_key == MATERIAL_KEY_PASTINA
    assert selection.horizonte_meses == 5
    assert selection.modelo == "prophet_ipim_cac_var_materials"
    assert selection.regresores == ("ipim_nivel_general", "cac_var_materials")
    assert selection.mae is None
    assert selection.mape is None
    assert selection.folds is None
    assert selection.confiabilidad == "no_calibrada"
    assert selection.origen_decision == ORIGEN_DECISION_MATERIAL_DEFAULT
    assert selection.no_calibrado is True
    assert "no existe una calibracion exacta" in selection.justificacion


def test_resuelve_fallback_a_prophet_base_para_material_sin_configuracion() -> None:
    selection = resolve_model_selection("material-desconocido", 3)

    assert selection.material_key == "material-desconocido"
    assert selection.horizonte_meses == 3
    assert selection.modelo == "prophet_base"
    assert selection.regresores == ()
    assert selection.mae is None
    assert selection.mape is None
    assert selection.folds is None
    assert selection.origen_decision == ORIGEN_DECISION_GLOBAL_FALLBACK


def test_fallback_global_queda_marcado_como_no_calibrado() -> None:
    selection = resolve_model_selection("material-desconocido", 12)

    assert selection.no_calibrado is True
    assert selection.confiabilidad == "no_calibrada"
    assert "fallback operativo" in selection.justificacion


def test_no_se_usa_un_modelo_global_unico_para_todos_los_materiales() -> None:
    cemento = resolve_model_selection(MATERIAL_KEY_CEMENTO_PORTLAND, 3)
    pastina = resolve_model_selection(MATERIAL_KEY_PASTINA, 3)
    membrana = resolve_model_selection(MATERIAL_KEY_MEMBRANA_MEGAFLEX, 3)

    assert cemento.modelo == "prophet_ipim_nivel_general"
    assert pastina.modelo == "prophet_ipim_cac_var_materials"
    assert membrana.modelo == "prophet_ipim_icc_var_materials"
    assert len({cemento.modelo, pastina.modelo, membrana.modelo}) == 3


def test_no_oculta_derrota_frente_al_baseline():
    selection = resolve_model_selection(MATERIAL_KEY_CEMENTO_PORTLAND, 3)
    assert selection.no_calibrado
    assert "No mejora el baseline" in selection.justificacion


def test_series_estimadas_no_aparecen_calibradas():
    for key in (MATERIAL_KEY_PASTINA, MATERIAL_KEY_MEMBRANA_MEGAFLEX):
        selection = resolve_model_selection(key, 3)
        assert selection.no_calibrado
        assert selection.confiabilidad == "no_calibrada"
        assert "precios estimados" in selection.justificacion


def test_benchmark_sin_proveniencia_no_reutiliza_metricas_legacy(monkeypatch):
    from app.modules.pricing.application import model_selector

    monkeypatch.setattr(model_selector, "_BENCHMARK_MANIFEST", {})
    assert model_selector._load_benchmark_selections() == ({}, {})


def test_benchmark_modificado_rechaza_proveniencia(tmp_path, monkeypatch):
    import shutil

    from app.modules.pricing.application import model_selector

    for name in ("manifest.json", "benchmarks.csv"):
        shutil.copy(model_selector.BENCHMARK_DIR / name, tmp_path / name)
    monkeypatch.setattr(model_selector, "BENCHMARK_DIR", tmp_path)
    assert model_selector._load_manifest()
    with (tmp_path / "benchmarks.csv").open("a") as handle:
        handle.write("modificado\n")
    assert model_selector._load_manifest() == {}
