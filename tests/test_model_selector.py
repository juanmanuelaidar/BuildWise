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
    assert selection.modelo == "prophet_ipim_icc_var_materials"
    assert selection.regresores == ("ipim_nivel_general", "icc_var_materials")
    assert selection.mae == Decimal("5.82")
    assert selection.mape == Decimal("4.22")
    assert selection.folds == 9
    assert selection.confiabilidad == "alta"
    assert selection.origen_decision == ORIGEN_DECISION_MATERIAL_HORIZONTE
    assert selection.no_calibrado is False


def test_resuelve_fallback_por_material_si_no_hay_horizonte_exacto() -> None:
    selection = resolve_model_selection(MATERIAL_KEY_PASTINA, 5)

    assert selection.material_key == MATERIAL_KEY_PASTINA
    assert selection.horizonte_meses == 5
    assert selection.modelo == "prophet_ipim_cac_labour_force"
    assert selection.regresores == ("ipim_nivel_general", "cac_labour_force")
    assert selection.mae == Decimal("97.97")
    assert selection.mape == Decimal("4.27")
    assert selection.folds == 9
    assert selection.confiabilidad == "media"
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

    assert cemento.modelo == "prophet_ipim_icc_var_materials"
    assert pastina.modelo == "prophet_ipim_cac_labour_force"
    assert membrana.modelo == "prophet_ipim_icc_var_general"
    assert len({cemento.modelo, pastina.modelo, membrana.modelo}) == 3
