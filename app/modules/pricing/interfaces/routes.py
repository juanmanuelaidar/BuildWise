from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.auth.infrastructure.models import Usuario
from app.modules.auth.interfaces.dependencies import get_current_user, require_admin
from app.modules.catalog.domain.repositories import MaterialRepository
from app.modules.catalog.interfaces.dependencies import get_material_repository
from app.modules.pricing.application.alerts import generar_alertas_proactivas
from app.modules.pricing.application.commercial_margins import (
    actualizar_margen_comercial as actualizar_margen_comercial_service,
)
from app.modules.pricing.application.commercial_margins import (
    crear_margen_comercial as crear_margen_comercial_service,
)
from app.modules.pricing.application.commercial_margins import (
    listar_margenes_comerciales as listar_margenes_comerciales_service,
)
from app.modules.pricing.application.commercial_prices import (
    calcular_precio_comercial,
)
from app.modules.pricing.application.contextual_purchase_recommendations import recomendar_estrategia_contextual
from app.modules.pricing.application.external_indices import list_external_indices, sync_external_index
from app.modules.pricing.application.forecast_service import (
    forecast_material,
)
from app.modules.pricing.application.historical_prices import (
    crear_precio_historico as crear_precio_historico_service,
)
from app.modules.pricing.application.historical_prices import (
    listar_precios_historicos as listar_precios_historicos_service,
)
from app.modules.pricing.application.historical_prices import (
    obtener_rango_precios_historicos as obtener_rango_precios_historicos_service,
)
from app.modules.pricing.application.imputation import impute_monthly_prices
from app.modules.pricing.application.priorities import priorizar_materiales_desde_forecast
from app.modules.pricing.application.purchase_optimization import (
    PurchaseOptimizationInputItem,
    generar_recomendacion_operativa_compra,
    optimizar_compra_con_presupuesto,
)
from app.modules.pricing.application.purchase_recommendations import recomendar_momento_compra
from app.modules.pricing.application.purchase_strategies import comparar_estrategias_compra
from app.modules.pricing.application.series import (
    PrecioSerieInput,
    calcular_variacion_entre_fechas,
    construir_serie_mensual,
    construir_serie_observaciones,
    construir_serie_precios,
    evaluar_anomalias_detectadas,
)
from app.modules.pricing.domain.exceptions import MaterialNotFoundException
from app.modules.pricing.domain.repositories import PricingRepository
from app.modules.pricing.infrastructure.models import Alerta, ExternalIndexValue, PrecioHistorico
from app.modules.pricing.interfaces.dependencies import get_pricing_repository
from app.modules.pricing.interfaces.schemas import (
    AlertaBatchUpdate,
    AlertaRead,
    AnomalyEvaluationCreate,
    AnomalyEvaluationRead,
    CommercialMarginCreate,
    CommercialMarginRead,
    CommercialMarginUpdate,
    CommercialPriceRead,
    ContextualPurchaseRecommendationCreate,
    ContextualPurchaseRecommendationRead,
    ExternalIndexSyncRequest,
    ExternalIndexSyncResponse,
    ExternalIndexValueRead,
    ForecastResponseRead,
    MaterialCriticidadCreate,
    MaterialCriticidadResponseRead,
    OperationalPurchaseRecommendationCreate,
    OperationalPurchaseRecommendationRead,
    PrecioHistoricoCreate,
    PrecioHistoricoRangoRead,
    PrecioHistoricoRead,
    PriceImputationRequest,
    PriceImputationResponse,
    PriceVariationBetweenDatesRead,
    PuntoSeriePrecioRead,
    PurchaseOptimizationCreate,
    PurchaseOptimizationRead,
    PurchaseRecommendationCreate,
    PurchaseRecommendationRead,
    PurchaseStrategyComparisonCreate,
    PurchaseStrategyComparisonRead,
    PurchaseTemporalSimulationCreate,
    PurchaseTemporalSimulationRead,
)
from app.shared.database.session import get_db

router = APIRouter(tags=["precios historicos"])
USAR_SELECTOR_MODELO_FORECAST = True

@router.get("/precios-historicos/rango", response_model=PrecioHistoricoRangoRead)
def obtener_rango_precios_historicos(
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
) -> PrecioHistoricoRangoRead:
    return PrecioHistoricoRangoRead(**obtener_rango_precios_historicos_service(db))


@router.get("/precios-historicos", response_model=list[PrecioHistoricoRead])
def listar_precios_historicos(
    material_id: int | None = None,
    desde: date | None = None,
    hasta: date | None = None,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
) -> list[PrecioHistorico]:
    return listar_precios_historicos_service(db, material_id=material_id, desde=desde, hasta=hasta)


@router.get("/indices-externos", response_model=list[ExternalIndexValueRead])
def listar_indices_externos(
    series_id: str | None = None,
    source_name: str | None = None,
    desde: date | None = None,
    hasta: date | None = None,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
) -> list[ExternalIndexValue]:
    return list_external_indices(
        db,
        series_id=series_id,
        source_name=source_name,
        start_date=desde,
        end_date=hasta,
    )


@router.post("/indices-externos/sync", response_model=ExternalIndexSyncResponse)
def sincronizar_indice_externo(
    payload: ExternalIndexSyncRequest,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(require_admin),
) -> ExternalIndexSyncResponse:
    result = sync_external_index(
        db,
        series_id=payload.series_id,
        source_name=payload.source_name,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
    return ExternalIndexSyncResponse(
        source_name=result.source_name,
        series_id=result.series_id,
        inserted=result.inserted,
        updated=result.updated,
        unchanged=result.unchanged,
    )


@router.get("/materiales/{material_id}/precios", response_model=list[PrecioHistoricoRead])
def listar_precios_por_material(
    material_id: int,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> list[PrecioHistorico]:
    if material_repo.get_by_id(material_id) is None:
        raise MaterialNotFoundException(material_id)
    return pricing_repo.get_historical_prices(material_id, date(2000, 1, 1))


@router.get("/materiales/{material_id}/serie-precios", response_model=list[PuntoSeriePrecioRead])
def obtener_serie_precios_material(
    material_id: int,
    desde: date | None = None,
    hasta: date | None = None,
    agrupacion: str = "dia",
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
):
    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    precios = pricing_repo.get_historical_prices(material_id, desde or date(2000, 1, 1))
    if hasta:
        precios = [p for p in precios if p.fecha <= hasta]

    registros = [
        PrecioSerieInput(
            fecha=precio.fecha,
            precio_normalizado=precio.precio_normalizado,
            unidad_base=material.unidad_base,
            fuente=precio.fuente.nombre if precio.fuente else None,
            numero_comprobante=precio.numero_comprobante,
            registro_id=getattr(precio, "id", None),
            origen_dato=getattr(precio, "origen_dato", None),
            metodo_estimacion=getattr(precio, "metodo_estimacion", None),
            presentacion_nombre=getattr(getattr(precio, "presentacion", None), "nombre_presentacion", None),
            presentacion_cantidad_base=getattr(getattr(precio, "presentacion", None), "cantidad_base", None),
            presentacion_unidad=getattr(getattr(precio, "presentacion", None), "unidad_presentacion", None),
            precio_original=getattr(precio, "precio_original", None),
            moneda=getattr(precio, "moneda", None),
            created_at=getattr(precio, "created_at", None),
        )
        for precio in precios
    ]
    if agrupacion == "mensual":
        return construir_serie_mensual(registros, material_nombre=material.nombre)
    if agrupacion == "observaciones":
        return construir_serie_observaciones(registros, material_nombre=material.nombre)
    if agrupacion != "dia":
        raise HTTPException(status_code=422, detail="La agrupacion debe ser 'observaciones', 'dia' o 'mensual'")
    return construir_serie_precios(registros, material_nombre=material.nombre)


@router.post("/materiales/{material_id}/anomalias/evaluacion", response_model=AnomalyEvaluationRead)
def evaluar_anomalias_material(
    material_id: int,
    payload: AnomalyEvaluationCreate,
    desde: date | None = None,
    hasta: date | None = None,
    agrupacion: str = "dia",
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> AnomalyEvaluationRead:
    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    precios = pricing_repo.get_historical_prices(material_id, desde or date(2000, 1, 1))
    if hasta:
        precios = [p for p in precios if p.fecha <= hasta]

    registros = [
        PrecioSerieInput(
            fecha=precio.fecha,
            precio_normalizado=precio.precio_normalizado,
            unidad_base=material.unidad_base,
            fuente=precio.fuente.nombre if precio.fuente else None,
            numero_comprobante=precio.numero_comprobante,
            registro_id=getattr(precio, "id", None),
        )
        for precio in precios
    ]
    if agrupacion == "mensual":
        serie = construir_serie_mensual(registros, material_nombre=material.nombre)
    elif agrupacion == "observaciones":
        serie = construir_serie_observaciones(registros, material_nombre=material.nombre)
    elif agrupacion == "dia":
        serie = construir_serie_precios(registros, material_nombre=material.nombre)
    else:
        raise HTTPException(status_code=422, detail="La agrupacion debe ser 'observaciones', 'dia' o 'mensual'")
    result = evaluar_anomalias_detectadas(serie, set(payload.fechas_confirmadas))
    return AnomalyEvaluationRead(
        total_puntos=result.total_puntos,
        total_detectadas=result.total_detectadas,
        total_confirmadas=result.total_confirmadas,
        verdaderos_positivos=result.verdaderos_positivos,
        falsos_positivos=result.falsos_positivos,
        falsos_negativos=result.falsos_negativos,
        precision=result.precision,
        recall=result.recall,
        f1=result.f1,
        exactitud=result.exactitud,
        fechas_detectadas=result.fechas_detectadas,
        fechas_confirmadas=result.fechas_confirmadas,
        coincidencias=result.coincidencias,
        baseline_umbral_pct=result.baseline_umbral_pct,
        baseline_total_detectadas=result.baseline_total_detectadas,
        baseline_verdaderos_positivos=result.baseline_verdaderos_positivos,
        baseline_falsos_positivos=result.baseline_falsos_positivos,
        baseline_falsos_negativos=result.baseline_falsos_negativos,
        baseline_precision=result.baseline_precision,
        baseline_recall=result.baseline_recall,
        baseline_f1=result.baseline_f1,
        baseline_fechas_detectadas=result.baseline_fechas_detectadas,
    )


@router.get("/materiales/{material_id}/variacion-entre-fechas", response_model=PriceVariationBetweenDatesRead)
def obtener_variacion_entre_fechas_material(
    material_id: int,
    fecha_desde: date,
    fecha_hasta: date,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> PriceVariationBetweenDatesRead:
    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    precios = pricing_repo.get_historical_prices(material_id, date(2000, 1, 1))
    registros = [
        PrecioSerieInput(
            fecha=precio.fecha,
            precio_normalizado=precio.precio_normalizado,
            unidad_base=material.unidad_base,
            fuente=precio.fuente.nombre if precio.fuente else None,
            numero_comprobante=precio.numero_comprobante,
        )
        for precio in precios
    ]

    try:
        result = calcular_variacion_entre_fechas(registros, fecha_desde, fecha_hasta)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PriceVariationBetweenDatesRead(
        material_id=material.id,
        material_nombre=material.nombre,
        unidad_base=material.unidad_base,
        fecha_desde_solicitada=fecha_desde,
        fecha_hasta_solicitada=fecha_hasta,
        fecha_desde_usada=result.fecha_desde,
        fecha_hasta_usada=result.fecha_hasta,
        precio_desde=result.precio_desde,
        precio_hasta=result.precio_hasta,
        variacion_porcentual=result.variacion_porcentual,
    )


@router.get("/materiales/{material_id}/forecast", response_model=ForecastResponseRead, response_model_exclude_none=True)
def obtener_forecast_material(
    material_id: int,
    horizonte_meses: int = 3,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> ForecastResponseRead:
    if horizonte_meses < 1 or horizonte_meses > 12:
        raise HTTPException(status_code=422, detail="El horizonte_meses debe estar entre 1 y 12")

    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    forecast_result = forecast_material(
        material,
        horizonte_meses,
        pricing_repo,
        usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
    )

    return ForecastResponseRead(
        material_id=material.id,
        material_nombre=material.nombre,
        unidad_base=material.unidad_base,
        horizonte_meses=horizonte_meses,
        modelo=forecast_result.modelo,
        supuesto_regresores=forecast_result.supuesto_regresores,
        ultima_fecha_observada=forecast_result.dataset[-1].ds,
        ultimo_precio_observado=Decimal(f"{forecast_result.dataset[-1].y:.2f}"),
        metricas=forecast_result.metricas,
        puntos=forecast_result.forecast,
        seleccion_modelo=forecast_result.seleccion_modelo,
    )


@router.post("/materiales/{material_id}/recomendacion-compra", response_model=PurchaseRecommendationRead)
def recomendar_momento_compra_material(
    material_id: int,
    payload: PurchaseRecommendationCreate,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> PurchaseRecommendationRead:
    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    result = recomendar_momento_compra(
        material,
        payload.horizonte_meses,
        payload.criticidad,
        payload.cantidad_objetivo,
        pricing_repo,
        usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
    )
    return PurchaseRecommendationRead(
        material_id=result.material_id,
        material_key=result.material_key,
        horizonte_meses=result.horizonte_meses,
        decision=result.decision,
        variacion_esperada_pct=result.variacion_esperada_pct,
        precio_actual=getattr(result, "precio_actual", None),
        precio_proyectado_horizonte=getattr(result, "precio_proyectado_horizonte", None),
        cantidad_objetivo=getattr(result, "cantidad_objetivo", None),
        impacto_economico_estimado=getattr(result, "impacto_economico_estimado", None),
        mape=getattr(result, "mape", None),
        umbral_decision_pct=getattr(result, "umbral_decision_pct", None),
        supera_umbral_decision=getattr(result, "supera_umbral_decision", False),
        confiabilidad=result.confiabilidad,
        criticidad=result.criticidad,
        justificacion=result.justificacion,
        advertencias=list(result.advertencias),
    )


@router.post("/materiales/{material_id}/recomendacion-contextual", response_model=ContextualPurchaseRecommendationRead)
def recomendar_estrategia_contextual_material(
    material_id: int,
    payload: ContextualPurchaseRecommendationCreate,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> ContextualPurchaseRecommendationRead:
    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    result = recomendar_estrategia_contextual(
        material,
        fase_obra=payload.fase_obra,
        tolerancia_riesgo=payload.tolerancia_riesgo,
        cantidad_objetivo=payload.cantidad_objetivo,
        horizonte_meses=payload.horizonte_meses,
        fecha_objetivo_uso=payload.fecha_objetivo_uso,
        pricing_repo=pricing_repo,
        usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
    )
    return ContextualPurchaseRecommendationRead(
        material_id=result.material_id,
        material_key=result.material_key,
        fase_obra=result.fase_obra,
        fecha_objetivo_uso=result.fecha_objetivo_uso,
        horizonte_meses=result.horizonte_meses,
        tolerancia_riesgo=result.tolerancia_riesgo,
        criticidad=result.criticidad,
        decision=result.decision,
        variacion_esperada_pct=result.variacion_esperada_pct,
        precio_actual=result.precio_actual,
        precio_proyectado_horizonte=result.precio_proyectado_horizonte,
        precio_proyectado_optimista=result.precio_proyectado_optimista,
        precio_proyectado_pesimista=result.precio_proyectado_pesimista,
        cantidad_objetivo=result.cantidad_objetivo,
        impacto_economico_estimado=result.impacto_economico_estimado,
        mape=result.mape,
        umbral_decision_pct=result.umbral_decision_pct,
        supera_umbral_decision=result.supera_umbral_decision,
        confiabilidad=result.confiabilidad,
        justificacion=result.justificacion,
        advertencias=list(result.advertencias),
    )


@router.post(
    "/materiales/{material_id}/comparacion-estrategias-compra",
    response_model=PurchaseStrategyComparisonRead,
)
def comparar_estrategias_compra_material(
    material_id: int,
    payload: PurchaseStrategyComparisonCreate,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> PurchaseStrategyComparisonRead:
    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    result = comparar_estrategias_compra(
        material,
        payload.horizonte_meses,
        payload.cantidad_objetivo,
        pricing_repo,
        porcentaje_compra_inmediata=payload.porcentaje_compra_inmediata,
        usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
    )

    return PurchaseStrategyComparisonRead(
        material_id=result.material_id,
        material_key=result.material_key,
        horizonte_meses=result.horizonte_meses,
        cantidad_objetivo=result.cantidad_objetivo,
        porcentaje_compra_inmediata=result.porcentaje_compra_inmediata,
        precio_actual=result.precio_actual,
        precio_proyectado_horizonte=result.precio_proyectado_horizonte,
        variacion_esperada_pct=result.variacion_esperada_pct,
        confiabilidad=result.confiabilidad,
        estrategias=[
            {
                "nombre": estrategia.nombre,
                "costo_estimado": estrategia.costo_estimado,
                "diferencia_vs_mejor_ars": getattr(estrategia, "diferencia_vs_mejor_ars", Decimal("0.00")),
                "diferencia_vs_mejor_pct": getattr(estrategia, "diferencia_vs_mejor_pct", Decimal("0.0000")),
                "riesgo": estrategia.riesgo,
                "descripcion": estrategia.descripcion,
            }
            for estrategia in result.estrategias
        ],
        mejor_estrategia=result.mejor_estrategia,
        ahorro_estimado=result.ahorro_estimado,
        umbral_decision_pct=getattr(result, "umbral_decision_pct", Decimal("0.0000")),
        ventaja_significativa=getattr(result, "ventaja_significativa", False),
        justificacion=result.justificacion,
        advertencias=list(result.advertencias),
    )


@router.post(
    "/materiales/{material_id}/simulacion-escenarios-compra",
    response_model=PurchaseTemporalSimulationRead,
)
def simular_escenarios_temporales_compra_material(
    material_id: int,
    payload: PurchaseTemporalSimulationCreate,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> PurchaseTemporalSimulationRead:
    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    simulaciones = []
    for horizonte in payload.horizontes_meses:
        result = comparar_estrategias_compra(
            material,
            horizonte,
            payload.cantidad_objetivo,
            pricing_repo,
            porcentaje_compra_inmediata=payload.porcentaje_compra_inmediata,
            usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
        )
        simulaciones.append(
            PurchaseStrategyComparisonRead(
                material_id=result.material_id,
                material_key=result.material_key,
                horizonte_meses=result.horizonte_meses,
                cantidad_objetivo=result.cantidad_objetivo,
                porcentaje_compra_inmediata=result.porcentaje_compra_inmediata,
                precio_actual=result.precio_actual,
                precio_proyectado_horizonte=result.precio_proyectado_horizonte,
                variacion_esperada_pct=result.variacion_esperada_pct,
                confiabilidad=result.confiabilidad,
                estrategias=result.estrategias,
                mejor_estrategia=result.mejor_estrategia,
                ahorro_estimado=result.ahorro_estimado,
                umbral_decision_pct=result.umbral_decision_pct,
                ventaja_significativa=result.ventaja_significativa,
                justificacion=result.justificacion,
                advertencias=list(result.advertencias),
            )
        )

    simulaciones.sort(key=lambda s: s.horizonte_meses)

    return PurchaseTemporalSimulationRead(
        material_id=material.id,
        material_key=simulaciones[0].material_key,
        cantidad_objetivo=payload.cantidad_objetivo,
        porcentaje_compra_inmediata=payload.porcentaje_compra_inmediata,
        simulaciones=simulaciones,
    )


@router.post("/compras/optimizar-presupuesto", response_model=PurchaseOptimizationRead)
def optimizar_presupuesto_compra(
    payload: PurchaseOptimizationCreate,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> PurchaseOptimizationRead:
    result = optimizar_compra_con_presupuesto(
        presupuesto_total=payload.presupuesto_total,
        horizonte_meses=payload.horizonte_meses,
        materiales=[
            PurchaseOptimizationInputItem(
                material_id=item.material_id,
                cantidad_objetivo=item.cantidad_objetivo,
                criticidad=item.criticidad,
                porcentaje_minimo_compra_inmediata=item.porcentaje_minimo_compra_inmediata,
            )
            for item in payload.materiales
        ],
        material_repo=material_repo,
        pricing_repo=pricing_repo,
        usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
    )

    return PurchaseOptimizationRead(
        presupuesto_total=result.presupuesto_total,
        presupuesto_utilizado=result.presupuesto_utilizado,
        presupuesto_restante=result.presupuesto_restante,
        horizonte_meses=result.horizonte_meses,
        estado_optimizacion=result.estado_optimizacion,
        items=[
            {
                "material_id": item.material_id,
                "material_key": item.material_key,
                "cantidad_objetivo": item.cantidad_objetivo,
                "cantidad_recomendada_comprar_ahora": item.cantidad_recomendada_comprar_ahora,
                "cantidad_recomendada_postergar": item.cantidad_recomendada_postergar,
                "precio_actual": item.precio_actual,
                "precio_proyectado_horizonte": item.precio_proyectado_horizonte,
                "costo_compra_ahora": item.costo_compra_ahora,
                "costo_futuro_estimado": item.costo_futuro_estimado,
                "ahorro_unitario_estimado": item.ahorro_unitario_estimado,
                "ahorro_total_estimado": item.ahorro_total_estimado,
                "impacto_economico_pct": item.impacto_economico_pct,
                "accion_recomendada": item.accion_recomendada,
                "criticidad": item.criticidad,
                "peso_criticidad": item.peso_criticidad,
                "confiabilidad": item.confiabilidad,
                "fecha_base_observada": item.fecha_base_observada,
            }
            for item in result.items
        ],
        ahorro_total_estimado=result.ahorro_total_estimado,
        justificacion=result.justificacion,
        advertencias=list(result.advertencias),
        fecha_base_calculo=result.fecha_base_calculo,
    )


@router.post("/compras/recomendacion-operativa", response_model=OperationalPurchaseRecommendationRead)
def generar_recomendacion_operativa(
    payload: OperationalPurchaseRecommendationCreate,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> OperationalPurchaseRecommendationRead:
    result = generar_recomendacion_operativa_compra(
        presupuesto_total=payload.presupuesto_total,
        horizonte_meses=payload.horizonte_meses,
        materiales=[
            PurchaseOptimizationInputItem(
                material_id=item.material_id,
                cantidad_objetivo=item.cantidad_objetivo,
                criticidad=item.criticidad,
                porcentaje_minimo_compra_inmediata=item.porcentaje_minimo_compra_inmediata,
            )
            for item in payload.materiales
        ],
        material_repo=material_repo,
        pricing_repo=pricing_repo,
        usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
    )

    return OperationalPurchaseRecommendationRead(
        fecha_calculo=result.fecha_calculo,
        horizonte_meses=result.horizonte_meses,
        presupuesto_total=result.presupuesto_total,
        presupuesto_utilizado=result.presupuesto_utilizado,
        presupuesto_restante=result.presupuesto_restante,
        ahorro_total_estimado=result.ahorro_total_estimado,
        decision_resumen=result.decision_resumen,
        items=[
            {
                "material_id": item.material_id,
                "material_key": item.material_key,
                "accion_recomendada": item.accion_recomendada,
                "cantidad_comprar_ahora": item.cantidad_comprar_ahora,
                "cantidad_postergar": item.cantidad_postergar,
                "impacto_economico_estimado": item.impacto_economico_estimado,
                "impacto_economico_pct": item.impacto_economico_pct,
                "confianza": item.confianza,
                "criticidad": item.criticidad,
                "recomendacion_simple": item.recomendacion_simple,
                "mejor_estrategia": item.mejor_estrategia,
                "ventaja_estrategia_significativa": item.ventaja_estrategia_significativa,
                "explicacion": item.explicacion,
                "fecha_base_observada": item.fecha_base_observada,
            }
            for item in result.items
        ],
        supuestos=list(result.supuestos),
        advertencias=list(result.advertencias),
    )


@router.post("/materiales/criticidad", response_model=MaterialCriticidadResponseRead)
def priorizar_materiales_por_criticidad(
    payload: MaterialCriticidadCreate,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    current_user: Usuario = Depends(get_current_user),
) -> MaterialCriticidadResponseRead:
    if payload.alpha == 0 and payload.beta == 0:
        raise HTTPException(status_code=422, detail="alpha y beta no pueden ser ambos cero")
    return priorizar_materiales_desde_forecast(payload, material_repo, pricing_repo)


@router.get("/admin/margenes", response_model=list[CommercialMarginRead])
def listar_margenes_comerciales(
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(require_admin),
) -> list[CommercialMarginRead]:
    margins = listar_margenes_comerciales_service(db)
    return [CommercialMarginRead.model_validate(margin) for margin in margins]


@router.post("/admin/margenes", response_model=CommercialMarginRead, status_code=status.HTTP_201_CREATED)
def crear_margen_comercial(
    payload: CommercialMarginCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(require_admin),
) -> CommercialMarginRead:
    margin = crear_margen_comercial_service(db, **payload.model_dump())
    return CommercialMarginRead.model_validate(margin)


@router.patch("/admin/margenes/{margin_id}", response_model=CommercialMarginRead)
def actualizar_margen_comercial(
    margin_id: int,
    payload: CommercialMarginUpdate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(require_admin),
) -> CommercialMarginRead:
    margin = actualizar_margen_comercial_service(db, margin_id=margin_id, update_data=payload.model_dump(exclude_unset=True))
    return CommercialMarginRead.model_validate(margin)


# --- Alertas Proactivas ---

@router.post("/alertas/generar", status_code=status.HTTP_201_CREATED)
def ejecutar_generacion_alertas(
    db: Session = Depends(get_db),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    admin: Usuario = Depends(require_admin),
):
    """
    Ejecuta el motor de alertas proactivas. Solo disponible para administradores.
    """
    generadas = generar_alertas_proactivas(db, pricing_repo)
    return {"mensaje": f"Proceso completado. Se generaron {generadas} alertas nuevas."}


@router.get("/alertas", response_model=list[AlertaRead])
def listar_alertas(
    solo_no_leidas: bool = False,
    material_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
) -> list[AlertaRead]:
    """
    Lista las alertas proactivas generadas por el sistema.
    """
    query = db.query(Alerta)
    if solo_no_leidas:
        query = query.filter(Alerta.leida.is_(False))
    if material_id:
        query = query.filter(Alerta.material_id == material_id)

    return query.order_by(Alerta.created_at.desc()).all()


@router.patch("/alertas/lectura")
def marcar_alertas_como_leidas(
    payload: AlertaBatchUpdate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
):
    """
    Marca un conjunto de alertas como leidas.
    """
    db.query(Alerta).filter(Alerta.id.in_(payload.alerta_ids)).update(
        {"leida": payload.leida}, synchronize_session=False
    )
    db.commit()
    return {"mensaje": f"{len(payload.alerta_ids)} alertas actualizadas."}


@router.get("/materiales/{material_id}/precio-comercial", response_model=CommercialPriceRead)
def obtener_precio_comercial_material(
    material_id: int,
    presentacion_id: int | None = None,
    product_key: str | None = None,
    horizonte_meses: int = 3,
    material_repo: MaterialRepository = Depends(get_material_repository),
    pricing_repo: PricingRepository = Depends(get_pricing_repository),
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
) -> CommercialPriceRead:
    if horizonte_meses < 1 or horizonte_meses > 12:
        raise HTTPException(status_code=422, detail="El horizonte_meses debe estar entre 1 y 12")

    material = material_repo.get_by_id(material_id)
    if material is None:
        raise MaterialNotFoundException(material_id)

    result = calcular_precio_comercial(
        material=material,
        pricing_repo=pricing_repo,
        db=db,
        horizonte_meses=horizonte_meses,
        presentation_id=presentacion_id,
        product_key=product_key,
        usar_selector_modelo=USAR_SELECTOR_MODELO_FORECAST,
    )
    return CommercialPriceRead(
        material_id=result.material_id,
        material_key=result.material_key,
        presentation_id=result.presentation_id,
        product_key=result.product_key,
        costo_base_actual=result.costo_base_actual,
        costo_base_proyectado=result.costo_base_proyectado,
        margen_ganancia_pct=result.margen_ganancia_pct,
        origen_margen=result.origen_margen,
        precio_final_actual=result.precio_final_actual,
        precio_final_proyectado=result.precio_final_proyectado,
        ganancia_unitaria_actual=result.ganancia_unitaria_actual,
        ganancia_unitaria_proyectada=result.ganancia_unitaria_proyectada,
        ultima_fecha_observada=getattr(result, "ultima_fecha_observada", None),
        advertencias=list(result.advertencias),
    )


@router.post("/materiales/{material_id}/imputar-precios", response_model=PriceImputationResponse)
def imputar_precios_material(
    material_id: int,
    payload: PriceImputationRequest,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(require_admin),
) -> PriceImputationResponse:
    result = impute_monthly_prices(
        db,
        material_id=material_id,
        start_date=payload.start_date,
        end_date=payload.end_date,
        index_series_id=payload.index_series_id,
        source_name=payload.source_name,
        metodo_estimacion=payload.metodo_estimacion,
    )
    return PriceImputationResponse(
        material_id=result.material_id,
        source_name=result.source_name,
        series_id=result.series_id,
        metodo_estimacion=result.metodo_estimacion,
        inserted=result.inserted,
        updated=result.updated,
        skipped_real_months=result.skipped_real_months,
        generated_months=result.generated_months,
    )


@router.post("/precios-historicos", response_model=PrecioHistoricoRead, status_code=status.HTTP_201_CREATED)
def crear_precio_historico(
    payload: PrecioHistoricoCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(require_admin),
) -> PrecioHistorico:
    return crear_precio_historico_service(db, **payload.model_dump(), usuario_id=getattr(current_user, "id", None))
