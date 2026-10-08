from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from statistics import median


@dataclass(frozen=True)
class PrecioSerieInput:
    fecha: date
    precio_normalizado: Decimal
    unidad_base: str
    fuente: str | None = None
    numero_comprobante: str | None = None
    registro_id: int | None = None
    origen_dato: str | None = None
    metodo_estimacion: str | None = None
    presentacion_nombre: str | None = None
    presentacion_cantidad_base: Decimal | None = None
    presentacion_unidad: str | None = None
    precio_original: Decimal | None = None
    moneda: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class PuntoSeriePrecio:
    fecha: date
    precio_promedio_normalizado: Decimal
    unidad_base: str
    precio_equivalente_25kg: Decimal | None
    precio_equivalente_50kg: Decimal | None
    cantidad_registros: int
    cantidad_facturas: int
    fuentes: list[str]
    variacion_porcentual_anterior: Decimal | None
    es_anomalia: bool = False
    severidad_anomalia: str | None = None
    score_anomalia: int | None = None
    confianza_anomalia: Decimal | None = None
    motivo_anomalia: str | None = None
    precio_esperado_anomalia: Decimal | None = None
    residuo_anomalia_pct: Decimal | None = None
    limite_residuo_anomalia_pct: Decimal | None = None
    rango_esperado_min_anomalia: Decimal | None = None
    rango_esperado_max_anomalia: Decimal | None = None
    tipo_anomalia: str | None = None
    explicacion_anomalia: str | None = None
    variables_relevantes_anomalia: list[str] | None = None
    observacion_id: int | None = None
    origen_dato: str | None = None
    metodo_estimacion: str | None = None
    presentacion_nombre: str | None = None
    presentacion_cantidad_base: Decimal | None = None
    presentacion_unidad: str | None = None
    precio_original: Decimal | None = None
    moneda: str | None = None
    numero_comprobante: str | None = None
    created_at: datetime | None = None
    origenes_dato: list[str] | None = None


@dataclass(frozen=True)
class AnomalyDetectionMetadata:
    motivo: str
    severidad: str
    score: int
    confianza: Decimal
    precio_esperado: Decimal
    residuo_pct: Decimal
    limite_residuo_pct: Decimal
    rango_esperado_min: Decimal
    rango_esperado_max: Decimal
    tipo: str
    explicacion: str
    variables_relevantes: list[str]


@dataclass(frozen=True)
class VariacionEntreFechas:
    fecha_desde: date
    fecha_hasta: date
    precio_desde: Decimal
    precio_hasta: Decimal
    variacion_porcentual: Decimal


@dataclass(frozen=True)
class AnomalyEvaluationResult:
    total_puntos: int
    total_detectadas: int
    total_confirmadas: int
    verdaderos_positivos: int
    falsos_positivos: int
    falsos_negativos: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None
    exactitud: Decimal | None
    fechas_detectadas: list[date]
    fechas_confirmadas: list[date]
    coincidencias: list[date]
    baseline_umbral_pct: Decimal
    baseline_total_detectadas: int
    baseline_verdaderos_positivos: int
    baseline_falsos_positivos: int
    baseline_falsos_negativos: int
    baseline_precision: Decimal | None
    baseline_recall: Decimal | None
    baseline_f1: Decimal | None
    baseline_fechas_detectadas: list[date]


@dataclass(frozen=True)
class AnomalyStabilityResult:
    ventanas: int
    jaccard_promedio: Decimal | None
    jaccard_minimo: Decimal | None
    jaccard_maximo: Decimal | None


def _quantize(value: Decimal, places: str = "0.0001") -> Decimal:
    return value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def _contar_facturas(registros: list[PrecioSerieInput]) -> int:
    comprobantes = {registro.numero_comprobante for registro in registros if registro.numero_comprobante}
    return len(comprobantes) if comprobantes else len(registros)


def _usa_equivalencias_bolsa(unidad_base: str, fuentes: list[str]) -> bool:
    return unidad_base == "kg" and "Factura compra" in fuentes


def _origenes_dato(registros: list[PrecioSerieInput]) -> list[str]:
    return sorted({registro.origen_dato for registro in registros if registro.origen_dato})


def _mediana(valores: list[float]) -> float | None:
    if not valores:
        return None
    return float(median(valores))


def _mad(valores: list[float]) -> float:
    if len(valores) < 2:
        return 0.0
    centro = median(valores)
    return float(median([abs(valor - centro) for valor in valores]))


def _pct_cambio(actual: float, referencia: float) -> float:
    if referencia == 0:
        return 0.0
    return abs((actual - referencia) / referencia) * 100


def _pct_diferencia_firmada(actual: float, referencia: float) -> float:
    if referencia == 0:
        return 0.0
    return ((actual - referencia) / referencia) * 100


def _punto_retrasado(puntos: list[PuntoSeriePrecio], index: int, lag: int) -> PuntoSeriePrecio:
    return puntos[index - lag] if index - lag >= 0 else puntos[0]


def _limite_adaptativo_desde_valores(valores: list[float], fallback: float = 0.0) -> Decimal:
    if len(valores) < 4:
        return Decimal(f"{fallback:.6f}")

    ordenados = sorted(valores)
    mitad = len(ordenados) // 2
    lower_half = ordenados[:mitad]
    upper_half = ordenados[mitad + (0 if len(ordenados) % 2 == 0 else 1) :]
    q1 = median(lower_half) if lower_half else ordenados[0]
    q3 = median(upper_half) if upper_half else ordenados[-1]
    iqr = q3 - q1
    mediana = median(ordenados)
    mad = median([abs(valor - mediana) for valor in ordenados]) if len(ordenados) >= 2 else 0
    limite_iqr = q3 + (1.5 * iqr)
    limite_mad = mediana + (3.0 * mad)
    limite = max(fallback, limite_iqr, limite_mad)
    return Decimal(f"{limite:.6f}")


def _baseline_tendencia_local(puntos: list[PuntoSeriePrecio], index: int) -> float | None:
    ventana = [float(item.precio_promedio_normalizado) for item in puntos[max(0, index - 3) : index]]
    return _mediana(ventana)


def _gap_porcentual(valor_actual: float, referencia: float | None) -> float:
    if referencia is None:
        return 0.0
    return _pct_cambio(valor_actual, referencia)


ANOMALY_FEATURE_LABELS = [
    "posición temporal",
    "mes calendario",
    "trimestre",
    "precio anterior",
    "precio hace 2 observaciones",
    "precio hace 3 observaciones",
    "precio hace 6 observaciones",
    "variación anterior",
    "variación hace 2 meses",
    "variación hace 3 meses",
    "promedio móvil 3 observaciones",
    "promedio móvil 6 observaciones",
    "dispersión reciente 3 observaciones",
    "dispersión reciente 6 observaciones",
    "cantidad de registros previos",
    "referencia estacional",
    "desvío estacional previo",
    "pendiente local",
]


def _ultimo_precio_hasta_fecha(registros: list[PrecioSerieInput], fecha_objetivo: date) -> tuple[date, Decimal] | None:
    candidatos = [registro for registro in registros if registro.fecha <= fecha_objetivo]
    if not candidatos:
        return None
    ultimo = max(candidatos, key=lambda registro: registro.fecha)
    return ultimo.fecha, ultimo.precio_normalizado


def calcular_variacion_entre_fechas(
    registros: list[PrecioSerieInput],
    fecha_desde: date,
    fecha_hasta: date,
) -> VariacionEntreFechas:
    if fecha_hasta <= fecha_desde:
        raise ValueError("fecha_hasta debe ser posterior a fecha_desde")

    punto_desde = _ultimo_precio_hasta_fecha(registros, fecha_desde)
    if punto_desde is None:
        raise ValueError("No hay precio historico para fecha_desde")

    punto_hasta = _ultimo_precio_hasta_fecha(registros, fecha_hasta)
    if punto_hasta is None:
        raise ValueError("No hay precio historico para fecha_hasta")

    fecha_real_desde, precio_desde = punto_desde
    fecha_real_hasta, precio_hasta = punto_hasta
    if precio_desde == 0:
        raise ValueError("No se puede calcular variacion con precio_desde en cero")

    variacion = _quantize(((precio_hasta - precio_desde) / precio_desde) * Decimal("100"))
    return VariacionEntreFechas(
        fecha_desde=fecha_real_desde,
        fecha_hasta=fecha_real_hasta,
        precio_desde=_quantize(precio_desde),
        precio_hasta=_quantize(precio_hasta),
        variacion_porcentual=variacion,
    )


def construir_serie_precios(
    registros: list[PrecioSerieInput],
    material_nombre: str | None = None,
) -> list[PuntoSeriePrecio]:
    grupos: dict[date, list[PrecioSerieInput]] = defaultdict(list)
    for registro in registros:
        grupos[registro.fecha].append(registro)

    puntos: list[PuntoSeriePrecio] = []
    precio_anterior: Decimal | None = None

    for fecha_punto in sorted(grupos):
        registros_fecha = grupos[fecha_punto]
        total = sum((registro.precio_normalizado for registro in registros_fecha), Decimal("0"))
        promedio = _quantize(total / Decimal(len(registros_fecha)))
        unidad_base = registros_fecha[0].unidad_base
        fuentes = sorted({registro.fuente for registro in registros_fecha if registro.fuente})
        usa_equivalencias = _usa_equivalencias_bolsa(unidad_base, fuentes)

        variacion = None
        if precio_anterior is not None and precio_anterior != 0:
            variacion = _quantize(((promedio - precio_anterior) / precio_anterior) * Decimal("100"))

        puntos.append(
            PuntoSeriePrecio(
                fecha=fecha_punto,
                precio_promedio_normalizado=promedio,
                unidad_base=unidad_base,
                precio_equivalente_25kg=_quantize(promedio * Decimal("25")) if usa_equivalencias else None,
                precio_equivalente_50kg=_quantize(promedio * Decimal("50")) if usa_equivalencias else None,
                cantidad_registros=len(registros_fecha),
                cantidad_facturas=_contar_facturas(registros_fecha),
                fuentes=fuentes,
                variacion_porcentual_anterior=variacion,
                origenes_dato=_origenes_dato(registros_fecha),
            )
        )
        precio_anterior = promedio

    return _aplicar_anomalias_random_forest(puntos, material_nombre=material_nombre)


def construir_serie_observaciones(
    registros: list[PrecioSerieInput],
    material_nombre: str | None = None,
) -> list[PuntoSeriePrecio]:
    puntos: list[PuntoSeriePrecio] = []
    precio_anterior: Decimal | None = None
    ordenados = sorted(
        enumerate(registros),
        key=lambda item: (item[1].fecha, item[1].registro_id if item[1].registro_id is not None else item[0]),
    )
    for _, registro in ordenados:
        precio = _quantize(registro.precio_normalizado)
        fuentes = [registro.fuente] if registro.fuente else []
        usa_equivalencias = _usa_equivalencias_bolsa(registro.unidad_base, fuentes)
        variacion = None
        if precio_anterior is not None and precio_anterior != 0:
            variacion = _quantize(((precio - precio_anterior) / precio_anterior) * Decimal("100"))
        puntos.append(
            PuntoSeriePrecio(
                fecha=registro.fecha,
                precio_promedio_normalizado=precio,
                unidad_base=registro.unidad_base,
                precio_equivalente_25kg=_quantize(precio * Decimal("25")) if usa_equivalencias else None,
                precio_equivalente_50kg=_quantize(precio * Decimal("50")) if usa_equivalencias else None,
                cantidad_registros=1,
                cantidad_facturas=1,
                fuentes=fuentes,
                variacion_porcentual_anterior=variacion,
                observacion_id=registro.registro_id,
                origen_dato=registro.origen_dato,
                metodo_estimacion=registro.metodo_estimacion,
                presentacion_nombre=registro.presentacion_nombre,
                presentacion_cantidad_base=registro.presentacion_cantidad_base,
                presentacion_unidad=registro.presentacion_unidad,
                precio_original=registro.precio_original,
                moneda=registro.moneda,
                numero_comprobante=registro.numero_comprobante,
                created_at=registro.created_at,
                origenes_dato=[registro.origen_dato] if registro.origen_dato else [],
            )
        )
        precio_anterior = precio
    return _aplicar_anomalias_random_forest(puntos, material_nombre=material_nombre)


def _features_anomalia_mensual(
    puntos: list[PuntoSeriePrecio],
    index: int,
    mismo_mes_anterior: PuntoSeriePrecio | None = None,
) -> list[float]:
    punto = puntos[index]
    anterior = _punto_retrasado(puntos, index, 1)
    hace_2 = _punto_retrasado(puntos, index, 2)
    hace_3 = _punto_retrasado(puntos, index, 3)
    hace_6 = _punto_retrasado(puntos, index, 6)
    precios_previos_3 = [float(item.precio_promedio_normalizado) for item in puntos[max(0, index - 3) : index]]
    precios_previos_6 = [float(item.precio_promedio_normalizado) for item in puntos[max(0, index - 6) : index]]
    variacion_anterior = float(anterior.variacion_porcentual_anterior or Decimal("0"))
    variacion_2 = float(hace_2.variacion_porcentual_anterior or Decimal("0"))
    variacion_3 = float(hace_3.variacion_porcentual_anterior or Decimal("0"))
    mad_3 = _mad(precios_previos_3)
    mad_6 = _mad(precios_previos_6)
    precio_estacional = float(mismo_mes_anterior.precio_promedio_normalizado) if mismo_mes_anterior else float(anterior.precio_promedio_normalizado)
    desvio_estacional_anterior = _gap_porcentual(float(anterior.precio_promedio_normalizado), precio_estacional)
    pendiente_local = _gap_porcentual(float(anterior.precio_promedio_normalizado), float(hace_3.precio_promedio_normalizado))
    return [
        float(index),
        float(punto.fecha.month),
        float(((punto.fecha.month - 1) // 3) + 1),
        float(anterior.precio_promedio_normalizado),
        float(hace_2.precio_promedio_normalizado),
        float(hace_3.precio_promedio_normalizado),
        float(hace_6.precio_promedio_normalizado),
        variacion_anterior,
        variacion_2,
        variacion_3,
        sum(precios_previos_3) / len(precios_previos_3) if precios_previos_3 else float(anterior.precio_promedio_normalizado),
        sum(precios_previos_6) / len(precios_previos_6) if precios_previos_6 else float(anterior.precio_promedio_normalizado),
        mad_3,
        mad_6,
        float(anterior.cantidad_registros),
        precio_estacional,
        desvio_estacional_anterior,
        pendiente_local,
    ]


def _intervalo_reentrenamiento_anomalias(puntos: list[PuntoSeriePrecio]) -> int:
    """Reentrena con una cadencia apropiada al muestreo mensual o denso."""
    meses = len({(p.fecha.year, p.fecha.month) for p in puntos})
    observaciones_por_mes = len(puntos) / max(meses, 1)
    return 6 if observaciones_por_mes <= 1.5 else 30


def _clasificar_severidad_anomalia(residual_pct: Decimal, residual_limit: Decimal, score: int, required_signals: int) -> str:
    if residual_limit <= 0:
        return "media" if score <= required_signals else "alta"

    ratio = residual_pct / residual_limit
    if ratio >= Decimal("2.2") or score >= required_signals + 1:
        return "alta"
    if ratio >= Decimal("1.5") or score >= required_signals:
        return "media"
    return "leve"


def _confianza_anomalia(
    score: int,
    required_signals: int,
    residual_pct: Decimal,
    residual_limit: Decimal,
    incertidumbre_modelo_pct: Decimal,
) -> Decimal:
    base = Decimal(score) / Decimal("4")
    if residual_limit > 0:
        residual_boost = min(Decimal("0.25"), residual_pct / (residual_limit * Decimal("4")))
    else:
        residual_boost = Decimal("0")
    evidence_boost = Decimal("0.10") if score >= required_signals + 1 else Decimal("0")
    uncertainty_penalty = min(Decimal("0.20"), incertidumbre_modelo_pct / Decimal("100"))
    return _quantize(max(min(base + residual_boost + evidence_boost - uncertainty_penalty, Decimal("1")), Decimal("0")) * Decimal("100"))


def _variables_relevantes_anomalia(importances: list[float], limit: int = 3) -> list[str]:
    ranked = sorted(enumerate(importances), key=lambda item: item[1], reverse=True)
    return [ANOMALY_FEATURE_LABELS[index] for index, importance in ranked[:limit] if importance > 0 and index < len(ANOMALY_FEATURE_LABELS)]


def _clasificar_tipo_anomalia(
    residual_signal: bool,
    variacion_signal: bool,
    seasonal_signal: bool,
    trend_signal: bool,
) -> str:
    if trend_signal and variacion_signal:
        return "cambio_sostenido"
    if seasonal_signal:
        return "desvio_estacional"
    if variacion_signal:
        return "salto_puntual"
    if trend_signal:
        return "desvio_tendencia"
    if residual_signal:
        return "residuo_extremo"
    return "mixta"


def _explicar_anomalia(
    actual: Decimal,
    predicted: Decimal,
    residual_signed_pct: Decimal,
    residual_limit: Decimal,
    tipo: str,
    signals: list[str],
) -> str:
    direccion = "por encima" if actual >= predicted else "por debajo"
    tipo_labels = {
        "cambio_sostenido": "cambio sostenido",
        "desvio_estacional": "desvío estacional",
        "salto_puntual": "salto puntual",
        "desvio_tendencia": "desvío de tendencia",
        "residuo_extremo": "residuo extremo",
        "mixta": "señal mixta",
    }
    tipo_label = tipo_labels.get(tipo, tipo.replace("_", " "))
    residual_value = Decimal(f"{abs(residual_signed_pct):.2f}")
    residual_limit_value = Decimal(f"{residual_limit:.2f}")
    base = (
        f"El precio observado estuvo {residual_value}% {direccion} de lo esperado. "
        f"Superó el margen normal de {residual_limit_value}%."
    )
    if signals:
        if len(signals) == 1:
            señales = signals[0]
        elif len(signals) == 2:
            señales = f"{signals[0]} y {signals[1]}"
        else:
            señales = ", ".join(signals[:-1]) + f" y {signals[-1]}"
        return f"{base} El sistema lo clasificó como {tipo_label} porque activó {señales}."
    return f"{base} El sistema lo clasificó como {tipo_label}."


def _es_material_cemento(material_nombre: str | None) -> bool:
    return bool(material_nombre and "cement" in material_nombre.lower())


def _detector_profile(material_nombre: str | None) -> dict[str, Decimal | int]:
    if _es_material_cemento(material_nombre):
        return {
            "residual_floor": Decimal("24.000000"),
            "uncertainty_multiplier": Decimal("1.90"),
            "variacion_floor": Decimal("16.000000"),
            "trend_floor": Decimal("12.000000"),
            "seasonal_floor": Decimal("13.000000"),
            "required_signals": 3,
            "confidence_floor": Decimal("65.0000"),
            "max_anomalies_per_month": 8,
        }
    return {
        "residual_floor": Decimal("16.000000"),
        "uncertainty_multiplier": Decimal("1.55"),
        "variacion_floor": Decimal("13.000000"),
        "trend_floor": Decimal("9.000000"),
        "seasonal_floor": Decimal("10.000000"),
        "required_signals": 3,
        "confidence_floor": Decimal("58.0000"),
        "max_anomalies_per_month": 6,
    }


def _detectar_anomalias_random_forest(
    puntos: list[PuntoSeriePrecio],
    material_nombre: str | None = None,
) -> dict[int, AnomalyDetectionMetadata]:
    if len(puntos) < 6:
        return {}

    from sklearn.ensemble import RandomForestRegressor

    trainable_indexes = [index for index in range(5, len(puntos))]
    if not trainable_indexes:
        return {}

    model_kwargs = dict(
        n_estimators=50,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
    )

    profile = _detector_profile(material_nombre)
    evaluaciones: list[tuple[int, float, float, float, float, float, float | None, float, float, int, int, float, list[float]]] = []
    previous_same_month: list[PuntoSeriePrecio | None] = []
    month_state: dict[int, tuple[int, PuntoSeriePrecio, PuntoSeriePrecio | None]] = {}
    for punto in puntos:
        state = month_state.get(punto.fecha.month)
        if state is None:
            previous_same_month.append(None)
            month_state[punto.fecha.month] = (punto.fecha.year, punto, None)
        elif state[0] == punto.fecha.year:
            previous_same_month.append(state[2])
            month_state[punto.fecha.month] = (state[0], punto, state[2])
        else:
            previous_same_month.append(state[1])
            month_state[punto.fecha.month] = (punto.fecha.year, punto, state[1])
    features_by_index = [
        _features_anomalia_mensual(puntos, index, previous_same_month[index])
        for index in range(len(puntos))
    ]
    residuals_historial: list[float] = []
    variaciones_historial: list[float] = []
    model = None
    last_fit_index = -1
    refit_interval = _intervalo_reentrenamiento_anomalias(puntos)
    max_training_points = 365
    prediction_cache: dict[int, tuple[float, list[float], list[float]]] = {}
    for index in trainable_indexes:
        if model is None or index - last_fit_index >= refit_interval:
            train_start = max(1, index - max_training_points)
            train_indexes = range(train_start, index)
            x_train = [features_by_index[train_index] for train_index in train_indexes]
            y_train = [float(puntos[train_index].precio_promedio_normalizado) for train_index in train_indexes]
            if len(x_train) < 5:
                continue
            model = RandomForestRegressor(**model_kwargs)
            model.fit(x_train, y_train)
            last_fit_index = index
            batch_end = min(len(puntos), index + refit_interval)
            batch_indexes = list(range(index, batch_end))
            batch_features = [features_by_index[batch_index] for batch_index in batch_indexes]
            batch_predictions = model.predict(batch_features)
            tree_batch_predictions = [tree.predict(batch_features) for tree in model.estimators_]
            importances = [float(value) for value in model.feature_importances_]
            for offset, batch_index in enumerate(batch_indexes):
                prediction_cache[batch_index] = (
                    float(batch_predictions[offset]),
                    [float(tree_values[offset]) for tree_values in tree_batch_predictions],
                    importances,
                )
        predicted, tree_predictions, feature_importances = prediction_cache[index]
        prediction_center = sum(tree_predictions) / len(tree_predictions)
        prediction_variance = sum((value - prediction_center) ** 2 for value in tree_predictions) / len(tree_predictions)
        prediction_std = prediction_variance**0.5
        incertidumbre_modelo_pct = _pct_cambio(predicted + prediction_std, predicted)
        actual = float(puntos[index].precio_promedio_normalizado)
        residual_pct = _pct_cambio(actual, predicted)
        residual_signed_pct = _pct_diferencia_firmada(actual, predicted)
        tendencia_local = _baseline_tendencia_local(puntos, index)
        tendencia_gap = _gap_porcentual(actual, tendencia_local)
        mismo_mes_anterior = previous_same_month[index]
        seasonal_gap = (
            _gap_porcentual(actual, float(mismo_mes_anterior.precio_promedio_normalizado))
            if mismo_mes_anterior is not None
            else None
        )
        variacion_actual = abs(float(puntos[index].variacion_porcentual_anterior or Decimal("0")))
        residual_limit = _limite_adaptativo_desde_valores(residuals_historial, fallback=float(profile["residual_floor"]))
        residual_limit = max(residual_limit, Decimal(f"{incertidumbre_modelo_pct * float(profile['uncertainty_multiplier']):.6f}"))
        variacion_limit = _limite_adaptativo_desde_valores(variaciones_historial, fallback=float(profile["variacion_floor"]))
        trend_limit = max(Decimal(f"{profile['trend_floor']:.6f}"), variacion_limit * Decimal("0.85"))
        seasonal_limit = max(Decimal(f"{profile['seasonal_floor']:.6f}"), variacion_limit)
        residual_signal = Decimal(f"{residual_pct:.6f}") > residual_limit
        variacion_signal = Decimal(f"{variacion_actual:.6f}") > variacion_limit
        seasonal_signal = seasonal_gap is not None and Decimal(f"{seasonal_gap:.6f}") > seasonal_limit
        trend_signal = Decimal(f"{tendencia_gap:.6f}") > trend_limit if tendencia_local is not None else False
        score = int(sum((residual_signal, variacion_signal, seasonal_signal, trend_signal)))
        required_signals = int(profile["required_signals"])
        evaluaciones.append(
            (
                index,
                residual_pct,
                predicted,
                incertidumbre_modelo_pct,
                variacion_actual,
                float(residual_limit),
                seasonal_gap,
                tendencia_gap,
                float(variacion_limit),
                score,
                required_signals,
                residual_signed_pct,
                feature_importances,
            )
        )
        residuals_historial.append(residual_pct)
        variaciones_historial.append(variacion_actual)

    if not evaluaciones:
        return {}

    anomalies: dict[int, AnomalyDetectionMetadata] = {}
    for (
        index,
        residual_pct,
        predicted,
        incertidumbre_modelo_pct,
        variacion_actual,
        residual_limit_float,
        seasonal_gap,
        tendencia_gap,
        variacion_limit_float,
        score,
        required_signals,
        residual_signed_pct,
        feature_importances,
    ) in evaluaciones:
        residual_decimal = Decimal(f"{residual_pct:.6f}")
        variacion = puntos[index].variacion_porcentual_anterior
        if variacion is None:
            continue

        residual_limit = Decimal(f"{residual_limit_float:.6f}")
        variacion_limit = Decimal(f"{variacion_limit_float:.6f}")
        trend_limit = max(Decimal(f"{profile['trend_floor']:.6f}"), variacion_limit * Decimal("0.85"))
        seasonal_limit = max(Decimal(f"{profile['seasonal_floor']:.6f}"), variacion_limit)
        residual_signal = residual_decimal > residual_limit
        variacion_signal = Decimal(f"{abs(variacion_actual):.6f}") > variacion_limit
        seasonal_signal = seasonal_gap is not None and Decimal(f"{seasonal_gap:.6f}") > seasonal_limit
        trend_signal = Decimal(f"{tendencia_gap:.6f}") > trend_limit
        if not (score >= required_signals or (residual_decimal > (residual_limit * Decimal("1.90")) and score >= required_signals - 1)):
            continue

        severidad = _clasificar_severidad_anomalia(residual_decimal, residual_limit, score, required_signals)
        incertidumbre_decimal = Decimal(f"{incertidumbre_modelo_pct:.6f}")
        confianza = _confianza_anomalia(score, required_signals, residual_decimal, residual_limit, incertidumbre_decimal)
        signalos = []
        señales_resumen = []
        if residual_signal:
            signalos.append(f"Residuo: {Decimal(f'{residual_pct:.2f}').quantize(Decimal('0.00'))}%")
            señales_resumen.append("el precio se alejó demasiado de lo esperado")
        if variacion_signal:
            signalos.append(f"Variación mensual: {Decimal(f'{variacion_actual:.2f}').quantize(Decimal('0.00'))}%")
            señales_resumen.append("hubo un salto fuerte frente al mes anterior")
        if seasonal_signal and seasonal_gap is not None:
            signalos.append(f"Gap estacional: {Decimal(f'{seasonal_gap:.2f}').quantize(Decimal('0.00'))}%")
            señales_resumen.append("se alejó del comportamiento normal para ese mismo mes")
        if trend_signal:
            signalos.append(f"Desvío de tendencia: {Decimal(f'{tendencia_gap:.2f}').quantize(Decimal('0.00'))}%")
            señales_resumen.append("rompió la tendencia reciente")
        predicted_decimal = Decimal(f"{predicted:.4f}").quantize(Decimal("0.0001"))
        predicted_display = predicted_decimal.quantize(Decimal("0.00"))
        residual_display = Decimal(f"{residual_pct:.4f}").quantize(Decimal("0.0001"))
        residual_display_2 = residual_display.quantize(Decimal("0.00"))
        residual_signed_decimal = Decimal(f"{residual_signed_pct:.4f}").quantize(Decimal("0.0001"))
        residual_signed_display = residual_signed_decimal.quantize(Decimal("0.00"))
        residual_limit_display = residual_limit.quantize(Decimal("0.00"))
        uncertainty_display = Decimal(f"{incertidumbre_modelo_pct:.4f}").quantize(Decimal("0.00"))
        lower_expected = _quantize(predicted_decimal * (Decimal("1") - (residual_limit / Decimal("100"))))
        upper_expected = _quantize(predicted_decimal * (Decimal("1") + (residual_limit / Decimal("100"))))
        lower_expected_display = lower_expected.quantize(Decimal("0.00"))
        upper_expected_display = upper_expected.quantize(Decimal("0.00"))
        tipo = _clasificar_tipo_anomalia(residual_signal, variacion_signal, seasonal_signal, trend_signal)
        variables_relevantes = _variables_relevantes_anomalia(feature_importances)
        explicacion = _explicar_anomalia(
            puntos[index].precio_promedio_normalizado,
            predicted_display,
            residual_signed_display,
            residual_limit_display,
            tipo,
            señales_resumen,
        )
        motivo = (
            "Anomalia detectada por Random Forest (ensemble robusto): "
            f"precio esperado {predicted_display}, "
            f"rango normal {lower_expected_display} a {upper_expected_display}, "
            f"residuo {residual_display_2}%, "
            f"limite residuo {residual_limit_display}%, "
            f"incertidumbre modelo {uncertainty_display}%, "
            f"variacion entre observaciones {Decimal(f'{float(variacion):.4f}').quantize(Decimal('0.00'))}%, "
            f"tipo {tipo}, "
            f"score {score}/{4}; "
            + "; ".join(signalos)
        )
        anomalies[index] = AnomalyDetectionMetadata(
            motivo=motivo,
            severidad=severidad,
            score=score,
            confianza=confianza,
            precio_esperado=predicted_decimal,
            residuo_pct=residual_display,
            limite_residuo_pct=residual_limit,
            rango_esperado_min=lower_expected,
            rango_esperado_max=upper_expected,
            tipo=tipo,
            explicacion=explicacion,
            variables_relevantes=variables_relevantes,
        )

    confidence_floor = Decimal(f"{profile['confidence_floor']:.4f}")
    max_anomalies_per_month = int(profile["max_anomalies_per_month"])
    grouped_by_month: dict[str, list[tuple[int, AnomalyDetectionMetadata]]] = defaultdict(list)
    for index, anomaly in anomalies.items():
        if anomaly.confianza < confidence_floor:
            continue
        month_key = f"{puntos[index].fecha.year:04d}-{puntos[index].fecha.month:02d}"
        grouped_by_month[month_key].append((index, anomaly))

    if not grouped_by_month:
        return {}

    selected: dict[int, AnomalyDetectionMetadata] = {}
    for month_key in sorted(grouped_by_month):
        month_candidates = grouped_by_month[month_key]
        best_by_date: dict[date, tuple[int, AnomalyDetectionMetadata]] = {}
        for index, anomaly in month_candidates:
            date_key = puntos[index].fecha
            current = best_by_date.get(date_key)
            candidate_rank = (anomaly.score, anomaly.confianza, anomaly.residuo_pct, -index)
            if current is None:
                best_by_date[date_key] = (index, anomaly)
                continue
            current_index, current_anomaly = current
            current_rank = (current_anomaly.score, current_anomaly.confianza, current_anomaly.residuo_pct, -current_index)
            if candidate_rank > current_rank:
                best_by_date[date_key] = (index, anomaly)
        month_candidates = list(best_by_date.values())
        month_candidates.sort(
            key=lambda item: (
                item[1].score,
                item[1].confianza,
                item[1].residuo_pct,
                item[0],
            ),
            reverse=True,
        )
        for index, anomaly in month_candidates[:max_anomalies_per_month]:
            selected[index] = anomaly

    return dict(sorted(selected.items()))


def _aplicar_anomalias_random_forest(
    puntos: list[PuntoSeriePrecio],
    material_nombre: str | None = None,
) -> list[PuntoSeriePrecio]:
    anomalies = _detectar_anomalias_random_forest(puntos, material_nombre=material_nombre)
    if not anomalies:
        return puntos
    return [
        replace(
            punto,
            es_anomalia=(anomaly := anomalies.get(index)) is not None,
            severidad_anomalia=anomaly.severidad if anomaly else None,
            motivo_anomalia=anomaly.motivo if anomaly else None,
            score_anomalia=anomaly.score if anomaly else None,
            confianza_anomalia=anomaly.confianza if anomaly else None,
            precio_esperado_anomalia=anomaly.precio_esperado if anomaly else None,
            residuo_anomalia_pct=anomaly.residuo_pct if anomaly else None,
            limite_residuo_anomalia_pct=anomaly.limite_residuo_pct if anomaly else None,
            rango_esperado_min_anomalia=anomaly.rango_esperado_min if anomaly else None,
            rango_esperado_max_anomalia=anomaly.rango_esperado_max if anomaly else None,
            tipo_anomalia=anomaly.tipo if anomaly else None,
            explicacion_anomalia=anomaly.explicacion if anomaly else None,
            variables_relevantes_anomalia=anomaly.variables_relevantes if anomaly else None,
            observacion_id=punto.observacion_id,
        )
        for index, punto in enumerate(puntos)
    ]


def medir_estabilidad_anomalias(puntos: list[PuntoSeriePrecio]) -> AnomalyStabilityResult:
    if len(puntos) < 8:
        return AnomalyStabilityResult(ventanas=0, jaccard_promedio=None, jaccard_minimo=None, jaccard_maximo=None)

    ventanas = []
    min_window = 6
    for end_index in range(min_window, len(puntos) + 1):
        prefix = puntos[:end_index]
        anomalies = _detectar_anomalias_random_forest(prefix)
        ventanas.append(set(anomalies))

    if len(ventanas) < 2:
        return AnomalyStabilityResult(ventanas=len(ventanas), jaccard_promedio=None, jaccard_minimo=None, jaccard_maximo=None)

    similitudes: list[float] = []
    for anterior, actual in zip(ventanas, ventanas[1:], strict=False):
        union = anterior | actual
        interseccion = anterior & actual
        similitud = 1.0 if not union else len(interseccion) / len(union)
        similitudes.append(similitud)

    return AnomalyStabilityResult(
        ventanas=len(ventanas),
        jaccard_promedio=_quantize(Decimal(str(sum(similitudes) / len(similitudes)))),
        jaccard_minimo=_quantize(Decimal(str(min(similitudes)))),
        jaccard_maximo=_quantize(Decimal(str(max(similitudes)))),
    )


def evaluar_anomalias_detectadas(
    puntos: list[PuntoSeriePrecio],
    fechas_confirmadas: set[date],
) -> AnomalyEvaluationResult:
    detectadas = {punto.fecha for punto in puntos if punto.es_anomalia}
    baseline_umbral_pct = Decimal("8.0000")
    baseline_detectadas = {
        punto.fecha
        for punto in puntos
        if punto.variacion_porcentual_anterior is not None and abs(punto.variacion_porcentual_anterior) > baseline_umbral_pct
    }
    confirmadas = set(fechas_confirmadas)
    verdaderos_positivos = len(detectadas & confirmadas)
    falsos_positivos = len(detectadas - confirmadas)
    falsos_negativos = len(confirmadas - detectadas)
    verdaderos_negativos = len(puntos) - verdaderos_positivos - falsos_positivos - falsos_negativos
    baseline_verdaderos_positivos = len(baseline_detectadas & confirmadas)
    baseline_falsos_positivos = len(baseline_detectadas - confirmadas)
    baseline_falsos_negativos = len(confirmadas - baseline_detectadas)

    def _ratio(numerator: int, denominator: int) -> Decimal | None:
        if denominator <= 0:
            return None
        return _quantize(Decimal(numerator) / Decimal(denominator))

    precision = _ratio(verdaderos_positivos, verdaderos_positivos + falsos_positivos)
    recall = _ratio(verdaderos_positivos, verdaderos_positivos + falsos_negativos)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = _quantize((Decimal("2") * precision * recall) / (precision + recall))
    exactitud = _ratio(verdaderos_negativos + verdaderos_positivos, len(puntos))
    baseline_precision = _ratio(baseline_verdaderos_positivos, baseline_verdaderos_positivos + baseline_falsos_positivos)
    baseline_recall = _ratio(baseline_verdaderos_positivos, baseline_verdaderos_positivos + baseline_falsos_negativos)
    baseline_f1 = None
    if baseline_precision is not None and baseline_recall is not None and baseline_precision + baseline_recall > 0:
        baseline_f1 = _quantize((Decimal("2") * baseline_precision * baseline_recall) / (baseline_precision + baseline_recall))

    return AnomalyEvaluationResult(
        total_puntos=len(puntos),
        total_detectadas=len(detectadas),
        total_confirmadas=len(confirmadas),
        verdaderos_positivos=verdaderos_positivos,
        falsos_positivos=falsos_positivos,
        falsos_negativos=falsos_negativos,
        precision=precision,
        recall=recall,
        f1=f1,
        exactitud=exactitud,
        fechas_detectadas=sorted(detectadas),
        fechas_confirmadas=sorted(confirmadas),
        coincidencias=sorted(detectadas & confirmadas),
        baseline_umbral_pct=baseline_umbral_pct,
        baseline_total_detectadas=len(baseline_detectadas),
        baseline_verdaderos_positivos=baseline_verdaderos_positivos,
        baseline_falsos_positivos=baseline_falsos_positivos,
        baseline_falsos_negativos=baseline_falsos_negativos,
        baseline_precision=baseline_precision,
        baseline_recall=baseline_recall,
        baseline_f1=baseline_f1,
        baseline_fechas_detectadas=sorted(baseline_detectadas),
    )


def construir_serie_mensual(
    registros: list[PrecioSerieInput],
    material_nombre: str | None = None,
) -> list[PuntoSeriePrecio]:
    grupos: dict[date, list[PrecioSerieInput]] = defaultdict(list)
    for registro in registros:
        mes = date(registro.fecha.year, registro.fecha.month, 1)
        grupos[mes].append(registro)

    puntos: list[PuntoSeriePrecio] = []
    precio_anterior: Decimal | None = None

    for mes in sorted(grupos):
        registros_mes = grupos[mes]
        total = sum((registro.precio_normalizado for registro in registros_mes), Decimal("0"))
        promedio = _quantize(total / Decimal(len(registros_mes)))
        unidad_base = registros_mes[0].unidad_base
        fuentes = sorted({registro.fuente for registro in registros_mes if registro.fuente})
        usa_equivalencias = _usa_equivalencias_bolsa(unidad_base, fuentes)

        variacion = None
        if precio_anterior is not None and precio_anterior != 0:
            variacion = _quantize(((promedio - precio_anterior) / precio_anterior) * Decimal("100"))

        puntos.append(
            PuntoSeriePrecio(
                fecha=mes,
                precio_promedio_normalizado=promedio,
                unidad_base=unidad_base,
                precio_equivalente_25kg=_quantize(promedio * Decimal("25")) if usa_equivalencias else None,
                precio_equivalente_50kg=_quantize(promedio * Decimal("50")) if usa_equivalencias else None,
                cantidad_registros=len(registros_mes),
                cantidad_facturas=_contar_facturas(registros_mes),
                fuentes=fuentes,
                variacion_porcentual_anterior=variacion,
                origenes_dato=_origenes_dato(registros_mes),
            )
        )
        precio_anterior = promedio

    return _aplicar_anomalias_random_forest(puntos, material_nombre=material_nombre)
