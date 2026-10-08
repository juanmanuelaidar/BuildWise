# Anomalias con Random Forest

> **Revisión metodológica de septiembre de 2026:** la versión inicial incluía dos variables del valor objetivo y un intervalo de reentrenamiento demasiado largo; se proponen correcciones en el PR #20. Las marcas históricas deben regenerarse antes de utilizarlas como evidencia de calidad del detector. La dispersión entre árboles y el puntaje de confianza son indicadores heurísticos, no intervalos o probabilidades calibradas. La comparación de precisión/recall/F1 requiere fechas reales etiquetadas.

## Objetivo

Detectar meses con comportamiento atipico en la serie historica de precios sin depender de un umbral porcentual fijo comun para todos los materiales.

La deteccion puede aplicarse tanto sobre una serie mensualizada como sobre una secuencia de observaciones; el resultado corresponde a los puntos realmente evaluados. En la tesis, la serie mensual de Cemento Portland constituye el caso de referencia.

## Por que no usar un umbral fijo

Un umbral unico, por ejemplo `8%`, es facil de explicar pero puede ser rigido:

- puede ser demasiado sensible para materiales naturalmente volatiles;
- puede ser demasiado permisivo para materiales estables;
- no aprende el patron historico propio de cada serie;
- obliga a defender un porcentaje fijo aunque los materiales tengan comportamientos distintos.

Por eso se reemplaza por una deteccion basada en `RandomForestRegressor`, que aprende un precio mensual esperado a partir del comportamiento historico del material.

## Enfoque implementado

La implementacion esta en `app/modules/pricing/application/series.py`.

Para cada material:

1. Se agrupan los precios por mes.
2. Se calcula el precio promedio mensual normalizado.
3. Se calcula la variacion porcentual contra el mes anterior.
4. Se entrena un `RandomForestRegressor` sobre la propia serie mensual.
5. El modelo estima el precio esperado para cada mes evaluable.
6. Se calcula el residuo porcentual entre precio observado y precio esperado.
7. Se mide la incertidumbre interna del modelo a partir de la dispersion de predicciones entre arboles.
8. Se combinan cuatro senales: residuo, variacion temporal, desvio estacional y desvio de tendencia. En la configuracion actual se requieren tres senales, salvo la excepcion de residuo extremo (mas de 1,90 veces su limite) acompañado de al menos dos senales. El umbral del residuo combina reglas robustas, un piso por material y dispersion heuristica del ensemble.
9. Se expone el rango normal esperado, el tipo operativo de anomalia, una explicacion legible y las variables mas relevantes del Random Forest.
10. Solo si se dispone de fechas etiquetadas y confirmadas, puede compararse el detector con el baseline simple de variacion mensual mayor a `8%`.

## Variables usadas

El modelo usa features simples y trazables:

- indice temporal del mes dentro de la serie;
- mes calendario;
- precio del mes anterior;
- variacion porcentual anterior;
- promedio movil corto de precios previos;
- rezagos de precio y variacion;
- dispersion robusta reciente;
- distancia estacional calculada exclusivamente con observaciones anteriores;
- cantidad de registros de la observacion anterior.

Las diferencias entre el precio observado actual y la tendencia o referencia estacional se calculan *despues* de estimar el precio esperado y forman parte de las senales de contraste, no de las variables predictoras del Random Forest.

Estas variables permiten capturar tendencia, estacionalidad simple, inercia de precio y robustez de la muestra mensual.

## Regla de marca

El modelo no marca anomalias por superar un porcentaje fijo.

La regla es:

```text
residuo porcentual observado > limite dinamico de residuos
```

El limite dinamico combina el piso minimo por material y los limites derivados de dos estadisticos robustos:

```text
max(piso, Q3 + 1.5 * IQR, mediana + 3 * MAD)
```

Donde `IQR` representa el rango intercuartil y `MAD` la desviacion absoluta mediana de los residuos porcentuales acumulados.

Ese limite se vuelve mas conservador si el Random Forest muestra alta dispersion entre sus arboles. En terminos practicos, si el modelo no tiene un precio esperado estable, el sistema exige un residuo mayor antes de marcar una alerta.

Ademas del residuo, la marca requiere evidencia complementaria: variacion mensual atipica, desvio contra tendencia local o desvio contra una referencia estacional cuando existe. Esto reduce falsos positivos por cambios esperables de mercado.

## Series cortas

Si la serie tiene menos de 6 puntos evaluables, no se entrena Random Forest y no se fuerzan anomalias.

Esto evita marcar puntos atipicos sin evidencia suficiente. En esos casos, la salida conserva la serie y deja `es_anomalia = false`.

## Salida

Cada punto mensual puede incluir:

- `es_anomalia`: indica si el mes fue marcado como atipico;
- `severidad_anomalia`: clasifica la magnitud relativa de la desviacion detectada;
- `score_anomalia`: cantidad de senales que respaldan la alerta;
- `confianza_anomalia`: puntaje heuristico ajustado por evidencia y dispersion entre arboles; no representa una probabilidad calibrada;
- `precio_esperado_anomalia`: estimacion puntual del Random Forest;
- `rango_esperado_min_anomalia` y `rango_esperado_max_anomalia`: banda normal estimada a partir del limite dinamico de residuo;
- `residuo_anomalia_pct`: distancia porcentual entre precio observado y esperado;
- `limite_residuo_anomalia_pct`: margen dinamico usado para evaluar el residuo;
- `tipo_anomalia`: clasificacion operativa (`salto_puntual`, `cambio_sostenido`, `desvio_tendencia`, `desvio_estacional`, `residuo_extremo` o `mixta`);
- `explicacion_anomalia`: lectura resumida en lenguaje claro;
- `variables_relevantes_anomalia`: principales variables usadas por el Random Forest para esa evaluacion;
- `motivo_anomalia`: describe la deteccion, incluyendo precio esperado, rango normal, residuo porcentual, incertidumbre, variacion mensual y senales activadas.

Ejemplo conceptual:

```text
Anomalia detectada por Random Forest: precio esperado 120.0000, residuo 35.0000%, incertidumbre modelo 4.2000%, variacion mensual 41.6667% y score 3/4
```

## Comparacion contra baseline

Para comprobar si el procedimiento aporta ventajas frente a una regla fija, la evaluacion de anomalias permite cargar fechas confirmadas y calcular metricas para dos enfoques:

- detector Random Forest con residuo dinamico;
- baseline simple que marca cualquier mes con variacion mensual mayor al `8%`.

La salida incluye precision, recall, F1, falsos positivos y falsos negativos del detector principal, mas las metricas equivalentes del baseline. Estas comparaciones solo son concluyentes cuando existe un conjunto de fechas realmente etiquetadas; no deben inferirse ventajas empiricas a partir de los puntajes internos del propio detector.

## Limitaciones

Random Forest detecta atipicidad, no explica causalidad.

Una anomalia puede deberse a:

- cambio real de proveedor;
- cambio de presentacion;
- error de carga;
- shock de mercado;
- dato estimado inconsistente;
- salto normal en una serie muy volatil.

La marca debe interpretarse como una alerta para revisar el mes, no como una prueba automatica de error.

La severidad no reemplaza la revision humana. Solo ayuda a priorizar que meses merecen atencion primero: una anomalia `alta` sugiere una desviacion mucho mas fuerte respecto de la banda aprendida por el modelo, mientras que una `leve` esta apenas por encima del umbral dinamico.

## Criterio de defensa

La ventaja metodologica frente al umbral fijo es que el sistema aprende el comportamiento historico del material y evalua el residuo contra una banda propia de la serie.

Esto permite decir:

```text
La anomalia no se define por un porcentaje arbitrario, sino por la distancia entre el precio observado y el precio esperado por un modelo entrenado sobre el patron historico del material.
```
