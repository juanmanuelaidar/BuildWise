# Mediciones Forecasting

> **Actualización 08/10/2026:** el selector utiliza la [reevaluación ex ante](CORRECCIONES_MVP_20261008.md), con evidencia en `db/benchmarks/mvp_2026_03/`. Los modelos y cifras siguientes se conservan como historial experimental; no describen la selección vigente.

> **Estado de auditoría (28/09/2026): resultados históricos PRELIMINARES.** Las corridas con regresores externos de este documento usaron valores reales del período de test en vez de proyecciones calculadas en el corte de cada fold. Sus MAPE sirven para reconstruir la exploración original, pero NO validan precisión operacional ex ante ni permiten promover un nuevo modelo por mejor desempeño hasta repetir los experimentos con el procedimiento corregido (PR #20). El Prophet base sin regresores no está afectado por esa filtración concreta. Además, la corrida histórica informa 1624 precios de Cemento mientras el CSV canónico actualmente versionado contiene 1626 registros reales; se debe congelar y documentar un único snapshot al recalcular las métricas. Para Membrana Megaflex hay 8 registros reales al corte de marzo de 2026 y un noveno en abril, fuera del período del TFG. Las cifras y clasificaciones que siguen son el registro de mediciones originales, **no recomendaciones vigentes**.

## Para que sirve este documento

Este archivo resume, de forma simple, el estado actual del forecasting en `BuildWise`.

La idea es responder rapido estas preguntas:

- que material es la referencia principal;
- cual es el modelo productivo vigente;
- que candidatos experimentales lo mejoran;
- que tan confiables son los otros materiales;
- y que paso cuando se aplico la misma bateria experimental a los tres productos.

## Idea principal

Hoy la referencia metodologica del proyecto es `Cemento Portland`.

La lectura corta es:

- baseline productivo vigente: `prophet_ipim_nivel_general`;
- mejor candidata experimental por regresores: `prophet_ipim_icc_var_materials`;
- segunda mejor candidata experimental: `prophet_ipim_cac_var_materials`;
- mejor resultado puntual a `3 meses`: `ensemble_simple_top2`;
- la misma exploracion profunda ya se corrio tambien sobre `Pastina` y `Membrana Megaflex`;
- no existe un unico mejor modelo para todos los materiales.

## Como leer los numeros

- metrica principal: `MAPE`
- metricas de apoyo: `MAE` y cantidad de `folds`
- efectividad informal: `100 - MAPE`
- no se elige modelo por grafica visual
- no se mezclan horizontes `3`, `6` y `12` meses como si fueran equivalentes

## Datos base

Supuestos principales de estas mediciones:

- material principal: `Cemento Portland`
- variable objetivo: `precio_promedio_normalizado`
- serie mensual desde `2022-01-01`
- backtesting temporal

Datos de `Cemento Portland` usados como referencia:

- registros crudos: `1624`
- puntos mensuales: `51`
- puntos diarios: `759`
- rango observado: `2022-01-03` a `2026-03-25`

## Cemento Portland

### Resumen rapido

La evolucion del ajuste fue esta:

- `Prophet` base: `MAPE 13.90%`
- mejor familia previa con regresores monetarios: `prophet_oficial_mayorista`, `MAPE 7.74%`
- baseline productivo vigente con `IPIM`: `prophet_ipim_nivel_general`, `MAPE 4.93%`
- mejor candidata experimental actual: `prophet_ipim_icc_var_materials`, `MAPE 4.22%`

### Baseline productivo vigente

| Modelo | Regresores | Horizonte | MAE | MAPE | Efectividad informal |
|---|---|---:|---:|---:|---:|
| `prophet_ipim_nivel_general` | `ipim_nivel_general` | `3m` | 6.76 | 4.93% | 95.07% |
| `prophet_ipim_nivel_general` | `ipim_nivel_general` | `6m` | - | 7.51% | 92.49% |
| `prophet_ipim_nivel_general` | `ipim_nivel_general` | `12m` | - | 10.38% | 89.62% |

### Mejores candidatos experimentales

| Modelo | Regresores | Horizonte | MAPE | Efectividad informal | Lectura |
|---|---|---:|---:|---:|---|
| `prophet_ipim_icc_var_materials` | `ipim_nivel_general` + `icc var_materials` | `3m` | 4.22% | 95.78% | mejor candidata por regresores |
| `prophet_ipim_icc_var_materials` | `ipim_nivel_general` + `icc var_materials` | `6m` | 5.52% | 94.48% | mejor candidata por regresores |
| `prophet_ipim_icc_var_materials` | `ipim_nivel_general` + `icc var_materials` | `12m` | 4.51% | 95.49% | mejor candidata por regresores |
| `prophet_ipim_cac_var_materials` | `ipim_nivel_general` + `cac var_materials` | `3m` | 4.33% | 95.67% | segunda mejor candidata |
| `prophet_ipim_cac_var_materials` | `ipim_nivel_general` + `cac var_materials` | `6m` | 5.99% | 94.01% | segunda mejor candidata |
| `prophet_ipim_cac_var_materials` | `ipim_nivel_general` + `cac var_materials` | `12m` | 4.69% | 95.31% | segunda mejor candidata |
| `ensemble_simple_top2` | promedio simple entre las dos mejores variantes de `3m` | `3m` | 4.08% | 95.92% | mejor resultado puntual |

### Que significa esto

- `IPIM` mejoro fuerte frente a las combinaciones anteriores con `dolar_*` e `ipc`.
- despues de esa mejora, `ICC var_materials` y `CAC var_materials` volvieron a bajar el error.
- `ICC var_materials` es hoy la mejor candidata experimental estable y defendible.
- `ensemble_simple_top2` da el mejor `3m`, pero es menos simple de justificar que un modelo con regresores claros.

## Otros materiales

`Pastina` y `Membrana Megaflex` se pueden pronosticar, pero no tienen la misma solidez metodologica que cemento.

Motivo:

- `Cemento Portland` usa serie real, densa y continua;
- `Pastina` y `Membrana Megaflex` usan series hibridas con datos `REAL` y `ESTIMADO`.
- aun asi, a ambos se les aplico la misma bateria experimental usada en cemento: baseline `IPIM`, lags, medias moviles, variaciones, `CAC`, `ICC` y ensemble simple.

### Mejores resultados por material y horizonte

| Material | Horizonte | Mejor modelo medido | MAPE | Efectividad informal |
|---|---:|---|---:|---:|
| `Cemento Portland` | `3m` | `ensemble_simple_top2` | 4.08% | 95.92% |
| `Cemento Portland` | `3m` | `prophet_ipim_icc_var_materials` | 4.22% | 95.78% |
| `Cemento Portland` | `6m` | `prophet_ipim_icc_var_materials` | 5.52% | 94.48% |
| `Cemento Portland` | `12m` | `prophet_ipim_icc_var_materials` | 4.51% | 95.49% |
| `Pastina` | `3m` | `prophet_ipim_nivel_general_lags` | 3.37% | 96.63% |
| `Pastina` | `6m` | `ensemble_simple_top2` | 4.99% | 95.01% |
| `Pastina` | `12m` | `prophet_ipim_cac_var_materials` | 4.94% | 95.06% |
| `Membrana Megaflex` | `3m` | `prophet_ipim_nivel_general_lags` | 6.70% | 93.30% |
| `Membrana Megaflex` | `6m` | `prophet_ipim_nivel_general_lags` | 7.23% | 92.77% |
| `Membrana Megaflex` | `12m` | `prophet_mayorista` | 11.17% | 88.83% |

### Lectura corta

- `Pastina` mejoro fuerte cuando se le aplico la bateria profunda, sobre todo con lags y con combinaciones `IPIM + CAC/ICC`.
- `Membrana Megaflex` tambien mejoro bastante en `3m` y `6m`, especialmente con lags.
- `Membrana Megaflex` sigue siendo el caso mas debil de los tres, y en `12m` la exploracion nueva no supero al mejor modelo previo documentado.
- no conviene imponer un solo modelo para todos los materiales.

## Mejor modelo por producto

Si se toma el mejor resultado actualmente disponible para cada producto, la foto queda asi:

| Producto | Mejor modelo actual | Horizonte donde mejor se sostiene | Criterio de lectura |
|---|---|---|---|
| `Cemento Portland` | `prophet_ipim_icc_var_materials` | `3m`, `6m` y `12m` | mejor candidata por regresores, con mejora consistente frente al baseline productivo |
| `Pastina` | `prophet_ipim_nivel_general_lags` | `3m` | mejor resultado corto; en horizontes mayores cambian los ganadores |
| `Membrana Megaflex` | `prophet_ipim_nivel_general_lags` | `3m` y `6m` | mejor mejora nueva en corto y mediano plazo |

Observacion importante:

- en `Cemento Portland` hay una mejor candidata bastante estable entre horizontes;
- en `Pastina` y `Membrana Megaflex` no hay un ganador unico absoluto para todos los horizontes;
- por eso el criterio correcto no es elegir un unico modelo global, sino seleccionar por material y horizonte.

## Confiabilidad relativa

| Material | Datos reales | Datos estimados | Folds | Confiabilidad relativa |
|---|---:|---:|---:|---|
| `Cemento Portland` | 1624 precios | 0 | 9 | alta |
| `Pastina` | 10 registros | 41 | 9 | media |
| `Membrana Megaflex` | 9 registros | 43 | 5 | media-baja |

Interpretacion:

- `Cemento Portland` es la base mas fuerte para defender resultados de tesis.
- un `MAPE` bueno en un material hibrido no vale lo mismo que en una serie totalmente real.
- la efectividad informal sirve para comunicar, pero no reemplaza al `MAPE`.

## Criterio metodologico

Las decisiones de modelo se tomaron con estas reglas:

- primero se comparo contra baselines simples;
- despues se probaron regresores monetarios e inflacionarios;
- cuando esas combinaciones empezaron a amesetarse, se probaron regresores sectoriales mas defendibles;
- despues se aplico la misma bateria experimental a los tres materiales para no sobreconcluir solo desde cemento;
- una mejora solo cuenta si baja el error y no empeora demasiado la estabilidad entre folds;
- no se promueve un modelo solo porque la curva “se vea bien”.

## Como elegir modelo si se agregan mas productos

Si en el futuro se incorporan mas materiales, la seleccion del mejor modelo debe seguir este mismo criterio:

1. construir una serie mensual comparable sobre `precio_promedio_normalizado`;
2. correr un baseline simple y un baseline productivo con `IPIM`;
3. evaluar la misma bateria experimental:
   - lags
   - medias moviles
   - variaciones
   - combinaciones con `CAC`
   - combinaciones con `ICC`
   - ensemble simple si tiene sentido
4. comparar por horizonte `3`, `6` y `12` meses por separado;
5. elegir el modelo con mejor `MAPE`, controlando tambien:
   - `MAE`
   - cantidad de `folds`
   - estabilidad entre folds
   - coherencia economica de los regresores
6. no promover un modelo si mejora apenas el error pero se vuelve mucho mas inestable;
7. documentar el mejor modelo por material, no asumir que el ganador de un producto sirve para otro.

En otras palabras:

- primero se mira `MAPE`;
- despues se valida estabilidad;
- despues se revisa si el modelo es defendible economicamente;
- recien entonces se recomienda como mejor modelo para ese nuevo producto.

## Estado actual de decision

Hoy conviene distinguir dos planos:

- productivo vigente: `prophet_ipim_nivel_general`
- mejor candidata experimental: `prophet_ipim_icc_var_materials`

Eso significa que el sistema actual puede sostener su baseline productivo, pero ya existe evidencia seria de variantes experimentales mejores en los tres materiales, aunque con distinta fortaleza metodologica.

Lectura material por material:

- `Cemento Portland`: la mejor candidata experimental sigue siendo `prophet_ipim_icc_var_materials`.
- `Pastina`: la exploracion profunda encontro mejoras claras frente al baseline `IPIM`; en `3m` gano `prophet_ipim_nivel_general_lags`.
- `Membrana Megaflex`: la exploracion profunda tambien encontro mejoras claras en `3m` y `6m`, pero no alcanzo para superar el mejor `12m` historico ya documentado.

## Reproducibilidad

Estas mediciones dependen de cerrar bien el bootstrap minimo reproducible de tesis.

Piezas clave:

- dataset canonico: `db/bootstrap/cemento_portland_historico.csv`
- importador oficial: `app/operations/bootstrap/import_cemento_canonico.py`
- materiales hibridos: `Pastina` y `Membrana Megaflex`
- regresores base: `ipc`, `dolar_*`, `ipim_nivel_general`

Flujo objetivo:

1. `alembic upgrade head`
2. `python -m app.operations.bootstrap.seed`
3. `python -m app.operations.bootstrap.import_cemento_canonico`
4. `python -m app.operations.bootstrap.import_pastina`
5. `python -m app.operations.bootstrap.import_membrana_megaflex`
6. `python -m app.operations.bootstrap.import_external_indices_snapshot`
7. `python -m app.operations.bootstrap.validate_minimum_dataset`

## Siguiente paso

Lo que falta cerrar con claridad es:

- si `ensemble_simple_top2` queda solo como evidencia exploratoria o como candidata real;
- si `prophet_ipim_icc_var_materials` pasa de experimental a recomendacion formal;
- si los nuevos mejores modelos de `Pastina` y `Membrana Megaflex` deben reemplazar formalmente a los previamente documentados;
- terminar de endurecer la reproducibilidad del bootstrap.
