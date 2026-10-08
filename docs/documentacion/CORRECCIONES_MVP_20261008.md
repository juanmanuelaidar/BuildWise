# Correcciones del MVP y reevaluación — 8 de octubre de 2026

Esta entrega corrige el código y conserva el alcance del MVP presentado. El borrador de la tesis todavía no se modifica. Consolida las correcciones temporales de la rama `fix/tesis-validacion-temporal-20260928` con la reparación del runtime, el selector y la precisión del optimizador.

## Cambios de comportamiento

- El backend vuelve a cargar: se retiran las referencias al módulo inexistente `economic_price` y la conversión incompleta de precios. Los análisis conservan los precios registrados y normalizados; los comprobantes originales no se reescriben. No se inventan descuentos ni reglas fiscales. La funcionalidad comercial de márgenes existente se conserva.
- Cada fold proyecta índices futuros usando solo el historial hasta su corte. Las tasas mensuales, que pueden ser negativas o cero, se proyectan por persistencia. Los niveles positivos usan crecimiento compuesto sobre los últimos doce registros, respetando los meses transcurridos.
- El Random Forest no recibe el precio objetivo ni la variación contemporánea como predictores. Reentrena cada seis puntos mensuales o treinta puntos de series densas. Las anomalías se regeneran; sus bandas siguen siendo heurísticas.
- El selector carga resultados ex ante con hashes de fuentes y del CSV de métricas. Se elimina el respaldo de métricas antiguas codificadas en el código. Un horizonte no evaluado reutiliza una configuración, pero no hereda el MAPE de otro horizonte. Si cambia el historial, se retiran sus métricas de referencia.
- La selección permanece exploratoria y se comunica como no calibrada para decisiones fuertes: falta una evaluación externa independiente. También se advierte sobre datos estimados, pocos folds y derrotas frente al baseline simple. Las advertencias quedan visibles fuera del acordeón de detalles.
- Las cantidades compradas se resuelven en incrementos de 0,0001. Después de convertir a Decimal se comprueba el presupuesto monetario publicado y la compra mínima. La cantidad postergada se calcula como objetivo menos compra inmediata. El caso precio 100.000 y presupuesto 15 compra 0,0001 por 10, en vez de redondear a una compra de 20.
- Se invalida la firma de snapshots anteriores y se regeneran 36 snapshots del bootstrap: tres materiales por horizontes de uno a doce meses. Se conserva la procedencia real/estimada al serializarlos. Prophet utiliza el runtime incluido en su wheel si no existe una instalación global de CmdStan; se fija `cmdstanpy==1.2.5`, compatible con `prophet==1.1.6`.

## Evidencia experimental congelada

Período: enero de 2022 a marzo de 2026, 51 meses por material. Cemento tiene 1626 registros reales; Pastina, 10 reales y 41 estimados; Membrana, 8 reales y 43 estimados. La observación de Membrana de abril se utiliza en los snapshots operativos del bootstrap, pero queda fuera de esta evaluación. Por ese motivo esos snapshots no muestran las métricas de referencia de marzo como si validaran el historial de abril.

El entrenamiento inicial tiene 24 meses. Los bloques de prueba no se solapan y tienen la longitud del horizonte: 9 folds a tres meses, 4 a seis y 2 a doce. La selección compara tres configuraciones Prophet ejecutables por material; no agota todas las familias experimentales del repositorio. `naive_last` repite el último precio observado y sirve de comparación explícita.

| Material | Horizonte | Configuración Prophet seleccionada | MAPE | MAE en ARS/kg | Folds | MAPE último precio |
|---|---:|---|---:|---:|---:|---:|
| Cemento | 3 | IPIM | 10,35% | 14,17 | 9 | 7,55% |
| Cemento | 6 | Sin regresores | 18,43% | 24,38 | 4 | 14,11% |
| Cemento | 12 | IPIM | 20,91% | 30,01 | 2 | 27,49% |
| Pastina | 3 | IPIM + variación materiales CAC | 5,52% | 119,08 | 9 | 6,95% |
| Pastina | 6 | IPIM + variación materiales CAC | 9,75% | 210,49 | 4 | 12,70% |
| Pastina | 12 | IPIM + variación materiales CAC | 21,15% | 465,17 | 2 | 22,64% |
| Membrana | 3 | IPIM + variación materiales ICC | 8,61% | 592,84 | 9 | 8,31% |
| Membrana | 6 | IPIM + variación materiales ICC | 14,05% | 964,42 | 4 | 16,91% |
| Membrana | 12 | IPIM + variación general ICC | 15,77% | 1113,76 | 2 | 30,83% |

Cemento a tres y seis meses y Membrana a tres no mejoran el baseline de último precio. Esas derrotas se conservan como resultados. Las métricas de Pastina y Membrana describen series híbridas, no exactitud sobre precios reales independientes. `100 - MAPE` no representa una exactitud estadística.

Los archivos de `db/benchmarks/mvp_2026_03/` conservan resultados por configuración, predicciones por fold, dispersión del MAPE, anomalías detectadas, casos ilustrativos del solver y el manifiesto de fuentes, dependencias y código. El SHA del manifiesto identifica la base de la corrida; los hashes de implementación identifican las modificaciones locales evaluadas.

## Reproducción y actualización

Con dependencias instaladas y la configuración requerida del proyecto:

```bash
ENVIRONMENT=test AUTH_SECRET_KEY=buildwise-local-test-secret python -m app.experiments.pricing.mvp_reassessment
ENVIRONMENT=test AUTH_SECRET_KEY=buildwise-local-test-secret python -m app.experiments.pricing.mvp_snapshots
```

El segundo comando reemplaza `tmp/forecast_snapshots.json` y está destinado al bootstrap nuevo, con IDs 1, 2 y 3. En una base existente, con otros IDs o precios, ejecutar el precomputador contra esa base:

```bash
FORECAST_ALLOW_SYNCHRONOUS_COMPUTE=true python -m app.modules.pricing.application.precompute_forecasts --horizontes 1 2 3 4 5 6 7 8 9 10 11 12
```

La API puede mantener deshabilitado el cálculo síncrono. Cambiar precios o índices exige volver a precomputar. Antes del despliegue se deben verificar los snapshots con la base de destino.

## Verificación y límites

Se verifican la suite integral del backend con cobertura, lint y las pruebas, lint y compilación del frontend. También se ejecutan los modelos Prophet y CBC reales para producir los archivos de evidencia y los snapshots. Las cuatro pruebas de integración PostgreSQL requieren una base migrada y se ejecutan en CI; no deben confundirse con los tests unitarios locales.

Quedan para la revisión de tesis y la validación posterior: actualizar tablas y conclusiones con estos resultados, obtener fechas de publicación de índices para una validación estricta de disponibilidad histórica, validar anomalías con etiquetas externas y medir ahorro realizado frente a una política de compra explícita. Los casos del solver verifican factibilidad; no demuestran ahorro histórico. Una evaluación independiente o anidada permitirá revisar la etiqueta de calibración sin usar los mismos folds para elegir y validar modelos.
