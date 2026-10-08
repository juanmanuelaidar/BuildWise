# Protocolo de reevaluación de la tesis BuildWise

**Corte experimental fijo:** enero de 2022 a marzo de 2026. **Horizontes:** 3, 6 y 12 meses. No incorporar observaciones posteriores a la fecha de corte en el dataset que sustenta las nuevas tablas del TFG.

## Antes de ejecutar

1. La dependencia ausente `economic_price` se resolvió retirando la conversión incompleta y conservando los precios registrados. Ver [correcciones y resultados del 08/10/2026](CORRECCIONES_MVP_20261008.md). No introducir nuevas reglas comerciales sin definición de dominio.
2. Guardar el SHA del commit utilizado y el hash del dataset canónico: `sha256sum db/bootstrap/cemento_portland_historico.csv`. Actualmente el CSV versionado contiene 1626 registros entre 2022-01-03 y 2026-03-25; el reporte experimental anterior indicaba 1624. La diferencia exige reconstruir las métricas con **un único** snapshot, no corregir la tabla por suposición.
3. Reconstruir la base mínima siguiendo la secuencia versionada del proyecto: `alembic upgrade head`, `python -m app.operations.bootstrap.seed`, importadores canónicos de cemento, pastina y membrana, carga del snapshot de índices y `python -m app.operations.bootstrap.validate_minimum_dataset`.
4. Verificar dependencias de Prophet y CmdStan. Conservar el snapshot exacto y registrar fechas de publicación de regresores, si están disponibles. **La proyección por fold elimina el uso de los valores del período de prueba, pero las fuentes actuales no permiten demostrar la disponibilidad pública efectiva de todos los índices en sus meses de referencia.**

## Ejecución y criterios

La reevaluación congelada ejecutada está en `db/benchmarks/mvp_2026_03/`. Se reproduce sin base de datos con `python -m app.experiments.pricing.mvp_reassessment`; los experimentos adicionales indicados a continuación requieren la base mínima.

1. Ejecutar `pytest -q tests/test_series.py tests/test_temporal_leakage_tesis.py -o addopts=''` y luego la suite integral. No interpretar una cobertura alta como validación del desempeño predictivo.
2. Ejecutar el experimento de referencia con la corrección ex ante, por ejemplo: `python -m app.experiments.pricing.cemento_forecast_plateau --material "Cemento Portland" --horizontes 3 6 12 --modelos prophet_ipim_nivel_general prophet_ipim_icc_var_materials --sin-ensemble --output-csv tmp/experiments/cemento_exante.csv`. Verificar previamente que IPIM e ICC estén cargados y alineados. Para el ensemble y variantes adicionales, repetir con la batería experimental pertinente.
3. Conservar por fold fechas de corte, regresores estimados, predicciones, observaciones reales, MAPE y MAE; no utilizar valores reales del test como entradas de Prophet, SARIMAX o XGBoost. Comparar cada horizonte por separado, informando la cantidad de folds y su dispersión.
4. Distinguir Cemento Portland, cuyos precios son reales, de Pastina (10 reales/41 estimados) y Membrana Megaflex (8 reales/43 estimados) al corte de marzo de 2026. La ejecución sobre estas dos últimas series solo demuestra el funcionamiento del procesamiento sobre datos híbridos; sus MAPE no acreditan capacidad predictiva sobre precios reales independientes.
5. Tras la corrección del Random Forest, regenerar las marcas de anomalías y registrar cambios respecto de la versión anterior. No informar precisión, recall ni F1 sin fechas verdaderamente etiquetadas por revisión externa. La banda basada en el residuo y la dispersión entre árboles sigue siendo heurística.
6. Para evaluar beneficio económico del DSS, construir una simulación histórica integral contra una política base explícita (por ejemplo, comprar todo inmediatamente), respetando el presupuesto, cantidades y la información disponible en cada origen. Hasta completar esa evaluación, las pruebas actuales demuestran factibilidad de restricciones y contratos del solver, **no** ahorros históricos.

## Actualización del documento

Sustituir las tablas 2.3, 2.4 y 2.5 y la figura 2.1 **solo después** de registrar la corrida ex ante reproducible. Actualizar el selector y las conclusiones si las métricas nuevas lo justifican. Si un modelo no mejora al baseline bajo las nuevas condiciones, reportarlo como resultado del experimento en lugar de conservar la clasificación anterior.

Los datos anteriores se preservan únicamente como historial del proceso de desarrollo y permanecen identificados como mediciones condicionadas por regresores observados a posteriori.
