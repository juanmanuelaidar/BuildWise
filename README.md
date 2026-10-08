<p align="center">
  <img src="frontend/bwlogo.png" alt="BuildWise" width="220" />
</p>

# BuildWise

[![codecov](https://codecov.io/gh/juanmanuelaidar/BuildWise/graph/badge.svg?branch=main)](https://codecov.io/gh/juanmanuelaidar/BuildWise)

BuildWise es un sistema para analizar precios de materiales de construccion, proyectar su evolucion y apoyar decisiones de compra.

Hoy el proyecto cubre tres frentes:

- catalogo e historial de precios;
- forecasting de materiales;
- recomendaciones y optimizacion de compra.

El criterio central del proyecto es que la decision economica no la toma un modelo generativo. El sistema calcula forecast, anomalías, presupuesto y recomendacion con datos propios; la IA conversacional solo interpreta pedidos y redacta explicaciones.

## Que resuelve

Permite responder preguntas como:

- cuanto aumento un material;
- cuanto cuesta realmente por kg, unidad o metro;
- como evoluciono el precio en el tiempo;
- cuanto podria costar comprar ahora o despues;
- como asignar un presupuesto limitado entre materiales.

## Estado actual

El repositorio ya tiene:

- backend con FastAPI;
- base PostgreSQL con Docker;
- frontend web;
- autenticacion con roles;
- carga y consulta de precios historicos;
- normalizacion automatica de precios;
- graficos y filtros;
- forecast para materiales soportados;
- endpoints de recomendacion y optimizacion de compra.

## Levantar el proyecto

### 1. Configuracion local

```bash
cp .env.example .env
```

Completar en `.env`:

```env
OPENAI_API_KEY=replace-with-your-token
OPENAI_BASE_URL=https://ai.cloud.um.edu.ar/api/v1
OPENAI_MODEL=gemma4-26b
```

Si se usa Claude como respaldo de la capa conversacional:

```env
CHAT_PROVIDER=anthropic
ANTHROPIC_API_KEY=replace-with-your-token
ANTHROPIC_MODEL=replace-with-your-claude-model
```

La integracion de IA es intercambiable: `CHAT_PROVIDER=openai` usa un endpoint compatible con `/chat/completions`, mientras que `CHAT_PROVIDER=anthropic` usa la API de Claude. Los calculos de precios, forecast, presupuesto y recomendaciones no dependen del proveedor generativo.

### 2. Ejecutar

```bash
docker compose up -d --build
```

### 3. URLs utiles

```text
Frontend: http://localhost:3000
API: http://localhost:8000
Swagger: http://localhost:8000/docs
```

### 4. Comandos rapidos

```bash
docker compose ps
docker compose down
```

## Usuarios de prueba

Admin:

```text
usuario: admin
clave: admin123
```

Cliente:

```text
usuario: cliente
clave: cliente123
```

## Stack

- Backend: FastAPI, SQLAlchemy, PostgreSQL, Alembic, Pydantic
- Frontend: React, Vite, Material UI, Tailwind CSS, Chart.js
- Infra local: Docker Compose

## Estructura principal

```text
app/
  modules/
    auth/
    catalog/
    pricing/
    health/
  shared/
  operations/

frontend/
docs/documentacion/
db/
tests/
```

## Arquitectura

El backend sigue un monolito modular con capas:

- `domain`: reglas de negocio;
- `application`: casos de uso;
- `infrastructure`: persistencia y adaptadores tecnicos;
- `interfaces`: rutas HTTP y schemas.

Dependencias esperadas:

```text
interfaces -> application -> domain
infrastructure -> application/domain
```

## Forecasting

El foco metodologico principal esta en `Cemento Portland`.

Puntos importantes:

- la variable objetivo es `precio_promedio_normalizado`;
- la metrica principal es `MAPE`;
- no se elige modelo por grafica visual;
- `Cemento Portland` es la serie mas fuerte metodologicamente;
- `Pastina` y `Membrana Megaflex` usan series mas hibridas.

Estado resumido:

- baseline productivo vigente para cemento: `prophet_ipim_nivel_general`;
- mejor candidata experimental actual: `prophet_ipim_icc_var_materials`;
- segunda mejor candidata experimental: `prophet_ipim_cac_var_materials`.
- `Pastina` y `Membrana Megaflex` ya pasaron por la misma bateria experimental profunda.

Mejores estimaciones hoy:

- `Cemento Portland`
  - `3m`: `ensemble_simple_top2`
  - mejor candidata por regresores: `prophet_ipim_icc_var_materials`
- `Pastina`
  - `3m`: `prophet_ipim_nivel_general_lags`
  - `6m`: `ensemble_simple_top2`
  - `12m`: `prophet_ipim_cac_var_materials`
- `Membrana Megaflex`
  - `3m`: `prophet_ipim_nivel_general_lags`
  - `6m`: `prophet_ipim_nivel_general_lags`
  - `12m`: `prophet_mayorista`

Detalle completo:

- [docs/documentacion/MEDICIONES_FORECASTING.md](docs/documentacion/MEDICIONES_FORECASTING.md)
- [docs/documentacion/DECISIONES_TESIS.md](docs/documentacion/DECISIONES_TESIS.md)

## Optimizacion de compras

La epica de optimizacion ya tiene backend implementado como cierre DSS de compra trazable para el MVP. La salida no se limita a graficos: transforma forecast, criticidad y presupuesto en una recomendacion operativa.

Capacidades disponibles:

- recomendacion de momento de compra;
- comparacion de estrategias;
- optimizacion con restriccion presupuestaria;
- priorizacion de materiales criticos;
- recomendacion operativa trazable por presupuesto (`/compras/recomendacion-operativa`).

La optimizacion bajo presupuesto usa `PuLP`.

Detalle metodologico:

- [docs/HU/DISENO_EPICA_5.md](docs/HU/DISENO_EPICA_5.md)
- [docs/documentacion/DEMO_DSS.md](docs/documentacion/DEMO_DSS.md)

## Asistente IA unificado

El asistente IA unificado concentra consultas generales, RAG operativo y presupuestacion guiada en una sola experiencia. Recupera datos del backend antes de responder y muestra fuentes usadas para que la respuesta sea trazable. Cuando detecta una necesidad de compra, abre un panel de validacion editable y genera una propuesta para el MVP. La capa conversacional no calcula la decision por si sola: interpreta, valida y redacta sobre resultados ya resueltos por backend.

Alcance implementado:

- RAG operativo sobre catalogo, precios historicos, fuentes, indices externos, forecast, recomendaciones y margenes segun rol;
- chips visibles con `RAG backend`, intencion clasificada, material resuelto, horizonte y fuentes recuperadas;
- clasificacion formal de intencion: `HISTORICO`, `FORECAST`, `RECOMENDACION`, `PRESUPUESTO`, `CATALOGO`, `ADMIN` o `FUERA_ALCANCE`;
- auditoria persistente por consulta con usuario, pregunta, fuentes, fallback, duracion y respuesta, visible para admin en `Auditoria IA`;
- metricas agregadas de auditoria y bateria canonica para medir estabilidad del RAG;
- medicion de determinismo RAG sobre consultas repetidas, comparando intencion, material, horizonte, fuentes, contexto y fallback;
- resolucion de materiales por nombre, alias y errores frecuentes del MVP;
- productos soportados: `Cemento Portland`, `Pastina` y `Membrana Megaflex`;
- interpretacion IA de producto, cantidad, fase de obra, fecha/horizonte, presupuesto y tolerancia al riesgo, limitada al catalogo MVP;
- validacion editable antes de ejecutar calculos de recomendacion;
- presupuesto actual con precio vigente;
- comparacion contra forecast y recomendacion contextual;
- decision `COMPRAR_AHORA`, `POSTERGAR`, `ESCALONAR` o `SIN_VENTAJA_CLARA`;
- redaccion final asistida por IA sin alterar importes ni decisiones calculadas por backend;
- salida auditable con precio actual, precio proyectado, diferencia, MAPE, umbral, fuente de datos y advertencias.

El presupuesto del MVP parte de precios registrados y cantidades confirmadas. Impuestos, flete, descuentos comerciales y stock minimo quedan fuera del calculo base salvo que se formalicen explicitamente como regla del dominio.

Detalle funcional y demo:

- [docs/HU/DISENO_EPICA_8.md](docs/HU/DISENO_EPICA_8.md)
- [docs/documentacion/DEMO_ASISTENTE_COMERCIAL.md](docs/documentacion/DEMO_ASISTENTE_COMERCIAL.md)
- [docs/documentacion/RAG_OPERATIVO_ASISTENTE.md](docs/documentacion/RAG_OPERATIVO_ASISTENTE.md)

## Bootstrap y reproducibilidad

La prioridad actual del proyecto es cerrar un entorno reproducible de tesis con un dataset minimo gobernado desde el repo.

El flujo objetivo de bootstrap es:

```bash
make bootstrap-all
```

Ese flujo debe dejar cargados:

- `Cemento Portland`;
- `Pastina`;
- `Membrana Megaflex`;
- regresores base como `ipc`, `dolar_*` e `ipim_nivel_general`.

La fuente canonica de cemento es:

- `db/bootstrap/cemento_portland_historico.csv`

Importador oficial:

- `app/operations/bootstrap/import_cemento_canonico.py`

## Endpoints utiles

Autenticacion:

```http
POST /auth/login
GET /auth/me
```

Catalogo:

```http
GET /materiales
GET /presentaciones
GET /fuentes
```

Precios:

```http
GET /precios-historicos
POST /precios-historicos
GET /materiales/{material_id}/precios
GET /materiales/{material_id}/serie-precios
```

## Migraciones y tests

Migraciones:

```bash
.venv/bin/python -m alembic upgrade head
.venv/bin/python -m alembic current
.venv/bin/python -m alembic check
```

Tests:

```bash
.venv/bin/python -m pytest -q
```

Verificacion enfocada del DSS:

```bash
.venv/bin/python -m pytest --no-cov tests/test_purchase_recommendations.py tests/test_purchase_strategies.py tests/test_purchase_optimization.py
```

## Documentacion

- [Correcciones y reevaluación del MVP (08/10/2026)](docs/documentacion/CORRECCIONES_MVP_20261008.md): cambios de código, métricas ex ante, límites y actualización de snapshots.

- [docs/documentacion/MEDICIONES_FORECASTING.md](docs/documentacion/MEDICIONES_FORECASTING.md): metricas, backtesting y comparativas
- [docs/documentacion/ENTREGABLE_METODOLOGIA_TIF3.md](docs/documentacion/ENTREGABLE_METODOLOGIA_TIF3.md): borrador principal del marco metodologico y analisis de resultados
- [docs/documentacion/GUIA_MARCO_METODOLOGICO_Y_RESULTADOS.md](docs/documentacion/GUIA_MARCO_METODOLOGICO_Y_RESULTADOS.md): lineamientos de escritura segun TIF 3
- [docs/documentacion/ESTADO_MVP.md](docs/documentacion/ESTADO_MVP.md): alcance, evidencia y checklist end-to-end del MVP
- [docs/documentacion/DECISIONES_TESIS.md](docs/documentacion/DECISIONES_TESIS.md): decisiones metodologicas
- [docs/documentacion/REGISTRO_DECISIONES_TRABAJO.md](docs/documentacion/REGISTRO_DECISIONES_TRABAJO.md): decisiones operativas de cierre MVP
- [docs/documentacion/DEMO_DSS.md](docs/documentacion/DEMO_DSS.md): guion de demo reproducible del DSS
- [docs/documentacion/DEMO_ASISTENTE_COMERCIAL.md](docs/documentacion/DEMO_ASISTENTE_COMERCIAL.md): guion de demo del asistente IA unificado
- [docs/documentacion/RAG_OPERATIVO_ASISTENTE.md](docs/documentacion/RAG_OPERATIVO_ASISTENTE.md): diseno del RAG operativo del asistente
- [docs/documentacion/GUIA_FLUJOS_UI.md](docs/documentacion/GUIA_FLUJOS_UI.md): recorrido funcional de la interfaz
- [docs/documentacion/GLOSARIO_FUNCIONAL.md](docs/documentacion/GLOSARIO_FUNCIONAL.md): terminos principales usados por el sistema
- [docs/documentacion/ANOMALIAS_RANDOM_FOREST.md](docs/documentacion/ANOMALIAS_RANDOM_FOREST.md): deteccion de anomalias con Random Forest
- [docs/HU/HU.md](docs/HU/HU.md): historias de usuario y estado funcional
- [docs/HU/DISENO_EPICA_1.md](docs/HU/DISENO_EPICA_1.md): gestion y preparacion de datos
- [docs/HU/DISENO_EPICA_2.md](docs/HU/DISENO_EPICA_2.md): analisis y visualizacion
- [docs/HU/DISENO_EPICA_3.md](docs/HU/DISENO_EPICA_3.md): prediccion de precios
- [docs/HU/DISENO_EPICA_4.md](docs/HU/DISENO_EPICA_4.md): proyeccion de costos de obra
- [docs/HU/DISENO_EPICA_5.md](docs/HU/DISENO_EPICA_5.md): optimizacion de compras
- [docs/HU/DISENO_EPICA_6.md](docs/HU/DISENO_EPICA_6.md): asistencia conversacional
- [docs/documentacion/DISENO_SELECTOR_MODELOS.md](docs/documentacion/DISENO_SELECTOR_MODELOS.md): seleccion de modelos
- [docs/documentacion/DISENO_INTEGRACION_SELECTOR_FORECAST.md](docs/documentacion/DISENO_INTEGRACION_SELECTOR_FORECAST.md): integracion futura del selector

## Nota de alcance

El `README` queda intencionalmente corto. La idea es que sirva para entender rapido:

- que hace el proyecto;
- como levantarlo;
- donde esta el estado real;
- y en que documento profundizar cada tema.
