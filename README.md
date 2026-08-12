# Retail Insights AI

Sistema de IA aplicada al negocio retail. Analiza datos de ventas, predice ventas futuras, expone todo vía API REST y permite consultar los datos en lenguaje natural con un agente conversacional.

## Estado del proyecto

| Fase | Estado | Descripción |
|------|--------|-------------|
| 0 — Setup | ✅ | Estructura, dependencias |
| 1 — EDA | ✅ | Análisis exploratorio + dataset limpio |
| 2 — ML | ✅ | RandomForest predictor (R²=0.05) |
| 3 — FastAPI | ✅ | API REST con /predict, /insights |
| 4 — Agente IA | ✅ | LangGraph + Gemini (chat en lenguaje natural) |
| 5 — n8n | ⬜ | Automatización (pendiente) |
| 6 — Despliegue | ⬜ | Producción (pendiente) |

---

## Quick start

### 1. Clonar y crear entorno

```bash
git clone https://github.com/LuckertK22/Retail-Insights-AI.git
cd Retail-Insights-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env y agregar tu GEMINI_API_KEY
```

### 3. Generar el modelo (si no existe)

```bash
python ml/train_model.py
```

### 4. Arrancar la API

```bash
python -m uvicorn api.main:app --reload
```

### 5. Abrir la interfaz

- **Scalar UI (recomendada):** http://localhost:8000/scalar
- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

---

## Endpoints

### `GET /insights/`
Estadísticas del dataset (sin necesidad del modelo).

```json
{
  "total_sales": 2297200.86,
  "total_profit": 286397.29,
  "profit_margin": 0.1247,
  "avg_order_value": 229.86,
  "top_categories": [
    {"name": "Technology", "value": 836154.03},
    {"name": "Furniture", "value": 741999.80},
    {"name": "Office Supplies", "value": 719047.03}
  ],
  "worst_profit_subcategories": [
    {"name": "Tables", "value": -17725.86},
    {"name": "Bookcases", "value": -12819.49}
  ],
  "sales_by_region": [
    {"name": "West", "value": 710219.68},
    {"name": "East", "value": 678781.24}
  ]
}
```

### `POST /predict/`
Predice las ventas de un pedido. Requiere:

```json
{
  "Category": "Furniture",
  "Region": "West",
  "Segment": "Consumer",
  "Ship_Mode": "Standard Class",
  "Discount": 0.2,
  "Quantity": 3,
  "Year": 2017,
  "Month": 11,
  "DayOfWeek": 0
}
```

Respuesta:
```json
{"predicted_sales": 237.41, "model": "RandomForestRegressor"}
```

### `POST /chat/`
Agente conversacional con IA. Recibe texto libre y responde en lenguaje natural.

**Prompt de ejemplo:**
> "¿Cuánto vendería si vendo 5 Chairs en West con 20% de descuento?"

**Respuesta:**
> "Basándome en los datos históricos, para un pedido de 5 Chairs en West con 20% de descuento, el modelo predice ventas aproximadas de $214.05."

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                     Cliente / Frontend                   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI (uvicorn)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ /predict │  │ /insights│  │  /chat   │  │ /health │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┘ │
│       │              │              │                   │
│  ┌────▼────┐   ┌─────▼────┐  ┌────▼─────┐            │
│  │ Model   │   │ Insights  │  │ LangGraph│            │
│  │ Service │   │ Service   │  │  Agent   │            │
│  └────┬────┘   └──────────┘  └────┬─────┘            │
│       │                            │                   │
│  ┌────▼────┐                  ┌───▼─────┐             │
│  │ model.  │                  │ Gemini   │             │
│  │ pkl     │                  │ API      │             │
│  └─────────┘                  └─────────┘             │
└─────────────────────────────────────────────────────────┘
```

### Flujo del agente LangGraph

```
Usuario → "agent" (Gemini decide) → ¿tool_calls?
                                      ↓ sí
                                   "tools"
                                      ↓
                          consultar_insights() o predecir_ventas()
                                      ↓
                                   "agent" → Respuesta
```

---

## Estructura del proyecto

```
Retail-Insights-AI/
├── data/
│   ├── superstore.csv          # Dataset original
│   └── superstore_clean.csv   # Dataset limpio (40 columnas)
├── notebooks/
│   └── exploracion.ipynb      # Análisis EDA paso a paso
├── ml/
│   ├── train_model.py         # Script de entrenamiento
│   └── model.pkl             # Modelo RandomForest (NO en git)
├── api/
│   ├── main.py                # App FastAPI + middlewares
│   ├── models.py              # Esquemas Pydantic
│   ├── routers/
│   │   ├── predict.py         # POST /predict/
│   │   ├── insights.py        # GET /insights/
│   │   └── chat.py            # POST /chat/
│   ├── services/
│   │   ├── model_service.py   # Gestor del modelo (singleton)
│   │   └── insight_service.py # Cache de insights
│   └── agent/
│       ├── tools.py           # Tools del agente (LangChain)
│       └── graph.py           # Grafo LangGraph
├── automation/                 # (pendiente) workflows n8n
├── .env.example               # Variables de entorno template
├── requirements.txt            # Dependencias versionadas
├── pyproject.toml             # Config de linting
├── Dockerfile                 # Containerización
└── .github/workflows/ci.yml   # GitHub Actions CI
```

---

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `GEMINI_API_KEY` | API key de Google Gemini | **requerido** |
| `ALLOWED_ORIGINS` | Orígenes CORS permitidos | `http://localhost:3000,http://localhost:8000` |
| `API_BASE` | Base URL del API (para tools del agente) | `http://localhost:8000` |

---

## Stack tecnológico

| Área | Herramienta |
|------|-------------|
| Datos | Pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| ML | scikit-learn, RandomForest, joblib |
| Backend | FastAPI, Pydantic, Uvicorn |
| Agente IA | LangChain, LangGraph, Gemini Flash |
| API Docs | Scalar (UI), Swagger (/docs) |
| Container | Docker |
| CI/CD | GitHub Actions |

---

## Hallazgos clave (Fase 1)

1. **`Sales` muy sesgada**: media $229.86 vs mediana $54.49. Se usa `log1p()` para entrenar.
2. **Tables y Bookcases pierden dinero**: sub-categorías con profit negativo (1,871 pedidos con pérdida).
3. **Concentración en Q4**: Nov ($352K) y Dic ($325K) son los meses más altos. West lidera regiones, Consumer lidera segmentos.

---

## Modelo ML — Resultados

| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| LinearRegression | $205.84 | $757.47 | 0.029 |
| **RandomForest** | **$203.34** | **$749.34** | **0.049** |

R² bajo (5%) es normal para ventas retail individuales — el dataset es ruidoso.

---

## Desarrollo

### Correr con Docker

```bash
docker build -t retail-insights .
docker run -p 8000:8000 --env-file .env retail-insights
```

### Tests de los endpoints

```bash
# Health
curl http://localhost:8000/health

# Insights
curl http://localhost:8000/insights/

# Predict
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"Category":"Furniture","Region":"West","Segment":"Consumer","Ship_Mode":"Standard Class","Discount":0.2,"Quantity":3,"Year":2017,"Month":11}'

# Chat
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message":"¿Cuál es la categoría que más vende?"}'
```

---

## Commits recientes

| Commit | Descripción |
|--------|-------------|
| `e30487d` | Chore: deps versionados, Dockerfile, pyproject.toml, CI, DayOfWeek |
| `e0364e3` | Refactor: logging, error handling, API_BASE via env |
| `c4f26ab` | Refactor: service layer, CORS via env, cache insights |
| `8562749` | Feat: FastAPI + LangGraph agent con Gemini |
| `a26e247` | Feat: train_model.py + RandomForest (R²=0.049) |
| `8970de6` | Chore: setup inicial + Fase 1 |
