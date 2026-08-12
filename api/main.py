"""
api/main.py

App FastAPI principal. Define los routers, carga el modelo al iniciar
y expone Swagger en /docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference
from api.routers import predict, insights, chat
import joblib
from pathlib import Path

app = FastAPI(
    title="Retail Insights AI",
    description="API para predecir ventas y consultar insights del Superstore dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/predict", tags=["Predict"])
app.include_router(insights.router, prefix="/insights", tags=["Insights"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

MODEL_PATH = Path(__file__).parent.parent / "ml" / "model.pkl"
model = None


@app.on_event("startup")
def load_model():
    """Carga el modelo una sola vez al arrancar la API."""
    global model
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado: {MODEL_PATH}")


@app.get("/", tags=["Root"])
def root():
    return {"message": "Retail Insights AI API. Docs en /docs | Scalar en /scalar"}


@app.get("/scalar", include_in_schema=False)
def scalar():
    return get_scalar_api_reference(title="Retail Insights AI")


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model is not None}
