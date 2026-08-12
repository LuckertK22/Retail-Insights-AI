"""
api/main.py

App FastAPI principal. Define los routers, carga el modelo al iniciar
y expone Swagger en /docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference
from api.routers import predict, insights, chat
from api.services.model_service import model_service
from api.services.insight_service import insight_service
import os
from pathlib import Path

app = FastAPI(
    title="Retail Insights AI",
    description="API para predecir ventas y consultar insights del Superstore dataset.",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(predict.router, prefix="/predict", tags=["Predict"])
app.include_router(insights.router, prefix="/insights", tags=["Insights"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])


@app.on_event("startup")
def startup():
    """Carga el modelo y cachea insights al arrancar."""
    model_service.load()
    insight_service.load()


@app.on_event("shutdown")
def shutdown():
    """Limpieza al apagar."""
    print("API apagándose")


@app.get("/", tags=["Root"])
def root():
    return {"message": "Retail Insights AI API. Docs en /docs | Scalar en /scalar"}


@app.get("/scalar", include_in_schema=False)
def scalar():
    return get_scalar_api_reference(title="Retail Insights AI")


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "insights_cached": insight_service.is_loaded(),
    }
