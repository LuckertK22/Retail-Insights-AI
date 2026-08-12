"""
api/main.py

App FastAPI principal. Define los routers, carga el modelo al iniciar
y expone Swagger en /docs.
"""

import logging
import os
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scalar_fastapi import get_scalar_api_reference
from api.routers import predict, insights, chat
from api.services.model_service import model_service
from api.services.insight_service import insight_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail Insights AI",
    description="API para predecir ventas y consultar insights del Superstore dataset.",
    version="1.0.0",
)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
).split(",")

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


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor", "path": str(request.url.path)},
    )


@app.on_event("startup")
def startup():
    """Carga el modelo y cachea insights al arrancar."""
    logger.info("Iniciando API...")
    model_service.load()
    insight_service.load()
    logger.info("API lista")


@app.on_event("shutdown")
def shutdown():
    """Limpieza al apagar."""
    logger.info("API apagándose")


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
