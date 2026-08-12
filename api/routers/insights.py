"""
api/routers/insights.py

Endpoint GET /insights — devuelve estadísticas precalculadas del dataset.
"""

from fastapi import APIRouter
from api.models import InsightsOutput
from api.services.insight_service import insight_service

router = APIRouter()


@router.get("/", response_model=InsightsOutput)
def get_insights():
    """
    Devuelve estadísticas del dataset.
    """
    return insight_service.get()
