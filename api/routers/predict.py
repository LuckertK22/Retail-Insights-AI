"""
api/routers/predict.py

Endpoint POST /predict — recibe datos de un pedido y devuelve predicción de ventas.
"""

from fastapi import APIRouter
import numpy as np
from api.models import PredictInput, PredictOutput
from api.services.model_service import model_service

router = APIRouter()

FEATURES = [
    "Quantity", "Discount", "Year", "Month", "DayOfWeek", "IsDiscounted",
    "Segment_Consumer", "Segment_Corporate", "Segment_Home Office",
    "Category_Furniture", "Category_Office Supplies", "Category_Technology",
    "Region_Central", "Region_East", "Region_South", "Region_West",
    "Ship Mode_First Class", "Ship Mode_Same Day", "Ship Mode_Second Class", "Ship Mode_Standard Class",
]


def build_feature_vector(input_data: PredictInput) -> np.ndarray:
    """Construye el vector de features en el mismo orden que el modelo fue entrenado."""

    return np.array([[
        input_data.Quantity,
        input_data.Discount,
        input_data.Year,
        input_data.Month,
        0,
        1 if input_data.Discount > 0 else 0,
        1 if input_data.Segment == "Consumer" else 0,
        1 if input_data.Segment == "Corporate" else 0,
        1 if input_data.Segment == "Home Office" else 0,
        1 if input_data.Category == "Furniture" else 0,
        1 if input_data.Category == "Office Supplies" else 0,
        1 if input_data.Category == "Technology" else 0,
        1 if input_data.Region == "Central" else 0,
        1 if input_data.Region == "East" else 0,
        1 if input_data.Region == "South" else 0,
        1 if input_data.Region == "West" else 0,
        1 if input_data.Ship_Mode == "First Class" else 0,
        1 if input_data.Ship_Mode == "Same Day" else 0,
        1 if input_data.Ship_Mode == "Second Class" else 0,
        1 if input_data.Ship_Mode == "Standard Class" else 0,
    ]])


@router.post("/", response_model=PredictOutput)
def predict_sales(input_data: PredictInput):
    """
    Predice las ventas de un pedido.
    """
    X = build_feature_vector(input_data)
    y_log = model_service.predict(X)
    predicted_sales = float(np.expm1(y_log)[0])

    return PredictOutput(
        predicted_sales=round(predicted_sales, 2),
        model=type(model_service.model).__name__,
    )
