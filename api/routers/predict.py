"""
api/routers/predict.py

Endpoint POST /predict — recibe datos de un pedido y devuelve predicción de ventas.
"""

from fastapi import APIRouter
import numpy as np
import joblib
from pathlib import Path
from api.models import PredictInput, PredictOutput

router = APIRouter()

MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "model.pkl"
FEATURES = [
    "Quantity", "Discount", "Year", "Month", "DayOfWeek", "IsDiscounted",
    "Segment_Consumer", "Segment_Corporate", "Segment_Home Office",
    "Category_Furniture", "Category_Office Supplies", "Category_Technology",
    "Region_Central", "Region_East", "Region_South", "Region_West",
    "Ship Mode_First Class", "Ship Mode_Same Day", "Ship Mode_Second Class", "Ship Mode_Standard Class",
]


def build_feature_vector(input_data: PredictInput) -> np.ndarray:
    """Construye el vector de features en el mismo orden que el modelo fue entrenado."""

    segment_map = {"Consumer": 1, "Corporate": 0, "Home Office": 0}
    segment_hh = {"Consumer": 0, "Corporate": 1, "Home Office": 0}
    segment_ho = {"Consumer": 0, "Corporate": 0, "Home Office": 1}

    cat_fur = 1 if input_data.Category == "Furniture" else 0
    cat_off = 1 if input_data.Category == "Office Supplies" else 0
    cat_tech = 1 if input_data.Category == "Technology" else 0

    region_central = 1 if input_data.Region == "Central" else 0
    region_east = 1 if input_data.Region == "East" else 0
    region_south = 1 if input_data.Region == "South" else 0
    region_west = 1 if input_data.Region == "West" else 0

    ship_first = 1 if input_data.Ship_Mode == "First Class" else 0
    ship_same = 1 if input_data.Ship_Mode == "Same Day" else 0
    ship_second = 1 if input_data.Ship_Mode == "Second Class" else 0
    ship_standard = 1 if input_data.Ship_Mode == "Standard Class" else 0

    day_of_week = 0
    is_discounted = 1 if input_data.Discount > 0 else 0

    return np.array([[
        input_data.Quantity,
        input_data.Discount,
        input_data.Year,
        input_data.Month,
        day_of_week,
        is_discounted,
        segment_map.get(input_data.Segment, 0),
        segment_hh.get(input_data.Segment, 0),
        segment_ho.get(input_data.Segment, 0),
        cat_fur,
        cat_off,
        cat_tech,
        region_central,
        region_east,
        region_south,
        region_west,
        ship_first,
        ship_same,
        ship_second,
        ship_standard,
    ]])


@router.post("/", response_model=PredictOutput)
def predict_sales(input_data: PredictInput):
    """
    Predice las ventas de un pedido.
    """

    model = joblib.load(MODEL_PATH)
    X = build_feature_vector(input_data)
    y_log = model.predict(X)
    predicted_sales = float(np.expm1(y_log)[0])

    return PredictOutput(
        predicted_sales=round(predicted_sales, 2),
        model=type(model).__name__,
    )
