"""
api/models.py

Esquemas Pydantic que validan los datos de entrada y salida de la API.
"""

from typing import Annotated
from pydantic import BaseModel, Field


class PredictInput(BaseModel):
    """Datos de entrada para predecir ventas de un pedido."""

    Category: Annotated[
        str,
        Field(description="Categoria del producto", example="Furniture")
    ] = Field(pattern="^(Furniture|Office Supplies|Technology)$")
    Region: Annotated[
        str,
        Field(description="Region", example="West")
    ] = Field(pattern="^(Central|East|South|West)$")
    Segment: Annotated[
        str,
        Field(description="Segmento del cliente", example="Consumer")
    ] = Field(pattern="^(Consumer|Corporate|Home Office)$")
    Ship_Mode: Annotated[
        str,
        Field(description="Modo de envio", example="Standard Class")
    ] = Field(pattern="^(First Class|Same Day|Second Class|Standard Class)$")
    Discount: float = Field(..., ge=0, le=1, example=0.2)
    Quantity: int = Field(..., ge=1, example=3)
    Year: int = Field(..., ge=2014, le=2017, example=2017)
    Month: int = Field(..., ge=1, le=12, example=11)
    DayOfWeek: int = Field(default=0, ge=0, le=6, example=0, description="Dia de la semana: 0=Lunes, 6=Domingo")


class PredictOutput(BaseModel):
    """Respuesta del endpoint /predict."""

    predicted_sales: float
    model: str


class InsightItem(BaseModel):
    """Un item individual dentro de un insight."""

    name: str
    value: float


class InsightsOutput(BaseModel):
    """Respuesta del endpoint /insights — estadisticas del dataset."""

    total_sales: float
    total_profit: float
    profit_margin: float
    avg_order_value: float
    top_categories: list[InsightItem]
    worst_profit_subcategories: list[InsightItem]
    sales_by_region: list[InsightItem]
