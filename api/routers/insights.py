"""
api/routers/insights.py

Endpoint GET /insights — devuelve estadísticas precalculadas del dataset.
"""

from fastapi import APIRouter
import pandas as pd
from pathlib import Path
from api.models import InsightsOutput, InsightItem

router = APIRouter()

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "superstore_clean.csv"


def compute_insights() -> InsightsOutput:
    """Calcula todos los insights a partir del CSV limpio."""

    df = pd.read_csv(DATA_PATH)

    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    profit_margin = total_profit / total_sales if total_sales > 0 else 0
    avg_order_value = float(df["Sales"].mean())

    top_categories = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    top_categories = [
        InsightItem(name=name, value=round(val, 2))
        for name, val in top_categories.items()
    ]

    worst_profit_sub = (
        df.groupby("Sub-Category")["Profit"]
        .sum()
        .sort_values(ascending=True)
        .head(5)
    )
    worst_profit_sub = [
        InsightItem(name=name, value=round(val, 2))
        for name, val in worst_profit_sub.items()
    ]

    sales_by_region = (
        df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )
    sales_by_region = [
        InsightItem(name=name, value=round(val, 2))
        for name, val in sales_by_region.items()
    ]

    return InsightsOutput(
        total_sales=round(total_sales, 2),
        total_profit=round(total_profit, 2),
        profit_margin=round(profit_margin, 4),
        avg_order_value=round(avg_order_value, 2),
        top_categories=top_categories,
        worst_profit_subcategories=worst_profit_sub,
        sales_by_region=sales_by_region,
    )


@router.get("/", response_model=InsightsOutput)
def get_insights():
    """
    Devuelve estadísticas del dataset.
    """
    return compute_insights()
