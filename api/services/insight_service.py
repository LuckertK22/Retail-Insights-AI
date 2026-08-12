"""
api/services/insight_service.py

Servicio que cachea los insights del dataset al startup.
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "superstore_clean.csv"


class InsightService:
    def __init__(self):
        self.insights = None

    def load(self):
        if self.insights is None:
            try:
                df = pd.read_csv(DATA_PATH)
                self.insights = self._compute(df)
                logger.info(f"Insights cacheados: {len(df)} filas")
            except Exception as e:
                logger.error(f"Error al cargar insights: {e}")
                raise

    def get(self):
        if self.insights is None:
            raise RuntimeError("Insights no cargados")
        return self.insights

    def is_loaded(self):
        return self.insights is not None

    def _compute(self, df):
        total_sales = float(df["Sales"].sum())
        total_profit = float(df["Profit"].sum())

        top_categories = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).head(5)
        worst_profit_sub = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=True).head(5)
        sales_by_region = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

        return {
            "total_sales": round(total_sales, 2),
            "total_profit": round(total_profit, 2),
            "profit_margin": round(total_profit / total_sales, 4) if total_sales > 0 else 0,
            "avg_order_value": round(float(df["Sales"].mean()), 2),
            "top_categories": [{"name": name, "value": round(val, 2)} for name, val in top_categories.items()],
            "worst_profit_subcategories": [{"name": name, "value": round(val, 2)} for name, val in worst_profit_sub.items()],
            "sales_by_region": [{"name": name, "value": round(val, 2)} for name, val in sales_by_region.items()],
        }


insight_service = InsightService()
