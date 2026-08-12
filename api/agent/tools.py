"""
api/agent/tools.py

Tools (funciones) que el agente LangGraph puede invocar.
"""

import os
import httpx
from langchain_core.tools import tool

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")


@tool
def consultar_insights() -> str:
    """
    Consulta estadísticas e insights del dataset de ventas retail.
    Úsala cuando el usuario pregunte sobre:
    - ventas totales, profit, margen
    - top categorías o sub-categorías
    - ventas por región o segmento
    - qué productos/categorías venden más o menos
    - rentabilidad

    Returns:
        Un resumen con los insights del dataset en texto legible.
    """
    try:
        resp = httpx.get(f"{API_BASE}/insights/", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        lines = [
            f"Ventas totales: ${data['total_sales']:,.2f}",
            f"Profit total: ${data['total_profit']:,.2f}",
            f"Margen de ganancia: {data['profit_margin']*100:.2f}%",
            f"Ticket promedio: ${data['avg_order_value']:,.2f}",
            "",
            "Top categorías por ventas:",
        ]
        for c in data["top_categories"]:
            lines.append(f"  - {c['name']}: ${c['value']:,.2f}")

        lines.append("")
        lines.append("Peores sub-categorías por profit (pierden dinero):")
        for c in data["worst_profit_subcategories"]:
            lines.append(f"  - {c['name']}: ${c['value']:,.2f}")

        lines.append("")
        lines.append("Ventas por región:")
        for c in data["sales_by_region"]:
            lines.append(f"  - {c['name']}: ${c['value']:,.2f}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error al consultar insights: {e}"


@tool
def predecir_ventas(
    category: str,
    region: str,
    segment: str,
    ship_mode: str,
    discount: float,
    quantity: int,
    year: int,
    month: int,
) -> str:
    """
    Predice las ventas de un pedido dados sus características.

    Args:
        category: Categoría del producto (Furniture, Office Supplies, Technology)
        region: Región (Central, East, South, West)
        segment: Segmento del cliente (Consumer, Corporate, Home Office)
        ship_mode: Modo de envío (First Class, Same Day, Second Class, Standard Class)
        discount: Descuento aplicado (0.0 a 1.0, ejemplo: 0.2 = 20%)
        quantity: Cantidad de unidades
        year: Año del pedido (2014-2017)
        month: Mes del pedido (1-12)

    Returns:
        La predicción de ventas en dólares.
    """
    try:
        payload = {
            "Category": category,
            "Region": region,
            "Segment": segment,
            "Ship_Mode": ship_mode,
            "Discount": discount,
            "Quantity": quantity,
            "Year": year,
            "Month": month,
        }
        resp = httpx.post(f"{API_BASE}/predict/", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return f"Ventas predichas: ${data['predicted_sales']:,.2f} (modelo: {data['model']})"
    except Exception as e:
        return f"Error al predecir ventas: {e}"
