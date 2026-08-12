"""
Fase 2 — Entrenamiento del modelo predictivo de ventas.

Uso:
    python train_model.py

Dependencias: scikit-learn, joblib, pandas, numpy
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "superstore_clean.csv"
MODEL_PATH = SCRIPT_DIR / "model.pkl"

FEATURES = [
    "Quantity", "Discount", "Year", "Month", "DayOfWeek", "IsDiscounted",
    "Segment_Consumer", "Segment_Corporate", "Segment_Home Office",
    "Category_Furniture", "Category_Office Supplies", "Category_Technology",
    "Region_Central", "Region_East", "Region_South", "Region_West",
    "Ship Mode_First Class", "Ship Mode_Same Day", "Ship Mode_Second Class", "Ship Mode_Standard Class",
]


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_features(df):
    X = df[FEATURES]
    y_log = np.log1p(df["Sales"])
    return X, y_log


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = np.expm1(lr.predict(X_test))
    y_test_orig = np.expm1(y_test)
    results["LinearRegression"] = {
        "model": lr,
        "mae": mean_absolute_error(y_test_orig, y_pred_lr),
        "rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred_lr)),
        "r2": r2_score(y_test_orig, y_pred_lr),
    }

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = np.expm1(rf.predict(X_test))
    results["RandomForest"] = {
        "model": rf,
        "mae": mean_absolute_error(y_test_orig, y_pred_rf),
        "rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred_rf)),
        "r2": r2_score(y_test_orig, y_pred_rf),
    }

    return results


def print_results(results):
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  MAE  : ${r['mae']:.2f}")
        print(f"  RMSE : ${r['rmse']:.2f}")
        print(f"  R²   : {r['r2']:.4f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nMejor modelo: {best_name} (R²={results[best_name]['r2']:.4f})")
    return best_name


def save_model(model, path):
    joblib.dump(model, path)
    print(f"\nModelo guardado: {path}")


def main():
    print("Cargando datos...")
    df = load_data(DATA_PATH)
    print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

    print("Preparando features...")
    X, y_log = prepare_features(df)

    print("Dividiendo train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("Entrenando modelos...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name = print_results(results)
    best_model = results[best_name]["model"]

    print("\n" + "=" * 60)
    save_model(best_model, MODEL_PATH)
    print("\nListo. Para usar el modelo:")
    print(f"  model = joblib.load('{MODEL_PATH}')")
    print(f"  prediction = model.predict(X_new)  # X_new debe tener las {len(FEATURES)} features en el mismo orden")


if __name__ == "__main__":
    main()
