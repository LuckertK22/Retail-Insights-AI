"""
api/services/model_service.py

Servicio que gestiona el modelo ML.
Carga el modelo una sola vez y lo provee a los routers.
"""

import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "model.pkl"


class ModelService:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = joblib.load(MODEL_PATH)
            print(f"Modelo cargado: {MODEL_PATH}")

    def predict(self, features):
        return self.model.predict(features)

    def is_loaded(self):
        return self.model is not None


model_service = ModelService()
