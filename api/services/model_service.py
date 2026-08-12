"""
api/services/model_service.py

Servicio que gestiona el modelo ML.
Carga el modelo una sola vez y lo provee a los routers.
"""

import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "model.pkl"


class ModelService:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"Modelo cargado: {MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error al cargar modelo: {e}")
                raise

    def predict(self, features):
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
        return self.model.predict(features)

    def is_loaded(self):
        return self.model is not None


model_service = ModelService()
