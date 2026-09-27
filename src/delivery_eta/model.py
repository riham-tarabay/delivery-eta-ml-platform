from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "distance_km",
    "order_hour",
    "traffic_index",
    "weather",
    "pickup_wait_min",
    "rider_experience_months",
    "is_weekend",
]
MODEL_VERSION = "eta-rf-v1"


class EtaPredictor:
    def __init__(self, pipeline: Any, model_version: str = MODEL_VERSION):
        self.pipeline = pipeline
        self.model_version = model_version

    @classmethod
    def load(cls, path: str | Path) -> "EtaPredictor":
        artifact = joblib.load(path)
        return cls(artifact["pipeline"], artifact.get("model_version", MODEL_VERSION))

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        values = self.pipeline.predict(frame)
        return [round(max(0.0, float(value)), 1) for value in values]
