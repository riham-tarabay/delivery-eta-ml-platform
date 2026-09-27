"""Serving adapter for the locally trained Olist ETA artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from delivery_eta.olist import FEATURE_COLUMNS, validate_feature_contract


class OlistEtaPredictor:
    """Load and serve an Olist artifact only when its feature contract matches."""

    def __init__(self, pipeline: Any, model_version: str, feature_columns: list[str]):
        validate_feature_contract(feature_columns)
        if feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                "Olist artifact feature contract does not match the serving contract"
            )
        self.pipeline = pipeline
        self.model_version = model_version
        self.feature_columns = feature_columns

    @classmethod
    def load(cls, path: str | Path) -> OlistEtaPredictor:
        artifact = joblib.load(path)
        required = {"pipeline", "model_version", "feature_columns"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Olist artifact is missing keys: {sorted(missing)}")
        return cls(
            artifact["pipeline"], artifact["model_version"], artifact["feature_columns"]
        )

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        frame = pd.DataFrame(rows, columns=self.feature_columns)
        if frame.empty:
            return []
        values = self.pipeline.predict(frame)
        return [round(max(0.0, float(value)), 3) for value in values]
