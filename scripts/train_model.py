"""Train a reproducible baseline model on explicitly synthetic delivery data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
NUMERIC = [
    "distance_km",
    "order_hour",
    "traffic_index",
    "pickup_wait_min",
    "rider_experience_months",
    "is_weekend",
]
CATEGORICAL = ["weather"]
FEATURES = NUMERIC + CATEGORICAL


def make_synthetic_data(
    rows: int = 6000, seed: int = 17
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    weather = rng.choice(
        ["clear", "rain", "wind", "heat"], rows, p=[0.62, 0.18, 0.12, 0.08]
    )
    frame = pd.DataFrame(
        {
            "distance_km": rng.uniform(0.5, 25, rows).round(2),
            "order_hour": rng.integers(0, 24, rows),
            "traffic_index": rng.uniform(0, 10, rows).round(2),
            "weather": weather,
            "pickup_wait_min": rng.uniform(0, 18, rows).round(2),
            "rider_experience_months": rng.integers(0, 121, rows),
            "is_weekend": rng.integers(0, 2, rows),
        }
    )
    rush = frame.order_hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
    weather_cost = frame.weather.map(
        {"clear": 0, "rain": 7, "wind": 3, "heat": 2}
    ).to_numpy()
    eta = (
        5
        + frame.distance_km * 3.1
        + frame.traffic_index * 1.15
        + weather_cost
        + frame.pickup_wait_min * 0.9
        + rush * 5
        + frame.is_weekend * 1.5
        - frame.rider_experience_months * 0.025
        + rng.normal(0, 2.8, rows)
    )
    return frame[FEATURES], np.maximum(eta.to_numpy(), 5)


def train(output_dir: Path = ARTIFACT_DIR) -> dict[str, float | int | str]:
    X, y = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=17
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100, min_samples_leaf=2, random_state=17, n_jobs=-1
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "model_version": "eta-rf-v1",
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "mae_minutes": round(float(mean_absolute_error(y_test, predictions)), 3),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "data_note": "Synthetic data; metrics are not evidence of real-world delivery accuracy.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "model_version": metrics["model_version"]},
        output_dir / "eta_model.joblib",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
