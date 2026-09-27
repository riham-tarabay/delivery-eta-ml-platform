"""Train and evaluate purchase-time ETA models on local Olist data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from delivery_eta.olist import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    TIME_COLUMN,
    load_olist_orders,
    split_chronologically,
    validate_feature_contract,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data/raw/olist"
DEFAULT_ARTIFACT = ROOT / "artifacts/olist"
DEFAULT_REPORT = ROOT / "reports/olist"
MODEL_VERSION = "olist-eta-v1"


def _pipeline(model: object) -> Pipeline:
    validate_feature_contract(FEATURE_COLUMNS)
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _metrics(
    y_true: pd.Series, prediction: np.ndarray, coverage: float = 1.0
) -> dict[str, float]:
    errors = np.abs(y_true.to_numpy() - prediction)
    return {
        "mae_days": round(float(mean_absolute_error(y_true, prediction)), 4),
        "rmse_days": round(float(np.sqrt(mean_squared_error(y_true, prediction))), 4),
        "p90_absolute_error_days": round(float(np.quantile(errors, 0.90)), 4),
        "coverage": round(float(coverage), 4),
    }


def train(
    raw_dir: Path = DEFAULT_RAW,
    artifact_dir: Path = DEFAULT_ARTIFACT,
    report_dir: Path = DEFAULT_REPORT,
) -> dict:
    frame = load_olist_orders(raw_dir, include_target=True)
    train_frame, validation_frame, test_frame = split_chronologically(frame)
    X_train, y_train = train_frame[FEATURE_COLUMNS], train_frame[TARGET]
    X_validation, y_validation = (
        validation_frame[FEATURE_COLUMNS],
        validation_frame[TARGET],
    )
    X_test, y_test = test_frame[FEATURE_COLUMNS], test_frame[TARGET]

    models = {
        "ridge": _pipeline(Ridge(alpha=10.0)),
        "random_forest": _pipeline(
            RandomForestRegressor(
                n_estimators=180,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=17,
                n_jobs=-1,
            )
        ),
    }
    validation_metrics: dict[str, dict[str, float]] = {
        "median_baseline": _metrics(
            y_validation, np.full(len(y_validation), y_train.median())
        ),
        "promise_baseline": _metrics(
            y_validation, validation_frame["promise_days"].to_numpy()
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        validation_metrics[name] = _metrics(y_validation, model.predict(X_validation))

    selected_name = min(
        ("ridge", "random_forest"),
        key=lambda name: validation_metrics[name]["mae_days"],
    )
    selected_model = models[selected_name]
    selected_model.fit(
        pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation])
    )
    test_prediction = selected_model.predict(X_test)
    test_metrics = {
        "selected_model": selected_name,
        "median_baseline": _metrics(y_test, np.full(len(y_test), y_train.median())),
        "promise_baseline": _metrics(y_test, test_frame["promise_days"].to_numpy()),
        selected_name: _metrics(y_test, test_prediction),
    }

    scored = test_frame[
        [
            "order_id",
            TIME_COLUMN,
            "customer_state",
            "seller_state",
            "promise_days",
            TARGET,
        ]
    ].copy()
    scored["prediction_days"] = test_prediction
    scored["absolute_error_days"] = (scored[TARGET] - scored["prediction_days"]).abs()
    scored["signed_error_days"] = scored["prediction_days"] - scored[TARGET]
    error_summary = (
        scored.groupby("customer_state", dropna=False)
        .agg(
            orders=("order_id", "count"),
            mae_days=("absolute_error_days", "mean"),
            mean_signed_error_days=("signed_error_days", "mean"),
        )
        .sort_values(["orders", "mae_days"], ascending=[False, False])
        .head(20)
        .reset_index()
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eda = {
        "rows_after_eligibility": len(frame),
        "purchase_start": str(frame[TIME_COLUMN].min()),
        "purchase_end": str(frame[TIME_COLUMN].max()),
        "target_mean_days": round(float(frame[TARGET].mean()), 4),
        "target_median_days": round(float(frame[TARGET].median()), 4),
        "target_p90_days": round(float(frame[TARGET].quantile(0.9)), 4),
        "missing_features": frame[FEATURE_COLUMNS].isna().sum().to_dict(),
        "split_rows": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "unique_orders": int(frame.order_id.nunique()),
        "feature_columns": FEATURE_COLUMNS,
    }
    metrics = {
        "model_version": MODEL_VERSION,
        "dataset": "Brazilian E-Commerce Public Dataset by Olist",
        "license": "CC BY-NC-SA 4.0; noncommercial use only unless separately authorized",
        "target": "delivery_days = order_delivered_customer_date - order_purchase_timestamp",
        "validation": validation_metrics,
        "test": test_metrics,
        "eda": eda,
        "data_note": "Local-only Olist data; raw data and trained artifacts are intentionally gitignored.",
    }
    joblib.dump(
        {
            "pipeline": selected_model,
            "model_version": MODEL_VERSION,
            "feature_columns": FEATURE_COLUMNS,
        },
        artifact_dir / "eta_olist_model.joblib",
    )
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (report_dir / "eda_summary.json").write_text(json.dumps(eda, indent=2) + "\n")
    error_summary.to_csv(report_dir / "error_by_customer_state.csv", index=False)
    scored.sort_values("absolute_error_days", ascending=False).head(100).to_csv(
        report_dir / "largest_test_errors.csv", index=False
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(json.dumps(train(args.raw_dir, args.artifact_dir, args.report_dir), indent=2))


if __name__ == "__main__":
    main()
