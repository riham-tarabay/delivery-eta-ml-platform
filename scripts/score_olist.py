"""Score local Olist orders with a trained purchase-time model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from delivery_eta.olist import (
    FEATURE_COLUMNS,
    load_olist_orders,
    validate_feature_contract,
)


def score(raw_dir: Path, artifact: Path, output: Path) -> int:
    validate_feature_contract(FEATURE_COLUMNS)
    frame = load_olist_orders(raw_dir, include_target=False)
    model_artifact = joblib.load(artifact)
    if model_artifact.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError(
            "Artifact feature contract does not match the current Olist features"
        )
    prediction = model_artifact["pipeline"].predict(frame[FEATURE_COLUMNS])
    result = frame[["order_id", "order_purchase_timestamp"]].copy()
    result["predicted_delivery_days"] = (
        pd.Series(prediction).clip(lower=0).round(3).to_numpy()
    )
    result["model_version"] = model_artifact["model_version"]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/olist"))
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/olist/eta_olist_model.joblib")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/olist/predictions.csv")
    )
    args = parser.parse_args()
    print(f"scored_rows={score(args.raw_dir, args.artifact, args.output)}")


if __name__ == "__main__":
    main()
