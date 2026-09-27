"""Run offline batch inference from an orders CSV."""

import argparse
from pathlib import Path

import pandas as pd

from delivery_eta.model import FEATURE_COLUMNS, EtaPredictor

REQUIRED = ["order_id", *FEATURE_COLUMNS]


def score_file(input_path: Path, output_path: Path, model_path: Path) -> int:
    frame = pd.read_csv(input_path)
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    predictor = EtaPredictor.load(model_path)
    features = frame[FEATURE_COLUMNS].to_dict(orient="records")
    frame["predicted_eta_minutes"] = predictor.predict(features)
    frame["model_version"] = predictor.model_version
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/eta_model.joblib")
    )
    args = parser.parse_args()
    rows = score_file(args.input_csv, args.output_csv, args.model)
    print(f"Scored {rows} rows with {args.model}; wrote {args.output_csv}")


if __name__ == "__main__":
    main()
