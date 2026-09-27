from delivery_eta.model import EtaPredictor
from scripts.train_model import train


def test_training_artifact_loads_and_serves_prediction(tmp_path):
    metrics = train(tmp_path)
    predictor = EtaPredictor.load(tmp_path / "eta_model.joblib")
    sample = {
        "distance_km": 4.2,
        "order_hour": 18,
        "traffic_index": 6.1,
        "weather": "rain",
        "pickup_wait_min": 3.0,
        "rider_experience_months": 12,
        "is_weekend": False,
    }
    prediction = predictor.predict([sample])[0]
    assert predictor.model_version == "eta-rf-v1"
    assert 0 < prediction < 200
    assert metrics["test_rows"] == 1200
    assert metrics["mae_minutes"] < 5
    assert "Synthetic" in metrics["data_note"]


def test_offline_batch_scoring_writes_predictions(tmp_path):
    import pandas as pd

    from scripts.score_batch import score_file

    model_dir = tmp_path / "model"
    train(model_dir)
    source = tmp_path / "orders.csv"
    output = tmp_path / "scored" / "predictions.csv"
    pd.DataFrame(
        [
            {
                "order_id": "batch-1",
                "distance_km": 2.5,
                "order_hour": 10,
                "traffic_index": 3.0,
                "weather": "clear",
                "pickup_wait_min": 2.0,
                "rider_experience_months": 8,
                "is_weekend": 0,
            }
        ]
    ).to_csv(source, index=False)
    assert score_file(source, output, model_dir / "eta_model.joblib") == 1
    result = pd.read_csv(output)
    assert result.loc[0, "order_id"] == "batch-1"
    assert result.loc[0, "predicted_eta_minutes"] > 0
    assert result.loc[0, "model_version"] == "eta-rf-v1"
