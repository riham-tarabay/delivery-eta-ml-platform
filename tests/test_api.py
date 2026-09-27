from fastapi.testclient import TestClient

from delivery_eta.api import create_app, get_redis_client
from delivery_eta.model import EtaPredictor


class StubPipeline:
    def predict(self, frame):
        return [30.0] * len(frame)


def client():
    return TestClient(create_app(EtaPredictor(StubPipeline(), "test-v1")))


def test_liveness_and_readiness():
    with client() as c:
        assert c.get("/health/live").json() == {"status": "alive"}
        assert c.get("/health/ready").json()["model_version"] == "test-v1"


def test_single_prediction_and_validation():
    payload = {
        "order_id": "order-123",
        "distance_km": 4.2,
        "order_hour": 18,
        "traffic_index": 6.1,
        "weather": "rain",
        "pickup_wait_min": 3,
        "rider_experience_months": 12,
        "is_weekend": False,
    }
    with client() as c:
        response = c.post("/v1/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["predicted_eta_minutes"] == 30.0
        assert response.json()["model_version"] == "test-v1"
        payload["order_hour"] = 25
        assert c.post("/v1/predict", json=payload).status_code == 422


def test_batch_prediction_and_metrics():
    order = {
        "order_id": "order-1",
        "distance_km": 3.0,
        "order_hour": 12,
        "traffic_index": 2.0,
        "weather": "clear",
        "pickup_wait_min": 1,
        "rider_experience_months": 24,
        "is_weekend": True,
    }
    with client() as c:
        response = c.post(
            "/v1/predict/batch",
            json={"orders": [order, {**order, "order_id": "order-2"}]},
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2
        assert "eta_predictions_total" in c.get("/metrics").text


class FakeRedis:
    def __init__(self):
        self.entries = []

    def xadd(self, stream, fields, **kwargs):
        self.entries.append((stream, fields))
        return "1-0"


def test_event_ingestion_enqueues_validated_event():
    fake = FakeRedis()
    application = create_app(EtaPredictor(StubPipeline(), "test-v1"))
    application.dependency_overrides[get_redis_client] = lambda: fake
    payload = {
        "event_id": "5f28bdab-6011-4f58-88e1-47642fe612c5",
        "order_id": "order-event-1",
        "features": {
            "distance_km": 4.2,
            "order_hour": 18,
            "traffic_index": 6.1,
            "weather": "rain",
            "pickup_wait_min": 3,
            "rider_experience_months": 12,
            "is_weekend": False,
        },
    }
    with TestClient(application) as c:
        response = c.post("/v1/events", json=payload)
    assert response.status_code == 202
    assert response.json()["accepted"] is True
    assert fake.entries[0][0] == "delivery:events"
