from fastapi.testclient import TestClient

from delivery_eta.api import create_app
from delivery_eta.model import EtaPredictor


class OlistStub:
    model_version = "olist-test-v1"

    def predict(self, rows):
        return [12.345] * len(rows)


class BaseStub:
    def predict(self, rows):
        return [30.0] * len(rows)


def payload(order_id="olist-1"):
    return {
        "order_id": order_id,
        "purchase_hour": 10,
        "purchase_dow": 2,
        "purchase_month": 5,
        "is_weekend": 0,
        "promise_days": 20.0,
        "item_count": 1,
        "product_count": 1,
        "seller_count": 1,
        "price_total": 100.0,
        "freight_total": 15.0,
        "payment_value_total": 115.0,
        "payment_installments_max": 1,
        "product_weight_g_mean": 500.0,
        "product_volume_cm3_mean": 1200.0,
        "customer_state": "SP",
        "seller_state": "RJ",
        "payment_type_mode": "credit_card",
        "product_category_mode": "health_beauty",
    }


def test_olist_readiness_and_prediction_contract():
    application = create_app(EtaPredictor(BaseStub(), "base-test-v1"), OlistStub())
    with TestClient(application) as client:
        assert client.get("/health/olist-ready").json() == {
            "status": "ready",
            "model_version": "olist-test-v1",
        }
        response = client.post("/v1/olist/predict", json=payload())
        assert response.status_code == 200
        assert response.json()["predicted_delivery_days"] == 12.345
        assert response.json()["feature_contract_version"] == "olist-features-v1"
        invalid = {**payload(), "unexpected_outcome": 999}
        assert client.post("/v1/olist/predict", json=invalid).status_code == 422


def test_olist_batch_and_missing_model_behavior():
    with TestClient(
        create_app(EtaPredictor(BaseStub(), "base-test-v1"), OlistStub())
    ) as client:
        response = client.post(
            "/v1/olist/predict/batch",
            json={"orders": [payload(), payload("olist-2")]},
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2

    with TestClient(create_app(EtaPredictor(BaseStub(), "base-test-v1"))) as client:
        assert client.get("/health/olist-ready").status_code == 503
        assert client.post("/v1/olist/predict", json=payload()).status_code == 503
