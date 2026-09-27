import json
import os
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated, Any

import redis
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from delivery_eta.model import EtaPredictor
from delivery_eta.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DeliveryEvent,
    PredictionRequest,
    PredictionResult,
)

PREDICTIONS = Counter(
    "eta_predictions_total", "Number of successful ETA predictions", ["mode"]
)
LATENCY = Histogram("eta_prediction_seconds", "Prediction latency in seconds", ["mode"])


def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
    )


def create_app(predictor: EtaPredictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            path = os.getenv("MODEL_PATH", "artifacts/eta_model.joblib")
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Model artifact not found at {path}; run scripts/train_model.py"
                )
            app.state.predictor = EtaPredictor.load(path)
        yield

    app = FastAPI(
        title="Delivery ETA ML Platform",
        version="1.0.0",
        description="Portfolio demonstration of online inference, batch scoring, and event-driven prediction.",
        lifespan=lifespan,
    )
    app.state.predictor = predictor

    @app.get("/health/live", tags=["operations"])
    def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["operations"])
    def readiness(request: Request) -> dict[str, str]:
        if request.app.state.predictor is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {
            "status": "ready",
            "model_version": request.app.state.predictor.model_version,
        }

    @app.get("/metrics", tags=["operations"])
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/predict", response_model=PredictionResult, tags=["inference"])
    def predict(body: PredictionRequest, request: Request) -> PredictionResult:
        started = perf_counter()
        model: EtaPredictor = request.app.state.predictor
        try:
            eta = model.predict([body.model_dump(exclude={"order_id"})])[0]
        except Exception as exc:
            raise HTTPException(status_code=500, detail="prediction failed") from exc
        PREDICTIONS.labels(mode="online").inc()
        LATENCY.labels(mode="online").observe(perf_counter() - started)
        return PredictionResult(
            order_id=body.order_id,
            predicted_eta_minutes=eta,
            model_version=model.model_version,
        )

    @app.post(
        "/v1/predict/batch", response_model=BatchPredictionResponse, tags=["inference"]
    )
    def predict_batch(
        body: BatchPredictionRequest, request: Request
    ) -> BatchPredictionResponse:
        started = perf_counter()
        model: EtaPredictor = request.app.state.predictor
        rows = [item.model_dump(exclude={"order_id"}) for item in body.orders]
        values = model.predict(rows)
        results = [
            PredictionResult(
                order_id=item.order_id,
                predicted_eta_minutes=value,
                model_version=model.model_version,
            )
            for item, value in zip(body.orders, values, strict=True)
        ]
        PREDICTIONS.labels(mode="batch").inc(len(results))
        LATENCY.labels(mode="batch").observe(perf_counter() - started)
        return BatchPredictionResponse(
            predictions=results, model_version=model.model_version
        )

    @app.post("/v1/events", status_code=202, tags=["streaming"])
    def ingest_event(
        body: DeliveryEvent, client: Annotated[redis.Redis, Depends(get_redis_client)]
    ) -> dict[str, Any]:
        try:
            client.xadd(
                "delivery:events",
                {"payload": json.dumps(body.model_dump(mode="json"))},
                maxlen=100000,
                approximate=True,
            )
        except redis.RedisError as exc:
            raise HTTPException(
                status_code=503, detail="event queue unavailable"
            ) from exc
        return {
            "accepted": True,
            "event_id": str(body.event_id),
            "stream": "delivery:events",
        }

    return app


app = create_app()
