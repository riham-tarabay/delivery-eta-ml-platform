"""Redis Streams consumer: event -> model inference -> PostgreSQL -> result stream."""

import json
import logging
import os
import time

import redis

from delivery_eta.database import initialize, save_prediction
from delivery_eta.model import EtaPredictor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("eta-worker")
STREAM = "delivery:events"
GROUP = "eta-workers"


def run() -> None:
    client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
    )
    model = EtaPredictor.load(os.getenv("MODEL_PATH", "artifacts/eta_model.joblib"))
    initialize()
    try:
        client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    consumer = os.getenv("CONSUMER_NAME", "worker-1")
    log.info("worker started model_version=%s", model.model_version)
    while True:
        batches = client.xreadgroup(
            GROUP, consumer, {STREAM: ">"}, count=25, block=5000
        )
        for _, messages in batches:
            for message_id, fields in messages:
                try:
                    event = json.loads(fields["payload"])
                    features = event["features"]
                    eta = model.predict([features])[0]
                    save_prediction(
                        event["event_id"], event["order_id"], eta, model.model_version
                    )
                    client.xadd(
                        "delivery:predictions",
                        {
                            "payload": json.dumps(
                                {
                                    "event_id": event["event_id"],
                                    "order_id": event["order_id"],
                                    "predicted_eta_minutes": eta,
                                    "model_version": model.model_version,
                                }
                            )
                        },
                        maxlen=100000,
                        approximate=True,
                    )
                    client.xack(STREAM, GROUP, message_id)
                except Exception as exc:
                    log.exception("event processing failed id=%s", message_id)
                    client.xadd(
                        "delivery:dead-letter",
                        {"message_id": message_id, "error": str(exc)[:500], **fields},
                    )
                    client.xack(STREAM, GROUP, message_id)
        time.sleep(0.05)


if __name__ == "__main__":
    run()
