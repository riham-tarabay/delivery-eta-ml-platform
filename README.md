# Delivery ETA ML Platform

A portfolio project demonstrating a small end-to-end ML service for delivery ETA prediction: reproducible model training, online REST inference, batch scoring, event-driven inference with Redis Streams, persistence in PostgreSQL, Docker Compose, Kubernetes deployment manifests, health checks, metrics, and tests.

> **Data and claims:** the model is trained on deterministic synthetic data created by this repository. It is not trained on Snoonu, Qatar, customer, or real delivery data. Reported metrics are only a reproducibility check and must not be presented as real-world accuracy or production impact. This is a portfolio system, not a production deployment.

## Architecture

```text
Client --POST /v1/predict--> FastAPI --> sklearn pipeline
Client --POST /v1/predict/batch--------------^ 
Client --POST /v1/events--> Redis Stream --> worker --> sklearn pipeline
                                             |           |--> PostgreSQL prediction record
                                             |           `--> delivery:predictions stream
                                             `--> delivery:dead-letter on processing failure
```

The API serves synchronous online and batch predictions. Event requests are acknowledged after being added to a Redis Stream; a consumer group worker scores them asynchronously, persists results, and emits a result event. A unique event ID and database constraint make result persistence idempotent for duplicate deliveries. Failed events are recorded in a dead-letter stream for inspection.

## Requirements

- Python 3.11+ for local development
- Docker and Docker Compose for the integrated stack
- `kubectl` and a Kubernetes cluster only for the optional deployment exercise

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements-dev.txt
python scripts/train_model.py
pytest -q
```

The training command creates an ignored model artifact and metrics file under `artifacts/`. To run the API without Docker after training:

```bash
PYTHONPATH=src uvicorn delivery_eta.api:app --reload
```

Interactive API documentation: `http://localhost:8000/docs`.

## Run the integrated services

```bash
cp .env.example .env
docker compose up --build
```

The API is available at `http://localhost:8000`; the API image trains its reproducible sample model during build. Set non-default credentials in a local `.env` before using anything beyond local development. Do not use the example password in shared environments.

## Example requests

Single prediction:

```bash
curl -X POST http://localhost:8000/v1/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"order_id":"demo-001","distance_km":4.2,"order_hour":18,"traffic_index":6.1,"weather":"rain","pickup_wait_min":3,"rider_experience_months":12,"is_weekend":false}'
```

Batch prediction uses `POST /v1/predict/batch` with `{"orders":[...]}`. Event ingestion uses `POST /v1/events` with a UUID `event_id`, `order_id`, and nested `features`; successful requests return HTTP 202. The worker writes predictions to the `predictions` table and `delivery:predictions` Redis stream. See OpenAPI docs for exact schemas.

## Operational endpoints

- `GET /health/live` — process liveness
- `GET /health/ready` — model readiness and model version
- `GET /metrics` — Prometheus-format request count and inference latency

## Offline batch scoring

Prepare a CSV with `order_id` plus all model feature columns, then run:

```bash
PYTHONPATH=src python scripts/score_batch.py orders.csv scored.csv
```

The output preserves input columns and adds `predicted_eta_minutes` and `model_version`.

## Train and inspect the model

```bash
python scripts/train_model.py
cat artifacts/metrics.json
```

The training script uses a fixed seed, train/test split, preprocessing pipeline, and regression metrics. Since the generated data follows a synthetic formula, those scores only check the implementation; they do not validate real operational performance.

## Kubernetes exercise

1. Build and publish the Docker image under your GitHub Container Registry namespace.
2. Create namespace `eta-platform` and a Kubernetes Secret named `delivery-eta-secrets` containing `DATABASE_URL` and `REDIS_URL` for approved external services.
3. Replace `ghcr.io/OWNER/...` in `infra/k8s/*.yaml` with the exact immutable image tag.
4. Apply the manifests and check rollouts, probes, logs, resource use, and API responses.

```bash
kubectl create namespace eta-platform
kubectl -n eta-platform create secret generic delivery-eta-secrets \\
  --from-literal=DATABASE_URL='...' --from-literal=REDIS_URL='...'
kubectl apply -f infra/k8s/api.yaml
kubectl apply -f infra/k8s/worker.yaml
kubectl -n eta-platform rollout status deployment/eta-api
kubectl -n eta-platform rollout status deployment/eta-worker
```

Do not commit the secret or paste credentials into GitHub. The manifests are a learning/portfolio example and require review, network policies, external secret management, autoscaling decisions, backups, and production observability before real use.

## Testing

`pytest -q` covers deterministic data generation, model save/load and inference, offline CSV scoring, request validation, online/request-batch response contracts, event enqueueing, readiness, and metrics exposure. Redis/PostgreSQL integration and Kubernetes deployment checks should be run against the Docker Compose stack/cluster before describing those components as tested.

## Limitations and next steps

- Synthetic training data; no real delivery dataset or verified business performance.
- No authentication on the sample API; do not expose it publicly without adding access control.
- The demo does not implement a full feature store, model registry, drift response policy, or autoscaling strategy.
- Add integration tests for Redis Streams/PostgreSQL, load tests, and a documented model rollback procedure before treating it as operationally hardened.
