# Interview walkthrough

## 60-second explanation

I built a small delivery ETA ML platform to practice the systems around a model, not just model training. It trains a reproducible scikit-learn baseline on explicitly synthetic data, exposes online and request-batch REST inference through FastAPI, and has a separate CSV-based offline scoring command. For asynchronous event processing, the API writes events to Redis Streams; a worker scores them, stores results in PostgreSQL, emits a result stream, and sends failures to a dead-letter stream. Docker Compose runs the local stack, and Kubernetes manifests show how I would deploy the stateless API and worker with probes and resource requests/limits. The sample data is synthetic, so I make no claim about real delivery accuracy or production scale.

## Be ready to explain

- **Why synthetic data?** It keeps the project runnable without private data. It is useful to test plumbing, not to make business or model-quality claims.
- **Why both online and batch?** Interactive callers need a quick response; offline scoring handles files independently of API request latency.
- **Why Redis Streams?** It gives a simple durable queue with consumer groups for the asynchronous demo. A larger system might use Kafka or a managed event service depending on throughput, replay, and platform standards.
- **What is persisted?** The worker stores event ID, order ID, prediction, model version, and timestamp in PostgreSQL. The event ID primary key prevents duplicate database rows.
- **What is monitored?** The API exposes prediction counters and inference latency in Prometheus format, plus liveness/readiness endpoints.
- **What would change before production?** Real representative data, model governance/registry, authentication, data contracts, integration and load tests, privacy controls, retries/pending-message recovery, alerting, drift monitoring, rollout/rollback, and production-grade secrets/network policies.

## Honest scope

This is an individually developed portfolio project, not evidence of leading a team, serving real customers, or operating a production ML system. Explain design decisions and limitations plainly; do not present synthetic-data scores as expected operational accuracy.
