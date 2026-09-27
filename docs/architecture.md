# Architecture decisions

## ADR-001: Separate request/response and event-driven inference

Online prediction is synchronous for low-latency request/response use. Event ingestion is asynchronous through Redis Streams so the API can accept work without waiting for scoring and persistence. The worker acknowledges only after persistence and result publication; failures are copied to a dead-letter stream for inspection.

## ADR-002: Reproducible synthetic baseline

A deterministic synthetic dataset makes the project runnable without private or licensed data. Its metrics are not a claim of real-world accuracy. Replace the generator with governed, representative data and re-evaluate before any operational use.

## ADR-003: Containerized dependencies, externalized production state

Docker Compose provides local Redis and PostgreSQL. Kubernetes manifests deploy stateless API and worker containers while expecting approved external Redis/PostgreSQL endpoints and Kubernetes Secrets. The included manifests are a starting point, not a production blueprint.

## ADR-004: Governed optional Olist serving path

The Olist model is served through a separate, opt-in FastAPI contract rather than silently replacing the original demo model. `OLIST_MODEL_PATH` enables the path, while the artifact loader verifies the exact feature list and model metadata before serving. `/health/olist-ready` is independent from the base service readiness check, and responses include both the model version and `olist-features-v1` contract version. This keeps the public image runnable without restricted Olist data while making the real-data model deployable through an approved artifact volume or model registry.
