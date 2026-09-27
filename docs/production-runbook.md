# Production-style runbook

This runbook describes the engineering controls demonstrated by the repository. It is not an authorization to deploy the Olist model to a live customer-facing system.

## Build and validate

Run the complete CI-equivalent checks before a release:

```bash
python -m pip install -r requirements-dev.txt
ruff check src scripts tests
ruff format --check src scripts tests
pytest -q
```

For the local Olist experiment, download the data through the pinned script, verify the recorded SHA-256, train the artifact, inspect the model card and metrics, and run local batch inference. Raw data and trained artifacts must remain outside Git.

## Serving contract

The optional Olist API is enabled only when `OLIST_MODEL_PATH` points to an artifact containing a pipeline, model version, and exact `olist-features-v1` feature list. Startup does not silently treat a missing Olist artifact as ready. Use `/health/olist-ready` as the model-specific readiness check. The API exposes `/v1/olist/predict` and `/v1/olist/predict/batch`; unknown request fields and invalid ranges are rejected by Pydantic.

The artifact loader verifies the feature contract before serving. A mismatch is a deployment failure, not a warning. The service reports model version and contract version in responses so a caller can correlate predictions with a release.

## Deployment checklist

1. Build from a pinned commit and immutable container tag.
2. Supply the approved model artifact through an external artifact store or mounted volume; do not bake restricted Olist data into a public image.
3. Configure Redis/PostgreSQL secrets through the platform secret manager.
4. Verify non-root execution, dropped capabilities, read-only root filesystem, resource limits, and health probes.
5. Confirm `/health/live`, `/health/ready`, and `/health/olist-ready` before traffic is enabled.
6. Run a representative holdout smoke test and compare output distributions with the approved release.
7. Record model version, feature-contract version, dataset provenance, and approval owner.

## Monitoring

Monitor request rate, latency, error rate, prediction distribution, feature null rates, unseen categories, data freshness, eligible-target coverage, and segment MAE when labels become available. Alert on contract violations, missing artifacts, readiness failures, abrupt shifts in prediction distributions, and materially degraded segment performance.

## Rollback and incident response

If a release fails readiness, violates the feature contract, or shows a material regression, stop promotion and restore the previous immutable image and model-artifact pair. Preserve request samples without personal content, logs, metrics, and the failed model metadata for review. Do not silently retrain against a changed dataset. Create a new version after root-cause analysis, holdout evaluation, provenance review, and documentation updates.
