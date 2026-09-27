# Delivery ETA ML Platform

A portfolio project demonstrating an end-to-end delivery ETA workflow: point-in-time feature engineering, chronological evaluation, baseline/model comparison, error analysis, reproducible offline inference, REST inference, event-driven scoring, Docker Compose, Kubernetes manifests, health checks, metrics, and tests.

> **Data and claims:** The real-data workflow uses the anonymised [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), whose canonical metadata reports **CC BY-NC-SA 4.0**. It is used locally for a noncommercial, retrospective demonstration. Raw data and generated model artifacts are not committed. Results do not establish current operational accuracy, production readiness, or professional ML experience.

## What changed

The original deterministic synthetic demo remains available for API compatibility and fast tests. The upgraded Olist workflow adds:

- order-level aggregation across orders, items, products, sellers, customers, and payments;
- a purchase-time feature contract that rejects target and post-delivery fields;
- a chronological train/validation/test split;
- median and recorded-promise baselines plus Ridge and Random Forest comparisons;
- MAE, RMSE, P90 absolute error, and eligible-order coverage;
- EDA summary, largest-error output, and customer-state error slices;
- deterministic local batch scoring with artifact/schema checks; and
- provenance, licensing, leakage, selection-bias, and deployment limitations.

The target is `delivery_days`: elapsed time from `order_purchase_timestamp` to `order_delivered_customer_date`, restricted to eligible delivered orders. `order_estimated_delivery_date` is treated as a recorded customer-promise baseline, not as a target or historical stream of model predictions.

## AI Engineer evidence

This repository is intentionally structured as a professional-style case study rather than a claim of prior employment. It demonstrates Python/scikit-learn modeling, large-table analysis, evaluation methodology, system integration, versioned serving contracts, deployment hardening, monitoring hooks, and technical documentation. The optional Olist service uses `OLIST_MODEL_PATH`, validates the exact feature contract before serving, exposes `/health/olist-ready`, and returns model and contract versions with predictions.

See [`docs/model-card.md`](docs/model-card.md) for intended use and limitations, [`docs/production-runbook.md`](docs/production-runbook.md) for release/monitoring/rollback controls, and [`docs/cv-case-study.md`](docs/cv-case-study.md) for truthful CV wording and one concrete interview example.

## Olist local workflow

The repository deliberately does not redistribute the dataset. Download it locally and record its provenance/hash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python scripts/download_olist.py
```

Train, compare models, and generate reports:

```bash
PYTHONPATH=src python scripts/train_olist_model.py
cat artifacts/olist/metrics.json
cat reports/olist/eda_summary.json
```

The command writes ignored outputs under `artifacts/olist/` and `reports/olist/`. Score the raw local Olist orders with the selected artifact:

```bash
PYTHONPATH=src python scripts/score_olist.py \
  --raw-dir data/raw/olist \
  --artifact artifacts/olist/eta_olist_model.joblib \
  --output artifacts/olist/predictions.csv
```

The feature builder aggregates one-to-many item rows before splitting and excludes delivery timestamps, final status, reviews, and other outcome fields from the purchase-time matrix. Missing delivery timestamps are not silently imputed as targets; they are outside the regression eligibility set and the resulting completion-selection bias is documented.

See [`data/README.md`](data/README.md) for the license, provenance, local-data policy, and source link. See [`docs/olist-provenance.json`](docs/olist-provenance.json) for the pinned archive hash, [`docs/olist-results.md`](docs/olist-results.md) for verified experiment results, and [`docs/dataset-and-scope-verification.md`](docs/dataset-and-scope-verification.md) for the dataset comparison and bounded scope.

## Existing online/event-driven service

The original service contract remains available for the lightweight synthetic demonstration:

```text
Client --POST /v1/predict--> FastAPI --> sklearn pipeline
Client --POST /v1/predict/batch--------------^
Client --POST /v1/events--> Redis Stream --> worker --> sklearn pipeline
                                             |           |--> PostgreSQL prediction record
                                             |           `--> delivery:predictions stream
                                             `--> delivery:dead-letter on failure
```

Run the original reproducible artifact and tests:

```bash
python scripts/train_model.py
pytest -q
PYTHONPATH=src uvicorn delivery_eta.api:app --reload
```

The API provides `/health/live`, `/health/ready`, `/metrics`, `/v1/predict`, `/v1/predict/batch`, and `/v1/events`. It has no authentication and must not be exposed publicly without access control.

## Integrated services and deployment validation

```bash
cp .env.example .env
docker compose up --build
```

The Kubernetes manifests in `infra/k8s/` include non-root execution, dropped Linux capabilities, read-only root filesystems, resource requests/limits, and health probes. They require an immutable image tag, external secrets, and review before any real deployment. The Olist model is intentionally not baked into the public image because the dataset is noncommercial and local-only.

## Evaluation scope and limitations

The real-data experiment is deliberately bounded:

- **In scope:** retrospective purchase-time regression, chronological order-level splitting, feature availability checks, EDA, model comparison, segment error analysis, deterministic inference, and container/configuration checks.
- **Out of scope:** real-time carrier tracking, revised ETA streams, traffic/weather signals, online retraining, survival modeling for undelivered orders, causal claims, current operational benchmarking, SLAs, and commercial/public redistribution without permission.

A successful run is evidence of a reproducible portfolio workflow. It is not evidence of live logistics impact or professional ML experience.
