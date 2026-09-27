# CV-ready case study: Delivery ETA ML Platform

## Recommended project entry

**Delivery ETA ML Platform — Python, scikit-learn, FastAPI, Redis Streams, PostgreSQL, Docker, Kubernetes, GitHub Actions**
Public repository: https://github.com/riham-tarabay/delivery-eta-ml-platform

- Built a reproducible, leakage-aware delivery-time regression workflow over **96,470 eligible orders** from the anonymised Olist e-commerce dataset, aggregating order-item, product, seller, customer, and payment tables to order grain.
- Designed chronological train/validation/test evaluation and compared median, recorded-promise, Ridge, and Random Forest baselines; the selected model achieved **3.7092-day MAE** and **5.2115-day RMSE** on a held-out chronological test split.
- Integrated the model with a versioned FastAPI serving contract, artifact/feature-schema verification, batch inference, readiness probes, Prometheus metrics, Docker/Kubernetes deployment controls, and CI checks; validated with **12 automated tests** plus Ruff lint/format gates.
- Authored a model card, production-style runbook, provenance record, error-analysis outputs, and explicit limitations covering selection bias, temporal generalization, leakage, and the dataset's CC BY-NC-SA noncommercial license.

## Short version for a one-page CV

**Delivery ETA ML Platform | Python, scikit-learn, FastAPI, Docker/Kubernetes** — Built a leakage-aware Olist delivery-time regression pipeline over 96K eligible orders with chronological evaluation, baseline/model comparison, error analysis, versioned serving contracts, CI, and deployment hardening; achieved 3.71-day MAE on a held-out historical test split. [GitHub](https://github.com/riham-tarabay/delivery-eta-ml-platform)

## One concrete interview example

> I built a delivery ETA platform to demonstrate the full ML lifecycle rather than only model fitting. The key design decision was to define the prediction point first: purchase time. I aggregated the Olist order-item, product, seller, customer, and payment tables to one row per order, then explicitly excluded delivery timestamps, final status, reviews, and other outcome fields from the feature matrix. I used a chronological split so future orders could not leak into training, compared against both a median baseline and the recorded delivery promise, and inspected error slices instead of reporting one score. The final Random Forest reached 3.71 days MAE on the held-out historical split. I then added a versioned FastAPI contract, artifact compatibility checks, readiness probes, batch inference, CI tests, and a model card. I present it as portfolio evidence—not professional employment—and I document the historical-data, selection-bias, licensing, and no-real-time-signal limitations.

## How to describe it accurately

Use **“built,” “implemented,” “evaluated,” and “deployed in a reproducible portfolio environment.”** Do not say that it served production customers, improved a company's KPI, handled live carrier data, or proves professional ML experience. The repository demonstrates job-relevant engineering ability; the employment-history section of the CV should remain factual.
