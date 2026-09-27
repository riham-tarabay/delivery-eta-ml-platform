# Model card: Olist purchase-time delivery estimate

## Model summary

- **Model version:** `olist-eta-v1`
- **Task:** regression of elapsed days from purchase to observed customer delivery
- **Serving contract:** `olist-features-v1`
- **Model selected:** Random Forest regressor inside a scikit-learn preprocessing pipeline
- **Training grain:** one row per order
- **Primary metric:** mean absolute error (MAE), supplemented by RMSE and P90 absolute error

## Intended use

This model is intended for a **noncommercial, retrospective portfolio demonstration** of leakage-aware feature engineering, model comparison, and production-style serving. It is not approved for customer-facing commitments, operational routing, or SLA decisions.

The model expects features available at purchase time: calendar features, a recorded delivery-promise duration, order/item aggregates, coarse customer/seller geography, payment aggregates, and product aggregates. The API rejects unknown fields and exposes a versioned feature-contract identifier.

## Data and provenance

The model uses the anonymised Brazilian E-Commerce Public Dataset by Olist, covering historical orders from 2016–2018. The canonical Kaggle metadata reports **CC BY-NC-SA 4.0**. The raw archive is downloaded locally, hash-checked, and excluded from Git. See [`olist-provenance.json`](olist-provenance.json).

The target is available only for orders with a valid observed customer-delivery timestamp. The modeling population therefore excludes undelivered/canceled cases and is subject to completion-selection bias.

## Evaluation

The chronological test split contains 14,471 orders. The selected Random Forest produced:

- **MAE:** 3.7092 days
- **RMSE:** 5.2115 days
- **P90 absolute error:** 7.5543 days

The median baseline produced 5.0866-day MAE. The recorded-promise baseline produced 10.7225-day MAE in this split. These are historical offline metrics and must not be interpreted as current operational performance.

## Limitations and risks

- No live carrier scans, traffic, weather, or revised-ETA history are available.
- The historical period may not represent current logistics operations.
- Orders without observed delivery are not modeled as censored observations.
- One-to-many item tables must remain aggregated at order grain to prevent duplicate labels.
- Delivery timestamps, final status, reviews, and other outcome fields are forbidden in the purchase-time matrix.
- `order_estimated_delivery_date` is a customer promise baseline; it is not a log of model-generated ETA revisions.
- The dataset license is noncommercial. Do not redistribute the data, publicly host derived artifacts, or use the model commercially without separate permission/legal review.

## Monitoring and change control

A deployment owner should monitor feature null rates, unseen categorical levels, prediction distribution, data freshness, segment MAE, and the proportion of orders eligible for evaluation. Retraining requires a new model version, a new provenance record, a chronological holdout, a model-card update, and review of leakage and license assumptions.
