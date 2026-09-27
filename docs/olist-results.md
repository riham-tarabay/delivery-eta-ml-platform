# Olist experiment results

This report was generated from a local download of the canonical Olist archive. The raw archive and trained model are intentionally not committed. See [`olist-provenance.json`](olist-provenance.json) for the source, license, and archive hash.

## Data exploration

The archive contains 99,441 orders and 112,650 order-item rows. The purchase-time feature builder reduces the eligible target set to **96,470 delivered orders** with valid, non-negative elapsed delivery time. The modeling grain is one row per order; item, product, seller, and payment tables are aggregated before splitting.

The eligible target has a mean of **12.5582 days**, a median of **10.2175 days**, and a P90 of **23.0964 days**. The chronological split contains 67,529 training rows, 14,470 validation rows, and 14,471 test rows. The source includes historical orders from 2016–2018 and has no real-time tracking or revised-ETA stream.

The feature audit found sparse missingness in payment aggregates and 16 missing product-attribute aggregates. These are handled through training-time imputers. The feature contract excludes delivery timestamps, final status, reviews, and other outcome fields.

## Model comparison

Validation is used for model selection. The final selected model is refit on train plus validation and evaluated once on the chronological test set.

| Model | Validation MAE (days) | Test MAE (days) | Test RMSE (days) | Test P90 absolute error (days) |
|---|---:|---:|---:|---:|
| Median baseline | 5.6939 | 5.0866 | 6.2835 | 8.9060 |
| Recorded-promise baseline | 14.9102 | 10.7225 | 13.1124 | 20.9521 |
| Ridge | 5.5744 | — | — | — |
| Random Forest | 4.8446 | **3.7092** | **5.2115** | **7.5543** |

The Random Forest improves test MAE by approximately **27.1%** relative to the median baseline. This is an offline historical comparison, not a current operational accuracy claim.

## Error analysis

The training command writes `reports/olist/error_by_customer_state.csv` and `reports/olist/largest_test_errors.csv` locally. These reports quantify error by customer state and inspect the largest absolute misses. The intended analysis is to compare under- and over-prediction across time, geography, order size/freight, seller/product segments, promise lead time, and missingness groups.

The key interpretation limits are selection bias from excluding orders without observed delivery timestamps, possible distribution shift after the historical period, and the absence of live carrier/traffic/weather signals. The recorded promise is a useful baseline, but it is not a historical log of model-generated ETAs.

## Reproduction

```bash
PYTHONPATH=src python scripts/download_olist.py
PYTHONPATH=src python scripts/train_olist_model.py
PYTHONPATH=src python scripts/score_olist.py
```

The generated model is a local artifact under `artifacts/olist/`; it is excluded by `.gitignore`.
