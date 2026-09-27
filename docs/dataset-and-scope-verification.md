# Delivery ETA Dataset and Scope Verification

## Decision

Use the **Brazilian E-Commerce Public Dataset by Olist** for the upgrade, but frame the project as a **noncommercial, retrospective offline ETA demonstration**. Do not present it as a current production system, real-time tracking model, or evidence of professional ML experience.

The canonical Olist archive is described as anonymised commercial data supplied by Olist and covering approximately 100,000 orders from 2016–2018 [1]. The verified archive contains 99,441 order rows and 96,476 non-null customer-delivery timestamps. It includes purchase, approval, carrier-handoff, actual customer-delivery, and promised-delivery timestamps, plus customer, seller, item, product, payment, and review tables [2].

## Candidate comparison

### 1. Olist Brazilian E-Commerce — recommended

**Why it fits:** The target can be defined as elapsed time from `order_purchase_timestamp` to `order_delivered_customer_date`, restricted to eligible completed deliveries. The related tables support order-level feature engineering, meaningful exploratory analysis, sliced error analysis, baseline comparisons, and reproducible inference tests.

**Important fields:**

- Purchase time: `order_purchase_timestamp`
- Observed target endpoint: `order_delivered_customer_date`
- Recorded customer promise: `order_estimated_delivery_date`
- Pre-delivery candidates: customer/seller geography, aggregated item counts, product attributes, payment fields, price, freight, and carefully justified shipping-limit features

**License and provenance:** Kaggle's canonical metadata identifies Olist as the publisher and reports **CC BY-NC-SA 4.0** [3]. Attribution, a license link, and share-alike obligations apply. Commercial authorization and the treatment of hosted models or derived databases are not established by the available evidence. Obtain written permission or legal review before commercial use, public hosting, or redistribution of the data or derived artifacts.

**Risks:** Delivery timestamps are missing for undelivered or canceled orders, so filtering to completed deliveries introduces selection bias. Item rows are one-to-many per order, so naïve joins can duplicate labels. Carrier handoff, final status, delivery timestamps, review fields, and other post-outcome values must be excluded from a purchase-time model. The estimated date is a recorded promise, not a historical stream of model-generated ETA revisions.

### 2. DataCo SMART SUPPLY CHAIN — fallback only

The Mendeley Data version contains a large structured file and reports **CC BY 4.0** metadata [4]. It has order and shipping timestamps, `Days for shipping (real)`, and `Days for shipment (scheduled)`, but no final customer-arrival timestamp. It is therefore suitable for shipping-duration or late-delivery prediction, not a verified customer delivery ETA. It also contains repeated order-level rows and sensitive customer fields that must be scrubbed.

### 3. Delivery Logistics Dataset — reject for this upgrade

The closest matching Kaggle pages explicitly describe the data as synthetic. The time fields use malformed 1970 placeholder encodings, duplicate pages disagree about row counts and licensing, and no independently verified observed delivery events were found. It should not be used to claim a real-data ETA upgrade.

## Bounded implementation scope

1. **Target and eligibility**
   - Predict elapsed delivery time at purchase time for orders with a valid observed customer-delivery timestamp.
   - Keep undelivered/canceled records out of the regression target and document the resulting completion-selection bias.
   - Compare against a historical/median baseline and the recorded estimated-delivery promise.
   - Do not predict the recorded promise itself.

2. **Point-in-time feature contract**
   - Use order-level features only.
   - Aggregate item, product, seller, and payment tables before splitting.
   - Include purchase calendar features, coarse customer/seller geography, item counts, distinct sellers/products, price/freight totals, product attributes, and payment type/installments/value where available at scoring time.
   - Exclude delivery, carrier, final-status, review, and other post-outcome fields.

3. **Data exploration**
   - Profile row counts, keys, duplicates, join cardinality, nulls, date inconsistencies, target eligibility, target distribution, promise distribution, and time coverage.
   - Analyze target and promise behavior by month, state/region, seller, product category, order size, and missingness.

4. **Model comparison**
   - Use a chronological order-level train/validation/test split.
   - Compare a median or historical baseline, a regularized linear model, and one tree-based model.
   - Report MAE, RMSE, P90 absolute error, and eligible-order coverage.

5. **Error analysis**
   - Break down residuals by month, region, promised lead time, order size/freight, seller/product segment, and missingness/status groups.
   - Quantify systematic under- and over-prediction instead of reporting only one aggregate score.

6. **Inference and deployment validation**
   - Add deterministic batch or API inference using the trained artifact.
   - Test schema validation, feature-contract/leakage assertions, one-to-many aggregation, missing fields, unseen categories, train/serve parity, deterministic output, and artifact loading.
   - Validate the container and Kubernetes manifests against a representative holdout. Treat this as reproducibility validation, not proof of production readiness.

7. **Explicitly out of scope**
   - Real-time carrier tracking, revised ETA streams, traffic/weather signals, online retraining, survival/censoring modeling, causal claims, current operational benchmarking, production SLA/scale claims, and commercial/public redistribution without permission.

## Sources

[1]: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "Brazilian E-Commerce Public Dataset by Olist"
[2]: https://www.kaggle.com/api/v1/datasets/view/olistbr/brazilian-ecommerce "Olist dataset API metadata"
[3]: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 legal code"
[4]: https://data.mendeley.com/datasets/8gx2fvg2k6/5 "DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS, Mendeley Data version 5"
[5]: https://creativecommons.org/licenses/by/4.0/legalcode "Creative Commons Attribution 4.0 legal code"
