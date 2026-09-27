"""Point-in-time feature engineering for the Olist public e-commerce dataset.

The module deliberately keeps raw data outside the repository. It aggregates all
one-to-many tables to order grain before modeling and excludes outcome fields
from the purchase-time feature matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

TARGET = "delivery_days"
TIME_COLUMN = "order_purchase_timestamp"
NUMERIC_FEATURES = [
    "purchase_hour",
    "purchase_dow",
    "purchase_month",
    "is_weekend",
    "promise_days",
    "item_count",
    "product_count",
    "seller_count",
    "price_total",
    "freight_total",
    "payment_value_total",
    "payment_installments_max",
    "product_weight_g_mean",
    "product_volume_cm3_mean",
]
CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "payment_type_mode",
    "product_category_mode",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _read(root: Path, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(root / name, **kwargs)


def _mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    return values.mode().iat[0] if not values.empty else "unknown"


def load_olist_orders(raw_dir: str | Path, include_target: bool = True) -> pd.DataFrame:
    """Build one row per order from the canonical Olist CSV files."""
    root = Path(raw_dir)
    orders = _read(root, "olist_orders_dataset.csv", parse_dates=DATE_COLUMNS)
    customers = _read(root, "olist_customers_dataset.csv")[
        ["customer_id", "customer_state"]
    ]
    sellers = _read(root, "olist_sellers_dataset.csv")[["seller_id", "seller_state"]]
    items = _read(
        root, "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"]
    )
    payments = _read(root, "olist_order_payments_dataset.csv")
    products = _read(root, "olist_products_dataset.csv")
    translations = _read(root, "product_category_name_translation.csv")

    items = items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
    ).merge(translations, on="product_category_name", how="left")
    items["product_volume_cm3"] = (
        items["product_length_cm"]
        * items["product_height_cm"]
        * items["product_width_cm"]
    )
    item_features = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        product_weight_g_mean=("product_weight_g", "mean"),
        product_volume_cm3_mean=("product_volume_cm3", "mean"),
        product_category_mode=("product_category_name_english", _mode),
    )

    payment_features = payments.groupby("order_id", as_index=False).agg(
        payment_value_total=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
        payment_type_mode=("payment_type", _mode),
    )

    frame = (
        orders.merge(customers, on="customer_id", how="left", validate="one_to_one")
        .merge(item_features, on="order_id", how="left", validate="one_to_one")
        .merge(payment_features, on="order_id", how="left", validate="one_to_one")
    )
    # Seller state is aggregated after the item table has been reduced to order grain.
    seller_states = (
        items[["order_id", "seller_id"]]
        .drop_duplicates()
        .merge(sellers, on="seller_id", how="left", validate="many_to_one")
        .groupby("order_id", as_index=False)
        .agg(seller_state=("seller_state", _mode))
    )
    frame = frame.merge(seller_states, on="order_id", how="left", validate="one_to_one")

    purchase = frame[TIME_COLUMN]
    frame["purchase_hour"] = purchase.dt.hour
    frame["purchase_dow"] = purchase.dt.dayofweek
    frame["purchase_month"] = purchase.dt.month
    frame["is_weekend"] = (purchase.dt.dayofweek >= 5).astype(int)
    frame["promise_days"] = (
        frame["order_estimated_delivery_date"] - purchase
    ).dt.total_seconds() / 86400

    if include_target:
        frame[TARGET] = (
            frame["order_delivered_customer_date"] - purchase
        ).dt.total_seconds() / 86400
        eligible = (
            frame["order_status"].eq("delivered")
            & frame[TARGET].notna()
            & frame[TARGET].ge(0)
        )
        frame = frame.loc[eligible].copy()

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("unknown").astype(str)
    frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    required = ["order_id", TIME_COLUMN, *FEATURE_COLUMNS]
    if include_target:
        required.append(TARGET)
    if frame[required].isna().all(axis=0).any():
        empty = frame[required].columns[frame[required].isna().all(axis=0)].tolist()
        raise ValueError(f"Required feature columns are entirely missing: {empty}")
    return frame


def split_chronologically(
    frame: pd.DataFrame, train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by purchase time so future orders never enter training."""
    if not frame[TIME_COLUMN].is_monotonic_increasing:
        frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    train_end = int(len(frame) * train_fraction)
    validation_end = train_end + int(len(frame) * validation_fraction)
    return (
        frame.iloc[:train_end],
        frame.iloc[train_end:validation_end],
        frame.iloc[validation_end:],
    )


def validate_feature_contract(columns: Iterable[str]) -> None:
    """Reject target/outcome fields if they appear in a purchase-time matrix."""
    forbidden = {
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "order_status",
        "review_score",
        "delivery_days",
        "delivery_minutes",
    }
    leaked = sorted(forbidden.intersection(set(columns)))
    if leaked:
        raise ValueError(f"Outcome/post-delivery fields are forbidden: {leaked}")
