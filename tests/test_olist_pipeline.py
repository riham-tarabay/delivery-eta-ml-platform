from pathlib import Path

import pandas as pd
import pytest

from delivery_eta.olist import (
    FEATURE_COLUMNS,
    load_olist_orders,
    split_chronologically,
    validate_feature_contract,
)


def _write_fixture(root: Path) -> None:
    pd.DataFrame(
        [
            {
                "order_id": "o1",
                "customer_id": "c1",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-01-01 10:00:00",
                "order_approved_at": "2018-01-01 10:05:00",
                "order_delivered_carrier_date": "2018-01-02 10:00:00",
                "order_delivered_customer_date": "2018-01-05 10:00:00",
                "order_estimated_delivery_date": "2018-01-10 00:00:00",
            },
            {
                "order_id": "o2",
                "customer_id": "c2",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-01-02 10:00:00",
                "order_approved_at": "2018-01-02 10:05:00",
                "order_delivered_carrier_date": "2018-01-03 10:00:00",
                "order_delivered_customer_date": "2018-01-07 10:00:00",
                "order_estimated_delivery_date": "2018-01-12 00:00:00",
            },
            {
                "order_id": "o3",
                "customer_id": "c3",
                "order_status": "canceled",
                "order_purchase_timestamp": "2018-01-03 10:00:00",
                "order_approved_at": "2018-01-03 10:05:00",
                "order_delivered_carrier_date": "",
                "order_delivered_customer_date": "",
                "order_estimated_delivery_date": "2018-01-13 00:00:00",
            },
        ]
    ).to_csv(root / "olist_orders_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "customer_id": "c1",
                "customer_unique_id": "u1",
                "customer_zip_code_prefix": 1,
                "customer_city": "a",
                "customer_state": "SP",
            },
            {
                "customer_id": "c2",
                "customer_unique_id": "u2",
                "customer_zip_code_prefix": 2,
                "customer_city": "b",
                "customer_state": "RJ",
            },
            {
                "customer_id": "c3",
                "customer_unique_id": "u3",
                "customer_zip_code_prefix": 3,
                "customer_city": "c",
                "customer_state": "MG",
            },
        ]
    ).to_csv(root / "olist_customers_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "seller_id": "s1",
                "seller_zip_code_prefix": 1,
                "seller_city": "a",
                "seller_state": "SP",
            },
            {
                "seller_id": "s2",
                "seller_zip_code_prefix": 2,
                "seller_city": "b",
                "seller_state": "RJ",
            },
        ]
    ).to_csv(root / "olist_sellers_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "order_id": "o1",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "shipping_limit_date": "2018-01-02",
                "price": 10,
                "freight_value": 2,
            },
            {
                "order_id": "o1",
                "order_item_id": 2,
                "product_id": "p2",
                "seller_id": "s2",
                "shipping_limit_date": "2018-01-02",
                "price": 20,
                "freight_value": 3,
            },
            {
                "order_id": "o2",
                "order_item_id": 1,
                "product_id": "p1",
                "seller_id": "s1",
                "shipping_limit_date": "2018-01-03",
                "price": 15,
                "freight_value": 2,
            },
        ]
    ).to_csv(root / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "product_id": "p1",
                "product_category_name": "cat",
                "product_name_lenght": 1,
                "product_description_lenght": 1,
                "product_photos_qty": 1,
                "product_weight_g": 100,
                "product_length_cm": 10,
                "product_height_cm": 2,
                "product_width_cm": 3,
            },
            {
                "product_id": "p2",
                "product_category_name": "cat2",
                "product_name_lenght": 1,
                "product_description_lenght": 1,
                "product_photos_qty": 1,
                "product_weight_g": 200,
                "product_length_cm": 10,
                "product_height_cm": 2,
                "product_width_cm": 3,
            },
        ]
    ).to_csv(root / "olist_products_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "order_id": "o1",
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_installments": 2,
                "payment_value": 32,
            },
            {
                "order_id": "o2",
                "payment_sequential": 1,
                "payment_type": "boleto",
                "payment_installments": 1,
                "payment_value": 17,
            },
        ]
    ).to_csv(root / "olist_order_payments_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "product_category_name": "cat",
                "product_category_name_english": "category-a",
            },
            {
                "product_category_name": "cat2",
                "product_category_name_english": "category-b",
            },
        ]
    ).to_csv(root / "product_category_name_translation.csv", index=False)


def test_olist_loader_aggregates_to_order_grain(tmp_path):
    _write_fixture(tmp_path)
    frame = load_olist_orders(tmp_path)
    assert len(frame) == 2
    assert frame.order_id.is_unique
    assert frame.loc[frame.order_id.eq("o1"), "item_count"].item() == 2
    assert frame.loc[frame.order_id.eq("o1"), "seller_count"].item() == 2
    assert frame.loc[frame.order_id.eq("o1"), "price_total"].item() == 30
    assert frame[TARGET].min() > 0


def test_chronological_split_is_ordered():
    frame = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            )
        }
    )
    train, validation, test = split_chronologically(frame, 0.5, 0.25)
    assert (
        train.iloc[-1].order_purchase_timestamp
        < validation.iloc[0].order_purchase_timestamp
    )
    assert (
        validation.iloc[-1].order_purchase_timestamp
        < test.iloc[0].order_purchase_timestamp
    )


def test_feature_contract_rejects_outcomes():
    validate_feature_contract(FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="Outcome"):
        validate_feature_contract([*FEATURE_COLUMNS, "order_delivered_customer_date"])


TARGET = "delivery_days"
