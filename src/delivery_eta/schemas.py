from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Weather = Literal["clear", "rain", "wind", "heat"]


class DeliveryFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: float = Field(ge=0.1, le=100)
    order_hour: int = Field(ge=0, le=23)
    traffic_index: float = Field(ge=0, le=10)
    weather: Weather
    pickup_wait_min: float = Field(ge=0, le=90)
    rider_experience_months: int = Field(ge=0, le=240)
    is_weekend: bool


class PredictionRequest(DeliveryFeatures):
    order_id: str = Field(min_length=1, max_length=100)


class PredictionResult(BaseModel):
    order_id: str
    predicted_eta_minutes: float = Field(ge=0)
    model_version: str


class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequest] = Field(min_length=1, max_length=5000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResult]
    model_version: str


class DeliveryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    order_id: str = Field(min_length=1, max_length=100)
    features: DeliveryFeatures


class OlistFeatures(BaseModel):
    """Point-in-time Olist features accepted by the serving contract."""

    model_config = ConfigDict(extra="forbid")

    purchase_hour: int = Field(ge=0, le=23)
    purchase_dow: int = Field(ge=0, le=6)
    purchase_month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    promise_days: float = Field(ge=0, le=120)
    item_count: int = Field(ge=1, le=100)
    product_count: int = Field(ge=1, le=100)
    seller_count: int = Field(ge=1, le=100)
    price_total: float = Field(ge=0, le=100000)
    freight_total: float = Field(ge=0, le=100000)
    payment_value_total: float = Field(ge=0, le=100000)
    payment_installments_max: int = Field(ge=1, le=36)
    product_weight_g_mean: float | None = Field(default=None, ge=0, le=100000)
    product_volume_cm3_mean: float | None = Field(default=None, ge=0, le=10000000)
    customer_state: str = Field(min_length=1, max_length=2)
    seller_state: str = Field(min_length=1, max_length=2)
    payment_type_mode: str = Field(min_length=1, max_length=32)
    product_category_mode: str = Field(min_length=1, max_length=128)


class OlistPredictionRequest(OlistFeatures):
    order_id: str = Field(min_length=1, max_length=100)


class OlistPredictionResult(BaseModel):
    order_id: str
    predicted_delivery_days: float = Field(ge=0)
    model_version: str
    feature_contract_version: str = "olist-features-v1"


class OlistBatchPredictionRequest(BaseModel):
    orders: list[OlistPredictionRequest] = Field(min_length=1, max_length=1000)


class OlistBatchPredictionResponse(BaseModel):
    predictions: list[OlistPredictionResult]
    model_version: str
    feature_contract_version: str = "olist-features-v1"
