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
