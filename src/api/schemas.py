"""API request and response schemas."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str

