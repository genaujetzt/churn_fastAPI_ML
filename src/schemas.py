from typing import Any

from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int
    churn: int


class PredictionResponseChurn(BaseModel):
    prediction: int
    probability_no_churn: float
    probability_churn: float


class TrainingConfigChurn(BaseModel):
    model_type: str
    hyperparameters: dict


class ErrorResponse(BaseModel):
    code: int
    message: str
    details: Any = None
