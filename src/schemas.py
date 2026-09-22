from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid") #запретим лишнее

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
    model_config = ConfigDict(extra="forbid")

    model_type: Literal[
        "logreg",
        "random_forest",
    ]

    hyperparameters: dict = Field(
        examples=[
            {
                "C": 1.0,
                "max_iter": 1000,
            }
        ]
    )


class ErrorResponse(BaseModel):
    code: int
    message: str
    details: Any = None
