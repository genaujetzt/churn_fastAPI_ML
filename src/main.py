import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sklearn.utils._param_validation import InvalidParameterError


from src import train
from src.schemas import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
)
from src.dataset import (
    preview_dataset,
    dataset_info,
    preprocessing,
    traintest_split,
    num_cols,
    cat_cols,
)

app = FastAPI()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("HTTP error %s: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "details": None,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    logger.error("Validation error: %s", exc.errors())

    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": "Validation error",
            "details": exc.errors(),
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_exception_handler(
    request: Request, exc: FileNotFoundError
):
    logger.error("File not found: %s", exc)

    return JSONResponse(
        status_code=404,
        content={
            "code": 404,
            "message": "Dataset not found",
            "details": str(exc),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error("Value error: %s", exc)

    return JSONResponse(
        status_code=400,
        content={"code": 400, "message": str(exc), "details": None},
    )


@app.exception_handler(TypeError)
async def type_error_handler(request: Request, exc: TypeError):
    logger.error("Type error: %s", exc)

    return JSONResponse(
        status_code=400,
        content={
            "code": 400,
            "message": "Invalid parameters",
            "details": str(exc),
        },
    )


@app.exception_handler(InvalidParameterError)
async def invalid_parameter_exception_handler(
    request: Request, exc: InvalidParameterError
):
    logger.error("Invalid model parameter: %s", exc)

    return JSONResponse(
        status_code=400,
        content={
            "code": 400,
            "message": "Invalid model parameters",
            "details": str(exc),
        },
    )


@app.get("/")
def home():
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn,
    summary="Predict customer churn",
    description="""
    Predict whether a customer is likely to churn.

    Example request:
    {
        "monthly_fee": 1,
        "usage_hours": 200,
        "support_requests": 2,
        "account_age_months": 1,
        "failed_payments": 1,
        "region": "asia",
        "device_type": "desktop",
        "payment_method": "card",
        "autopay_enabled": 0
    }

    Example response:
    {
        "prediction": 1
    }
    """,
)
def predict(features: FeatureVectorChurn):
    with train.model_lock:
        current_model = train.model

    if current_model is None:
        raise HTTPException(status_code=503, detail="Model is not trained")
    data = features.model_dump()
    df = pd.DataFrame([data], columns=num_cols + cat_cols)

    prediction = current_model.predict(df)[0]
    probabilities = current_model.predict_proba(df)[0]

    logger.info("Prediction requested")

    return {
        "prediction": int(prediction),
        "probability_no_churn": float(probabilities[0]),
        "probability_churn": float(probabilities[1]),
    }


@app.get("/dataset/preview")
def dataset_preview(n: int = 5):
    return preview_dataset("data/churn_dataset.csv", n)


@app.get("/dataset/info")
def info_dataset():
    return dataset_info("data/churn_dataset.csv")


@app.get("/dataset/split-info")
def split_info():
    return traintest_split(*preprocessing("data/churn_dataset.csv"))[-1]


@app.post("/model/train")
def train_model(config: TrainingConfigChurn):

    X, y = preprocessing("data/churn_dataset.csv")

    X_train, X_test, y_train, y_test, _ = traintest_split(X, y)

    model = train.train_churn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        config.model_type,
        config.hyperparameters,
    )

    return train.metrics


@app.get("/model/status")
def model_status():
    return {
        "trained": train.model is not None,
        "trained_at": train.trained_at,
        "metrics": train.metrics,
        "model_type": train.model_type,
        "hyperparameters": train.hyperparameters,
    }


@app.get("/model/schema")
def model_schema():
    return FeatureVectorChurn.model_json_schema()


@app.get("/model/metrics")
def model_metrics(limit: int = 5, model_type: str | None = None):
    history = train.load_archive()

    if model_type is not None:
        history = [
            item for item in history if item["model_type"] == model_type
        ]

    if not history:
        raise HTTPException(
            status_code=404, detail="Training history is empty"
        )

    return history[-limit:]


@app.get("/health")
def health():
    dataset_available = Path("data/churn_dataset.csv").exists()

    return {
        "status": "ok",
        "model_available": train.model is not None,
        "dataset_available": dataset_available,
    }
