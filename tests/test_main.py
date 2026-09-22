from fastapi.testclient import TestClient
import pytest

from sklearn.model_selection import train_test_split

from src.main import app
from src.dataset import preprocessing, traintest_split
from tests.test_train import make_test_data
from src import train

client = TestClient(app)


def test_training_and_prediction_flow():

    # 1. Проверяем, что датасет читается
    response = client.get("/dataset/info")

    assert response.status_code == 200

    data = response.json()

    assert data["rows"] > 0
    assert "churn" in data["features"]

    # 2. Запускаем обучение модели
    response = client.post(
        "/model/train",
        json={
            "model_type": "logreg",
            "hyperparameters": {"C": 1.0, "max_iter": 1000},
        },
    )

    assert response.status_code == 200

    metrics = response.json()

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # 3. Проверяем статус модели
    response = client.get("/model/status")

    assert response.status_code == 200

    status = response.json()

    assert status["trained"] is True
    assert status["model_type"] == "logreg"
    assert status["metrics"] is not None

    # 4. Делаем prediction
    response = client.post(
        "/predict",
        json={
            "monthly_fee": 50,
            "usage_hours": 200,
            "support_requests": 2,
            "account_age_months": 12,
            "failed_payments": 0,
            "region": "asia",
            "device_type": "desktop",
            "payment_method": "card",
            "autopay_enabled": 1,
        },
    )

    assert response.status_code == 200

    prediction = response.json()

    assert "prediction" in prediction
    assert "probability_no_churn" in prediction
    assert "probability_churn" in prediction

    assert prediction["prediction"] in [0, 1]
    assert 0 <= prediction["probability_no_churn"] <= 1
    assert 0 <= prediction["probability_churn"] <= 1


def test_predict_without_model(monkeypatch):
    from src import train

    monkeypatch.setattr(train, "model", None)

    response = client.post(
        "/predict",
        json={
            "monthly_fee": 50,
            "usage_hours": 200,
            "support_requests": 2,
            "account_age_months": 12,
            "failed_payments": 0,
            "region": "asia",
            "device_type": "desktop",
            "payment_method": "card",
            "autopay_enabled": 1,
        },
    )

    assert response.status_code == 503

    data = response.json()

    assert data["code"] == 503
    assert data["message"] == "Model is not trained"


def test_predict_invalid_data():

    response = client.post(
        "/predict",
        json={
            "monthly_fee": "hello",
            "usage_hours": 200,
            "support_requests": 2,
            "account_age_months": 12,
            "failed_payments": 0,
            "region": "asia",
            "device_type": "desktop",
            "payment_method": "card",
            "autopay_enabled": 1,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["code"] == 422
    assert data["message"] == "Validation error"


def test_train_unknown_model():

    response = client.post(
        "/model/train", json={"model_type": "catboost", "hyperparameters": {}}
    )

    assert response.status_code == 422

    data = response.json()

    assert data["code"] == 422


def test_train_invalid_hyperparameter():
    response = client.post(
        "/model/train",
        json={"model_type": "logreg", "hyperparameters": {"C": -1}},
    )

    assert response.status_code == 400

    data = response.json()

    assert data["code"] == 400
    assert data["message"] == "Invalid model parameters"
    assert data["details"] is not None


def test_predict_rejects_extra_field():

    response = client.post(
        "/predict",
        json={
            "monthly_fee": 50,
            "usage_hours": 200,
            "support_requests": 2,
            "account_age_months": 12,
            "failed_payments": 0,
            "region": "asia",
            "device_type": "desktop",
            "payment_method": "card",
            "autopay_enabled": 1,
            "unexpected": 123,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["code"] == 422


def test_invalid_dataset_columns(tmp_path):

    path = tmp_path / "bad.csv"

    df = make_test_data().drop(
        columns=["payment_method"]
    )

    df.to_csv(path, index=False)

    with pytest.raises(ValueError):
        preprocessing(str(path))


def test_training_with_missing_values():

    df = make_test_data()

    df.loc[0, "monthly_fee"] = float("nan")
    df.loc[1, "region"] = None

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = train.train_churn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "logreg",
        {
            "C": 1.0,
            "max_iter": 1000,
        },
    )

    assert model is not None


def test_failed_training_keeps_previous_model():

    X, y = preprocessing(
        "data/churn_dataset.csv"
    )

    X_train, X_test, y_train, y_test, _ = (
        traintest_split(X, y)
    )

    train.train_churn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "logreg",
        {
            "C": 1.0,
            "max_iter": 1000,
        },
    )

    previous_model = train.model
    previous_metrics = train.metrics.copy()

    with pytest.raises(Exception):
        train.train_churn_model(
            X_train,
            X_test,
            y_train,
            y_test,
            "logreg",
            {
                "C": -1,
            },
        )

    assert train.model is previous_model
    assert train.metrics == previous_metrics

