import pytest
import pandas as pd

from src.dataset import preprocessing, traintest_split
from src.train import train_churn_model

DATASET_PATH = "data/churn_dataset.csv"


def get_train_test_data():
    X, y = preprocessing(DATASET_PATH)

    return traintest_split(X, y)


def test_logreg_training():
    X_train, X_test, y_train, y_test, _ = get_train_test_data()

    model = train_churn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "logreg",
        {"C": 1.0, "max_iter": 1000},
    )

    assert model is not None
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_random_forest_training():
    X_train, X_test, y_train, y_test, _ = get_train_test_data()

    model = train_churn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "random_forest",
        {"n_estimators": 100, "max_depth": 5},
    )

    assert model is not None
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_training_metrics():
    X_train, X_test, y_train, y_test, _ = get_train_test_data()

    train_churn_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "logreg",
        {"C": 1.0, "max_iter": 1000},
    )

    from src import train

    assert train.metrics is not None
    assert "accuracy" in train.metrics
    assert "f1" in train.metrics
    assert "roc_auc" in train.metrics

    assert 0 <= train.metrics["accuracy"] <= 1
    assert 0 <= train.metrics["f1"] <= 1
    assert 0 <= train.metrics["roc_auc"] <= 1


def test_unknown_model_type():
    X, y = preprocessing(DATASET_PATH)

    X_train, X_test, y_train, y_test, _ = traintest_split(X, y)

    with pytest.raises(ValueError):
        train_churn_model(X_train, X_test, y_train, y_test, "catboost", {})


def make_test_data():
    return pd.DataFrame(
        {
            "monthly_fee": [10, 20, 30, 40, 50, 60, 70, 80],
            "usage_hours": [100, 90, 80, 70, 60, 50, 40, 30],
            "support_requests": [0, 0, 1, 1, 2, 2, 3, 4],
            "account_age_months": [24, 20, 18, 15, 12, 8, 5, 2],
            "failed_payments": [0, 0, 0, 1, 1, 2, 2, 3],
            "region": [
                "asia",
                "asia",
                "europe",
                "europe",
                "asia",
                "europe",
                "asia",
                "europe",
            ],
            "device_type": [
                "desktop",
                "mobile",
                "desktop",
                "mobile",
                "desktop",
                "mobile",
                "desktop",
                "mobile",
            ],
            "payment_method": [
                "card",
                "card",
                "cash",
                "card",
                "cash",
                "card",
                "cash",
                "card",
            ],
            "autopay_enabled": [1, 1, 1, 0, 0, 0, 0, 0],
            "churn": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
