from datetime import datetime
import pickle
import json
import logging

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.dataset import num_cols, cat_cols

MODEL_PATH = "model/model.pkl"

model = None
trained_at = None
metrics = None
model_type = None
hyperparameters = None

logger = logging.getLogger(__name__)


def load_churn_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_churn_model(model, trained_at, metrics, model_type, hyperparameters):
    data = {
        "model": model,
        "trained_at": trained_at,
        "metrics": metrics,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(data, file)


data = load_churn_model(MODEL_PATH)

model = data["model"]
trained_at = data["trained_at"]
metrics = data["metrics"]
model_type = data["model_type"]
hyperparameters = data["hyperparameters"]

print("Model loaded")


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)


def train_churn_model(
    X_train, X_test, y_train, y_test, model_type_input, hyperparameters_input
):

    global model
    global trained_at
    global metrics
    global model_type
    global hyperparameters

    if model_type_input == "logreg":
        classifier = LogisticRegression(random_state=42, **hyperparameters_input)

    elif model_type_input == "random_forest":
        classifier = RandomForestClassifier(random_state=42, **hyperparameters_input)

    else:
        raise ValueError(f"Unknown model type: {model_type_input}")

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    model.fit(X_train, y_train)
    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "f1": f1_score(y_test, model.predict(X_test)),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }
    model_type = model_type_input
    hyperparameters = hyperparameters_input
    trained_at = datetime.now().isoformat()

    save_churn_model(
        model,
        trained_at,
        metrics,
        model_type,
        hyperparameters,
    )

    save_to_archive(trained_at, model_type, hyperparameters, metrics)

    logger.info(
        "Model trained: type=%s, hyperparameters=%s, metrics=%s",
        model_type,
        hyperparameters,
        metrics,
    )

    return model


ARCHIVE_PATH = "model/history.json"


def save_to_archive(timestamp, model_type, hyperparameters, metrics):
    try:
        with open(ARCHIVE_PATH, "r") as file:
            history = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(
        {
            "timestamp": timestamp,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }
    )

    with open(ARCHIVE_PATH, "w") as file:
        json.dump(history, file, indent=4)


def load_archive():
    try:
        with open(ARCHIVE_PATH, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []
