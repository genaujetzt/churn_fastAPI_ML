from datetime import datetime
import pickle
import json
import logging
import threading

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
model_lock = threading.Lock()


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


try:
    data = load_churn_model(MODEL_PATH)

    model = data["model"]
    trained_at = data["trained_at"]
    metrics = data["metrics"]
    model_type = data["model_type"]
    hyperparameters = data["hyperparameters"]

    logger.info("Model loaded successfully")

except (FileNotFoundError, pickle.UnpicklingError, EOFError, KeyError) as exc:
    model = None
    trained_at = None
    metrics = None
    model_type = None
    hyperparameters = None

    logger.warning("Model could not be loaded from %s: %s", MODEL_PATH, exc)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ),
    ]
)

ALLOWED_HYPERPARAMETERS = {
    "logreg": {
        "C",
        "max_iter",
        "solver",
        "tol",
        "fit_intercept",
    },
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "criterion",
    },
}


def train_churn_model(
    X_train, X_test, y_train, y_test, model_type_input, hyperparameters_input
):

    global model
    global trained_at
    global metrics
    global model_type
    global hyperparameters

    if model_type_input not in ALLOWED_HYPERPARAMETERS:
        raise ValueError(f"Unknown model type: {model_type_input}")

    allowed = ALLOWED_HYPERPARAMETERS[model_type_input]

    unknown = set(hyperparameters_input) - allowed

    if unknown:
        raise ValueError(f"Unsupported hyperparameters: {sorted(unknown)}")

    if model_type_input == "logreg":
        classifier = LogisticRegression(
            random_state=42, **hyperparameters_input
        )

    elif model_type_input == "random_forest":
        classifier = RandomForestClassifier(
            random_state=42, **hyperparameters_input
        )

    else:
        raise ValueError(f"Unknown model type: {model_type_input}")

    candidate_model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    candidate_model.fit(X_train, y_train)

    y_pred = candidate_model.predict(X_test)
    y_proba = candidate_model.predict_proba(X_test)[:, 1]

    candidate_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    candidate_model_type = model_type_input
    candidate_hyperparameters = hyperparameters_input.copy()
    candidate_trained_at = datetime.now().isoformat()

    # на всякий случай убедимся, что сохранение тоже прошло успешно
    save_churn_model(
        candidate_model,
        candidate_trained_at,
        candidate_metrics,
        candidate_model_type,
        candidate_hyperparameters,
    )

    save_to_archive(
        candidate_trained_at,
        candidate_model_type,
        candidate_hyperparameters,
        candidate_metrics,
    )

    # lock отрабатывает быстро
    with model_lock:
        model = candidate_model
        trained_at = candidate_trained_at
        metrics = candidate_metrics
        model_type = candidate_model_type
        hyperparameters = candidate_hyperparameters

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
    except (FileNotFoundError, json.JSONDecodeError):
        return []
