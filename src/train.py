from datetime import datetime
import pickle
import json
import logging
import threading
import os
import tempfile
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
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
training_lock = threading.Lock()  #Не позволяет двум /model/train обучаться одновременно
state_lock = threading.Lock() #Защищает короткую операцию публикации, на train отключаем лок


def load_churn_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _write_pickle_temp(data, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

            pickle.dump(data, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        return temp_path

    except Exception:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

        raise

def save_churn_model(
    model,
    trained_at,
    metrics,
    model_type,
    hyperparameters,
):
    data = {
        "model": model,
        "trained_at": trained_at,
        "metrics": metrics,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
    }

    _write_pickle_temp(data, MODEL_PATH)
    
    temp_path = None
    try:
        temp_path = _write_pickle_temp(
            data,
            MODEL_PATH,
        )

        os.replace(
            temp_path,
            MODEL_PATH,
        )

        temp_path = None

    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


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

def create_preprocessor(): #новый `ColumnTransformer` для каждого кандидата 
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
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
    X_train,
    X_test,
    y_train,
    y_test,
    model_type_input,
    hyperparameters_input,
):
    global model
    global trained_at
    global metrics
    global model_type
    global hyperparameters

    with training_lock:

        if model_type_input not in ALLOWED_HYPERPARAMETERS:
            raise ValueError(
                f"Unknown model type: {model_type_input}"
            )

        allowed = ALLOWED_HYPERPARAMETERS[model_type_input]

        unknown = (
            set(hyperparameters_input) - allowed
        )

        if unknown:
            raise ValueError(
                f"Unsupported hyperparameters: "
                f"{sorted(unknown)}"
            )

        if model_type_input == "logreg":
            classifier = LogisticRegression(
                random_state=42,
                **hyperparameters_input,
            )

        elif model_type_input == "random_forest":
            classifier = RandomForestClassifier(
                random_state=42,
                **hyperparameters_input,
            )

        candidate_model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                ("classifier", classifier),
            ]
        )
        logger.info(
            "Starting training: type=%s, hyperparameters=%s",
            model_type_input,
            hyperparameters_input,
        )
        # Глобальная рабочая модель НЕ меняется.
        candidate_model.fit(
            X_train,
            y_train,
        )

        y_pred = candidate_model.predict(X_test)

        y_proba = candidate_model.predict_proba(
            X_test
        )[:, 1]

        candidate_metrics = {
            "accuracy": accuracy_score(
                y_test,
                y_pred,
            ),
            "f1": f1_score(
                y_test,
                y_pred,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                y_proba,
            ),
        }

        candidate_trained_at = datetime.now().isoformat()

        candidate_model_type = model_type_input

        candidate_hyperparameters = (
            hyperparameters_input.copy()
        )

        history = load_archive()
        candidate_history = [
            *history,
            {
                "trained_at": candidate_trained_at,
                "model_type": candidate_model_type,
                "hyperparameters": candidate_hyperparameters,
                **candidate_metrics,
            },
        ]
        model_data = {
            "model": candidate_model,
            "trained_at": candidate_trained_at,
            "metrics": candidate_metrics,
            "model_type": candidate_model_type,
            "hyperparameters": candidate_hyperparameters,
        }
        persist_training_artifacts(
            model_data,
            candidate_history,
        )
        
        # И только после успешного persistence
        # публикуем candidate как рабочее состояние.
        with state_lock:
            model = candidate_model
            trained_at = candidate_trained_at
            metrics = candidate_metrics
            model_type = candidate_model_type
            hyperparameters = candidate_hyperparameters

        logger.info(
            "Model trained: type=%s, "
            "hyperparameters=%s, metrics=%s",
            candidate_model_type,
            candidate_hyperparameters,
            candidate_metrics,
        )

        return candidate_metrics

ARCHIVE_PATH = "model/history.json"

def _write_json_temp(data, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

            json.dump(
                data,
                temp_file,
                indent=4,
                ensure_ascii=False,
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())

        return temp_path

    except Exception:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

        raise


def save_to_archive(
    timestamp,
    model_type,
    hyperparameters,
    metrics,
):
    history = load_archive()

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

    
    temp_path = None

    try:
        temp_path = _write_json_temp(
            history,
            ARCHIVE_PATH,
        )

        os.replace(
            temp_path,
            ARCHIVE_PATH,
        )

        temp_path = None

    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

def load_archive():
    try:
        with open(ARCHIVE_PATH, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


#будет одновременно публиковать оба файла
def persist_training_artifacts(
    model_data: dict,
    history_data: list,
) -> None:
    model_temp = None
    history_temp = None

    try:
        # Сначала полностью сериализуем 
        model_temp = _write_pickle_temp(
            model_data,
            MODEL_PATH,
        )

        history_temp = _write_json_temp(
            history_data,
            ARCHIVE_PATH,
        )

        # после успешной сериализации обоих публикуем как рабочие файлы.
        os.replace(model_temp, MODEL_PATH)
        model_temp = None

        os.replace(history_temp, ARCHIVE_PATH)
        history_temp = None

    finally:
        if model_temp is not None:
            try:
                os.unlink(model_temp)
            except FileNotFoundError:
                pass

        if history_temp is not None:
            try:
                os.unlink(history_temp)
            except FileNotFoundError:
                pass