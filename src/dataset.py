import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.schemas import DatasetRowChurn

logger = logging.getLogger(__name__)


num_cols = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
cat_cols = ["region", "device_type", "payment_method"]


def preview_dataset(path: str, n: int = 5) -> list[dict]:
    df = pd.read_csv(path)
    logger.info("Dataset loaded for preview: %s", path)
    return df.head(n).to_dict(orient="records")


def dataset_info(path: str) -> dict:
    df = pd.read_csv(path)
    logger.info(
        "Dataset info requested: %s, rows=%d, columns=%d",
        path,
        df.shape[0],
        df.shape[1],
    )
    rows, columns = df.shape
    column_names = df.columns.to_list()
    distribution = df["churn"].value_counts(normalize=True)
    return {
        "rows": rows,
        "columns": columns,
        "features": column_names,
        "churn_distribution": distribution.to_dict(),
    }


def load_dataset(path: str) -> list[DatasetRowChurn]:
    df = pd.read_csv(path)
    logger.info("Loading dataset: %s", path)

    rows = []

    for _, row in df.iterrows():
        dataset_row = DatasetRowChurn(**row.to_dict())
        rows.append(dataset_row)

    logger.info("Dataset loaded successfully: %d rows", len(rows))

    return rows


def preprocessing(path: str) -> tuple[pd.DataFrame, pd.Series]:

    df = pd.read_csv(path)
    logger.info("Preprocessing dataset: %s, rows=%d", path, len(df))

    X, y = df.drop(["churn"], axis=1), df["churn"]
    return X, y


def traintest_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": y_train.value_counts(normalize=True).to_dict(),
        "test_churn_distribution": y_test.value_counts(normalize=True).to_dict(),
    }

    logger.info("Dataset split completed: train=%d, test=%d", len(X_train), len(X_test))

    return X_train, X_test, y_train, y_test, split_info
