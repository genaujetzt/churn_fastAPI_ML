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

EXPECTED_COLUMNS = (
    num_cols
    + cat_cols
    + ["churn"]
)

def validate_dataset(df: pd.DataFrame) -> None:

    if df.empty:
        raise ValueError("Dataset is empty")

    actual_columns = set(df.columns)
    expected_columns = set(EXPECTED_COLUMNS)

    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    if missing_columns or extra_columns:
        raise ValueError(
            "Invalid dataset columns. "
            f"Missing: {sorted(missing_columns)}. "
            f"Extra: {sorted(extra_columns)}."
        )

    if df["churn"].isna().any():
        raise ValueError(
            "Target column 'churn' contains missing values"
        )

    if not pd.api.types.is_numeric_dtype(
        df["churn"]
    ):
        raise ValueError(
            "Target column 'churn' must be numeric"
        )

    churn_values = set(
        df["churn"].unique()
    )

    if not churn_values.issubset({0, 1}):
        raise ValueError(
            "Target column 'churn' must contain only 0 and 1"
        )

    if churn_values != {0, 1}:
        raise ValueError(
            "Target column 'churn' must contain both 0 and 1"
        )

    for column in num_cols:
        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            raise ValueError(
                f"Numeric column '{column}' must be numeric"
            )


def read_validated_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    validate_dataset(df)

    logger.info(
        "Dataset validated: rows=%d, columns=%d",
        df.shape[0],
        df.shape[1],
    )

    return df


def preview_dataset(
    path: str,
    n: int = 5,
) -> list[dict]:

    df = read_validated_dataset(path)

    return df.head(n).to_dict(
        orient="records"
    )


def dataset_info(path: str) -> dict:

    df = read_validated_dataset(path)

    rows, columns = df.shape

    distribution = (
        df["churn"]
        .value_counts(normalize=True)
    )

    return {
        "rows": rows,
        "columns": columns,
        "features": df.columns.tolist(),
        "churn_distribution": (
            distribution.to_dict()
        ),
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


def preprocessing(
    path: str,
) -> tuple[pd.DataFrame, pd.Series]:

    df = read_validated_dataset(path)

    X = df.drop(
        ["churn"],
        axis=1,
    )

    y = df["churn"]

    return X, y


def traintest_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": y_train.value_counts(
            normalize=True
        ).to_dict(),
        "test_churn_distribution": y_test.value_counts(
            normalize=True
        ).to_dict(),
    }

    logger.info(
        "Dataset split completed: train=%d, test=%d", len(X_train), len(X_test)
    )

    return X_train, X_test, y_train, y_test, split_info
