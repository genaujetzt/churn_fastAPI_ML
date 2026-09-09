from src.dataset import preprocessing, traintest_split

DATASET_PATH = "data/churn_dataset.csv"


def test_preprocessing():
    X, y = preprocessing(DATASET_PATH)

    assert X is not None
    assert y is not None
    assert "churn" not in X.columns
    assert len(X) == len(y)


def test_train_test_split():
    X, y = preprocessing(DATASET_PATH)

    X_train, X_test, y_train, y_test, split_info = traintest_split(X, y)

    assert len(X_train) == 1600
    assert len(X_test) == 400
    assert len(y_train) == 1600
    assert len(y_test) == 400

    assert split_info["train_size"] == 1600
    assert split_info["test_size"] == 400


def test_churn_distribution():
    X, y = preprocessing(DATASET_PATH)

    X_train, X_test, y_train, y_test, _ = traintest_split(X, y)

    train_distribution = y_train.value_counts(normalize=True)
    test_distribution = y_test.value_counts(normalize=True)

    assert abs(train_distribution[1] - test_distribution[1]) < 0.01
