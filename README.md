# Churn Prediction Service

FastAPI-сервис для предсказания оттока клиентов (customer churn).

Сервис позволяет:

* загружать и анализировать churn dataset;
* разделять данные на train/test с сохранением распределения классов;
* обучать Logistic Regression или Random Forest;
* настраивать гиперпараметры модели через API;
* сохранять обученную модель и историю экспериментов;
* получать prediction и вероятность churn;
* проверять состояние сервиса через `/health`;
* получать информацию о модели, её метриках и схеме входных данных.

## Project structure

```text
churn/
├── data/
│   └── churn_dataset.csv
├── model/
│   ├── model.pkl
│   └── history.json
├── src/
│   ├── main.py
│   ├── dataset.py
│   ├── train.py
│   └── schemas.py
├── tests/
│   ├── test_dataset.py
│   ├── test_train.py
    ├── pytest.ini
│   └── test_main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset

The service expects a file:

```text
data/churn_dataset.csv
```

The dataset contains the following features:

| Feature              | Type   | Description                  |
| -------------------- | ------ | ---------------------------- |
| `monthly_fee`        | float  | Monthly customer fee         |
| `usage_hours`        | float  | Customer usage hours         |
| `support_requests`   | int    | Number of support requests   |
| `account_age_months` | int    | Account age in months        |
| `failed_payments`    | int    | Number of failed payments    |
| `region`             | string | Customer region              |
| `device_type`        | string | Customer device type         |
| `payment_method`     | string | Payment method               |
| `autopay_enabled`    | int    | Autopay flag (`0` or `1`)    |
| `churn`              | int    | Target variable (`0` or `1`) |

During training:

* missing values are handled during dataset preparation;
* numerical features are scaled with `StandardScaler`;
* categorical features are encoded with `OneHotEncoder`;
* preprocessing and the classifier are combined into one `Pipeline`.

## Local setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the service:

```bash
uvicorn src.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build the image:

```bash
docker build -t churn-service .
```

Run the container:

```bash
docker run --name churn-container -p 8000:8000 churn-service
```

The service will then be available at:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

## API examples

### Train model

`POST /model/train`

Example request:

```json
{
  "model_type": "logreg",
  "hyperparameters": {
    "C": 1.0,
    "max_iter": 1000
  }
}
```

Example response:

```json
{
  "accuracy": 0.79,
  "f1": 0.04,
  "roc_auc": 0.72
}
```

Random Forest example:

```json
{
  "model_type": "random_forest",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 5
  }
}
```

### Predict churn

`POST /predict`

Example request:

```json
{
  "monthly_fee": 50,
  "usage_hours": 200,
  "support_requests": 2,
  "account_age_months": 12,
  "failed_payments": 0,
  "region": "asia",
  "device_type": "desktop",
  "payment_method": "card",
  "autopay_enabled": 1
}
```

Example response:

```json
{
  "prediction": 1,
  "probability_no_churn": 0.18,
  "probability_churn": 0.82
}
```

### Model status

`GET /model/status`

Returns whether a model is currently available, when it was trained, its type, hyperparameters and metrics.

### Model schema

`GET /model/schema`

Returns the expected input fields and their types for `/predict`.

### Model metrics

`GET /model/metrics`

Returns training history and metrics.

Examples:

```text
GET /model/metrics
GET /model/metrics?limit=5
GET /model/metrics?model_type=logreg
```

### Health

`GET /health`

Returns service status and availability of the dataset and trained model.

Example:

```json
{
  "status": "ok",
  "model_available": true,
  "dataset_available": true
}
```

## Testing

Run all tests:

```bash
python -m pytest -v
```

The test suite contains unit tests for dataset preparation and model training, as well as integration tests for the FastAPI endpoints.

## Error handling

The service returns errors in a common format:

```json
{
  "code": 400,
  "message": "Error message",
  "details": null
}
```

Validation errors, missing datasets, invalid model configurations and unavailable models are handled by global exception handlers.

## Model persistence

The trained `Pipeline` is stored as:

```text
model/model.pkl
```

The file contains the complete preprocessing pipeline and trained classifier.

Training history is stored in:

```text
model/history.json
```

Each training record contains the timestamp, model type, hyperparameters and evaluation metrics.
