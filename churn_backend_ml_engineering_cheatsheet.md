# Churn FastAPI ML Service — инженерная памятка

## 0. Карта проекта

Типичная структура этого проекта:

```text
churn/
├── data/
│   └── churn_dataset.csv
├── model/
│   ├── model.pkl
│   └── history.json
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── dataset.py
│   ├── train.py
│   └── schemas.py
├── tests/
│   ├── test_dataset.py
│   ├── test_train.py
│   └── test_main.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

Ответственность модулей:

- `main.py` — FastAPI app, endpoints, exception handlers, API-level logging.
- `schemas.py` — Pydantic-модели входов/выходов API.
- `dataset.py` — чтение CSV, preprocessing, train/test split.
- `train.py` — preprocessing pipeline, создание/обучение моделей, метрики, persistence, training history.
- `tests/` — unit и integration tests.
- `Dockerfile` — инструкция сборки контейнера.

Для небольшого проекта необязательно создавать искусственные `api/`, `core/`, `services/` только ради слоёв. Структура должна отражать реальные обязанности, а не формально следовать шаблону.

---

# 1. Exceptions и handlers в FastAPI

## 1.1. Что такое exception

В Python exception — это объект, который сообщает об ошибочном состоянии:

```python
raise ValueError("Dataset is empty")
```

или:

```python
raise HTTPException(
    status_code=503,
    detail="Model is not trained"
)
```

Если exception никто не обработал, запрос обычно заканчивается 500.

## 1.2. Что такое exception handler

Handler — функция, которая говорит FastAPI:

> если во время обработки HTTP-запроса возникло исключение определённого типа, преврати его в конкретный HTTP-ответ.

Базовый принцип:

```text
endpoint
   ↓
exception
   ↓
global exception handler
   ↓
HTTP status + JSON
```

Пример:

```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "details": None,
        },
    )
```

Сам `HTTPException` отвечает на вопрос:

> какую ошибку мы хотим сообщить?

Handler отвечает:

> в каком формате клиент её получит?

## 1.3. Единый формат ошибки

Удобный контракт:

```json
{
  "code": 400,
  "message": "Invalid model parameters",
  "details": null
}
```

Например:

```python
class ErrorResponse(BaseModel):
    code: int
    message: str
    details: Any = None
```

## 1.4. Какие handlers нужны

### HTTPException

Используется для контролируемых API-ошибок:

```python
raise HTTPException(
    status_code=503,
    detail="Model is not trained",
)
```

Глобальный handler:

```python
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
```

### RequestValidationError

Возникает, когда входной JSON не соответствует Pydantic-схеме.

Например:

```json
{
  "monthly_fee": "hello"
}
```

Handler:

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
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
```

Важно: такой exception может возникнуть до входа в endpoint.

```text
HTTP request
   ↓
Pydantic validation
   ↓
ошибка
   ↓
endpoint может вообще не выполниться
```

### FileNotFoundError

Обычная Python-ошибка, например при чтении CSV:

```python
pd.read_csv(path)
```

Handler может преобразовать её в 404:

```python
@app.exception_handler(FileNotFoundError)
async def file_not_found_exception_handler(
    request: Request,
    exc: FileNotFoundError,
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
```

### ValueError

Подходит для ожидаемых ошибок бизнес-логики/подготовки данных:

```python
raise ValueError("Dataset is empty")
```

Handler:

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error("Value error: %s", exc)

    return JSONResponse(
        status_code=400,
        content={
            "code": 400,
            "message": str(exc),
            "details": None,
        },
    )
```

### TypeError

Полезен как дополнительный safety net:

```python
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
```

Но не стоит использовать один `Exception` handler вместо всех специализированных: он легко маскирует реальные баги.

---

# 2. Валидация входа и схема API

Pydantic-модель — это контракт API.

Например:

```python
class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int
```

Она определяет:

- набор полей;
- типы;
- обязательность;
- базовую валидацию.

## `model_dump()`

Pydantic object:

```python
features: FeatureVectorChurn
```

не является обычным `dict`.

Чтобы получить словарь:

```python
data = features.model_dump()
```

Получаем:

```python
{
    "monthly_fee": 50,
    "usage_hours": 200,
    ...
}
```

Это удобно для:

```python
df = pd.DataFrame([data])
```

## Схема для клиента

Можно автоматически вернуть Pydantic JSON Schema:

```python
@app.get("/model/schema")
def model_schema():
    return FeatureVectorChurn.model_json_schema()
```

Не нужно дублировать список признаков в отдельном `FEATURE_SCHEMA`, если он уже является частью `FeatureVectorChurn`.

---

# 3. Подготовка признаков

## Числовые и категориальные признаки

Явно разделяем:

```python
num_cols = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

cat_cols = [
    "region",
    "device_type",
    "payment_method",
]
```

## ColumnTransformer

Он позволяет применить разные preprocessing steps к разным колонкам:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            cat_cols,
        ),
    ]
)
```

## Pipeline

Вместо отдельного сохранения scaler/encoder/classifier:

```python
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)
```

Теперь `model` — единый объект.

Он содержит:

```text
Pipeline
├── preprocessor
│   ├── StandardScaler
│   └── OneHotEncoder
└── classifier
```

И именно этот Pipeline сериализуется:

```python
pickle.dump(model, file)
```

Поэтому при загрузке мы получаем preprocessing + trained model вместе.

---

# 4. Защита от несоответствия признаков при predict

Request:

```python
data = features.model_dump()
```

DataFrame:

```python
df = pd.DataFrame(
    [data],
    columns=num_cols + cat_cols,
)
```

Так API задаёт точный набор исходных признаков.

Дальше Pipeline сам применяет `ColumnTransformer`.

Главный принцип:

```text
training columns
      ↓
same feature contract
      ↓
prediction columns
```

---

# 5. Обучение и безопасное обновление модели

## Проблема

Плохой вариант:

```python
model = Pipeline(...)
model.fit(...)
```

Если `fit()` упадёт, глобальный `model` уже заменён на необученный Pipeline.

Получаем опасное состояние:

```text
старую модель потеряли
новая не обучилась
status → trained=true
predict → сломается
```

## Правильный pattern: candidate model

Сначала:

```python
candidate_model = Pipeline(...)
```

Потом:

```python
candidate_model.fit(...)
```

Потом:

```python
y_pred = candidate_model.predict(X_test)
y_proba = candidate_model.predict_proba(X_test)[:, 1]
```

Потом считаем метрики и только после успешного завершения всей операции обновляем рабочее состояние:

```python
with model_lock:
    model = candidate_model
    metrics = candidate_metrics
    trained_at = candidate_trained_at
    model_type = candidate_model_type
    hyperparameters = candidate_hyperparameters
```

Это можно описывать как **атомарную замену состояния**: рабочее состояние не изменяется, пока новый кандидат полностью не готов.

Это не database transaction и не абсолютная атомарность всех операций во всей системе; речь идёт о неделимой замене in-memory state после успешной подготовки кандидата.

## Зачем lock

```python
model_lock = threading.Lock()
```

Он нужен для короткого критического участка, а не для всего обучения.

Плохо:

```text
lock
  ↓
5–30 секунд fit
  ↓
unlock
```

Лучше:

```text
создать candidate
↓
fit candidate
↓
metrics
↓
persist
↓
lock на короткий swap
↓
unlock
```

В `/predict`:

```python
with train.model_lock:
    current_model = train.model
```

После этого prediction работает с локальной ссылкой `current_model`.

---

# 6. Обработка model.pkl

Нельзя делать безусловно:

```python
data = load_churn_model(MODEL_PATH)
```

на import-level без обработки ошибок.

Безопасный вариант:

```python
try:
    data = load_churn_model(MODEL_PATH)

    model = data["model"]
    trained_at = data["trained_at"]
    metrics = data["metrics"]
    model_type = data["model_type"]
    hyperparameters = data["hyperparameters"]

    logger.info("Model loaded successfully")

except (
    FileNotFoundError,
    pickle.UnpicklingError,
    EOFError,
    KeyError,
) as exc:
    model = None
    trained_at = None
    metrics = None
    model_type = None
    hyperparameters = None

    logger.warning(
        "Model could not be loaded from %s: %s",
        MODEL_PATH,
        exc,
    )
```

Критически важный принцип:

```text
model.pkl отсутствует/битый
        ↓
model = None
        ↓
FastAPI всё равно стартует
```

После этого `/predict` может вернуть контролируемый:

```json
{
  "code": 503,
  "message": "Model is not trained",
  "details": null
}
```

а `/model/train` остаётся доступным.

---

# 7. Hyperparameters: schema vs runtime validation

Плохо:

```python
hyperparameters: dict
```

не потому, что `dict` запрещён, а потому что он не описывает допустимые ключи.

Минимальная защита:

```python
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
```

Проверка:

```python
if model_type_input not in ALLOWED_HYPERPARAMETERS:
    raise ValueError(
        f"Unknown model type: {model_type_input}"
    )

allowed = ALLOWED_HYPERPARAMETERS[model_type_input]

unknown = set(hyperparameters_input) - allowed

if unknown:
    raise ValueError(
        f"Unsupported hyperparameters: {sorted(unknown)}"
    )
```

Это ловит:

```json
{
  "model_type": "logreg",
  "hyperparameters": {
    "random_state": 123
  }
}
```

до передачи аргументов в sklearn.

Для значений параметров дополнительно полезно обрабатывать sklearn parameter validation:

```python
from sklearn.utils._param_validation import InvalidParameterError
```

и возвращать 400 через global handler.

Пример:

```json
{
  "model_type": "logreg",
  "hyperparameters": {
    "C": -1
  }
}
```

---

# 8. Метрики

Для бинарного churn:

```python
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
```

Метрики:

```python
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
```

Почему ROC-AUC использует `y_proba`, а не `y_pred`:

```text
predict()
→ 0 / 1
→ итоговый класс
→ accuracy, F1

predict_proba()
→ вероятность класса
→ ranking/confidence
→ ROC-AUC
```

`[:, 1]` означает: взять вероятность положительного класса `churn=1`.

---

# 9. История экспериментов

`history.json` — это не application log.

Это журнал ML-экспериментов:

```json
{
  "timestamp": "...",
  "model_type": "logreg",
  "hyperparameters": {
    "C": 1.0
  },
  "accuracy": 0.82,
  "f1": 0.61,
  "roc_auc": 0.78
}
```

Запись добавляется после успешного обучения.

Для чтения:

```python
def load_archive():
    try:
        with open(ARCHIVE_PATH, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
```

API:

```text
GET /model/metrics
GET /model/metrics?limit=5
GET /model/metrics?model_type=logreg
```

---

# 10. Logging

## Logging ≠ print

Плохо для сервисного кода:

```python
print("Model loaded")
```

Лучше:

```python
import logging

logger = logging.getLogger(__name__)
```

И централизованная настройка:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
```

## Уровни

```text
DEBUG    — детальная диагностика
INFO     — обычные важные события
WARNING  — подозрительная ситуация
ERROR    — ошибка
CRITICAL — очень серьёзная ошибка
```

Примеры:

```python
logger.info("Model loaded")
logger.info("Prediction requested")
logger.warning("Model file is missing")
logger.error("HTTP error %s: %s", exc.status_code, exc.detail)
```

Для exception с traceback:

```python
logger.exception("Prediction failed")
```

## Что логировать

Минимум:

- загрузку dataset;
- успешную загрузку model;
- старт/успех training;
- prediction requests;
- ошибки.

Не надо писать в лог чувствительные или ненужные пользовательские данные.

Локально с `basicConfig` логи обычно идут в stdout/stderr терминала. Для файла нужен отдельный handler, например `FileHandler`; в production часто используют stdout + внешний сборщик логов.

---

# 11. Health endpoint

Простой health check:

```python
@app.get("/health")
def health():
    dataset_available = Path(
        "data/churn_dataset.csv"
    ).exists()

    return {
        "status": "ok",
        "model_available": train.model is not None,
        "dataset_available": dataset_available,
    }
```

Смысл:

```text
GET /health
→ сервис жив?
→ модель доступна?
→ dataset доступен?
```

Health endpoint отличается от `/model/status`:

- `/health` — operational state;
- `/model/status` — ML state и metadata.

---

# 12. Pytest

## Unit test

Проверяет одну маленькую единицу кода:

```python
def test_preprocessing():
    X, y = preprocessing(path)

    assert len(X) == len(y)
    assert "churn" not in X.columns
```

No FastAPI, no HTTP.

## Integration test

Проверяет взаимодействие нескольких частей:

```python
client = TestClient(app)

response = client.post("/model/train", json={...})
```

Например:

```text
TestClient
 ↓
FastAPI
 ↓
endpoint
 ↓
dataset
 ↓
train
 ↓
model
 ↓
JSON response
```

## `pytest.raises`

Проверяет ожидаемое исключение:

```python
with pytest.raises(ValueError):
    train_churn_model(...)
```

## `monkeypatch`

Временно меняет объект для теста:

```python
monkeypatch.setattr(train, "model", None)
```

Полезно для сценария:

```text
model unavailable
```

## TestClient

```python
from fastapi.testclient import TestClient

client = TestClient(app)
```

Позволяет тестировать FastAPI без запуска Uvicorn.

## Основные команды

```bash
python -m pytest -v
python -m pytest tests/test_train.py
python -m pytest tests/test_main.py::test_predict_without_model
```

---

# 13. Данные для тестов

Для unit-тестов лучше не зависеть от большого production-like CSV.

Хороший вариант — маленький фиксированный DataFrame:

```python
def make_test_data():
    return pd.DataFrame({
        "monthly_fee": [10, 20, 30, 40, 50, 60, 70, 80],
        ...
        "churn": [0, 0, 0, 0, 1, 1, 1, 1],
    })
```

Плюсы:

- воспроизводимость;
- маленький размер;
- тест контролирует вход;
- изменение основного CSV не ломает unit tests.

---

# 14. Docker

## Что такое Docker

Docker упаковывает приложение и его окружение в image.

```text
Dockerfile
    ↓
docker build
    ↓
image
    ↓
docker run
    ↓
container
```

## Dockerfile

Типовая схема:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY model ./model

EXPOSE 8000

CMD [
    "uvicorn",
    "src.main:app",
    "--host", "0.0.0.0",
    "--port", "8000"
]
```

`0.0.0.0` нужен, чтобы приложение было доступно снаружи контейнера.

## `.dockerignore`

Не отправляем в Docker build context:

```text
.venv/
__pycache__/
.pytest_cache/
.git/
tests/
*.pyc
```

Не исключаем `data/` и `model/`, если они нужны контейнеру.

## Команды

```bash
docker build -t churn-service .
docker images
docker run --name churn-container -p 8000:8000 churn-service
docker ps
docker ps -a
docker logs churn-container
docker stop churn-container
docker rm churn-container
```

После изменения файлов, которые попадают в image, нужен новый:

```bash
docker build -t churn-service .
```

Удаление локального `model.pkl` после уже выполненного `docker build` не меняет старый image: файл уже был скопирован внутрь него.

---

# 15. Virtual environment

Создать:

```bash
python -m venv .venv
```

Активировать Linux/macOS:

```bash
source .venv/bin/activate
```

Проверить:

```bash
which python
python --version
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Обновить requirements:

```bash
pip freeze > requirements.txt
```

Выйти:

```bash
deactivate
```

Важно: `.venv` — локальная среда разработки, а Docker имеет своё окружение.

---

# 16. Git

## Начало репозитория

```bash
git init
```

## Проверка состояния

```bash
git status
```

## Добавление

```bash
git add .
```

## Commit

```bash
git commit -m "Finalize churn service"
```

## История

```bash
git log --oneline --decorate -5
```

## Remote

```bash
git remote -v
```

Добавить:

```bash
git remote add origin https://github.com/user/repo.git
```

## Push

```bash
git branch -M main
git push -u origin main
```

## Если remote уже имеет отдельную историю

```bash
git pull origin main --allow-unrelated-histories --no-rebase
```

Если есть конфликт:

```bash
git status
```

После решения конфликта:

```bash
git add <file>
git commit
git push
```

## `.gitignore`

В репозиторий обычно не кладут:

```text
.venv/
__pycache__/
.pytest_cache/
*.pyc
```

Но `.gitignore` сам должен лежать в репозитории.

---

# 17. Code quality

## PEP 8

PEP 8 — style guide, а не программа.

Он задаёт правила:

- 4 пробела для indentation;
- `snake_case` для функций/переменных;
- `PascalCase` для классов;
- `UPPER_CASE` для констант;
- порядок импортов;
- читаемую длину строк;
- пустые строки между блоками.

## Black

Автоматический formatter:

```bash
black .
```

Проверка без изменения:

```bash
black --check .
```

Если нужен классический лимит 79:

```bash
black --line-length 79 .
```

## Ruff

Lint:

```bash
ruff check .
```

Ruff ищет в том числе:

- неиспользуемые imports;
- ошибки/подозрительные конструкции;
- style violations.

## pycodestyle

Классическая проверка PEP 8:

```bash
pycodestyle src tests
```

Black часто использует 88 символов, а классический PEP 8 — 79, поэтому они могут немного расходиться.

---

# 18. Типовой финальный workflow

Перед commit:

```bash
black .
ruff check .
pycodestyle src tests
python -m pytest -v
docker build -t churn-service .
```

Потом smoke test:

```text
GET /health
GET /docs
POST /model/train
GET /model/status
POST /predict
GET /model/metrics
GET /model/schema
```

Потом:

```bash
git status
git add .
git commit -m "Finalize churn service"
git push
```

---

# 19. Backend-принципы, которые стоит запомнить

## Separation of concerns

Один модуль — одна основная ответственность.

```text
API
schemas
data
ML/training
tests
```

## Single source of truth

Не дублируй одну и ту же схему в нескольких местах.

Например, если `FeatureVectorChurn` описывает input contract, `/model/schema` лучше строить из него, а не создавать второй ручной список.

## Validation at the boundary

Вход пользователя валидируем как можно раньше:

```text
HTTP request
 ↓
Pydantic
 ↓
business logic
```

## Fail safely

Отсутствие model.pkl не должно убивать весь сервис.

```text
model unavailable
↓
model = None
↓
service alive
```

## Never replace working state before success

Сначала candidate, потом commit/swap.

## Persistence vs logging

`model.pkl` — persistence модели.

`history.json` — история ML-экспериментов.

application logs — журнал работы сервиса.

Это три разных концепции.

## API contract

Endpoint должен иметь предсказуемые:

- вход;
- выход;
- status codes;
- error format.

## Reproducibility

Фиксированный `random_state`, versioned dependencies, deterministic tests и понятная training history делают ML-сервис воспроизводимым.

---

# 20. Как рассказывать этот проект на Senior ML Engineer интервью

Не стоит рассказывать:

> «Я сделала FastAPI, потом Docker, потом pytest...»

Лучше рассказывать через инженерные решения.

## Короткий вариант

> Я реализовала небольшой production-like churn prediction service на FastAPI и scikit-learn. Сервис принимает конфигурацию обучения через API, поддерживает Logistic Regression и Random Forest с параметрами, выполняет preprocessing внутри единого sklearn Pipeline и сохраняет Pipeline целиком, поэтому при inference используется ровно тот же preprocessing, что и при обучении.
>
> Я отдельно разделила API schemas, data preparation и training logic. Для API использовала Pydantic, включая schema для prediction input, training configuration и structured error response.
>
> Важной частью была обработка состояния модели. Я не заменяю рабочую модель до успешного завершения нового обучения: создаю candidate Pipeline, обучаю его, считаю метрики и только после успешного завершения atomically swap'лю модель и metadata. Это защищает сервис от неудачных retraining jobs.
>
> Для ошибок сделала централизованные FastAPI exception handlers, поэтому validation errors, недоступная модель и некорректные training parameters возвращаются в едином API-формате вместо технического 500.
>
> Историю тренировок сохраняю отдельно от application logs: для каждого эксперимента сохраняются timestamp, model type, hyperparameters и accuracy/F1/ROC-AUC. Это позволяет сравнивать конфигурации моделей.
>
> Для качества есть unit tests для data preparation и training logic и integration tests через FastAPI TestClient. Сервис также контейнеризован через Docker и имеет health endpoint для проверки доступности модели и датасета.

---

# 21. Если интервьюер спросит «почему Pipeline?»

Хороший ответ:

> Я не хотел хранить preprocessing отдельно от модели, потому что это создаёт риск train/inference mismatch. `ColumnTransformer + Pipeline` позволяет сериализовать preprocessing и estimator как единый объект и гарантирует, что inference проходит через тот же transformation graph.

---

# 22. «Почему StandardScaler и OneHotEncoder?»

> Числовые признаки имеют разные масштабы, поэтому для них используется scaling. Категориальные признаки требуют categorical encoding; OneHotEncoder преобразует их в числовое представление. `handle_unknown="ignore"` защищает inference от ранее невстречавшихся категорий.

---

# 23. «Почему ROC-AUC считаете на probability?»

> Accuracy и F1 работают с hard labels, а ROC-AUC оценивает ranking по score/probability. Поэтому для ROC-AUC используется `predict_proba(X)[:, 1]`, а не `predict(X)`.

---

# 24. «Что будет, если model.pkl повреждён?»

> Сервис не должен падать на import. Я перехватываю ошибки загрузки и оставляю `model=None`. Health endpoint показывает модель как недоступную, predict возвращает 503, а train endpoint остаётся доступным и может создать новую рабочую модель.

---

# 25. «Что будет, если retraining упадёт на C=-1?»

> Новый Pipeline создаётся как candidate и не становится глобальной рабочей моделью до успешного fit и расчёта метрик. Поэтому предыдущая рабочая модель продолжает обслуживать prediction requests.

---

# 26. «Почему не просто try/except внутри endpoint?»

> Потому что это приводит к дублированию и разным форматам ошибок. Глобальные exception handlers централизуют перевод Python/FastAPI exceptions в единый HTTP error contract.

---

# 27. «Почему нужен TestClient?»

> Unit tests проверяют отдельную ML/data логику. TestClient позволяет проверить интеграцию FastAPI с этой логикой без запуска реального HTTP-сервера: request проходит через routing, validation, endpoint и возвращает реальный response.

---

# 28. «Что бы вы улучшили дальше?»

Хороший senior-level ответ:

> Для production я бы вынес model registry/persistence из local filesystem, использовал object storage или model registry, добавил versioning моделей и схем, отдельно разделил training service и inference service при необходимости, добавил observability — structured logs, metrics и tracing, а также concurrency-safe model swap и более строгую схему hyperparameters. Для experiment tracking можно перейти с JSON на MLflow или аналогичный инструмент.

Важно не утверждать, что эти production-компоненты уже реализованы, если они не реализованы.

---

# 29. Что проект демонстрирует на самом деле

Этот проект — не только «я умею вызвать sklearn».

Он демонстрирует:

```text
Python
  ↓
data preprocessing
  ↓
ML pipeline
  ↓
model persistence
  ↓
API contract
  ↓
validation
  ↓
error handling
  ↓
logging
  ↓
tests
  ↓
Docker
  ↓
Git
```

Именно эту цепочку стоит держать в голове на backend/ML Engineer интервью.
