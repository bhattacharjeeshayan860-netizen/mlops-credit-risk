"""FastAPI application for real-time credit-risk inference.

Endpoints:
- GET /health: service health check
- POST /predict: model prediction with confidence score
- GET /metrics: Prometheus metrics endpoint
- POST /retrain: manual or automated retraining trigger (stub)
- GET /model/info: active model metadata
"""
from typing import Optional

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from src.predict import load_model_info, load_resources, make_prediction
from src.preprocessing import CreditRiskPreprocessor


class PredictionRequest(BaseModel):
    """Single-row credit-risk request payload."""

    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: Optional[float] = None
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: Optional[int] = None


PREDICTION_COUNT = Counter(
    "api_prediction_requests_total",
    "Total credit risk predictions",
    ["risk_label"],
)
PREDICTION_HISTOGRAM = Histogram(
    "credit_default_probability",
    "Distribution of predicted credit default probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

app = FastAPI(
    title="Real-Time ML Inference API",
    description="Credit-risk prediction API with monitoring and drift detection.",
    version="0.1.0",
)

Instrumentator().instrument(app).expose(app)

# These are populated when the API starts. They hold the same objects returned by
# load_resources() so we do not re-fetch from MLflow on every request.
_model: Pipeline | None = None
_preprocessor: CreditRiskPreprocessor | None = None


@app.on_event("startup")
def startup_event() -> None:
    """Load model and preprocessor once when the server starts."""
    global _model, _preprocessor
    _model, _preprocessor, _ = load_resources()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    """Run inference on a single set of features."""
    prediction_input = request.model_dump()

    result = make_prediction(
        prediction_input,
        model=_model,
        preprocessor=_preprocessor,
    )

    PREDICTION_COUNT.labels(risk_label=result["risk_label"]).inc()
    PREDICTION_HISTOGRAM.observe(result["default_probability"])

    return result


@app.get("/model/info")
def model_info() -> dict:
    """Return active model metadata."""
    return load_model_info()


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/retrain")
def retrain() -> dict:
    """Manual retraining trigger (full implementation in Week 7)."""
    return {"status": "retraining_triggered", "detail": "not implemented until Week 7"}
