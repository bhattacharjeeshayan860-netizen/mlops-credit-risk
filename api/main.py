"""FastAPI application for real-time credit-risk inference.

Planned endpoints:
- GET /health: service health check
- POST /predict: model prediction with confidence score
- GET /metrics: Prometheus metrics endpoint
- POST /retrain: manual or automated retraining trigger
- GET /model/info: active model metadata
"""
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Any

from pydantic import BaseModel
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.pipeline import Pipeline

from src.monitor import (
    log_prediction,
    reset_prediction_log,
    run_drift_check,
    get_monitoring_stats as calculate_monitoring_stats,
    get_latest_drift_report,
)
from src.predict import (
    load_resources,
    make_prediction,
    promote_if_better,
    update_local_fallback,
    CreditRiskPreprocessor,
)
from src.utils import MLFLOW_TRACKING_URI
from src.train import train_candidate
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Request model for input data."""
    # Define the expected fields for the prediction request    
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: Optional[float] = None
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: Optional[float] = None

PREDICTION_COUNT = Counter(
    "api_prediction_requests_total",
    "credit_risk_prediction_total",
    ["risk_label"]
)
PREDICTION_HISTOGRAM = Histogram(
    "credit_default_probability",
    "Distribution of predicted credit default probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


_model: Optional[Pipeline] = None
_preprocessor: Optional[CreditRiskPreprocessor] = None
_model_info: Optional[dict[str, Any]] = None
_drift_event_active: bool = False

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan handler to load model and preprocessor at startup."""
    del _app
    global _model, _preprocessor, _model_info
    try:
        logger.info("Loading resources at startup...")
        _model, _preprocessor, _model_info = load_resources()
        logger.info("Resources loaded successfully.")
        yield
    except Exception as e:
        logger.error(f"Failed to load resources at startup: {e}")
        # We don't raise here so the app can still start, 
        # but /ready will correctly indicate it's not ready.
        yield
    finally:
        logger.info("Lifespan shutdown.")

app = FastAPI(
    title="Real-Time ML Inference API",
    description="Credit-risk prediction API with monitoring and drift detection.",
    version="0.1.0",
    lifespan=lifespan
)

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
origins = [origin.strip() for origin in cors_origins_env.split(",")]
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")


def require_api_key(api_key: Optional[str]) -> None:
    """Require an API key for protected write endpoints when configured."""
    if API_AUTH_TOKEN and api_key != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Valid API key required")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Ensure reports directory exists
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Mount reports directory for static access
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

Instrumentator().instrument(app).expose(app)


@app.get("/reports/latest")
def get_latest_report_name() -> dict[str, str]:
    """Return the name of the latest drift report."""
    report_name = get_latest_drift_report()
    if report_name:
        return {"report_name": report_name}
    raise HTTPException(status_code=404, detail="No reports found.")


@app.post("/predict")
def predict(request: PredictionRequest, x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """Return the object (predictionRequest) as a dictionary."""
    require_api_key(x_api_key)
    prediction_input = request.model_dump()
    
    result, monitoring_data = make_prediction(prediction_input, model=_model, preprocessor=_preprocessor)
    log_prediction(monitoring_data, result)
    PREDICTION_COUNT.labels(risk_label=result["risk_label"]).inc()
    PREDICTION_HISTOGRAM.observe(result["default_probability"])
    return result


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    """Return active model metadata."""
    result = _model_info
    if result:
        return result 
    raise HTTPException(status_code=404, detail="Model info does not exist.")


@app.post("/monitor")
def monitor_drift(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """Run drift detection and return results."""
    require_api_key(x_api_key)
    result = run_drift_check()
    global _drift_event_active
    if not result["drift_detected"]:
        _drift_event_active = False
    elif not _drift_event_active:
        _drift_event_active = True
        training_result = train_candidate(persist_local_artifacts=False)
        promoted = promote_if_better(
            training_result["version"],
            training_result["run_id"],
        )
        if promoted:
            update_local_fallback()
            reset_prediction_log()
        result["retraining"] = training_result
        result["promoted"] = promoted
    return result


@app.post("/retrain")
def retrain_model(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """Trigger model retraining."""
    require_api_key(x_api_key)
    return monitor_drift(x_api_key)


@app.get("/monitoring/stats")
def monitoring_stats_endpoint() -> dict[str, Any]:
    """Return summary statistics from the prediction log."""
    return calculate_monitoring_stats()


@app.get("/ready")
def ready() -> dict[str, str]:
    """Return readiness status (process alive AND model loaded)."""
    try:
        if _model is not None:
            return {"status": "ready"}
        return {"status": "not_ready", "detail": "Model not yet loaded"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.get("/system/status")
def system_status() -> dict[str, Any]:
    """Return a comprehensive system health status."""
    results: dict[str, Any] = {
        "api": {"status": "operational", "detail": "API is running"},
        "model": {"status": "operational", "detail": "Model is loaded"},
        "mlflow": {"status": "operational", "detail": "MLflow is reachable"},
        "monitoring": {"status": "operational", "detail": "Monitoring is active"}
    }

    # 1. Model Check
    if _model is None:
        results["model"] = {"status": "failure", "detail": "Model not loaded"}
    elif _model_info is None:
        results["model"] = {"status": "warning", "detail": "Model loaded but metadata missing"}

    # 2. Monitoring Check
    try:
        calculate_monitoring_stats()
    except Exception as e:
        results["monitoring"] = {"status": "failure", "detail": str(e)}

    # 3. MLflow Check
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
    except Exception as e:
        results["mlflow"] = {"status": "warning", "detail": f"MLflow connectivity issue: {str(e)}"}

    return results
