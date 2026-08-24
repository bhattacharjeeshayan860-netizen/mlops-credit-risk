"""FastAPI application for real-time credit-risk inference.

Planned endpoints:
- GET /health: service health check
- POST /predict: model prediction with confidence score
- GET /metrics: Prometheus metrics endpoint
- POST /retrain: manual or automated retraining trigger
- GET /model/info: active model metadata
"""
from pydantic import BaseModel
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from src.monitor import log_prediction, reset_prediction_log, run_drift_check, get_monitoring_stats as calculate_monitoring_stats
from src.predict import load_resources, make_prediction, load_model_info, promote_if_better, reload_resources, update_local_fallback
from src.utils import MLFLOW_TRACKING_URI
from src.train import train_candidate
import mlflow

class PredictionRequest(BaseModel):
    """Request model for input data."""
    # Define the expected fields for the prediction request    
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: Optional[float]= None
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents : Optional[float]= None

PREDICTION_COUNT = Counter(
                            "api_prediction_requests_total",
                            "credit_risk_prediction_total",
                            ["risk_label"]
)
PREDICTION_HISTOGRAM = Histogram("credit_default_probability",
                                "Distribution of predicted credit default probabilities",
                                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


app = FastAPI(
    title="Real-Time ML Inference API",
    description="Credit-risk prediction API with monitoring and drift detection.",
    version="0.1.0",
)

_model = None
_preprocessor = None
_model_info = None

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

_drift_event_active = False

Instrumentator().instrument(app).expose(app)
@app.on_event("startup")
def startup_event():
    """Load model and preprocessor into memory at startup."""
    global _model, _preprocessor, _model_info
    _model,_preprocessor,_model_info= load_resources()

@app.post("/predict")
def predict(request: PredictionRequest)-> dict:
    """Return the object (predictionRequest) as a dictionary."""
    prediction_input= request.model_dump()
    
    result, monitoring_data = make_prediction(prediction_input, model=_model, preprocessor=_preprocessor)
    log_prediction(monitoring_data, result)
    PREDICTION_COUNT.labels(risk_label=result["risk_label"]).inc()
    PREDICTION_HISTOGRAM.observe(result["default_probability"])
    return result

@app.get("/model/info")
def model_info()-> dict:
    """Return active model metadata."""
    result = _model_info
    if result:
        return result 
    raise HTTPException(status_code=404, detail="Model info does not exist.")

@app.post("/monitor")
def monitor_drift() -> dict:
    """Run drift detection and return results."""
    result= run_drift_check()
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
            reload_resources()
            update_local_fallback()
            reset_prediction_log()
        result["retraining"] = training_result
        result["promoted"] = promoted
    return result

@app.post("/retrain")
def retrain_model() -> dict:
    """Trigger model retraining."""
    return monitor_drift()

@app.get("/monitoring/stats")
def monitoring_stats_endpoint() -> dict:
    """Return summary statistics from the prediction log."""
    return calculate_monitoring_stats()

@app.get("/ready")
def ready() -> dict[str, str]:
    """Return readiness status (process alive AND model loaded)."""
    try:
        from src.predict import _model
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
def system_status() -> dict:
    """Return a comprehensive system health status."""
    results = {
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
