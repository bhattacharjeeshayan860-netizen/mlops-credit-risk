"""FastAPI application for real-time credit-risk inference.

Planned endpoints:
- GET /health: service health check
- POST /predict: model prediction with confidence score
- GET /metrics: Prometheus metrics endpoint
- POST /retrain: manual or automated retraining trigger
- GET /model/info: active model metadata
"""
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from src.monitor import log_prediction_input, reset_prediction_log, run_drift_check
from src.predict import load_resources, make_prediction, load_model_info, promote_if_better, reload_resources, update_local_fallback
from src.train import train_candidate

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
    
    result,monitoring_data= make_prediction(prediction_input, model=_model, preprocessor=_preprocessor)
    log_prediction_input(monitoring_data)
    PREDICTION_COUNT.labels(risk_label=result["risk_label"]).inc()
    PREDICTION_HISTOGRAM.observe(result["default_probability"])
    return result

@app.get("/model/info")
def model_info()-> dict:
    """Return active model metadata."""
    result = _model_info
    if result:
        return result 
    raise ValueError("Model info does not exist.")

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

@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
