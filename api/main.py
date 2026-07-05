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
from src.predict import make_prediction, load_model_info

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


app = FastAPI(
    title="Real-Time ML Inference API",
    description="Credit-risk prediction API with monitoring and drift detection.",
    version="0.1.0",
)

@app.post("/predict")
def predict(request: PredictionRequest)-> dict:
    """Return the object (predictionRequest) as a dictionary."""
    prediction_input= request.model_dump()
    result= make_prediction(prediction_input)
    return result

@app.get("/model/info")
def model_info()-> dict:
    """Return active model metadata."""
    result= load_model_info()
    if result:
        return result 
    raise ValueError("Model info does not exist.")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
