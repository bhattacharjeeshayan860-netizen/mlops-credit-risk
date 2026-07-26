"""Tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    """The health endpoint should confirm that the API is running."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_inference_endpoint_returns_prediction() -> None:
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.76,
        "age": 45,
        "NumberOfTime30_59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.34,
        "MonthlyIncome": 5200,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }

    response = client.post("/predict",json=payload)
    assert response.status_code == 200

    data =response.json()
    assert "prediction" in data
    assert "default_probability" in data
    assert "risk_label" in data
def test_model_info_endpoint_returns_metadata() -> None:
    response =client.get("/model/info")
    assert response.status_code == 200
    data = response.json()

    assert "model_type" in data
    assert "version" in data
    assert "trained_at" in data

def test_metrics_endpoint_is_available() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
