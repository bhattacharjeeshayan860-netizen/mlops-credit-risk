"""Tests for the FastAPI inference service."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.monitor import log_prediction_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




@pytest.fixture
def client():
    """Create a FastAPI test client that runs startup/shutdown events."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client) -> None:
    """The health endpoint should confirm that the API is running."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inference_endpoint_returns_prediction(client) -> None:
    """The prediction endpoint should return a valid inference payload."""
    """for i in range(1005):
        test_input = {
            "RevolvingUtilizationOfUnsecuredLines": 0.50,
            "age": 20 + (i % 50),
            "NumberOfTime30_59DaysPastDueNotWorse": i % 3,
            "DebtRatio": 0.30,
            "MonthlyIncome": 4000 + i,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": i % 2,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60_89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1,
        }
    

        log_prediction_input(test_input)"""
    prediction_input = {
    "RevolvingUtilizationOfUnsecuredLines": 0.76,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.34,
    "MonthlyIncome": 5200,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
}
    response = client.post("/predict", json=prediction_input)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "default_probability" in data
    assert "risk_label" in data


def test_model_info_endpoint_returns_metadata(client) -> None:
    """The model info endpoint should expose metadata about the active model."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()

    assert "model_type" in data
    assert "version" in data
    assert "trained_at" in data


def test_metrics_endpoint_is_available(client) -> None:
    """The Prometheus metrics endpoint should be available."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))