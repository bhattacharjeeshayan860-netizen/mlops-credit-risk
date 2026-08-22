"""Tests for the FastAPI inference service."""

import sys
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient
import api.main as api_main
from src.preprocessing import CreditRiskPreprocessor
from src.train import build_model
from src.monitor import log_prediction_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




@pytest.fixture
def client(monkeypatch):
    """Create a FastAPI test client that runs startup/shutdown events."""
    training_data = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.8, 0.3, 0.6],
            "age": [30, 45, 60, 25],
            "NumberOfTime30_59DaysPastDueNotWorse": [0, 2, 0, 1],
            "DebtRatio": [0.2, 0.7, 0.3, 0.5],
            "MonthlyIncome": [4000.0, 5200.0, 7000.0, 2500.0],
            "NumberOfOpenCreditLinesAndLoans": [3, 8, 10, 2],
            "NumberOfTimes90DaysLate": [0, 1, 0, 2],
            "NumberRealEstateLoansOrLines": [0, 1, 2, 0],
            "NumberOfTime60_89DaysPastDueNotWorse": [0, 0, 1, 0],
            "NumberOfDependents": [0.0, 2.0, 1.0, 3.0],
        }
    )
    preprocessor = CreditRiskPreprocessor().fit(training_data)
    model = build_model().fit(preprocessor.transform(training_data), [0, 1, 0, 1])
    model_info = {
        "model_type": "LogisticRegression",
        "version": "test",
        "trained_at": "test",
    }
    monkeypatch.setattr(api_main, "load_resources", lambda: (model, preprocessor, model_info))
    monkeypatch.setattr(api_main, "load_model_info", lambda: model_info)

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client) -> None:
    """The health endpoint should confirm that the API is running."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inference_endpoint_returns_prediction(client) -> None:
    """The prediction endpoint should return a valid inference payload."""
    """used to create a rolling buffer of 1000 recent requests for drift detection
    for i in range(1005):
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
    

        result, X = make_prediction(
        prediction_input,
        model=_model,
        preprocessor=_preprocessor
        )

        log_prediction_input(X)"""
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