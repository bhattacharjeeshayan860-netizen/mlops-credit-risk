"""Tests for the FastAPI inference service."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
import api.main as api_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def client(monkeypatch):
    """Create a FastAPI test client that runs startup/shutdown events."""
    preprocessor = Mock()
    preprocessor.transform.side_effect = lambda frame: frame
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
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
        "NumberOfDependents": 2,
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


def test_monitoring_stats_returns_200(client) -> None:
    """The monitoring stats endpoint should return 200 OK."""
    response = client.get("/monitoring/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data


def test_monitoring_stats_empty_log(client, monkeypatch) -> None:
    """The monitoring stats endpoint should return zero-valued metrics when the log is empty."""
    import pandas as pd
    monkeypatch.setattr("src.monitor.load_prediction_log", lambda: pd.DataFrame())
    
    response = client.get("/monitoring/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 0
    assert data["high_risk_rate"] == 0.0
    assert data["avg_default_probability"] == 0.0
    assert data["recent_volume"] == 0


def test_ready_endpoint(client) -> None:
    """The ready endpoint should return the correct readiness state."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert "status" in response.json()


def test_cors_allows_vite_origin(client) -> None:
    """CORS should allow requests from the Vite frontend origin."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_system_status_endpoint(client) -> None:
    """The system status endpoint should return the expected structure."""
    response = client.get("/system/status")
    assert response.status_code == 200
    data = response.json()
    for key in ["api", "model", "mlflow", "monitoring"]:
        assert key in data
        assert "status" in data[key]
        assert "detail" in data[key]


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
