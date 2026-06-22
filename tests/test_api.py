"""Tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    """The health endpoint should confirm that the API is running."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
