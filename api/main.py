"""FastAPI application for real-time credit-risk inference.

Planned endpoints:
- GET /health: service health check
- POST /predict: model prediction with confidence score
- GET /metrics: Prometheus metrics endpoint
- POST /retrain: manual or automated retraining trigger
- GET /model/info: active model metadata
"""

from fastapi import FastAPI


app = FastAPI(
    title="Real-Time ML Inference API",
    description="Credit-risk prediction API with monitoring and drift detection.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
