"""Prediction utilities shared by the FastAPI service.

This module loads the trained model and preprocessing artifacts from MLflow
(or from a local fallback), applies inference-time preprocessing, and returns
prediction probabilities.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import CreditRiskPreprocessor
from src.utils import MLFLOW_ARTIFACT_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_THRESHOLD = 0.5

# Module-level cache. Populated once by load_resources() when the API starts.
_model: Pipeline | None = None
_preprocessor: CreditRiskPreprocessor | None = None
_model_info: dict[str, Any] | None = None


def get_latest_run_id() -> str:
    """Return the most recent run ID for the configured MLflow experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No runs found for experiment: {MLFLOW_EXPERIMENT_NAME}")
    return runs.iloc[0]["run_id"]


def load_model_from_mlflow(run_id: str | None = None) -> Pipeline:
    """Load a trained sklearn model from MLflow artifacts."""
    if run_id is None:
        run_id = get_latest_run_id()
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{MLFLOW_ARTIFACT_PATH}")


def load_preprocessor_from_mlflow(run_id: str | None = None) -> CreditRiskPreprocessor:
    """Load the fitted preprocessor from MLflow artifacts."""
    if run_id is None:
        run_id = get_latest_run_id()
    local_dir = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor")
    return joblib.load(Path(local_dir) / "preprocessor.pkl")


def load_model_info() -> dict[str, Any]:
    """Load model metadata saved at training time."""
    path = ARTIFACTS_DIR / "model_info.json"
    if not path.exists():
        raise FileNotFoundError(f"Model info not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_resources() -> tuple[Pipeline, CreditRiskPreprocessor, dict[str, Any]]:
    """Load model + preprocessor once, preferring MLflow, falling back to local files.

    This function is called by the API at startup. The returned objects are also
    cached as module-level globals so helper functions can use them directly.
    """
    global _model, _preprocessor, _model_info

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        _model = load_model_from_mlflow()
        _preprocessor = load_preprocessor_from_mlflow()
        _model_info = load_model_info()
        logger.info("Loaded model and preprocessor from MLflow.")
    except Exception:
        logger.exception("MLflow load failed; falling back to local artifacts.")
        _model = joblib.load(ARTIFACTS_DIR / "model.pkl")
        _preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
        _model_info = load_model_info()

    return _model, _preprocessor, _model_info


def make_prediction(
    input_data: dict[str, Any],
    model: Pipeline | None = None,
    preprocessor: CreditRiskPreprocessor | None = None,
) -> dict[str, Any]:
    """Return a credit-risk prediction for one API request payload.

    The `model` and `preprocessor` arguments are optional. If they are not
    provided, the function falls back to the module-level cache loaded by
    `load_resources()`.
    """
    model = model if model is not None else _model
    preprocessor = preprocessor if preprocessor is not None else _preprocessor

    if model is None or preprocessor is None:
        raise RuntimeError("Model and preprocessor are not loaded. Call load_resources() first.")

    df = pd.DataFrame([input_data])
    X = preprocessor.transform(df)

    default_probability = float(model.predict_proba(X)[:, 1][0])
    prediction = 1 if default_probability >= DEFAULT_THRESHOLD else 0
    risk_label = "high_risk" if prediction == 1 else "low_risk"

    return {
        "prediction": prediction,
        "default_probability": default_probability,
        "risk_label": risk_label,
        "model_version": _model_info.get("version") if _model_info else None,
        "mlflow_run_id": _model_info.get("mlflow_run_id") if _model_info else None,
    }
