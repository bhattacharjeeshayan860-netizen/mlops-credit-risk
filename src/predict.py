"""Prediction utilities shared by the FastAPI service.

This module loads the trained model and preprocessing artifacts from MLflow
(or from a local fallback), applies inference-time preprocessing, and returns
prediction probabilities.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import CreditRiskPreprocessor
from src.utils import MLFLOW_ARTIFACT_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def _import_mlflow():
    """Lazy-import MLflow so the API can start without it when using local artifacts."""
    import mlflow
    import mlflow.sklearn  # noqa: F401
    return mlflow

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_THRESHOLD = 0.5

# Module-level cache. Populated once by load_resources() when the API starts.
_model: Pipeline | None = None
_preprocessor: CreditRiskPreprocessor | None = None
_model_info: dict[str, Any] | None = None

def load_champion_model_preprocessor() -> tuple:
    """Load the champion model and preprocessor from MLflow.

    Returns:
        tuple: A tuple containing the loaded model and preprocessor.
    """
    mlflow= _import_mlflow()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    Client= MlflowClient()

    try:
        model_version= Client.get_model_version_by_alias(name= MLFLOW_EXPERIMENT_NAME,alias= "champion")
    except Exception as e:
        logger.error(f"Failed to retrieve champion model version from MLflow: {e}")
        raise RuntimeError("Failed to retrieve champion model version from MLflow.")
    if model_version.run_id is None:
        raise RuntimeError("Model version has no associated run_id.")
    run_id= model_version.run_id
    version_num=model_version.version
    print(f"Loading Champion Version: {version_num} (Run ID: {run_id})")
    model_uri= f"models:/{MLFLOW_EXPERIMENT_NAME}@champion"
    model= mlflow.sklearn.load_model(model_uri)

    # Load the preprocessor artifact from the same run
    local_dir= Client.download_artifacts(run_id=run_id, path="preprocessor/preprocessor.pkl")
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Preprocessor artifact not found at {local_dir}")
    preprocessor= joblib.load(local_dir)

    return model, preprocessor

def get_champion_metrics() -> dict[str, Any]:
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=MLFLOW_EXPERIMENT_NAME,
        alias="champion",
    )
    if model_version.run_id is None:
        raise RuntimeError("Model version has no associated run_id.")
    run = client.get_run(model_version.run_id)

    return {
        "run_id": model_version.run_id,
        "version": model_version.version,
        "val_roc_auc": run.data.metrics["val_roc_auc"],
    }

def get_candidate_metrics(run_id: str) -> dict[str, Any]:
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(run_id)

    return {
        "run_id": run_id,
        "val_roc_auc": run.data.metrics["val_roc_auc"],
    }

def set_registered_model_alias(new_version: str) -> None:
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    client.set_registered_model_alias(
        MLFLOW_EXPERIMENT_NAME,
        "champion",
        new_version,
    )

def promote_if_better(candidate_version: str, candidate_run_id: str) -> bool:
    champion = get_champion_metrics()
    candidate = get_candidate_metrics(candidate_run_id)

    if candidate["val_roc_auc"] > champion["val_roc_auc"]:
        set_registered_model_alias(candidate_version)
        logger.info(
            "Promoted version %s over version %s.",
            candidate_version,
            champion["version"],
        )
        return True

    logger.info(
        "Kept champion version %s because candidate version %s was not better.",
        champion["version"],
        candidate_version,
    )
    return False

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

    try:
        logger.info("trying MLflow.")
        _model,_preprocessor= load_champion_model_preprocessor()
        _model_info = load_model_info()
        logger.info("Loaded model and preprocessor from MLflow.")

    except Exception:
        logger.exception("Failed to load from MLflow, falling back to local artifacts.")
        _model = joblib.load(ARTIFACTS_DIR / "model.pkl")
        _preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
        _model_info = load_model_info()
        logger.info("Loaded model and preprocessor from local artifacts.")

    return _model, _preprocessor, _model_info

def reload_resources() -> tuple:
    """Reload model and preprocessor from MLflow."""
    try:
        logger.info("Reloading model and preprocessor from MLflow.")
        model, preprocessor = load_champion_model_preprocessor()
        model_info = load_model_info()
        global _model, _preprocessor, _model_info
        _model, _preprocessor, _model_info = model, preprocessor, model_info
        logger.info("Reloaded model and preprocessor from MLflow.")
    except Exception:
        logger.exception("Failed to reload from MLflow.")
    return _model, _preprocessor,_model_info
def make_prediction(
    input_data: dict[str, Any],
    model: Pipeline | None = None,
    preprocessor: CreditRiskPreprocessor | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
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
    result = {
        "prediction": prediction,
        "default_probability": default_probability,
        "risk_label": risk_label,
        "model_version": _model_info.get("version") if _model_info else None,
        "mlflow_run_id": _model_info.get("mlflow_run_id") if _model_info else None,
    }


    return result,X
