"""Shared utility functions for the MLOps project.

This module will contain reusable helpers for reading configuration, loading and
saving JSON artifacts, validating feature order, and managing project paths.
"""

import os

from dotenv import load_dotenv


load_dotenv()

# Keep MLflow client calls fast-fail when the tracking server is not reachable
# (e.g. during local testing / CI before mlflow-server is up). The FastAPI
# service falls back to local artifacts, so a short timeout is safe.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "0")


FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30_59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60_89DaysPastDueNotWorse",
    "NumberOfDependents",
]
FASTAPI_PORT = os.getenv("FASTAPI_PORT", "8000")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5000")
PROMETHEUS_PORT = os.getenv("PROMETHEUS_PORT", "9090")
GRAFANA_PORT = os.getenv("GRAFANA_PORT", "3000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_model")
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "model")
