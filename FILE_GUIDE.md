# File Guide

This guide explains what each file and folder is for.

## `data/raw/`

Stores the raw Kaggle dataset files before preprocessing. Raw data should not be modified directly.

Expected local files:

- `cs-training.csv`: training dataset from Kaggle
- `Data Dictionary.xls`: feature descriptions provided with the dataset

## `notebooks/01_eda.ipynb`

Exploratory data analysis notebook. This is where missing values, class imbalance, outliers, feature distributions, and baseline model behavior will be studied.

## `src/train.py`

Training pipeline entry point. It will load raw data, apply preprocessing, train the model, evaluate performance, save artifacts, and log runs to MLflow.

## `src/predict.py`

Prediction logic used by the API. It will load the trained model and preprocessing artifacts, apply inference-time preprocessing, and return predictions with confidence scores.

## `src/monitor.py`

Monitoring and drift detection logic. It will manage incoming request logs, compare live data against the reference dataset, and generate Evidently drift reports.

## `src/utils.py`

Shared helper functions for file loading, artifact saving, JSON handling, feature validation, and configuration.

## `api/main.py`

FastAPI application entry point. It will expose `/health`, `/predict`, `/metrics`, `/retrain`, and `/model/info`.

## `docker/Dockerfile`

Container image definition for the FastAPI application.

## `docker/docker-compose.yml`

Local orchestration file for FastAPI, MLflow, Prometheus, and Grafana.

## `docker/prometheus.yml`

Prometheus scrape configuration. It tells Prometheus to collect metrics from the FastAPI service every 15 seconds.

## `.github/workflows/ci.yml`

GitHub Actions workflow for running automated checks such as tests and linting.

## `tests/test_api.py`

API tests for endpoints such as `/health` and `/predict`.

## `mlflow/`

Reserved for MLflow-related configuration and local metadata when needed.

## `artifacts/`

Stores generated model and preprocessing artifacts such as `model.pkl`, `reference.csv`, and `medians.json`.

## `requirements.txt`

Python dependencies required to train the model, run the API, test the project, and support monitoring.
