# Project Task

## Task Name

Real-Time ML Inference API with Monitoring and Drift Detection

## What This Project Is About

This project builds a production-style machine learning system for credit-risk prediction. It uses the Kaggle Give Me Some Credit dataset to train a binary classification model, then serves that model through a FastAPI REST API.

The larger goal is to demonstrate the complete MLOps lifecycle:

- Train and evaluate a credit-risk model
- Save preprocessing artifacts required for inference
- Serve predictions through a REST API
- Track experiments and model versions with MLflow
- Monitor API metrics with Prometheus and Grafana
- Detect data drift with Evidently AI
- Trigger retraining when drift is detected
- Use Docker Compose to run the system locally
- Use GitHub Actions for testing and CI/CD

## Target Outcome

By the end of the build, the repository should look like a professional portfolio project suitable for DS/ML fresher roles, especially roles that value practical MLOps knowledge.

The project should show that the author can move from notebook experimentation to a deployable, monitored ML service.

## Current Week: Week 3 — MLflow Integration (in progress)

### Goals for this week

- Connect training runs to MLflow experiment tracking.
- Log model, parameters, metrics, and artifacts under a named experiment.
- Make the FastAPI API load the latest model/preprocessor from MLflow by default, with a local-file fallback.
- Keep a local-file fallback so the API still works when MLflow is unreachable.
- Ensure model metadata (`model_info.json`) includes the MLflow run ID and version.
- Keep Docker Compose working with the MLflow server on the internal network.

### Active problems / fixes needed

1. **MLflow tracking URI is empty** in `src/utils.py`. Use an environment variable with sensible defaults:
   - Local Python run: `http://127.0.0.1:5000`
   - Inside Docker: `http://mlflow-server:5000`
2. **Syntax error** in `src/train.py` line 162 (`model.named_steps["classifier"` is split across lines incorrectly).
3. **Wrong MLflow metric type**: `confusion_matrix` is being logged as a metric; it should be logged as an artifact (JSON) or printed only.
4. **Duplicate imports** of `datetime` and `time` in `src/train.py`.
5. **Bug in `load_model_from_mlflow`**: `get_latest_run_id()` already returns a string, but the code later tries `.empty` on that string.
6. **Bug in `load_preprocessor_from_mlflow`**: downloads a directory path, then passes the path string directly to `joblib.load()` instead of joining with `preprocessor.pkl`.
7. **Performance issue**: model and preprocessor are loaded from MLflow on every single `/predict` request. Load once at API startup and refresh only when needed.
8. **Missing `/retrain` endpoint** in `api/main.py` (planned but not implemented).
9. **Docker Compose**: `fastapi-app` needs `MLFLOW_TRACKING_URI` environment variable set to `http://mlflow-server:5000`.

### Deliverables to verify

- `python src/train.py` completes and creates an MLflow run under experiment `credit_risk_model`.
- `mlflow ui` (or the Docker MLflow server) shows the run with parameters, metrics, and artifacts.
- `uvicorn api.main:app` starts and `/predict` returns a valid prediction.
- `docker-compose up` brings up FastAPI + MLflow together and `/model/info` returns metadata.

## Build Roadmap

- Week 1 — EDA + baseline training (done)
- Week 2 — FastAPI `/predict` + `/health` + local file fallback (done)
- Week 3 — MLflow experiment tracking + model loading from MLflow (current)
- Week 4 — Docker + Docker Compose for FastAPI + MLflow
- Week 5 — Prometheus + Grafana monitoring
- Week 6 — Evidently AI drift detection
- Week 7 — GitHub Actions CI/CD + deployment
