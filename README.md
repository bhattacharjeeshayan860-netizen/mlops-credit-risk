# Real-Time ML Inference API with Monitoring and Drift Detection

An end-to-end MLOps project for serving a credit-risk machine learning model through a production-style API, tracking experiments with MLflow, monitoring service metrics with Prometheus/Grafana, and detecting data drift with Evidently.

This project is being built as a serious fresher portfolio project for DS/ML and MLOps roles. The goal is not only to train a model, but to show the full lifecycle: data preparation, model training, versioning, API inference, containerization, observability, drift monitoring, retraining, CI/CD, and deployment.

## Problem Statement

Financial institutions need reliable systems to estimate whether a customer is likely to default in the next two years. A model is useful only when it can be trained reproducibly, served through an API, monitored in production, and updated when data changes.

This project solves that problem using the Kaggle **Give Me Some Credit** dataset.

- **Task:** Binary classification
- **Target:** `SeriousDlqin2yrs`
- **Positive class:** Customer defaulted within two years
- **Main challenge:** Class imbalance, missing values, outliers, and production drift

## Why This Project Matters

Most beginner ML projects stop at notebooks. This project is designed to go beyond that by demonstrating practical engineering skills expected in real DS/ML teams:

- Building a reproducible training pipeline
- Serving predictions using FastAPI
- Tracking experiments and model versions with MLflow
- Handling inference-time null values safely
- Monitoring live API behavior with Prometheus
- Visualizing metrics with Grafana
- Detecting drift between training data and incoming requests
- Triggering retraining workflows through GitHub Actions
- Running the full system with Docker Compose

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data & ML | pandas, NumPy, scikit-learn, XGBoost |
| API | FastAPI, Pydantic |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently AI |
| Monitoring | Prometheus, Grafana |
| Packaging | Docker, Docker Compose |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Deployment Target | Render or Railway |

## Dataset

Dataset: [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)

The dataset contains customer financial and credit-history features. The model predicts whether a customer is likely to experience serious delinquency within two years.

Expected local raw files:

- `data/raw/cs-training.csv`
- `data/raw/Data Dictionary.xls`

Key preprocessing decisions:

- Impute missing `MonthlyIncome` with the training median
- Impute missing `NumberOfDependents` with the training median
- Save training medians to `artifacts/medians.json`
- Use `class_weight="balanced"` for class imbalance
- Clip or transform extreme values in `RevolvingUtilizationOfUnsecuredLines`
- Save a held-out reference dataset to `artifacts/reference.csv` for drift comparison

## System Architecture

```text
Client
  |
  | POST /predict
  v
FastAPI Inference Service
  |
  | Loads model + preprocessing artifacts
  v
Prediction Response
  |
  | Logs incoming request data
  v
Rolling Request Buffer
  |
  | Compared with reference.csv
  v
Evidently Drift Report
  |
  | Optional retraining trigger
  v
GitHub Actions + MLflow Model Versioning
```

Prometheus scrapes the FastAPI `/metrics` endpoint every 15 seconds, and Grafana displays operational dashboards.

## API Design

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check whether the API is running |
| `POST` | `/predict` | Return prediction and confidence score |
| `GET` | `/metrics` | Expose Prometheus metrics |
| `POST` | `/retrain` | Trigger retraining manually or automatically |
| `GET` | `/model/info` | Return active model version and training metadata |

Example prediction request:

```json
{
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
```

Expected response:

```json
{
  "prediction": 0,
  "confidence": 0.91,
  "model_version": "v1"
}
```

## Repository Structure

```text
.
├── api/
│   └── main.py
├── artifacts/
├── data/
│   └── raw/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── mlflow/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── monitor.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── FILE_GUIDE.md
├── PROJECT_TASK.md
├── requirements.txt
└── README.md
```

## Build Roadmap

| Week | Milestone | Output |
| --- | --- | --- |
| 1 | EDA and baseline model | Clean notebook, trained model, 85%+ ROC-AUC target |
| 2 | FastAPI inference service | Working `/predict` endpoint |
| 3 | MLflow integration | Logged experiments and model versions |
| 4 | Docker setup | FastAPI + MLflow running through Compose |
| 5 | Monitoring | Prometheus metrics and Grafana dashboard |
| 6 | Drift detection | Evidently report comparing reference vs live data |
| 7 | Retraining + CI/CD | Automated tests, retraining workflow, deployment |

## Local Setup

```bash
git clone <repo-url>
cd mloops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Planned Commands

Train model:

```bash
python src/train.py
```

Run API:

```bash
uvicorn api.main:app --reload
```

Run tests:

```bash
pytest
```

Run full stack:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Current Status

This repository currently contains the planned production-grade structure and documentation. Implementation will be added step by step following the roadmap above.

## Portfolio Highlights

This project is intended to demonstrate:

- Strong understanding of the ML lifecycle beyond notebooks
- Practical API development for real-time inference
- Awareness of production ML risks such as drift and monitoring gaps
- Ability to structure a clean, maintainable Python ML project
- Familiarity with tools used in modern ML engineering teams

## Author

Final-year BCA student building practical DS/ML/MLOps projects for high-quality fresher roles in India.
