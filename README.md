# Credit Risk Sentinel

Production-minded MLOps case study: real-time credit-risk inference, monitoring, and automated retraining.

This project shows how a notebook-grade ML idea becomes a service you can actually trust: reproducible training, durable preprocessing artifacts, containerized FastAPI inference, Prometheus/Grafana observability, Evidently AI drift detection, and a GitHub Actions-driven retraining path. It is shaped as a portfolio piece for DS, ML, and MLOps roles where hiring teams look for engineering judgment, not just model accuracy.

## Project Motive

Most student ML projects prove that a model can be trained. Fewer prove that it can be served, monitored, and maintained.

This repository closes that gap.

The business problem is simple: a financial institution wants to estimate whether a customer is likely to default within two years. The engineering problem is harder: that prediction has to be reproducible, robust to missing values and outliers, observable in production, and ready for drift-driven retraining.

That is the story this project tells.

## Recruiter / Hiring Signal

The message should be immediate: this is not a notebook dump or a single `model.pkl` file. It is a system.

The project demonstrates that I can structure ML work like software, move beyond experiments, and think in terms of reliability, maintainability, and operational impact.

## What Makes This Stand Out

- End-to-end production architecture: training → artifact versioning → containerized API → monitoring → retraining hook
- Reproducible stratified train/validation/test splits with class-aware model fitting
- Preprocessing learned from training data and reused at inference time via saved artifacts
- Saved artifacts for model, preprocessor, metrics, reference data, and imputation values
- FastAPI service with `/predict`, `/health`, `/metrics`, `/retrain`, and `/model/info` endpoints
- Prometheus metrics ingestion and Grafana dashboards for operational visibility
- Evidently AI drift detection comparing incoming requests against a held-out reference dataset
- GitHub Actions CI/CD for testing, building, and retraining automation
- Clean project structure that mirrors a deployable MLOps system

## What It Does

The model predicts the `SeriousDlqin2yrs` target from the Give Me Some Credit dataset.

Training and inference behavior:

- Binary classification on 150,000 records (~7% positive class)
- Logistic regression with `class_weight="balanced"` to handle default-rate imbalance
- Missing-value handling for `MonthlyIncome`, `NumberOfDependents`, and invalid `age` values using training-derived medians
- Outlier clipping for heavy-tailed credit features such as `RevolvingUtilizationOfUnsecuredLines`
- Saved preprocessing artifacts (`medians.json`, `reference.csv`, `preprocessor.pkl`) reused at inference time
- A held-out reference dataset for Evidently AI drift comparison
- Target: 85%+ ROC-AUC on a stratified held-out test set

## Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python |
| Data | pandas, NumPy |
| Modeling | scikit-learn, XGBoost |
| API | FastAPI, Uvicorn |
| Experiment Tracking | MLflow |
| Artifact Handling | joblib, JSON |
| Monitoring | Prometheus, Grafana |
| Drift Detection | Evidently AI |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

## Dataset

Dataset: [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)

Expected raw data files:

- `data/raw/cs-training.csv`
- `data/raw/Data Dictionary.xls`

The dataset contains 10 financial and credit-history features used to predict whether a borrower will experience 90-days-past-due delinquency within two years.

## System View

```text
Customer Request
  |
  v
FastAPI Inference Layer
  |
  v
Loaded Model + Preprocessing Artifact
  |
  v
Risk Score + Prediction
  |
  v
Metrics Endpoint  ---->  Prometheus  ---->  Grafana
  |
  v
Request Buffer  ---->  Evidently AI Drift Report
  |
  v
Retraining Trigger  ---->  GitHub Actions  ---->  New MLflow Model Version
```

Prometheus scrapes the `/metrics` endpoint every 15 seconds. Grafana turns those signals into operational dashboards. Evidently AI compares the rolling request buffer against the saved reference dataset to detect data drift.

## Docker Architecture

Four services orchestrated via Docker Compose on a shared internal network:

| Service | Port | Responsibility |
| --- | --- | --- |
| `fastapi-app` | 8000 | Prediction API and metrics endpoint |
| `mlflow-server` | 5000 | Experiment and model-version tracking |
| `prometheus` | 9090 | Metrics scraping and storage |
| `grafana` | 3000 | Dashboard visualization |

Shared volumes:

- `artifacts/` — model, preprocessor, reference data, and median values (shared between FastAPI and MLflow)
- `mlflow-db/` — MLflow backend store
- `grafana-data/` — persisted dashboards

Containers communicate by service name, not IP.

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
├── grafana-data/
├── mlflow/
├── mlflow-db/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── monitor.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── FILE_GUIDE.md
├── PROJECT_TASK.md
├── README.md
└── requirements.txt
```

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

## Run It Locally

Train the model:

```bash
python src/train.py
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Run tests:

```bash
pytest
```

## Run the Full Container Stack

```bash
docker compose -f docker/docker-compose.yml up --build
```

Access points:

- API: http://localhost:8000
- API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Portfolio Value

If you are reviewing this as a hiring signal, this project is meant to communicate that I can:

- Move from data prep to a service-oriented ML workflow
- Keep training and inference preprocessing consistent through saved artifacts
- Think in terms of artifacts, versions, and observability rather than notebooks
- Build toward monitoring, drift detection, and automated retraining instead of stopping at model fit
- Present ML work in a readable, maintainable, and production-aware way

In a high-signal interview, this gives a clean story: I can build, package, validate, and operate ML work instead of only training it.

## Author

Final-year BCA student building practical DS, ML, and MLOps projects for strong fresher roles in India.

Shayan Bhattacharjee
