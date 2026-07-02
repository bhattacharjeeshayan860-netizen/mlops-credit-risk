# Credit Risk Sentinel

Real-time credit-risk inference, monitoring, and drift detection built as a production-minded MLOps case study.

This project shows how a notebook-grade ML idea becomes a service you can actually trust: reproducible training, durable preprocessing artifacts, API inference, observability, and a clean path to retraining. It is shaped as a portfolio piece for DS, ML, and MLOps roles where hiring teams look for engineering judgment, not just model accuracy.

## Project Motive

Most student ML projects prove that a model can be trained. Fewer prove that it can be served, monitored, and maintained.

This repository exists to close that gap.

The business problem is simple: a financial institution wants to estimate whether a customer is likely to default within two years. The engineering problem is harder: that prediction has to be reproducible, explainable enough to trust, robust to missing values and outliers, and ready for production concerns like drift and retraining.

That is the story this project tells.

## Recruiter View

If someone opens this repository in a hiring process, the message should be immediate: this is not just a model file, it is a system.

The project demonstrates that I can move beyond experiments, structure ML work like software, and think in terms of reliability, maintainability, and operational impact.

## Why This Stands Out

This codebase is intentionally shaped to signal real-world ML engineering ability:

- Reproducible train/validation/test splits with stratified sampling
- Explicit preprocessing learned from training data and reused at inference time
- Saved artifacts for model, preprocessor, metrics, and reference data
- A FastAPI service surface for health and prediction workflows
- Foundation for monitoring, drift detection, and retraining automation
- A project structure that looks closer to a deployable system than a notebook dump

## What It Does

The model predicts the `SeriousDlqin2yrs` target from the Give Me Some Credit dataset.

Current training and preprocessing behavior includes:

- Binary classification with logistic regression and `class_weight="balanced"`
- Missing-value handling for `MonthlyIncome`, `NumberOfDependents`, and invalid `age` values
- Feature renaming for Kaggle column names with hyphens
- Outlier clipping for heavy-tailed credit features
- Saved preprocessing artifact metadata for inference-time reuse
- A held-out reference dataset for later drift comparison

The current API surface includes a working `/health` endpoint, with the prediction and observability pieces already scaffolded in the codebase.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Language | Python |
| Data | pandas, NumPy |
| Modeling | scikit-learn |
| API | FastAPI |
| Artifact Handling | joblib, JSON |
| Monitoring | Prometheus, Grafana |
| Drift Detection | Evidently |
| Containerization | Docker, Docker Compose |
| Testing | pytest |
| Experiment Tracking | MLflow |

## Dataset

Dataset: [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)

Expected raw data files:

- `data/raw/cs-training.csv`
- `data/raw/Data Dictionary.xls`

The dataset contains financial and credit-history signals used to predict delinquency risk.

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
Risk Score + Label
  |
  v
Metrics, Drift Checks, and Retraining Hooks
```

Prometheus is intended to scrape the service metrics endpoint, while Grafana is used to turn those signals into something visible and operational.

## Repository Structure

```text
.
├── api/
│   └── main.py
├── artifacts/
├── data/
│   └── raw/
├── docker/
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

Bring up the container stack:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Portfolio Value

If you are reviewing this as a hiring signal, this project is meant to communicate that I can:

- Move from data prep to a service-oriented ML workflow
- Handle preprocessing consistently between training and inference
- Think in terms of artifacts, not just notebooks
- Build toward monitoring and retraining instead of stopping at model fit
- Present ML work in a way that is readable, maintainable, and production-aware

In a high-signal interview, this gives a clean story: I can build, package, validate, and operate ML work instead of only training it.

## Current Status

The repository is in an active build phase. The training and preprocessing foundation is in place, the API skeleton exists, and the remaining work is focused on expanding the service, monitoring, and automation layers into a complete MLOps system.

## Author

Final-year BCA student building practical DS, ML, and MLOps projects for strong fresher roles in India.
