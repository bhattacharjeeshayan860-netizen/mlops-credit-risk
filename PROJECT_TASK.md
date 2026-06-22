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
