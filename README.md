# mloops: End-to-End MLOps Pipeline for Credit Risk Modeling

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/docker-compose-blue?logo=docker)](https://docs.docker.com/compose/)

`mloops` is a production-oriented MLOps framework designed to manage the complete lifecycle of a credit-risk prediction model. It bridges the gap between experimental machine learning and reliable real-time serving by integrating experiment tracking, stateful preprocessing, automated observability, and containerized deployment.

---

## 🏗 Architecture

The system is composed of a modular stack designed for scalability and observability:

```mermaid
graph TD
    subgraph "Training & Registry"
        A[Raw Data] --> B[Training Pipeline]
        B --> C{MLflow Server}
        C -->|Store| D[Model Registry]
        C -->|Store| E[Experiment Metrics]
    end

    subgraph "Inference Service"
        D --> F[FastAPI Service]
        F -->|Load| G[Preprocessing Artifacts]
        H[Live Requests] --> F
        F -->|Return| I[Risk Score]
    end

    subgraph "Observability Stack"
        F -->|Expose| J[Prometheus Metrics]
        J -->|Scrape| K[Prometheus Server]
        K -->|Visualize| L[Grafana Dashboards]
        F -->|Log Data| M[Prediction Log]
        M -->|Drift Detection| N[Evidently AI]
        N -->|Report| O[Drift Reports]
    end
```

### Current Status

| Feature | Status |
| :--- | :--- |
| **ML Training Pipeline** | ✅ Implemented |
| **MLflow Tracking & Registry** | ✅ Implemented |
| **FastAPI Inference API** | ✅ Implemented |
| **Prometheus Metrics Integration** | ✅ Implemented |
| **Grafana Dashboard Support** | ✅ Implemented |
| **Evidently Drift Detection** | ✅ Implemented, manually triggered |
| **Dockerized Stack** | ✅ Implemented |
| **Frontend Dashboard** | ⏳ Planned |
| **Automated Retraining Loop** | ⏳ Planned |
| **Authentication & HTTPS** | ⏳ Planned |

---

## 🛠 Setup & Installation

### Prerequisites
- **Windows**: PowerShell 5.1+
- **Linux/macOS**: Bash
- **Docker & Docker Compose** installed and running

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mloops
   ```

2. **Set up a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   First, ensure an MLflow server is running to receive logs:
   ```bash
   # Using Docker to run only the MLflow server
   docker compose -f docker/docker-compose.yml up -d mlflow-server

   # Run the training script
   python src/train.py
   ```
   *Note: The API is designed to fall back to local `artifacts/` if the MLflow server is unavailable.*

5. **Start the Inference API**
   ```bash
   uvicorn api.main:app --reload
   ```

6. **Run Tests**
   ```bash
   pytest
   ```

### Docker Compose (Full Stack)
Launch the entire MLOps stack (API, MLflow, Prometheus, Grafana) with one command:
```bash
docker compose -f docker/docker-compose.yml up -d --build
```

---

## 🔌 API Reference

The inference service is available at `http://localhost:8000`.

### Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check. |
| `POST` | `/predict` | Returns a credit-risk prediction and probability. |
| `GET` | `/model/info` | Returns metadata of the currently loaded model. |
| `POST` | `/monitor` | Triggers a drift detection report. |
| `GET` | `/metrics` | Prometheus scrape endpoint for API metrics. |

### Example Request: `/predict`

**Payload:**
```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.32,
  "age": 45,
  "NumberOfTime30_59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.15,
  "MonthlyIncome": 5000.0,
  "NumberOfOpenCreditLinesAndLoans": 12,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60_89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```

**Response (Example; values will vary depending on the loaded model and input):**
```json
{
  "prediction": 0,
  "default_probability": 0.042,
  "risk_label": "low_risk",
  "model_version": "0.1.0",
  "mlflow_run_id": "abc123def456"
}
```

---

## ⚙️ Configuration

Create a `.env` file in the root directory.

**`.env.example` content:**
```env
# API Configuration
FASTAPI_PORT=8000

# MLflow Configuration
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=credit_risk_model
MLFLOW_ARTIFACT_PATH=model
MLFLOW_PORT=5000

# Monitoring & Observability
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change-me
```

---

## 🐳 Docker Service Summary

| Service | Port | Description |
| :--- | :--- | :--- |
| `fastapi-app` | `8000` | Real-time inference API |
| `mlflow-server` | `5000` | Experiment tracking and registry |
| `prometheus` | `9090` | Metrics scraper |
| `grafana` | `3000` | Dashboard visualization |

---

## 🚀 Deployment Guide (Linux VPS)

To deploy this stack on a cloud instance:

1. **Clone and Configure**
   ```bash
   git clone <repository-url>
   cd mloops
   cp .env.example .env
   nano .env  # Update for production (e.g., change GRAFANA_ADMIN_PASSWORD)
   ```

2. **Launch Stack**
   ```bash
   docker compose -f docker/docker-compose.yml up -d --build
   ```

3. **Verify Health**
   ```bash
   docker compose -f docker/docker-compose.yml ps
   curl http://localhost:8000/health
   ```

4. **Persistence & Storage**
   The following storage mechanisms are in place:
   - **MLflow database and artifacts**: Persisted through bind mounts.
   - **Grafana data**: Persisted through a named Docker volume (`grafana-data`).
   - **API artifacts**: Currently bundled into the Docker image.

5. **Maintenance**
   - **View logs**: `docker compose -f docker/docker-compose.yml logs fastapi-app`
   - **Update deployment**: `docker compose -f docker/docker-compose.yml pull && docker compose -f docker/docker-compose.yml up -d`

---

## 🛡️ Production Checklist

- [ ] **HTTPS/TLS**: Use a reverse proxy (Nginx/Traefik) with Let's Encrypt.
- [ ] **Authentication**: Secure `/predict` and `/monitor` endpoints.
- [ ] **CORS**: Restrict access to your specific frontend domain.
- [ ] **Rate Limiting**: Prevent API abuse.
- [ ] **Secret Management**: Use a secure vault instead of plain `.env` files.
- [ ] **Backups**: Schedule regular backups of MLflow bind-mounted directories.
- [ ] **Drift Reporting**: Map the `reports/` directory to a persistent volume to view drift reports.

---

## ❓ Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **MLflow connection failures** | `MLFLOW_TRACKING_URI` mismatch | Check `.env` and ensure `mlflow-server` is running |
| **Missing model artifacts** | Model not yet trained | Run `python src/train.py` |
| **API startup failures** | Port conflict or dependency error | Check `docker logs fastapi-app` |
| **Docker health-check fails** | FastAPI service taking too long to start | Increase `start_period` in `docker-compose.yml` |
| **Empty drift logs** | Not enough prediction data collected | Log at least 1,000 requests to `prediction_log.csv` |

---

**Built by Shayan Bhattacharjee**
