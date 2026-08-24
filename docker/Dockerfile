# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user and required runtime directories.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/artifacts /app/data/raw

WORKDIR /app

# Install only the runtime dependencies the API actually imports.
COPY --chown=appuser:appuser requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy only the runtime code and artifacts needed by the API.
# The app supports MLflow-first loading with a local fallback, and drift-triggered
# retraining still reads the raw training CSV from data/raw/cs-training.csv.
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/raw/cs-training.csv ./data/raw/cs-training.csv
COPY --chown=appuser:appuser artifacts/model.pkl ./artifacts/model.pkl
COPY --chown=appuser:appuser artifacts/preprocessor.pkl ./artifacts/preprocessor.pkl
COPY --chown=appuser:appuser artifacts/model_info.json ./artifacts/model_info.json
COPY --chown=appuser:appuser artifacts/reference.csv ./artifacts/reference.csv

USER appuser

EXPOSE 8000

# Healthcheck uses the stdlib only; no curl installation needed.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
