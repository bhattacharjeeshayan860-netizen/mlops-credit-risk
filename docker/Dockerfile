# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Create a non-root user to run the API.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy only the Docker runtime requirements so test dependencies stay out of the image.
COPY --chown=appuser:appuser requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy ONLY the files the FastAPI service needs at runtime.
# .dockerignore blocks everything else that is not needed by the service.
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser artifacts/ ./artifacts/
COPY --chown=appuser:appuser data/raw/cs-training.csv ./data/raw/cs-training.csv

USER appuser

EXPOSE 8000

# Use Python's stdlib for the healthcheck so we don't need to install curl.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"]

# Bind to 0.0.0.0 so the container is reachable from outside.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
