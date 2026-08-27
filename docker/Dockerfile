# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user and required runtime directories.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/artifacts /app/data/raw /app/reports \
    && chown -R appuser:appuser /app/artifacts /app/data/raw /app/reports /app/reports \
    && chown -R appuser:appuser /app/artifacts /app/data/raw /app/reports

WORKDIR /app

# Install only the runtime dependencies the API actually imports.
COPY --chown=appuser:appuser requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy only the app code and runtime dependencies.
# The production API prefers MLflow model artifacts and does not require local
# generated artifacts or the raw training CSV to be baked into the image.
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/

USER appuser

EXPOSE 8000

# Healthcheck uses the stdlib only; no curl installation needed.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready').read()"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
