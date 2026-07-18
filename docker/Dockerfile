# Docker image for the FastAPI inference service.
#
# This container runs the API that loads a baked-in model, serves predictions,
# and exposes operational metrics for Prometheus. It is intentionally small:
# - only runtime dependencies are installed (see requirements-api.txt)
# - only the code/artifacts needed for inference are copied
# - the service runs as an unprivileged user
# - pip cache and build tools are removed in the same layer

# Pin to a specific image digest for reproducible builds.
# (python:3.11-slim as of 18 Jul 2026; update this when you intentionally upgrade)
FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime libraries and a compiler only for the duration of pip install,
# then remove build tools in the same layer so they don't bloat the final image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt && \
    apt-get purge -y --auto-remove gcc && \
    rm -rf /root/.cache/pip

# Create an unprivileged user to run the API.
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy only the modules and model artifacts required at runtime.
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser src/predict.py src/preprocessing.py src/utils.py ./src/
COPY --chown=appuser:appuser artifacts/model.pkl artifacts/preprocessor.pkl artifacts/model_info.json ./artifacts/

USER appuser

EXPOSE 8000

# Fast, non-blocking health check using the running Uvicorn process.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()" || exit 1

# Start the FastAPI application using Uvicorn.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
