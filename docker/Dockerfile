# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install curl for the HEALTHCHECK, then clean apt caches to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the API.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy ONLY the runtime requirements file first so dependency installation is cached
# unless requirements-api.txt itself changes.
COPY --chown=appuser:appuser requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy ONLY the files the FastAPI service needs at runtime.
# .dockerignore blocks everything else (notebooks, data, tests, VCS, etc.).
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser artifacts/ ./artifacts/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Bind to 0.0.0.0 so the container is reachable from outside.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
