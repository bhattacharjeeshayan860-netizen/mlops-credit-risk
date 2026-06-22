# Docker image for the FastAPI inference service.
#
# This container will run the API that loads the trained model, serves
# predictions, and exposes operational metrics for Prometheus.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
