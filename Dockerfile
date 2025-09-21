FROM python:3.11-slim

# System deps (xgboost needs libgomp1; matplotlib benefits from fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential git curl tini && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -ms /bin/bash appuser

WORKDIR /workspace

# Python deps first (better layer caching)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY . /workspace

# Defaults (overridden by .env via docker-compose)
ENV MLFLOW_TRACKING_URI="file:///workspace/mlruns" \
    PYTHONUNBUFFERED=1

USER appuser

# Base CMD is a no-op; compose overrides per service
CMD ["python", "-c", "print('Use docker-compose to run services')"]
