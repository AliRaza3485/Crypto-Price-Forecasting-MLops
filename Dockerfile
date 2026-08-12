# ============================================================
# Slim serving image for the BTC Next-Hour Return Forecaster API.
#
# Built (and pushed to Docker Hub) by CI -- NOT locally. Ships only what the
# serving app needs: the slim deps, the app code, config, and the trained model.
# Training tooling (mlflow, dvc, optuna, jupyter...) is deliberately excluded.
#
# NOTE: the drift reference (data/processed/X_train.parquet) is gitignored, so
# it is NOT in the image -> /predict and /predict/live work, but
# /monitoring/drift returns 503 until a reference is provided.
# ============================================================
FROM python:3.13-slim

# No .pyc files; flush stdout/stderr immediately so container logs are live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first so this layer is cached unless requirements-ci.txt changes.
COPY requirements-ci.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-ci.txt

# App code, config, and the trained model. PROJECT_ROOT in src/config.py resolves
# to /app, so config/config.yaml and models/rf_model.joblib are found here.
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

EXPOSE 8000

# Lightweight liveness check hitting the app's own /health (curl isn't in slim,
# so use Python's stdlib urllib).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
