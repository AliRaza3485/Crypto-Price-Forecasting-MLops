"""
Model training for Crypto-Price-Forecasting-MLops.

Trains the production forecaster on the chronological split files from
data/processed/, evaluates it on the held-out (newest) test set, logs everything
to MLflow on DagsHub, and registers the model so it becomes a versioned,
deployable artifact. A local copy is also saved to models/.

Why FIXED hyperparameters here (no tuning)?
  Hyperparameter SEARCH is experimentation — it lives in the notebook
  (notebooks/02_model_experiment.ipynb), where Optuna + TimeSeriesSplit picked
  the best RandomForest config. Production training must be fast, deterministic
  and reproducible, so this script just reads those winning params from
  config/config.yaml. To re-tune, do it in the notebook and update the config.

Auth — automation-friendly, NO browser:
  The notebook used dagshub.init() (browser OAuth login). This script instead
  reads MLflow credentials from a .env file (git-ignored) so it can run
  unattended (CI, scheduler). Required variables (see .env.example):
      MLFLOW_TRACKING_URI       DagsHub ".mlflow" endpoint
      MLFLOW_TRACKING_USERNAME  DagsHub username
      MLFLOW_TRACKING_PASSWORD  DagsHub access token
  MLflow reads these automatically once python-dotenv has loaded them.

We score the SAME way as the experiment notebook (RMSE / MAE / R² +
directional accuracy) so the numbers line up across DagsHub runs.

All paths/params come from config/config.yaml. Run from the project root:
    python -m src.models.model_training
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CONFIG, get_path
from src.models.model_meta import build_meta, save_meta

# NOTE: mlflow and python-dotenv are TRAINING-only deps and are deliberately
# NOT in the slim requirements-ci.txt (which also builds the serving image).
# They are imported lazily inside setup_mlflow()/train() so this module can be
# imported — and its pure helpers (load_splits, evaluate) tested in CI —
# without those heavy packages installed. Serving/CI stays slim; the EC2
# retrain box (requirements-train.txt) has them for real runs.

# MLflow prints a "🏃 View run ... at <url>" line (with an emoji) when a run
# ends. The default Windows console codec (cp1252) can't encode that emoji and
# would crash the script on exit with UnicodeEncodeError — fatal for the
# unattended/CI use this script is built for. Force UTF-8 on our streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- Load settings from config (nothing hard-coded) ---
_data_cfg = CONFIG["data"]
_split_cfg = CONFIG["split"]
_model_cfg = CONFIG["model"]
_mlflow_cfg = CONFIG["mlflow"]

PROCESSED_DIR = get_path(_data_cfg["processed_dir"])
MODEL_DIR = get_path(_model_cfg["model_dir"])
MODEL_PATH = MODEL_DIR / _model_cfg["model_file"]

RF_PARAMS = _model_cfg["params"]
RANDOM_STATE = _model_cfg["random_state"]
ALGORITHM = _model_cfg["algorithm"]

EXPERIMENT_NAME = _mlflow_cfg["experiment_name"]
RUN_NAME = _mlflow_cfg["run_name"]
REGISTERED_MODEL_NAME = _mlflow_cfg["registered_model_name"]

TARGET_COL = "target"
_PLACEHOLDER_TOKEN = "your_dagshub_token_here"  # sentinel from .env.example

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_splits():
    """Read the four chronological split files. y is the `target` return Series."""
    X_train = pd.read_parquet(PROCESSED_DIR / _split_cfg["x_train_file"])
    y_train = pd.read_parquet(PROCESSED_DIR / _split_cfg["y_train_file"])[TARGET_COL]
    X_test = pd.read_parquet(PROCESSED_DIR / _split_cfg["x_test_file"])
    y_test = pd.read_parquet(PROCESSED_DIR / _split_cfg["y_test_file"])[TARGET_COL]

    logger.info("X_train %s | X_test %s | features: %s",
                X_train.shape, X_test.shape, list(X_train.columns))
    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred) -> dict:
    """Regression metrics (identical to the experiment notebook)."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # Bonus: did we get the up/down direction right?
        "directional_acc": float((np.sign(y_true) == np.sign(y_pred)).mean()),
    }


def setup_mlflow() -> None:
    """Authenticate to DagsHub MLflow via .env, then select the experiment.

    Fails fast with a clear message if the token is missing or still the
    placeholder — so a bad run never silently logs nowhere.
    """
    from dotenv import load_dotenv  # lazy: training-only dep (see top-of-file note)
    import mlflow

    load_dotenv()  # .env -> environment variables (MLflow reads them itself)

    required = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing MLflow credentials in .env: {missing}. "
            "Copy .env.example to .env and fill in your DagsHub token."
        )
    if os.getenv("MLFLOW_TRACKING_PASSWORD") == _PLACEHOLDER_TOKEN:
        raise EnvironmentError(
            "MLFLOW_TRACKING_PASSWORD is still the placeholder. "
            "Put your real DagsHub token in .env."
        )

    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow -> %s  (experiment: %s)", mlflow.get_tracking_uri(), EXPERIMENT_NAME)


def train():
    """Train the tuned RandomForest, evaluate, log + register to MLflow."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    X_train, y_train, X_test, y_test = load_splits()
    setup_mlflow()

    with mlflow.start_run(run_name=RUN_NAME) as run:
        model = RandomForestRegressor(**RF_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = evaluate(y_test.values, preds)

        # --- Log params + metrics ---
        mlflow.log_param("model", ALGORITHM)
        mlflow.log_params(RF_PARAMS)
        mlflow.log_metrics(metrics)

        # --- Log + register the model artifact, with its input/output schema ---
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # --- Also keep a local copy (handy for offline serving / DVC) ---
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        # --- Sidecar metadata for /model/info (see model_meta.py) ---
        meta = build_meta(
            algorithm=ALGORITHM,
            params=RF_PARAMS,
            metrics=metrics,
            n_features=X_train.shape[1],
            feature_names=list(X_train.columns),
            mlflow_run_id=run.info.run_id,
            registered_model_name=REGISTERED_MODEL_NAME,
            source="initial_training",
        )
        save_meta(meta)

        logger.info("Run ID: %s", run.info.run_id)
        logger.info("Test metrics:")
        for k, v in metrics.items():
            logger.info("  %-16s: %.6f", k, v)
        logger.info("Registered '%s' | local copy -> %s", REGISTERED_MODEL_NAME, MODEL_PATH)

    return model, metrics


def main() -> None:
    train()


if __name__ == "__main__":
    main()
