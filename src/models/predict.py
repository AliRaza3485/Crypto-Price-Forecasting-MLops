"""
Inference for Crypto-Price-Forecasting-MLops.

Turns feature rows into predictions:
  * the predicted next-hour RETURN (what the model actually outputs), and
  * the reconstructed next-hour PRICE, when a current close is supplied:
        predicted_price = current_close * (1 + predicted_return)

Why a separate module (not inside the API)?
  Serving is just one caller. Keeping prediction here means the same logic is
  reused by the API, by batch scoring, and by tests — and it stays testable
  WITHOUT a running web server or a network connection.

Model source (config-driven):
  Defaults to the local models/rf_model.joblib written by training. The MLflow
  registry copy on DagsHub is the versioned source of truth; loading a specific
  registry version is a later option, kept out of the hot path for now.

Quick self-check from the project root:
    python -m src.models.predict
"""

import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import CONFIG, get_path

_model_cfg = CONFIG["model"]
MODEL_PATH = get_path(_model_cfg["model_dir"]) / _model_cfg["model_file"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once and cache it.

    lru_cache means the API loads the model on the first request only, then
    reuses it — no reloading the file on every prediction.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first with: "
            "python -m src.models.model_training"
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)
    return model


def feature_names() -> list[str]:
    """The exact feature columns (and order) the model was trained on."""
    return list(load_model().feature_names_in_)


def predict_return(features: pd.DataFrame) -> np.ndarray:
    """Predict the next-hour return for one or more feature rows.

    Columns are validated and reordered to the model's expected feature order,
    so callers don't have to worry about the order they pass features in.
    """
    model = load_model()
    expected = list(model.feature_names_in_)

    missing = [c for c in expected if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = features[expected]          # exact columns, exact order
    return model.predict(X)


def predict_price(features: pd.DataFrame, current_close) -> np.ndarray:
    """Predict the next-hour PRICE from features + the current close price."""
    returns = predict_return(features)
    return np.asarray(current_close) * (1.0 + returns)


def main() -> None:
    """Self-check: build one neutral feature row and print a prediction."""
    row = pd.DataFrame([{name: 0.0 for name in feature_names()}])
    ret = float(predict_return(row)[0])
    close = 60000.0
    logger.info("Feature columns (%d): %s", len(feature_names()), feature_names())
    logger.info("Neutral-row predicted return: %.6f", ret)
    logger.info("At close=%.2f -> predicted price: %.2f", close, close * (1 + ret))


if __name__ == "__main__":
    main()
