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

Live inference, offline by design:
  predict_from_candles() turns a raw (already-cleaned) OHLCV candles DataFrame
  into a full prediction dict. It does NOT call Binance or any network API —
  it only transforms a DataFrame that's already in memory. This keeps it:
    * unit-testable with a synthetic DataFrame, no internet needed, no
      flaky/rate-limited tests, no mocking a live API in CI, and
    * reusable: the FastAPI endpoint (next step) will do the network fetch,
      then hand the candles to this function. If the fetch logic ever changes
      (different exchange, different endpoint, cached data, backtesting on a
      historical CSV...), this function doesn't care and doesn't change.

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
from src.features.make_features import add_features

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


def predict_from_candles(candles: pd.DataFrame) -> dict:
    """Turn recent raw candles into one full live prediction. No network calls.

    This is the pure "brain" of live inference: given candles that are ALREADY
    in memory (already cleaned — same shape as data_ingestion.clean() returns:
    timestamp, open, high, low, close, volume), it figures out the current hour
    and predicts the next one. Fetching those candles from Binance is a
    separate concern and belongs in the API endpoint, not here — see the
    module docstring for why.

    Steps:
      1. add_features(candles) -> the 14 backward-looking features. This is the
         function that KEEPS the latest row (no target is computed, so nothing
         forces it to be dropped) — that's exactly the row we need: "given
         everything up to and including the current hour, describe this hour."
      2. If nothing comes back, the caller didn't send enough history (the
         longest lag/rolling window needs 24 prior hours) -> raise a clear
         ValueError instead of silently failing later.
      3. Take the LAST row of that result — that's the current hour.
      4. predict_return() on that one row -> the model's next-hour return.
      5. Rebuild the price: predicted_price = current_close * (1 + return).

    Args:
        candles: cleaned OHLCV DataFrame with columns
            [timestamp, open, high, low, close, volume].

    Returns:
        dict with:
          - as_of_time: ISO string, timestamp of the latest candle used
          - current_price: float, that candle's close
          - predicted_return: float, model's predicted next-hour return
          - predicted_price: float, current_price * (1 + predicted_return)
          - predicted_for_time: ISO string, as_of_time + 1 hour

    Raises:
        ValueError: if there isn't enough history to build even one full
            feature row (add_features() drops warm-up rows internally).
    """
    feat_df = add_features(candles)

    if feat_df.empty:
        raise ValueError(
            "Not enough candles to build features — need more history "
            "(the longest lag/rolling window requires 24 prior hours)."
        )

    last_row = feat_df.iloc[[-1]]  # keep as a 1-row DataFrame, not a Series

    as_of_time = pd.Timestamp(last_row["timestamp"].iloc[0])
    current_price = float(last_row["close"].iloc[0])

    predicted_return = float(predict_return(last_row)[0])
    predicted_price = current_price * (1.0 + predicted_return)
    predicted_for_time = as_of_time + pd.Timedelta(hours=1)

    logger.info(
        "Prediction as_of=%s current_price=%.2f predicted_return=%.6f predicted_price=%.2f",
        as_of_time, current_price, predicted_return, predicted_price,
    )

    return {
        "as_of_time": as_of_time.isoformat(),
        "current_price": current_price,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "predicted_for_time": predicted_for_time.isoformat(),
    }


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
