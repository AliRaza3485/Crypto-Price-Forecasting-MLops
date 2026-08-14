"""
FastAPI serving app for Crypto-Price-Forecasting-MLops.

Endpoints
  GET  /health         -> liveness + whether the model file is available.
  POST /predict         -> given the model's engineered feature row (and,
                            optionally, the current close price), returns the
                            predicted next-hour return -- plus the
                            reconstructed price when close is given.
  GET  /predict/live     -> fetches recent candles from Binance itself, builds
                            features, and returns a full live prediction. No
                            input needed from the caller.
  GET  /monitoring/drift -> fetches recent candles, builds features, and
                            compares their distribution to the training data;
                            returns a per-feature data-drift report (PSI + KS).
  GET  /predict/history   -> returns recently logged live predictions
                            (resolved + pending) from the background history
                            logger, for a "predicted vs actual" chart.

Contract
  /predict is the FEATURES-IN contract: the caller supplies the 14 engineered
  features the model was trained on. /predict/live is the fully-automatic
  version: it fetches its own data (via fetch_recent_candles) and builds its
  own features (via predict_from_candles), so a caller just hits the endpoint
  with no body at all.

Background history logging
  A lightweight APScheduler BackgroundScheduler runs INSIDE this process
  (started in lifespan(), stopped on shutdown) and, every
  config["history"]["poll_interval_minutes"], repeats the exact same work as
  GET /predict/live -- fetch candles, predict -- then logs the result to a
  small SQLite file (src/monitoring/history.py) and backfills actual_price
  for any earlier prediction whose target hour has now arrived. This is what
  GET /predict/history reads from. No separate cron job or service needed.

Run locally from the project root:
    uvicorn src.api.app:app --reload
Then open the interactive docs:  http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CONFIG
from src.data.data_ingestion import fetch_recent_candles
from src.models.model_meta import load_meta
from src.models.predict import (
    MODEL_PATH,
    load_model,
    predict_from_candles,
    predict_return,
)
from src.monitoring import history
from src.monitoring.drift import compute_drift, get_current_features

logger = logging.getLogger(__name__)

# --- Config for the live endpoint (nothing hard-coded) ---
_data_cfg = CONFIG["data"]
_serving_cfg = CONFIG["serving"]
_history_cfg = CONFIG["history"]

SYMBOL = _data_cfg["symbol"]
INTERVAL = _data_cfg["interval"]
LIVE_LOOKBACK = _serving_cfg["live_lookback"]
HISTORY_POLL_MINUTES = _history_cfg["poll_interval_minutes"]


def _log_live_prediction() -> None:
    """Scheduler job: fetch candles, predict, log the result, resolve older ones.

    Deliberately swallows and logs ANY exception -- a single failed poll
    (e.g. Binance hiccup) must never crash the scheduler thread, which would
    silently kill all FUTURE polls too.
    """
    try:
        candles = fetch_recent_candles(SYMBOL, INTERVAL, LIVE_LOOKBACK)
        result = predict_from_candles(candles)
        history.log_prediction(result)
        history.resolve_pending(candles)
    except Exception:
        logger.exception("Background history poll failed -- will retry next interval")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model cache, init the history DB, and start the background
    history-logging scheduler at startup; never crash the server if the
    model is missing, and always stop the scheduler cleanly on shutdown.

    If the model is missing we still start (so /health can report it) --
    only /predict and /predict/live will fail, with a clear 503. The
    scheduler job itself already guards against a missing model the same
    way (it just logs and retries next interval).
    """
    try:
        load_model()
    except FileNotFoundError:
        pass  # /health will show model_available=False

    history.init_db()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _log_live_prediction,
        "interval",
        minutes=HISTORY_POLL_MINUTES,
        id="log_live_prediction",
        # Fire once immediately on startup (not just after the first full
        # interval), so history isn't empty right after every deploy/restart.
        next_run_time=datetime.now(),
    )
    scheduler.start()

    yield

    scheduler.shutdown(wait=False)


app = FastAPI(
    title="BTC Next-Hour Return Forecaster",
    version="1.0.0",
    description="Predicts the next-hour BTC return (and price) from engineered features.",
    lifespan=lifespan,
)


class Features(BaseModel):
    """The 14 engineered features the model expects (see make_features.py)."""

    return_1h: float = Field(
        ..., description="Current hourly return (% change of close)"
    )
    return_lag_1: float
    return_lag_2: float
    return_lag_3: float
    return_lag_6: float
    return_lag_12: float
    return_lag_24: float
    close_over_ma_6: float = Field(
        ..., description="close / 6h moving average (scale-free)"
    )
    volatility_6: float = Field(..., description="std of returns over 6h")
    close_over_ma_12: float
    volatility_12: float
    close_over_ma_24: float
    volatility_24: float
    log_volume: float = Field(..., description="log1p(volume)")

    current_close: float | None = Field(
        default=None,
        description="Current close price. If provided, the predicted PRICE is returned too.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "return_1h": 0.0012,
                "return_lag_1": -0.0003,
                "return_lag_2": 0.0008,
                "return_lag_3": 0.0001,
                "return_lag_6": -0.0015,
                "return_lag_12": 0.0004,
                "return_lag_24": 0.0009,
                "close_over_ma_6": 1.001,
                "volatility_6": 0.004,
                "close_over_ma_12": 0.998,
                "volatility_12": 0.005,
                "close_over_ma_24": 1.002,
                "volatility_24": 0.006,
                "log_volume": 8.5,
                "current_close": 60000.0,
            }
        }
    }


class Prediction(BaseModel):
    predicted_return: float = Field(
        ..., description="Predicted next-hour return (fraction, e.g. 0.001 = +0.1%)"
    )
    predicted_price: float | None = Field(
        None, description="current_close * (1 + predicted_return), if close was given"
    )


class LivePrediction(BaseModel):
    """Response shape for GET /predict/live -- mirrors predict_from_candles()'s dict."""

    as_of_time: str = Field(..., description="ISO timestamp of the latest candle used")
    current_price: float = Field(..., description="Close price at as_of_time")
    predicted_return: float = Field(
        ..., description="Predicted next-hour return (fraction)"
    )
    predicted_price: float = Field(
        ..., description="current_price * (1 + predicted_return)"
    )
    predicted_for_time: str = Field(
        ..., description="ISO timestamp this prediction is FOR (as_of_time + 1h)"
    )


class FeatureDrift(BaseModel):
    """Per-feature drift result (one row of compute_drift()'s report)."""

    feature: str
    psi: float = Field(
        ..., description="Population Stability Index (>= 0.20 => drifted)"
    )
    psi_level: str = Field(..., description="stable | moderate | major")
    ks_pvalue: float = Field(..., description="KS test p-value (informational only)")
    drifted: bool


class DriftReport(BaseModel):
    """Response shape for GET /monitoring/drift -- mirrors compute_drift()'s dict."""

    n_features: int = Field(..., description="Total features checked")
    n_drifted: int = Field(..., description="How many features drifted (PSI >= major)")
    drift_detected: bool = Field(
        ..., description="True once n_drifted >= min_drifted_features"
    )
    features: list[FeatureDrift]


class HistoryEntry(BaseModel):
    """One logged prediction -- mirrors a row from history.get_history()."""

    as_of_time: str = Field(
        ..., description="ISO timestamp of the candle used for this prediction"
    )
    current_price: float
    predicted_return: float
    predicted_price: float
    predicted_for_time: str = Field(
        ..., description="ISO timestamp this prediction was FOR"
    )
    actual_price: float | None = Field(
        None,
        description="Real close price once predicted_for_time has passed; null while pending",
    )
    error: float | None = Field(
        None, description="actual_price - predicted_price; null while pending"
    )


class HistoryResponse(BaseModel):
    """Response shape for GET /predict/history."""

    hours: int = Field(..., description="How many hours back this response covers")
    count: int = Field(..., description="Number of entries returned")
    entries: list[HistoryEntry]


class ModelInfo(BaseModel):
    """Response shape for GET /model/info."""

    available: bool = Field(..., description="Whether metadata was found on disk")
    algorithm: str | None = Field(None, description="e.g. 'random_forest'")
    trained_at: str | None = Field(None, description="ISO timestamp of this model version")
    source: str | None = Field(
        None, description="'initial_training' or 'retrain_promoted'"
    )
    metrics: dict | None = Field(None, description="Test-set metrics for this version")
    n_features: int | None = None
    mlflow_run_id: str | None = None
    registered_model_name: str | None = None


@app.get("/health")
def health() -> dict:
    """Liveness check + whether the trained model file is present."""
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Metadata about the currently-served model version: algorithm, when it
    was (re)trained, its test metrics, and whether it came from initial
    training or a promoted retrain.

    Reads a small sidecar JSON written next to the model file at train/promote
    time (see src/models/model_meta.py) -- never touches the model itself, so
    this stays cheap and always reflects the model actually on disk.

    Returns available=False (not a 503) when no metadata exists yet -- e.g.
    a model trained before this endpoint existed, or a fresh deploy where
    training hasn't run. The caller can render "not yet reported" instead of
    treating it as an error.
    """
    meta = load_meta()
    if meta is None:
        return ModelInfo(available=False)
    return ModelInfo(available=True, **meta)


@app.post("/predict", response_model=Prediction)
def predict(payload: Features) -> Prediction:
    """Predict the next-hour return (and price, if current_close was supplied)."""
    data = payload.model_dump()
    current_close = data.pop(
        "current_close"
    )  # remove -> leaves exactly the 14 features
    row = pd.DataFrame([data])

    try:
        predicted_return = float(predict_return(row)[0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predicted_price = None
    if current_close is not None:
        predicted_price = current_close * (1.0 + predicted_return)

    return Prediction(
        predicted_return=predicted_return, predicted_price=predicted_price
    )


@app.get("/predict/live", response_model=LivePrediction)
def predict_live() -> LivePrediction:
    """Fully-automatic live prediction: fetch recent candles, build features,
    predict the next-hour return and price. No input needed from the caller.

    Error handling (never leak a raw stack trace to the caller):
      * Model not trained on disk       -> FileNotFoundError -> 503
      * Not enough candles for features -> ValueError         -> 503
      * Anything else going wrong while fetching from Binance
        (network error, Binance outage, bad response, etc.)   -> 503
        with a generic "upstream data source unavailable" message, since the
        exact exception type from a third-party client isn't something we
        want to hard-code or expose to callers.
    """
    try:
        candles = fetch_recent_candles(SYMBOL, INTERVAL, LIVE_LOOKBACK)
        result = predict_from_candles(candles)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Upstream data source unavailable"
        ) from exc

    return LivePrediction(**result)


@app.get("/monitoring/drift", response_model=DriftReport)
def monitoring_drift() -> DriftReport:
    """Data-drift check: compare RECENT live features to the TRAINING data.

    Fetches ~30 days of candles from Binance, builds the model features, and
    compares their distribution against X_train (per-feature PSI + KS test).
    Returns the same report as `python -m src.monitoring.drift`.

    Note: each call fetches a fresh batch from Binance, so it's heavier than a
    single prediction -- intended for periodic monitoring, not per-request use.

    Errors map to a clean 503 (never a raw stack trace):
      * reference X_train missing on disk -> FileNotFoundError -> 503
      * not enough recent candles         -> ValueError         -> 503
      * anything else fetching from Binance                     -> 503
    """
    try:
        current = get_current_features()
        report = compute_drift(current)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Upstream data source unavailable"
        ) from exc

    return DriftReport(**report)


@app.get("/predict/history", response_model=HistoryResponse)
def predict_history(hours: int = 24) -> HistoryResponse:
    """Recently logged live predictions (resolved + pending), newest first.

    Reads from the SQLite history the background scheduler has been writing
    to since this process started (see _log_live_prediction / lifespan()) --
    it does NOT hit Binance or the model on every call, so this is cheap.

    Args:
        hours: how far back to look (default 24). A fresh deploy will have
            an empty/short history until the scheduler has had time to run.
    """
    if hours <= 0:
        raise HTTPException(status_code=400, detail="hours must be a positive integer")

    entries = history.get_history(hours=hours)
    return HistoryResponse(hours=hours, count=len(entries), entries=entries)
