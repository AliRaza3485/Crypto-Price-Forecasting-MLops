"""
FastAPI serving app for Crypto-Price-Forecasting-MLops.

Endpoints
  GET  /health   -> liveness + whether the model file is available.
  POST /predict  -> given the model's engineered feature row (and, optionally,
                    the current close price), returns the predicted next-hour
                    return — plus the reconstructed price when close is given.

Contract
  This is the FEATURES-IN contract: the caller supplies the 14 engineered
  features the model was trained on. A live endpoint that fetches Binance and
  builds those features itself is a planned follow-up (see project roadmap).

Run locally from the project root:
    uvicorn src.api.app:app --reload
Then open the interactive docs:  http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import MODEL_PATH, load_model, predict_return


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model cache at startup if it exists; never crash the server.

    If the model is missing we still start (so /health can report it) — only
    /predict will fail, with a clear 503.
    """
    try:
        load_model()
    except FileNotFoundError:
        pass  # /health will show model_available=False
    yield


app = FastAPI(
    title="BTC Next-Hour Return Forecaster",
    version="1.0.0",
    description="Predicts the next-hour BTC return (and price) from engineered features.",
    lifespan=lifespan,
)


class Features(BaseModel):
    """The 14 engineered features the model expects (see make_features.py).

    Order here mirrors the model; predict_return() re-validates it anyway, so
    field order is for readability, not correctness.
    """

    return_1h: float = Field(..., description="Current hourly return (% change of close)")
    return_lag_1: float
    return_lag_2: float
    return_lag_3: float
    return_lag_6: float
    return_lag_12: float
    return_lag_24: float
    close_over_ma_6: float = Field(..., description="close / 6h moving average (scale-free)")
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
    predicted_return: float = Field(..., description="Predicted next-hour return (fraction, e.g. 0.001 = +0.1%)")
    predicted_price: float | None = Field(None, description="current_close * (1 + predicted_return), if close was given")


@app.get("/health")
def health() -> dict:
    """Liveness check + whether the trained model file is present."""
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features) -> Prediction:
    """Predict the next-hour return (and price, if current_close was supplied)."""
    data = payload.model_dump()
    current_close = data.pop("current_close")   # remove -> leaves exactly the 14 features
    row = pd.DataFrame([data])

    try:
        predicted_return = float(predict_return(row)[0])
    except FileNotFoundError as exc:
        # Model not trained yet -> service unavailable, with a clear message.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predicted_price = None
    if current_close is not None:
        predicted_price = current_close * (1.0 + predicted_return)

    return Prediction(predicted_return=predicted_return, predicted_price=predicted_price)
