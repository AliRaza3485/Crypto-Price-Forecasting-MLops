"""Unit tests for the inference module (src/models/predict.py)."""

import numpy as np
import pandas as pd
import pytest

from src.models.predict import (
    MODEL_PATH,
    feature_names,
    load_model,
    predict_from_candles,
    predict_price,
    predict_return,
)

# These tests need the trained model on disk. Skip cleanly if it isn't there
# (e.g. a fresh checkout before `python -m src.models.model_training`).
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Model not trained yet (models/rf_model.joblib missing)"
)


def _neutral_row() -> pd.DataFrame:
    """One feature row of zeros — always valid, order-independent."""
    return pd.DataFrame([{name: 0.0 for name in feature_names()}])


def _make_synthetic_candles(n_hours: int = 60) -> pd.DataFrame:
    """Build a small fake OHLCV history: gently rising close, no network needed.

    n_hours=60 is comfortably more than the longest lookback (24), so
    add_features() inside predict_from_candles() has enough history to
    produce at least one valid (non-NaN) row.
    """
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    # Gentle upward drift + a small wiggle, so returns/volatility aren't all zero.
    base = np.linspace(60000, 61000, n_hours)
    wiggle = np.sin(np.linspace(0, 15, n_hours)) * 50
    close = base + wiggle

    candles = pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 5,
        "high": close + 15,
        "low": close - 15,
        "close": close,
        "volume": np.linspace(100, 200, n_hours),
    })
    return candles


def test_model_loads_with_feature_names():
    model = load_model()
    assert hasattr(model, "feature_names_in_")
    assert len(model.feature_names_in_) == 14


def test_feature_names_count():
    assert len(feature_names()) == 14


def test_predict_return_finite_scalar():
    preds = predict_return(_neutral_row())
    assert len(preds) == 1
    assert np.isfinite(preds[0])


def test_predict_return_reorders_columns():
    """Shuffled column order must give the same prediction (we reorder internally)."""
    row = _neutral_row()
    shuffled = row[list(reversed(row.columns))]
    assert predict_return(row)[0] == pytest.approx(predict_return(shuffled)[0])


def test_predict_return_missing_column_raises():
    bad = _neutral_row().drop(columns=[feature_names()[0]])
    with pytest.raises(ValueError, match="Missing feature columns"):
        predict_return(bad)


def test_predict_price_reconstruction():
    row = _neutral_row()
    close = 60000.0
    ret = predict_return(row)[0]
    price = predict_price(row, close)[0]
    assert price == pytest.approx(close * (1 + ret))


def test_predict_from_candles_returns_expected_shape():
    """predict_from_candles() should return the expected keys as finite floats,
    and predicted_price must be consistent with current_price * (1 + predicted_return).
    """
    candles = _make_synthetic_candles(n_hours=60)

    result = predict_from_candles(candles)

    # --- keys present ---
    expected_keys = {
        "as_of_time",
        "current_price",
        "predicted_return",
        "predicted_price",
        "predicted_for_time",
    }
    assert expected_keys.issubset(result.keys())

    # --- values are finite floats ---
    assert np.isfinite(result["current_price"])
    assert np.isfinite(result["predicted_return"])
    assert np.isfinite(result["predicted_price"])

    # --- as_of_time / predicted_for_time are valid timestamps, 1h apart ---
    as_of = pd.Timestamp(result["as_of_time"])
    predicted_for = pd.Timestamp(result["predicted_for_time"])
    assert predicted_for == as_of + pd.Timedelta(hours=1)

    # --- price reconstruction formula must hold exactly ---
    expected_price = result["current_price"] * (1.0 + result["predicted_return"])
    assert result["predicted_price"] == pytest.approx(expected_price)


def test_predict_from_candles_too_few_rows_raises_value_error():
    """Too little history (not enough for even one full feature row) should
    raise a clear ValueError instead of silently returning garbage.
    """
    candles = _make_synthetic_candles(n_hours=5)  # far fewer than the 24 needed

    with pytest.raises(ValueError):
        predict_from_candles(candles)
