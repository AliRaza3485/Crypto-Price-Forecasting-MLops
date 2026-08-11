"""Unit tests for the inference module (src/models/predict.py)."""

import numpy as np
import pandas as pd
import pytest

from src.models.predict import (
    MODEL_PATH,
    feature_names,
    load_model,
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
