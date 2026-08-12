"""Unit tests for the data-drift module (src/monitoring/drift.py).

Fully offline: PSI/compute_drift run on synthetic in-memory data, and the one
function that touches the network (get_current_features) is exercised with
fetch_recent_candles monkeypatched to return synthetic candles. Nothing here
hits Binance or requires the trained model.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring import drift
from src.monitoring.drift import (
    compute_drift,
    format_report,
    get_current_features,
    psi,
)

# The 14 model features, in the order make_features.py produces them. Kept here
# so the drift tests are self-contained (no parquet / model file needed).
FEATURE_NAMES = [
    "return_1h", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_6",
    "return_lag_12", "return_lag_24", "close_over_ma_6", "volatility_6",
    "close_over_ma_12", "volatility_12", "close_over_ma_24", "volatility_24",
    "log_volume",
]


def _synthetic_reference(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A fake 'training' feature matrix: 14 columns of standard-normal noise."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({name: rng.normal(0.0, 1.0, n) for name in FEATURE_NAMES})


def _make_synthetic_candles(n_hours: int = 60) -> pd.DataFrame:
    """Small fake CLEANED OHLCV history (same shape clean() produces)."""
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    base = np.linspace(60000, 61000, n_hours)
    wiggle = np.sin(np.linspace(0, 15, n_hours)) * 50
    close = base + wiggle
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 5,
        "high": close + 15,
        "low": close - 15,
        "close": close,
        "volume": np.linspace(100, 200, n_hours),
    })


# --- PSI maths ---------------------------------------------------------------

def test_psi_identical_is_zero():
    x = _synthetic_reference(seed=0)["return_1h"].to_numpy()
    assert psi(x, x) == 0.0


def test_psi_increases_with_shift():
    x = _synthetic_reference(seed=0)["return_1h"].to_numpy()
    small = psi(x, x + 0.1)   # barely moved
    big = psi(x, x + 3.0)     # moved a lot
    assert big > small
    assert big > 0.2          # a 3-sigma shift is clearly "major"


# --- compute_drift -----------------------------------------------------------

def test_compute_drift_no_drift_on_same():
    ref = _synthetic_reference(seed=0)
    current = _synthetic_reference(seed=1)  # same distribution, different draw
    report = compute_drift(current, reference=ref)
    assert report["n_drifted"] == 0
    assert report["drift_detected"] is False


def test_compute_drift_detects_shift():
    ref = _synthetic_reference(seed=0)
    current = _synthetic_reference(seed=1).copy()
    shifted = ["volatility_6", "volatility_12", "volatility_24"]
    for col in shifted:
        current[col] = current[col] + 5.0  # push far outside the reference range

    report = compute_drift(current, reference=ref)
    drifted = {r["feature"] for r in report["features"] if r["drifted"]}
    assert drifted == set(shifted)
    assert report["drift_detected"] is True  # 3 >= min_drifted_features


def test_compute_drift_missing_feature_raises():
    ref = _synthetic_reference(seed=0)
    current = _synthetic_reference(seed=1).drop(columns=["return_1h"])
    with pytest.raises(ValueError, match="missing feature columns"):
        compute_drift(current, reference=ref)


def test_report_has_expected_keys():
    ref = _synthetic_reference(seed=0)
    report = compute_drift(_synthetic_reference(seed=1), reference=ref)

    assert {"n_features", "n_drifted", "drift_detected", "features"} <= report.keys()
    assert report["n_features"] == len(FEATURE_NAMES)
    for row in report["features"]:
        assert {"feature", "psi", "psi_level", "ks_pvalue", "drifted"} <= row.keys()

    # format_report should mention the verdict either way
    assert "VERDICT" in format_report(report)


# --- get_current_features (network mocked) -----------------------------------

def test_get_current_features_monkeypatched(monkeypatch):
    """With fetch_recent_candles mocked, get_current_features should build the
    14 features from synthetic candles -- no real Binance call."""
    candles = _make_synthetic_candles(n_hours=60)
    monkeypatch.setattr(drift, "fetch_recent_candles", lambda s, i, l: candles)

    features = get_current_features()
    assert len(features) > 0
    assert set(FEATURE_NAMES).issubset(features.columns)


def test_get_current_features_too_few_raises(monkeypatch):
    """Too few candles -> add_features yields no rows -> clean ValueError."""
    candles = _make_synthetic_candles(n_hours=5)  # far fewer than the 24 needed
    monkeypatch.setattr(drift, "fetch_recent_candles", lambda s, i, l: candles)

    with pytest.raises(ValueError):
        get_current_features()
