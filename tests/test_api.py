"""API tests for src/api/app.py using FastAPI's TestClient (no live server needed)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.predict import MODEL_PATH, feature_names

client = TestClient(app)


def _valid_payload(current_close=None) -> dict:
    """A complete, valid feature payload (neutral zeros), optionally with a close."""
    payload = {name: 0.0 for name in feature_names()}
    if current_close is not None:
        payload["current_close"] = current_close
    return payload


def _make_synthetic_candles(n_hours: int = 60) -> pd.DataFrame:
    """Small fake CLEANED OHLCV history: gently rising close, no network needed.

    n_hours=60 is comfortably more than the longest lookback (24), so
    predict_from_candles() (called inside /predict/live) has enough history
    to produce a valid prediction.
    """
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    base = np.linspace(60000, 61000, n_hours)
    wiggle = np.sin(np.linspace(0, 15, n_hours)) * 50
    close = base + wiggle

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 5,
            "high": close + 15,
            "low": close - 15,
            "close": close,
            "volume": np.linspace(100, 200, n_hours),
        }
    )


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_available"] is True


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not trained yet")
def test_predict_returns_float():
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["predicted_return"], float)
    assert body["predicted_price"] is None  # no close given


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not trained yet")
def test_predict_reconstructs_price():
    close = 60000.0
    resp = client.post("/predict", json=_valid_payload(current_close=close))
    assert resp.status_code == 200
    body = resp.json()
    expected = close * (1 + body["predicted_return"])
    assert body["predicted_price"] == pytest.approx(expected)


def test_predict_missing_feature_is_422():
    payload = _valid_payload()
    payload.pop(feature_names()[0])  # drop a required feature
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # pydantic validation error


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not trained yet")
def test_predict_live_success(monkeypatch):
    """/predict/live should work end-to-end WITHOUT hitting Binance, by
    monkeypatching fetch_recent_candles (as imported into src.api.app) to
    return synthetic candles instead of calling the real network.
    """
    candles = _make_synthetic_candles(n_hours=60)

    def _fake_fetch(symbol, interval, lookback):
        return candles

    monkeypatch.setattr("src.api.app.fetch_recent_candles", _fake_fetch)

    resp = client.get("/predict/live")
    assert resp.status_code == 200
    body = resp.json()

    expected_keys = {
        "as_of_time",
        "current_price",
        "predicted_return",
        "predicted_price",
        "predicted_for_time",
    }
    assert expected_keys.issubset(body.keys())

    for key in ("current_price", "predicted_return", "predicted_price"):
        assert np.isfinite(body[key])

    expected_price = body["current_price"] * (1.0 + body["predicted_return"])
    assert body["predicted_price"] == pytest.approx(expected_price)


def test_predict_live_upstream_failure_returns_503(monkeypatch):
    """If fetching candles fails for ANY reason (Binance down, network error,
    bad response, etc.), the endpoint must return a clean 503 -- never a raw
    stack trace. Doesn't need the model, since the failure happens before
    prediction is even attempted.
    """

    def _boom(symbol, interval, lookback):
        raise Exception("simulated Binance outage")

    monkeypatch.setattr("src.api.app.fetch_recent_candles", _boom)

    resp = client.get("/predict/live")
    assert resp.status_code == 503


def test_monitoring_drift_success(monkeypatch):
    """/monitoring/drift should return the drift report as JSON. Both the data
    fetch (get_current_features) and the maths (compute_drift) are mocked, so
    this test only checks the endpoint wiring + response shape -- no network,
    no dependence on X_train being on disk (drift maths is tested in
    tests/test_drift.py).
    """
    canned = {
        "n_features": 3,
        "n_drifted": 1,
        "drift_detected": False,
        "features": [
            {"feature": "volatility_24", "psi": 0.31, "psi_level": "major",
             "ks_pvalue": 0.0, "drifted": True},
            {"feature": "return_1h", "psi": 0.04, "psi_level": "stable",
             "ks_pvalue": 0.21, "drifted": False},
            {"feature": "log_volume", "psi": 0.15, "psi_level": "moderate",
             "ks_pvalue": 0.03, "drifted": False},
        ],
    }
    monkeypatch.setattr("src.api.app.get_current_features", lambda: None)
    monkeypatch.setattr("src.api.app.compute_drift", lambda current: canned)

    resp = client.get("/monitoring/drift")
    assert resp.status_code == 200
    body = resp.json()

    assert body["n_features"] == 3
    assert body["n_drifted"] == 1
    assert body["drift_detected"] is False
    assert len(body["features"]) == 3
    for row in body["features"]:
        assert {"feature", "psi", "psi_level", "ks_pvalue", "drifted"} <= row.keys()


def test_monitoring_drift_upstream_failure_returns_503(monkeypatch):
    """Any failure while building the current batch (Binance down, network
    error, etc.) must surface as a clean 503, not a raw stack trace.
    """

    def _boom():
        raise Exception("simulated Binance outage")

    monkeypatch.setattr("src.api.app.get_current_features", _boom)

    resp = client.get("/monitoring/drift")
    assert resp.status_code == 503
