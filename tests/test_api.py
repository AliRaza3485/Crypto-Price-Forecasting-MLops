"""API tests for src/api/app.py using FastAPI's TestClient (no live server needed)."""

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
    assert body["predicted_price"] is None   # no close given


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
    payload.pop(feature_names()[0])          # drop a required feature
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422           # pydantic validation error
