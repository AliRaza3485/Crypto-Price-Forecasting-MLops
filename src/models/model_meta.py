"""
Model metadata for Crypto-Price-Forecasting-MLops.

The trained model itself (rf_model.joblib) is just weights -- it can't tell
you what algorithm it is, when it was trained, what its test metrics were, or
which MLflow run produced it. This module writes a small sidecar JSON file
next to the model every time one is trained or promoted, and reads it back
for serving.

Why a sidecar file (not baked into app.py, not a DB)?
  * Zero extra infra -- it's one JSON file living alongside rf_model.joblib,
    so it travels with the model in the same volume mount / Docker layer.
  * Written at the exact moment the model file is written (train() and
    promote()), so it can never silently drift out of sync with the model
    that's actually being served.
  * Read fresh on every request (no caching) -- unlike the model itself,
    this is cheap to read and we want /model/info to reflect a retrain the
    moment it happens, without waiting for a process restart.

File location: <model_dir>/model_meta.json (e.g. models/model_meta.json).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import CONFIG, get_path

logger = logging.getLogger(__name__)

_model_cfg = CONFIG["model"]
MODEL_DIR = get_path(_model_cfg["model_dir"])
META_PATH = MODEL_DIR / "model_meta.json"


def build_meta(
    *,
    algorithm: str,
    params: dict,
    metrics: dict,
    n_features: int,
    feature_names: list[str],
    mlflow_run_id: str | None,
    registered_model_name: str | None,
    source: str,
    champion_metrics: dict | None = None,
) -> dict:
    """Assemble the metadata dict. Kept separate from save_meta() so tests can
    check its shape without touching the filesystem.
    """
    return {
        "algorithm": algorithm,
        "params": params,
        "metrics": metrics,
        "champion_metrics": champion_metrics,
        "n_features": n_features,
        "feature_names": feature_names,
        "mlflow_run_id": mlflow_run_id,
        "registered_model_name": registered_model_name,
        # How this version of the model came to be:
        #   "initial_training" -> python -m src.models.model_training
        #   "retrain_promoted" -> src.models.retrain, candidate beat champion
        "source": source,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def save_meta(meta: dict, path: Path = META_PATH) -> None:
    """Write metadata to disk. Never raises -- a failed metadata write must
    not fail (or roll back) an otherwise-successful training/promotion run;
    it just means /model/info stays stale until the next successful save.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved model metadata -> %s", path)
    except OSError:
        logger.exception("Failed to write model metadata to %s (non-fatal)", path)


def load_meta(path: Path = META_PATH) -> dict | None:
    """Read metadata from disk. Returns None if it doesn't exist yet (e.g. a
    model trained before this feature existed) or is unreadable -- callers
    treat that as "not yet reported", never as a hard error.
    """
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read model metadata from %s", path)
        return None
