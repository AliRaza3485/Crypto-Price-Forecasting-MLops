"""
Automated retraining orchestrator for Crypto-Price-Forecasting-MLops.

Implements a drift-gated retrain-and-validate loop:

    drift check -> (no drift & no --force) -> keep current model, exit
                -> (drift OR --force)       -> refresh data
                                             -> train candidate
                                             -> validate vs champion
                                             -> promote ONLY if candidate wins
                                             -> restart the serving container

Why a validation gate at all?
  Blindly overwriting the champion with whatever the latest training run
  produces is how you silently regress production. The gate makes promotion
  conditional on the candidate actually beating the champion on the SAME
  fresh test set, so a bad retrain (bad luck, bad data window, etc.) never
  reaches serving.

Honesty note (read this before panicking about "boring" results):
  For hourly BTC returns, the series is close to a random walk, so gains over
  the champion will usually be microscopic — most runs will legitimately KEEP
  the champion. That is correct behavior. The gate exists to prevent
  deploying a WORSE model, not to guarantee improvement every run.

Run from the project root:
    python -m src.models.retrain
    python -m src.models.retrain --force   # skip the drift gate, retrain anyway
"""

import argparse
import logging
import subprocess
import sys

import joblib

from sklearn.ensemble import RandomForestRegressor

# NOTE: mlflow is a TRAINING-only dep, NOT in the slim requirements-ci.txt.
# It is imported lazily inside promote()/log_rejection() so this module (and
# its pure helpers) can be imported and tested in CI without mlflow installed.
# The EC2 retrain box has it via requirements-train.txt for real runs.
from src.config import CONFIG, get_path
from src.data import data_ingestion, make_dataset, make_split
from src.features import make_features
from src.models.model_training import (
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    RF_PARAMS,
    evaluate,
    load_splits,
    setup_mlflow,
)
from src.models.model_meta import build_meta, save_meta
from src.monitoring.drift import compute_drift, format_report, get_current_features

# Same reason as model_training.py: MLflow prints an emoji on run-end that
# crashes the default Windows console codec (cp1252). Force UTF-8 so this
# script survives unattended (CI / scheduler) runs.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- Settings from config/config.yaml (nothing hard-coded) ---
_retrain_cfg = CONFIG["retrain"]
DRIFT_GATE = _retrain_cfg["drift_gate"]
PROMOTION_MARGIN = _retrain_cfg["promotion_margin"]
CONTAINER_NAME = _retrain_cfg["container_name"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def should_retrain(force: bool) -> tuple[bool, dict | None]:
    """Decide whether a retrain is warranted.

    --force always wins (manual override). Otherwise, if the drift gate is
    switched off in config, we retrain unconditionally too (config-level
    override). Only when the gate is on and not forced do we actually pull
    live data and run the drift check.
    """
    if force:
        logger.info(
            "--force passed -> skipping drift gate, retraining unconditionally."
        )
        return True, None

    if not DRIFT_GATE:
        logger.info("retrain.drift_gate=false in config -> retraining unconditionally.")
        return True, None

    logger.info("Checking for drift against the reference (training) distribution...")
    current = get_current_features()
    report = compute_drift(current)
    print(format_report(report))

    verdict = "DRIFT DETECTED" if report["drift_detected"] else "no significant drift"
    logger.info(
        "Drift verdict: %s (%d/%d features drifted)",
        verdict,
        report["n_drifted"],
        report["n_features"],
    )
    return report["drift_detected"], report


def refresh_data() -> None:
    """Regenerate every parquet file in data/ from fresh Binance data.

    Order matters: each stage reads the previous stage's output.
    """
    logger.info("Refreshing data: fetching latest raw OHLCV from Binance...")
    data_ingestion.main()

    logger.info("Refreshing data: cleaning raw candles...")
    make_dataset.main()

    logger.info("Refreshing data: rebuilding features...")
    make_features.main()

    logger.info("Refreshing data: rebuilding chronological train/test split...")
    make_split.main()


def train_candidate(X_train, y_train) -> RandomForestRegressor:
    """Fit a fresh challenger model with the same production hyperparameters."""
    model = RandomForestRegressor(**RF_PARAMS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def decide_promotion(
    cand_metrics: dict, champ_metrics: dict | None, margin: float
) -> bool:
    """Promote only if the candidate is actually better (lower RMSE) than the
    champion by at least `margin`. No champion yet (first-ever run) -> promote
    by default, since there's nothing to lose to.
    """
    if champ_metrics is None:
        return True
    return cand_metrics["rmse"] <= champ_metrics["rmse"] - margin


def promote(model, cand_metrics: dict, champ_metrics: dict | None, X_train) -> None:
    """Log the win to MLflow, register the model, overwrite the local champion
    joblib, and try to restart serving so it picks up the new weights.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    setup_mlflow()

    with mlflow.start_run(run_name="retrain_promoted") as run:
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_params(RF_PARAMS)
        mlflow.log_metrics(cand_metrics)
        if champ_metrics is not None:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champ_metrics.items()})
        mlflow.log_param("decision", "promoted")

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        logger.info("Run ID: %s", run.info.run_id)

    # Overwrite the local champion AFTER MLflow logging succeeded, so a
    # logging failure never leaves us with a half-promoted state.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Local champion overwritten -> %s", MODEL_PATH)

    # --- Sidecar metadata for /model/info (see model_meta.py) ---
    meta = build_meta(
        algorithm="RandomForestRegressor",
        params=RF_PARAMS,
        metrics=cand_metrics,
        n_features=X_train.shape[1],
        feature_names=list(X_train.columns),
        mlflow_run_id=run.info.run_id,
        registered_model_name=REGISTERED_MODEL_NAME,
        source="retrain_promoted",
        champion_metrics=champ_metrics,
    )
    save_meta(meta)

    restart_serving()


def restart_serving() -> None:
    """Restart the serving container so it picks up the new joblib.

    Non-fatal by design: the new model is already saved to disk at this
    point, so a failed restart just means someone needs to restart it
    manually — it must never raise and undo a successful promotion.
    """
    result = subprocess.run(
        ["docker", "restart", CONTAINER_NAME],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Restarted serving container '%s'.", CONTAINER_NAME)
    else:
        logger.warning(
            "Could not restart container '%s' (exit %d): %s",
            CONTAINER_NAME,
            result.returncode,
            result.stderr.strip(),
        )


def log_rejection(cand_metrics: dict, champ_metrics: dict | None) -> None:
    """Record that a retrain ran but the champion was kept — a visible trail
    on DagsHub even when nothing changes in production.
    """
    import mlflow

    setup_mlflow()
    with mlflow.start_run(run_name="retrain_rejected") as run:
        mlflow.log_metrics(cand_metrics)
        if champ_metrics is not None:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champ_metrics.items()})
        mlflow.log_param("decision", "rejected")
        logger.info("Run ID: %s", run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drift-gated retrain-and-validate orchestrator."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the drift gate and retrain unconditionally.",
    )
    args = parser.parse_args()

    do_retrain, _report = should_retrain(args.force)
    if not do_retrain:
        logger.info("No significant drift — keeping current model.")
        return

    # Load the champion BEFORE touching any data/artifacts, so we still have
    # a reference to compare against even after refresh_data() overwrites
    # the parquet files this run depends on.
    try:
        champion = joblib.load(MODEL_PATH)
        logger.info("Loaded current champion from %s", MODEL_PATH)
    except FileNotFoundError:
        champion = None
        logger.info("No champion found at %s (first-ever run).", MODEL_PATH)

    refresh_data()
    X_train, y_train, X_test, y_test = load_splits()

    logger.info("Training candidate model...")
    candidate = train_candidate(X_train, y_train)
    cand_metrics = evaluate(y_test.values, candidate.predict(X_test))

    # Both models are scored on the SAME fresh test set -> a fair
    # champion-vs-challenger comparison, not champion's old score vs
    # candidate's new score on different data.
    champ_metrics = (
        evaluate(y_test.values, champion.predict(X_test))
        if champion is not None
        else None
    )

    logger.info("Candidate metrics:")
    for k, v in cand_metrics.items():
        logger.info("  %-16s: %.6f", k, v)
    if champ_metrics is not None:
        logger.info("Champion metrics:")
        for k, v in champ_metrics.items():
            logger.info("  %-16s: %.6f", k, v)
    else:
        logger.info("Champion metrics: n/a (no champion yet)")

    if decide_promotion(cand_metrics, champ_metrics, PROMOTION_MARGIN):
        promote(candidate, cand_metrics, champ_metrics, X_train)
        logger.info("PROMOTED — candidate is now the champion.")
    else:
        log_rejection(cand_metrics, champ_metrics)
        logger.info("REJECTED — champion kept.")


if __name__ == "__main__":
    main()
