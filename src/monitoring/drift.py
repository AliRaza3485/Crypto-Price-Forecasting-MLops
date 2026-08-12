"""
Data-drift detection for Crypto-Price-Forecasting-MLops.

The model was trained on a fixed slice of history (``X_train``). Once it is
serving live, the incoming data can slowly drift away from what the model saw
during training -- the market gets more volatile, volume regimes change, etc.
When the *input feature distributions* move far enough, the model is effectively
being asked about conditions it never learned, so its predictions get less
trustworthy. This module measures that gap.

How it works
  * REFERENCE distribution = the training features (``X_train.parquet``), i.e.
    "what the model considers normal".
  * CURRENT distribution   = a batch of recent live features (supplied by the
    caller -- fetching it lives in the CLI/serving layer, not here, so this
    module stays pure and easy to test).

For each of the 14 features we compute two independent signals:
  * PSI (Population Stability Index) -- a single, interpretable shift number.
        < 0.10  stable | 0.10-0.20 moderate | >= 0.20 major/drift
  * KS test p-value -- probability the two samples share one distribution;
        a very small p-value means "they differ". Reported as extra evidence.

The DECISION metric is PSI: a feature is flagged as drifted when its PSI crosses
``psi_major``. (KS is famously over-sensitive on large samples -- with ~14k
reference rows it flags trivial differences -- so we keep it informational and
let the interpretable PSI threshold drive the verdict.) The overall report
raises the drift flag once at least ``min_drifted_features`` have drifted.

All thresholds come from config/config.yaml (nothing hard-coded here).
"""

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import CONFIG, get_path
from src.data.data_ingestion import fetch_recent_candles
from src.features.make_features import add_features

# --- Settings from config/config.yaml (nothing hard-coded) ---
_mon_cfg = CONFIG["monitoring"]
_data_cfg = CONFIG["data"]
KS_PVALUE_THRESHOLD = _mon_cfg["ks_pvalue_threshold"]
PSI_MODERATE = _mon_cfg["psi_moderate"]
PSI_MAJOR = _mon_cfg["psi_major"]
MIN_DRIFTED_FEATURES = _mon_cfg["min_drifted_features"]

# Which market + how much recent history to pull for the current batch.
SYMBOL = _data_cfg["symbol"]
INTERVAL = _data_cfg["interval"]
DRIFT_LOOKBACK = _mon_cfg["drift_lookback"]

# The reference is the training feature matrix written by src/data/make_split.py.
REFERENCE_PATH = get_path(CONFIG["data"]["processed_dir"]) / CONFIG["split"]["x_train_file"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_reference() -> pd.DataFrame:
    """Load the training features that define the 'normal' distribution.

    Cached: the reference never changes within a run, so we read it once.
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference features not found at {REFERENCE_PATH}. "
            "Run `python -m src.data.make_split` first."
        )
    return pd.read_parquet(REFERENCE_PATH)


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Steps:
      1. Cut the REFERENCE into `bins` equal-frequency buckets (deciles). Using
         the reference's own quantiles as the boundaries is what makes PSI a
         fair "did the shape move relative to training?" measure.
      2. Put both samples into those same buckets and take each bucket's share.
         Values outside the training range naturally fall into the edge buckets
         (that itself is a drift signal).
      3. PSI = sum over buckets of (cur% - ref%) * ln(cur% / ref%).

    A tiny epsilon guards against empty buckets (ln(0) / divide-by-zero). A
    near-constant reference (fewer distinct values than bins) returns 0.0, since
    a feature that barely varies can't meaningfully "drift" in shape.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Internal cut points = reference deciles, dropping the 0% and 100% ends.
    inner_quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    cut_points = np.unique(np.quantile(ref, inner_quantiles))
    if cut_points.size == 0:
        return 0.0  # reference has ~no variation -> no meaningful bins

    n_bins = cut_points.size + 1
    ref_counts = np.bincount(np.digitize(ref, cut_points), minlength=n_bins)
    cur_counts = np.bincount(np.digitize(cur, cut_points), minlength=n_bins)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_pvalue(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov p-value (probability of one distribution)."""
    return float(ks_2samp(np.asarray(reference, dtype=float),
                          np.asarray(current, dtype=float)).pvalue)


def _psi_level(value: float) -> str:
    """Human label for a PSI value, using the config thresholds."""
    if value >= PSI_MAJOR:
        return "major"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def compute_drift(current: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
    """Compare a batch of CURRENT features against the reference, per feature.

    Args:
      current:   a DataFrame containing (at least) the 14 model features. Extra
                 columns like timestamp/close are ignored.
      reference: override the reference (mainly for tests); defaults to X_train.

    Returns a report dict:
      {
        "n_features":   14,
        "n_drifted":    2,
        "drift_detected": bool,          # n_drifted >= min_drifted_features
        "features": [
            {"feature", "psi", "psi_level", "ks_pvalue", "drifted"}, ...
        ],
      }
    """
    if reference is None:
        reference = load_reference()

    features = list(reference.columns)
    missing = [f for f in features if f not in current.columns]
    if missing:
        raise ValueError(f"Current data is missing feature columns: {missing}")

    results = []
    for feature in features:
        ref_col = pd.to_numeric(reference[feature], errors="coerce").to_numpy(dtype=float)
        cur_col = pd.to_numeric(current[feature], errors="coerce").to_numpy(dtype=float)
        ref_col = ref_col[np.isfinite(ref_col)]
        cur_col = cur_col[np.isfinite(cur_col)]

        value = psi(ref_col, cur_col)
        results.append({
            "feature": feature,
            "psi": round(value, 4),
            "psi_level": _psi_level(value),
            "ks_pvalue": round(ks_pvalue(ref_col, cur_col), 4),
            "drifted": bool(value >= PSI_MAJOR),
        })

    n_drifted = sum(r["drifted"] for r in results)
    return {
        "n_features": len(features),
        "n_drifted": n_drifted,
        "drift_detected": bool(n_drifted >= MIN_DRIFTED_FEATURES),
        "features": results,
    }


# --- Live batch + CLI (this part touches the network) -----------------------

def get_current_features() -> pd.DataFrame:
    """Fetch a recent batch of candles from Binance and build model features.

    Uses the SAME add_features() as training and live prediction, so the
    'current' distribution is measured on features built identically to the
    reference -- no training/serving skew to muddy the drift signal.

    The window (monitoring.drift_lookback, ~30 days) is deliberately wide so
    there are plenty of rows to form real distributions, not just a few candles.
    """
    candles = fetch_recent_candles(SYMBOL, INTERVAL, DRIFT_LOOKBACK)
    features = add_features(candles)
    if features.empty:
        raise ValueError(
            "Not enough recent candles to build features for the drift check."
        )
    return features


def format_report(report: dict) -> str:
    """Render a drift report dict as a readable, aligned text table."""
    lines = [
        f"{'Feature':<18}{'PSI':>8}  {'Level':<10}{'KS p-value':>12}  {'Drift?':>7}",
        "-" * 60,
    ]
    for r in report["features"]:
        lines.append(
            f"{r['feature']:<18}{r['psi']:>8.4f}  {r['psi_level']:<10}"
            f"{r['ks_pvalue']:>12.4f}  {'YES' if r['drifted'] else 'no':>7}"
        )
    verdict = "DRIFT DETECTED" if report["drift_detected"] else "no significant drift"
    lines.append("-" * 60)
    lines.append(
        f"VERDICT: {verdict} ({report['n_drifted']}/{report['n_features']} features drifted)"
    )
    return "\n".join(lines)


def main() -> None:
    """Fetch recent live features, compare to training, print the drift report."""
    logger.info("Building current feature batch from recent Binance candles...")
    current = get_current_features()
    logger.info(
        "Current batch: %d rows (%s -> %s)",
        len(current), current["timestamp"].min(), current["timestamp"].max(),
    )

    report = compute_drift(current)
    print(format_report(report))


if __name__ == "__main__":
    main()
