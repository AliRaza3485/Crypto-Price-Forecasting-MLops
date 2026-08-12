"""
Feature engineering for Crypto-Price-Forecasting-MLops.

Reads the clean dataset from data/processed/ and builds model-ready features,
plus the prediction target, then writes the result back to data/processed/.

Design rules (very important for a time series):
  * FEATURES may only use past/current information (lags, rolling windows that
    look BACKWARD). Never anything from the future -> otherwise the model
    "cheats" and the offline score is a lie (data leakage).
  * The TARGET looks forward: the next-hour RETURN (% change), not the raw
    price. We predict a return and later rebuild the price at prediction time:
        predicted_price = current_close * (1 + predicted_return)

Why a return target instead of the raw price (Option 2, from the EDA)?
  * Price trends and keeps hitting new highs (52k -> 126k), so a tree model
    trained on old prices cannot extrapolate to unseen higher prices.
  * Returns are scale-free (they hover around 0 in every price regime), so the
    model generalises across price levels. This is the correct framing for
    non-stationary crypto prices.

What we keep / drop:
  * DROP raw open/high/low/volume  -> scale-dependent and (OHLC) ~1.00
    correlated; we already distilled them into returns / volatility / log_volume.
  * KEEP `close` as a REFERENCE column (NOT a feature) -> needed to rebuild the
    predicted price and to compute the actual price during evaluation.

Two-function split (added for live inference):
  * add_features(df)   -> ONLY the 14 backward-looking features. No target.
                           Keeps the latest row (needed to predict the NEXT hour
                           live, since that row has no target yet).
  * build_features(df) -> calls add_features(df), then adds the forward target
                           and drops NaN rows. Used for TRAINING. Output is
                           unchanged from before the refactor.

All paths/params come from config/config.yaml. Run from the project root:
    python -m src.features.make_features
"""

import logging

import numpy as np
import pandas as pd

from src.config import CONFIG, get_path

# --- Load settings from config (nothing hard-coded) ---
_data_cfg = CONFIG["data"]
_feat_cfg = CONFIG["features"]

INPUT_FILE = get_path(_data_cfg["processed_dir"]) / _data_cfg["processed_file"]
OUTPUT_FILE = get_path(_data_cfg["processed_dir"]) / _feat_cfg["featured_file"]

HORIZON = _feat_cfg["target_horizon"]
LAGS = _feat_cfg["lags"]
ROLL_WINDOWS = _feat_cfg["roll_windows"]

# Columns that are NOT features: the time index, the price reference used only
# for reconstruction/evaluation, and the target itself.
NON_FEATURE_COLS = ["timestamp", "close", "target"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ONLY the 14 backward-looking features to a clean OHLCV frame. No target.

    Every feature here looks BACKWARD (past/current close/volume only), so this
    function is safe to call on live data: it never needs a future price.

    Because there is no target computed in this function, the most RECENT row
    is never dropped for a missing target. This is exactly the row a live
    prediction service needs: "given everything known up to and including the
    current hour, what features describe this hour?" -> feed that row to the
    model to predict the NEXT hour's return.

    The only rows dropped here are the warm-up rows at the START of the
    dataframe, where lags/rolling windows don't have enough history yet
    (e.g. return_lag_24 needs 24 prior rows to exist).

    Returns:
        DataFrame with columns: timestamp, close (reference), + 14 features.
    """
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    # --- Base signal: hourly return (% change of close vs previous hour) ---
    df["return_1h"] = df["close"].pct_change()

    # --- Lagged returns: what the last N hours' returns were ---
    for lag in LAGS:
        df[f"return_lag_{lag}"] = df["return_1h"].shift(lag)

    # --- Trend & volatility from rolling windows (all look backward) ---
    for w in ROLL_WINDOWS:
        # Where is price vs its own moving average? (>1 above, <1 below) -> scale-free
        ma = df["close"].rolling(window=w).mean()
        df[f"close_over_ma_{w}"] = df["close"] / ma
        # How choppy has the market been? (std of recent returns)
        df[f"volatility_{w}"] = df["return_1h"].rolling(window=w).std()

    # --- Volume: tame the right-skew with a log transform ---
    df["log_volume"] = np.log1p(df["volume"])

    # --- Keep timestamp + close (reference) + the 14 features; drop raw OHLCV ---
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c not in ("open", "high", "low", "volume")
    ]
    df = df[["timestamp", "close", *feature_cols]]

    # --- Drop ONLY the warm-up NaN rows at the start (from lags/rolling windows) ---
    # NOTE: no target column exists yet, so dropna() here can never remove the
    # latest row -> the latest row always survives.
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        "add_features: dropped %d warm-up rows with NaNs -> %d usable rows (latest row kept).",
        n_before - len(df), len(df),
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scale-free features + a forward-return target to a clean OHLCV frame.

    Internally calls add_features(df) to get the 14 backward-looking features,
    then adds the forward-looking target on top. This function is for TRAINING
    only (it needs the future price to compute the target), so the last
    HORIZON row(s) are dropped -- that is expected and correct here.

    Output is identical to the pre-refactor single-function version: same
    rows, same columns, same values.
    """
    df = add_features(df)

    # --- TARGET: next-hour RETURN (scale-free), not the raw price ---
    #   target_t = close_{t+HORIZON} / close_t - 1
    df["target"] = df["close"].shift(-HORIZON) / df["close"] - 1

    # --- Reorder: timestamp + close (reference) + features + target ---
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df = df[["timestamp", "close", *feature_cols, "target"]]

    # --- Drop rows made NaN by the target (the last HORIZON rows) ---
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        "build_features: dropped %d rows with no future target -> %d usable rows.",
        n_before - len(df), len(df),
    )

    return df


def main() -> None:
    logger.info("Reading clean data from %s", INPUT_FILE)
    df = pd.read_parquet(INPUT_FILE)

    df = build_features(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    logger.info("Saved %d rows to %s", len(df), OUTPUT_FILE)
    logger.info("Target: next-%dh return (scale-free). Reference col: close.", HORIZON)
    logger.info("Built %d feature columns: %s", len(feature_cols), feature_cols)
    logger.info("Date range: %s  ->  %s", df["timestamp"].min(), df["timestamp"].max())


if __name__ == "__main__":
    main()
