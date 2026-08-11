"""
Data cleaning for Crypto-Price-Forecasting-MLops.

Reads the raw OHLCV snapshot from data/raw/, applies defensive cleaning
(duplicates, missing values, missing hours, ordering, sanity checks) and
writes a clean dataset to data/processed/.

Why defensive cleaning if the raw snapshot is already clean?
The EDA showed no nulls / duplicates / gaps in the historical file, so on
today's data these steps change nothing. But the SAME function will run on
LIVE data later (a missed hour, a repeated candle, a bad row), so we build
the safety net now.

All paths come from config/config.yaml. Run from the project root as a module:
    python -m src.data.make_dataset
"""

import logging

import pandas as pd

from src.config import CONFIG, get_path

# --- Load settings from config/config.yaml (nothing hard-coded here) ---
_data_cfg = CONFIG["data"]
RAW_FILE = get_path(_data_cfg["raw_dir"]) / _data_cfg["raw_file"]
OUTPUT_FILE = get_path(_data_cfg["processed_dir"]) / _data_cfg["processed_file"]

# The candle spacing we expect between consecutive rows (hourly data).
EXPECTED_FREQ = "1h"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply defensive cleaning to a raw OHLCV DataFrame and return a clean copy.

    Steps (each one only acts if there is something to fix):
      1. Sort by timestamp and drop exact duplicate rows.
      2. Drop duplicate timestamps (keep the last candle for that hour).
      3. Reindex onto a complete hourly grid so missing hours become explicit.
      4. Fill the gaps that step 3 created (forward-fill last known values).
      5. Drop any remaining rows with nulls and log a sanity check.
    """
    n_start = len(df)
    df = df.copy()

    # 1. Order by time (a time series must be chronological) and drop exact dups.
    df = df.sort_values("timestamp").reset_index(drop=True)
    exact_dups = df.duplicated().sum()
    if exact_dups:
        logger.warning("Dropping %d exact duplicate rows.", exact_dups)
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. Same hour appearing twice -> keep the most recent candle for that hour.
    ts_dups = df["timestamp"].duplicated().sum()
    if ts_dups:
        logger.warning("Dropping %d duplicate timestamps (keeping last).", ts_dups)
        df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    # 3. Build a complete hourly index from first to last timestamp. Any hour
    #    that is missing in the data will now show up as a row full of NaNs.
    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=EXPECTED_FREQ,
    )
    missing_hours = len(full_index) - len(df)
    if missing_hours:
        logger.warning("Found %d missing hour(s); reindexing onto full grid.", missing_hours)
    df = (
        df.set_index("timestamp")
        .reindex(full_index)
        .rename_axis("timestamp")
        .reset_index()
    )

    # 4. Fill the holes created by reindexing. For prices, carrying the last
    #    known value forward is the standard, leak-free choice (no future info).
    n_nulls = int(df[OHLCV_COLS].isnull().any(axis=1).sum())
    if n_nulls:
        logger.warning("Forward-filling %d row(s) with missing values.", n_nulls)
        df[OHLCV_COLS] = df[OHLCV_COLS].ffill()

    # 5. If the very first rows were NaN, ffill can't fix them -> drop them.
    remaining = int(df[OHLCV_COLS].isnull().any(axis=1).sum())
    if remaining:
        logger.warning("Dropping %d row(s) still null after fill.", remaining)
        df = df.dropna(subset=OHLCV_COLS).reset_index(drop=True)

    logger.info("Cleaning: %d rows in -> %d rows out.", n_start, len(df))
    return df


def main() -> None:
    logger.info("Reading raw data from %s", RAW_FILE)
    df = pd.read_parquet(RAW_FILE)

    df = clean(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    logger.info("Saved %d clean rows to %s", len(df), OUTPUT_FILE)
    logger.info("Date range: %s  ->  %s", df["timestamp"].min(), df["timestamp"].max())


if __name__ == "__main__":
    main()
