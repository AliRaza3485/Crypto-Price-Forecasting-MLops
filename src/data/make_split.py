"""
Train / test split for Crypto-Price-Forecasting-MLops.

Reads the featured dataset from data/processed/ and splits it into four files
(X_train, y_train, X_test, y_test), written back to data/processed/.

Why a CHRONOLOGICAL split (no shuffle)?
  This is a time series. If we shuffled rows randomly, future hours would leak
  into the training set and the model would "see the future" -> the offline
  score would be a lie (data leakage). So we split by TIME:
      first (1 - test_size) of the timeline -> TRAIN  (older data)
      last  test_size        of the timeline -> TEST   (newer data)
  The test set then behaves like real unseen future data.

Column roles (decided in make_features):
  * timestamp -> used as the INDEX of every file (kept for ordering and so we
    can later look up `close` to rebuild the predicted price). Not a feature.
  * close     -> REFERENCE only, dropped from X (feeding raw price would bring
    back the scale/extrapolation problem we avoided with a return target).
  * target    -> the label -> becomes y.
  * everything else -> the features -> become X.

All paths/params come from config/config.yaml. Run from the project root:
    python -m src.data.make_split
"""

import logging

import pandas as pd

from src.config import CONFIG, get_path

# --- Load settings from config (nothing hard-coded) ---
_data_cfg = CONFIG["data"]
_feat_cfg = CONFIG["features"]
_split_cfg = CONFIG["split"]

PROCESSED_DIR = get_path(_data_cfg["processed_dir"])
INPUT_FILE = PROCESSED_DIR / _feat_cfg["featured_file"]
TEST_SIZE = _split_cfg["test_size"]

# Columns that are NOT features (mirror of make_features).
TARGET_COL = "target"
REFERENCE_COL = "close"   # kept only for price reconstruction, never fed to the model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def split(df: pd.DataFrame):
    """Split a featured DataFrame into (X_train, X_test, y_train, y_test).

    timestamp becomes the index; `close` is dropped from X; `target` is y.
    The split is chronological — the last `TEST_SIZE` fraction of time is test.
    """
    df = df.sort_values("timestamp").set_index("timestamp")

    # Separate inputs (X) from the label (y). Drop the reference price from X.
    X = df.drop(columns=[TARGET_COL, REFERENCE_COL])
    y = df[[TARGET_COL]]

    # Chronological cut point: NO shuffle — order is preserved.
    n = len(df)
    split_idx = int(n * (1 - TEST_SIZE))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info("Total rows      : %d", n)
    logger.info("Train rows      : %d  (%s -> %s)", len(X_train), X_train.index.min(), X_train.index.max())
    logger.info("Test rows       : %d  (%s -> %s)", len(X_test), X_test.index.min(), X_test.index.max())
    logger.info("Feature columns : %d  %s", X.shape[1], list(X.columns))

    return X_train, X_test, y_train, y_test


def main() -> None:
    logger.info("Reading featured data from %s", INPUT_FILE)
    df = pd.read_parquet(INPUT_FILE)

    X_train, X_test, y_train, y_test = split(df)

    # Save all four files (timestamp travels along as the index).
    outputs = {
        _split_cfg["x_train_file"]: X_train,
        _split_cfg["y_train_file"]: y_train,
        _split_cfg["x_test_file"]: X_test,
        _split_cfg["y_test_file"]: y_test,
    }
    for filename, frame in outputs.items():
        path = PROCESSED_DIR / filename
        frame.to_parquet(path)   # index=True by default -> keeps timestamp
        logger.info("Saved %-16s %s", frame.shape, path.name)


if __name__ == "__main__":
    main()
