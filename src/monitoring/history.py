"""
Prediction history logging & resolution for Crypto-Price-Forecasting-MLops.

Every live prediction (GET /predict/live) is logged to a small SQLite
database, together with the actual price once enough time has passed for
the prediction's target hour to arrive. This turns single "point in time"
predictions into a track record the frontend can chart: predicted vs
actual, over time.

Design notes
  * SQLite, not a separate DB service -- the whole app already runs as one
    Docker container on one EC2 box, so a file-based DB avoids running (and
    persisting) a second service.
  * A NEW sqlite3 connection is opened per function call rather than kept
    open globally. The background scheduler (src/api/app.py) runs on its
    OWN thread (APScheduler's BackgroundScheduler), separate from the
    thread(s) serving HTTP requests -- sqlite3 connections aren't safe to
    share across threads, so "one connection per call" sidesteps that
    entirely instead of manually juggling check_same_thread=False + locks.
  * All timestamps are stored as naive UTC ISO strings -- this matches
    predict_from_candles()'s as_of_time / predicted_for_time, which come
    from Binance candles parsed with pd.to_datetime(..., unit="ms"), i.e.
    already naive UTC. Keeping everything naive avoids aware/naive
    comparison errors -- use datetime.utcnow() here, never
    datetime.now(timezone.utc).

Quick self-check from the project root:
    python -m src.monitoring.history
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import CONFIG, get_path

logger = logging.getLogger(__name__)

_history_cfg = CONFIG["history"]
DB_PATH = get_path(_history_cfg["db_dir"]) / _history_cfg["db_file"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_time TEXT NOT NULL UNIQUE,
    current_price REAL NOT NULL,
    predicted_return REAL NOT NULL,
    predicted_price REAL NOT NULL,
    predicted_for_time TEXT NOT NULL,
    actual_price REAL,
    error REAL
)
"""


def init_db() -> None:
    """Create the DB file + table if they don't exist yet. Safe to call repeatedly."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_TABLE)
    logger.info("History DB ready at %s", DB_PATH)


def log_prediction(result: dict) -> None:
    """Insert one live-prediction result (predict_from_candles()'s dict shape).

    as_of_time is UNIQUE -- if the scheduler ever runs twice for the same
    candle (e.g. a restart right after a poll), the duplicate insert is
    silently ignored instead of erroring or double-logging.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO predictions
                (as_of_time, current_price, predicted_return, predicted_price, predicted_for_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result["as_of_time"],
                result["current_price"],
                result["predicted_return"],
                result["predicted_price"],
                result["predicted_for_time"],
            ),
        )
    logger.info("Logged prediction as_of=%s", result["as_of_time"])


def resolve_pending(candles: pd.DataFrame) -> int:
    """Fill in actual_price/error for past predictions whose target hour has arrived.

    Looks up each still-unresolved row's predicted_for_time against the
    supplied candles (cleaned OHLCV -- same shape data_ingestion.clean()
    returns) and, where an exact hourly candle exists for that time, fills
    in the real close price and the prediction error.

    Args:
        candles: recent cleaned candles (e.g. the same batch just used for a
            live prediction) -- must have "timestamp" and "close" columns.

    Returns:
        How many rows were resolved.
    """
    lookup = {
        pd.Timestamp(ts).isoformat(): float(close)
        for ts, close in zip(candles["timestamp"], candles["close"])
    }
    if not lookup:
        return 0

    with sqlite3.connect(DB_PATH) as conn:
        pending = conn.execute(
            "SELECT id, predicted_for_time, predicted_price FROM predictions "
            "WHERE actual_price IS NULL"
        ).fetchall()

        resolved = 0
        for row_id, predicted_for_time, predicted_price in pending:
            actual_price = lookup.get(predicted_for_time)
            if actual_price is None:
                continue
            error = actual_price - predicted_price
            conn.execute(
                "UPDATE predictions SET actual_price = ?, error = ? WHERE id = ?",
                (actual_price, error, row_id),
            )
            resolved += 1

    if resolved:
        logger.info("Resolved %d pending prediction(s)", resolved)
    return resolved


def get_history(hours: int = 24) -> list[dict]:
    """Return logged predictions (resolved + pending) from the last `hours`, newest first."""
    now_naive_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = (now_naive_utc - timedelta(hours=hours)).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT as_of_time, current_price, predicted_return, predicted_price, "
            "predicted_for_time, actual_price, error FROM predictions "
            "WHERE as_of_time >= ? ORDER BY as_of_time DESC",
            (cutoff,),
        ).fetchall()

    return [dict(row) for row in rows]


def main() -> None:
    """Self-check: init the DB and print how many rows are currently stored."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    logger.info("History DB at %s has %d row(s)", DB_PATH, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
