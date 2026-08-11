"""
Data ingestion for Crypto-Price-Forecasting-MLops.

Pulls historical OHLCV candlestick data for a crypto symbol from the public
Binance API (no API key needed) and saves it as a Parquet file under data/raw/.
All settings (symbol, interval, lookback, paths) come from config/config.yaml.

Run from the project root as a module:
    python -m src.data.data_ingestion
"""

import logging

import pandas as pd
from binance.client import Client

from src.config import CONFIG, get_path

# --- Load settings from config/config.yaml (nothing hard-coded here) ---
_data_cfg = CONFIG["data"]
SYMBOL = _data_cfg["symbol"]
INTERVAL = _data_cfg["interval"]
LOOKBACK = _data_cfg["lookback"]
OUTPUT_FILE = get_path(_data_cfg["raw_dir"]) / _data_cfg["raw_file"]

# Binance returns 12 fields per candle; these are their names in order.
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_klines(symbol: str, interval: str, lookback: str) -> pd.DataFrame:
    """Download historical candlesticks from Binance into a DataFrame."""
    logger.info("Connecting to Binance (public API, no key needed)...")
    client = Client()

    logger.info("Fetching %s %s candles since '%s'...", symbol, interval, lookback)
    raw = client.get_historical_klines(symbol, interval, lookback)
    logger.info("Received %d candles.", len(raw))

    return pd.DataFrame(raw, columns=KLINE_COLUMNS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only time + OHLCV, fix data types, and sort by time."""
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()

    # Binance timestamps are milliseconds since epoch -> real datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    # Prices and volume arrive as text -> convert to numbers
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    df = df.rename(columns={"open_time": "timestamp"})
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    df = fetch_klines(SYMBOL, INTERVAL, LOOKBACK)
    df = clean(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    logger.info("Saved %d rows to %s", len(df), OUTPUT_FILE)
    logger.info("Date range: %s  ->  %s", df["timestamp"].min(), df["timestamp"].max())
    logger.info("Columns: %s", list(df.columns))


if __name__ == "__main__":
    main()
