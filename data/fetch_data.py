# =============================================================================
# data/fetch_data.py — Historical Price Data Downloader
#
# Downloads OHLCV data for the configured symbol from Yahoo Finance via
# yfinance, validates it, and saves it to a local CSV file for reuse.
#
# Usage:
#   python data/fetch_data.py
#
# The CSV is reused on subsequent runs to avoid repeated downloads.
# Delete the CSV file to force a fresh download.
# =============================================================================

import os
import sys
import pandas as pd
import yfinance as yf

# Allow imports from project root when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SYMBOL, START_DATE, END_DATE, DATA_DIR, DATA_FILE


def fetch_data(
    symbol: str = SYMBOL,
    start: str = START_DATE,
    end: str = END_DATE,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download (or load from cache) historical daily OHLCV price data.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker symbol (e.g. "AAPL", "RELIANCE.NS").
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.
    force_download : bool
        If True, re-downloads data even if a local CSV already exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.
        Index is a DatetimeIndex named "Date".
    """
    # ------------------------------------------------------------------
    # Use cached CSV if it exists and force_download is False
    # ------------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE) and not force_download:
        print(f"[INFO] Loading cached data from '{DATA_FILE}' ...")
        df = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True)
        print(f"[INFO] Loaded {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    # ------------------------------------------------------------------
    # Download fresh data from Yahoo Finance
    # ------------------------------------------------------------------
    print(f"[INFO] Downloading {symbol} data from Yahoo Finance ({start} -> {end}) ...")
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=True)

    if raw.empty:
        raise ValueError(
            f"[ERROR] No data returned for symbol '{symbol}'. "
            "Check the ticker or your internet connection."
        )

    # ------------------------------------------------------------------
    # Flatten MultiIndex columns (yfinance ≥ 0.2 returns MultiIndex)
    # ------------------------------------------------------------------
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only the standard OHLCV columns
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    # ------------------------------------------------------------------
    # Validate & clean
    # ------------------------------------------------------------------
    original_len = len(df)

    # 1. Drop rows where all OHLCV values are NaN
    df.dropna(how="all", inplace=True)

    # 2. Forward-fill isolated NaN cells (e.g. missing volume on holidays)
    df.ffill(inplace=True)

    # 3. Drop any remaining NaN rows
    df.dropna(inplace=True)

    removed = original_len - len(df)
    if removed:
        print(f"[WARN] Removed {removed} row(s) with missing values.")

    # 4. Sanity checks
    assert len(df) >= 252 * 5, (
        f"[ERROR] Only {len(df)} trading days found — need at least 5 years of data. "
        "Adjust START_DATE in config.py."
    )
    assert (df["Close"] > 0).all(), "[ERROR] Negative or zero Close prices detected."
    assert (df["Volume"] >= 0).all(), "[ERROR] Negative Volume values detected."

    # ------------------------------------------------------------------
    # Save to CSV
    # ------------------------------------------------------------------
    df.to_csv(DATA_FILE)
    print(f"[INFO] Saved {len(df)} rows to '{DATA_FILE}'")
    print(f"[INFO] Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"[INFO] Columns    : {list(df.columns)}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick statistical summary of the downloaded data."""
    print("\n" + "=" * 55)
    print(f"  Data Summary for {SYMBOL}")
    print("=" * 55)
    print(f"  Rows (trading days) : {len(df):,}")
    print(f"  Start               : {df.index[0].date()}")
    print(f"  End                 : {df.index[-1].date()}")
    print(f"  Close — Min         : ${df['Close'].min():.2f}")
    print(f"  Close — Max         : ${df['Close'].max():.2f}")
    print(f"  Close — Mean        : ${df['Close'].mean():.2f}")
    print(f"  Avg Daily Volume    : {df['Volume'].mean():,.0f}")
    print("=" * 55 + "\n")


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical stock data.")
    parser.add_argument("--symbol", type=str, default=SYMBOL,
                        help="Yahoo Finance ticker (default: from config.py)")
    parser.add_argument("--start",  type=str, default=START_DATE,
                        help="Start date YYYY-MM-DD (default: from config.py)")
    parser.add_argument("--end",    type=str, default=END_DATE,
                        help="End date YYYY-MM-DD (default: from config.py)")
    parser.add_argument("--force",  action="store_true",
                        help="Force re-download even if CSV already exists")
    args = parser.parse_args()

    data = fetch_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        force_download=args.force,
    )
    print_summary(data)
