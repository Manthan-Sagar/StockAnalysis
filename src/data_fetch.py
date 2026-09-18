"""
Data Acquisition Module
Fetches daily OHLCV equity data for NASDAQ large-cap stocks (AAPL, MSFT, TSLA)
using yfinance, validates data hygiene, and saves raw files.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import yfinance as yf

# Configuration constants
TICKERS: List[str] = ["AAPL", "MSFT", "TSLA"]
START_DATE: str = "2023-01-01"
END_DATE: str = "2024-12-31"
DATA_RAW_DIR: Path = Path("data/raw")


def fetch_ticker_data(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a given ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    start : str
        Start date string in 'YYYY-MM-DD' format.
    end : str
        End date string in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV DataFrame with Date as a column.
    """
    print(f"Fetching historical data for {ticker} ({start} to {end})...")
    # yf.Ticker(...).history provides exact auto-adjusted series reliably
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        # Fallback to yf.download if history returns empty
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.reset_index()

    # Standardize Date column
    if "Date" in df.columns:
        # Strip timezone if present so joins and comparisons are uniform
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()

    # Standardize column naming
    standard_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in standard_cols if c in df.columns]
    df = df[available_cols].copy()

    # Sort deterministically
    df = df.sort_values("Date").reset_index(drop=True)

    # Validate non-emptiness
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    return df


def validate_raw_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Perform sanity checks on downloaded ticker data.

    Parameters
    ----------
    df : pd.DataFrame
        Ticker dataframe.
    ticker : str
        Ticker symbol.
    """
    row_count = len(df)
    print(f"[{ticker}] Total trading days: {row_count}")

    # Check trading days count (~500 trading days over 2 years)
    if row_count < 450 or row_count > 550:
        print(f"Warning: Expected ~500 trading days for {ticker}, but got {row_count}.")

    # Check for duplicate dates
    duplicate_dates = df["Date"].duplicated().sum()
    if duplicate_dates > 0:
        raise ValueError(f"Found {duplicate_dates} duplicate dates in {ticker} data!")

    # Check for null values in key OHLCV fields
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Missing values detected in {ticker}:\n{null_counts}")
    else:
        print(f"[{ticker}] Zero duplicate dates and zero missing values.")


def fetch_all_and_combine(
    tickers: List[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path = DATA_RAW_DIR
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Fetch raw data for all tickers, save individual CSVs, combine into a
    long-format DataFrame with a 'Ticker' column, and save combined_raw.csv.

    Parameters
    ----------
    tickers : List[str]
        List of stock symbols.
    start : str
        Start date string.
    end : str
        End date string.
    output_dir : Path
        Directory to save raw CSV files.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame]
        Dictionary of per-ticker dataframes, and combined long-format DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_dfs: Dict[str, pd.DataFrame] = {}
    combined_list: List[pd.DataFrame] = []

    for ticker in tickers:
        df = fetch_ticker_data(ticker, start=start, end=end)
        validate_raw_data(df, ticker)

        # Save individual raw CSV
        individual_path = output_dir / f"{ticker}.csv"
        df.to_csv(individual_path, index=False)
        print(f"Saved {ticker} raw data -> {individual_path}")

        ticker_dfs[ticker] = df

        # Add Ticker column for long-format combination
        df_long = df.copy()
        df_long["Ticker"] = ticker
        # Reorder columns: Date, Ticker, Open, High, Low, Close, Volume
        cols = ["Date", "Ticker"] + [c for c in df_long.columns if c not in ["Date", "Ticker"]]
        df_long = df_long[cols]
        combined_list.append(df_long)

    # Combine into single long-format DataFrame
    combined_df = pd.concat(combined_list, ignore_index=True)
    combined_df = combined_df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    combined_path = output_dir / "combined_raw.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined long-format raw data -> {combined_path}")

    # Print summary diagnostics
    print("\n" + "=" * 50)
    print("COMBINED RAW DATA INFO:")
    print("=" * 50)
    combined_df.info()
    print("\n" + "=" * 50)
    print("COMBINED RAW DATA SUMMARY STATISTICS:")
    print("=" * 50)
    print(combined_df.groupby("Ticker").describe().T)

    return ticker_dfs, combined_df


if __name__ == "__main__":
    fetch_all_and_combine()
