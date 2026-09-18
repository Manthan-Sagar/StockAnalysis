"""
Feature Engineering Module
Implements clean data preprocessing and computes per-ticker technical indicators
with zero lookahead leakage, creates one-hot ticker encodings, and prepares
the dataset for model training and Power BI reporting.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# Constants
DATA_RAW_DIR: Path = Path("data/raw")
DATA_PROCESSED_DIR: Path = Path("data/processed")


def clean_raw_data(
    combined_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cleans raw combined data:
    1. Forward fills isolated missing values per ticker, drops remaining nulls.
    2. Performs inner join on Date across all tickers so only days all three traded are kept.
    3. Sorts deterministically by Ticker then Date.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Long-format raw DataFrame containing Date, Ticker, Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Cleaned, synchronized long-format DataFrame.
    """
    df = combined_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()

    # Sort deterministically
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    # Forward fill isolated missing values within each ticker, drop any remaining NaNs
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df.groupby("Ticker")[col].ffill()

    df = df.dropna().reset_index(drop=True)

    # Find common dates across all unique tickers (inner join on date)
    tickers = df["Ticker"].unique()
    common_dates = None
    for ticker in tickers:
        ticker_dates = set(df[df["Ticker"] == ticker]["Date"])
        if common_dates is None:
            common_dates = ticker_dates
        else:
            common_dates = common_dates.intersection(ticker_dates)

    print(f"Total synchronized trading dates common to all tickers: {len(common_dates)}")
    df = df[df["Date"].isin(common_dates)].copy()

    # Sort deterministically by Ticker, then Date
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = DATA_PROCESSED_DIR / "cleaned_data.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved synchronized clean data -> {cleaned_path}")

    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Wilder's 14-day Relative Strength Index (RSI).

    Parameters
    ----------
    series : pd.Series
        Price series (Close).
    period : int
        RSI lookback window (default: 14).

    Returns
    -------
    pd.Series
        RSI series bounded [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing (alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where loss is 0
    rsi = rsi.fillna(100.0)
    return rsi


def compute_ticker_features(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features for a single ticker's price series.
    Ensures zero cross-ticker leakage.

    Parameters
    ----------
    df_ticker : pd.DataFrame
        Price series for a single ticker sorted chronologically by Date.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with technical indicators and prediction target.
    """
    df = df_ticker.copy().sort_values("Date").reset_index(drop=True)
    close = df["Close"]
    volume = df["Volume"]

    # 1. Daily Return (% change of Close vs previous day)
    df["Daily_Return"] = close.pct_change() * 100.0

    # 2. Target: Next Day's Daily Return (shifted -1)
    df["Target_Next_Return"] = df["Daily_Return"].shift(-1)

    # 3. Simple Moving Averages
    df["SMA_10"] = close.rolling(window=10).mean()
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()

    # 4. Exponential Moving Averages and MACD
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 5. Rolling Volatility of Daily Returns
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    # 6. Bollinger Bands (20-day, 2 std) and Bollinger Width
    rolling_std_20 = close.rolling(window=20).std()
    df["Bollinger_Upper"] = df["SMA_20"] + (2.0 * rolling_std_20)
    df["Bollinger_Lower"] = df["SMA_20"] - (2.0 * rolling_std_20)
    df["Bollinger_Width"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df["SMA_20"]

    # 7. Relative Strength Index (RSI 14)
    df["RSI_14"] = compute_rsi(close, period=14)

    # 8. Volume Dynamics
    df["Volume_Change"] = volume.pct_change() * 100.0
    df["Volume_SMA_10"] = volume.rolling(window=10).mean()

    # 9. Spreads and Relative Valuation Ratios
    df["High_Low_Spread"] = (df["High"] - df["Low"]) / close
    df["Close_to_SMA20_Ratio"] = close / df["SMA_20"]

    # 10. Return Lags
    df["Return_Lag_1"] = df["Daily_Return"].shift(1)
    df["Return_Lag_2"] = df["Daily_Return"].shift(2)
    df["Return_Lag_3"] = df["Daily_Return"].shift(3)

    # Drop the first 50 rows per ticker (due to SMA_50 burn-in) and drop last row (target NaN)
    # Plus any remaining NaNs
    df = df.iloc[50:].copy()
    df = df.dropna(subset=["Target_Next_Return"]).copy()

    return df


def engineer_features(
    cleaned_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Engineers technical features across all tickers with strict per-ticker isolation,
    applies one-hot encoding for Ticker, and removes burn-in NaNs.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned long-format DataFrame.

    Returns
    -------
    pd.DataFrame
        Fully featured DataFrame ready for ML modeling and Power BI export.
    """
    featured_list: List[pd.DataFrame] = []

    for ticker, group in cleaned_df.groupby("Ticker"):
        print(f"Engineering features for {ticker} ({len(group)} rows)...")
        feat_df = compute_ticker_features(group)
        print(f"  -> {len(feat_df)} rows remaining after 50-day warm-up and target shift.")
        featured_list.append(feat_df)

    combined_featured = pd.concat(featured_list, ignore_index=True)

    # One-hot encode Ticker (Ticker_AAPL, Ticker_MSFT, Ticker_TSLA)
    ticker_dummies = pd.get_dummies(combined_featured["Ticker"], prefix="Ticker", dtype=float)
    combined_featured = pd.concat([combined_featured, ticker_dummies], axis=1)

    # Final dropna check
    initial_len = len(combined_featured)
    combined_featured = combined_featured.dropna().reset_index(drop=True)
    if len(combined_featured) < initial_len:
        print(f"Dropped {initial_len - len(combined_featured)} residual NaN rows.")

    # Sort deterministically
    combined_featured = combined_featured.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    featured_path = DATA_PROCESSED_DIR / "featured_data.csv"
    combined_featured.to_csv(featured_path, index=False)
    print(f"\nSaved engineered featured data -> {featured_path}")
    print(f"Total rows: {len(combined_featured)}, Total columns: {len(combined_featured.columns)}")

    return combined_featured


def run_feature_pipeline() -> pd.DataFrame:
    """
    Orchestrate loading raw data, cleaning, and feature engineering.
    """
    raw_path = DATA_RAW_DIR / "combined_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run data_fetch.py first.")

    raw_df = pd.read_csv(raw_path)
    clean_df = clean_raw_data(raw_df)
    featured_df = engineer_features(clean_df)
    return featured_df


if __name__ == "__main__":
    run_feature_pipeline()
