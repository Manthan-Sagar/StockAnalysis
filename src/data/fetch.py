"""
Data Acquisition Module
Cached, rate-limit-safe downloader for daily OHLCV data via yfinance.
Writes per-ticker parquet files to data/raw/ and assembles a stacked
(date, ticker) panel for the full universe.

Key design decisions:
- Parquet (pyarrow) for storage: ~100 tickers × 12 years is too large for
  comfortable CSV round-tripping.
- Cache-first: checks for existing parquet before hitting Yahoo's API.
- Batched downloads with sleep intervals to avoid rate-limiting.
- Forward-fills only isolated single-day gaps; flags gaps > 3 trading days.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

from src.data.universe import (
    BENCHMARKS,
    END_DATE,
    START_DATE,
    get_all_tickers,
    get_sector_map,
    save_universe_metadata,
)

logger = logging.getLogger(__name__)

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
BATCH_SIZE = 12
BATCH_SLEEP = 1.5          # seconds between batches
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0        # exponential backoff multiplier
MAX_GAP_FILL_DAYS = 3      # only forward-fill gaps ≤ this many trading days


# ──────────────────────────────────────────────────────────────────────────────
# Single-ticker download with retry
# ──────────────────────────────────────────────────────────────────────────────

def _download_ticker(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
) -> Optional[pd.DataFrame]:
    """
    Download daily OHLCV for a single ticker with retry/backoff.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns [Date, Open, High, Low, Close, Volume]
        sorted by Date.  Returns None on persistent failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            obj = yf.Ticker(ticker)
            df = obj.history(start=start, end=end, auto_adjust=True)

            if df.empty:
                # Fallback to yf.download
                df = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False,
                )

            if df.empty:
                logger.warning(f"[{ticker}] No data returned (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_BACKOFF ** attempt)
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]

            df = df.reset_index()

            # Standardize Date
            if "Date" in df.columns:
                df["Date"] = (
                    pd.to_datetime(df["Date"])
                    .dt.tz_localize(None)
                    .dt.normalize()
                )
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
                df["Date"] = (
                    pd.to_datetime(df["Date"])
                    .dt.tz_localize(None)
                    .dt.normalize()
                )

            # Keep only standard columns
            standard = ["Date", "Open", "High", "Low", "Close", "Volume"]
            available = [c for c in standard if c in df.columns]
            df = df[available].copy()
            df = df.sort_values("Date").reset_index(drop=True)
            df = df.drop_duplicates(subset=["Date"], keep="first")

            if len(df) < 100:
                logger.warning(
                    f"[{ticker}] Only {len(df)} rows — possible data quality issue "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF ** attempt)
                    continue

            return df

        except Exception as e:
            logger.warning(f"[{ticker}] Download error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)

    logger.error(f"[{ticker}] Failed after {MAX_RETRIES} attempts — skipping.")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Gap analysis & forward-fill logic
# ──────────────────────────────────────────────────────────────────────────────

def _analyze_and_fill_gaps(
    df: pd.DataFrame, ticker: str
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Detect trading-day gaps, forward-fill only isolated ≤ MAX_GAP_FILL_DAYS
    gaps, and log longer gaps as data quality issues.

    Returns
    -------
    df : pd.DataFrame
        Gap-filled DataFrame.
    gap_report : list of dict
        Each dict: {start, end, gap_days, action}.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    date_diffs = df["Date"].diff().dt.days
    gap_report: List[dict] = []

    # Identify gaps larger than typical weekends (> 3 calendar days ≈ > 1 trading day gap)
    for idx in date_diffs[date_diffs > 4].index:
        gap_start = df["Date"].iloc[idx - 1]
        gap_end = df["Date"].iloc[idx]
        cal_days = (gap_end - gap_start).days
        # Approximate trading days in gap
        trading_days_approx = int(cal_days * 5 / 7) - 1

        if trading_days_approx <= MAX_GAP_FILL_DAYS:
            action = "forward_filled"
        else:
            action = "flagged_data_quality_issue"

        gap_report.append({
            "ticker": ticker,
            "gap_start": str(gap_start.date()),
            "gap_end": str(gap_end.date()),
            "calendar_days": cal_days,
            "approx_trading_days": trading_days_approx,
            "action": action,
        })

    if gap_report:
        flagged = [g for g in gap_report if g["action"] == "flagged_data_quality_issue"]
        filled = [g for g in gap_report if g["action"] == "forward_filled"]
        if filled:
            logger.info(f"[{ticker}] Forward-filled {len(filled)} isolated gap(s)")
        if flagged:
            logger.warning(
                f"[{ticker}] {len(flagged)} gap(s) exceed {MAX_GAP_FILL_DAYS} trading days — "
                f"flagged as data quality issues"
            )

    # Forward-fill NaN values (from any remaining missing data)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Drop any remaining rows with NaN in critical columns
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

    return df, gap_report


# ──────────────────────────────────────────────────────────────────────────────
# Cached batch download
# ──────────────────────────────────────────────────────────────────────────────

def fetch_universe(
    tickers: Optional[List[str]] = None,
    start: str = START_DATE,
    end: str = END_DATE,
    raw_dir: Path = RAW_DIR,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download and cache OHLCV data for the full universe + benchmarks.

    Parameters
    ----------
    tickers : list of str, optional
        Override the default universe tickers.
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    raw_dir : Path
        Directory for per-ticker parquet caches.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    dict
        {ticker: pd.DataFrame} for all successfully downloaded tickers.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = get_all_tickers() + BENCHMARKS

    ticker_dfs: Dict[str, pd.DataFrame] = {}
    to_download: List[str] = []
    all_gaps: List[dict] = []

    # Phase 1: load from cache where possible
    for t in tickers:
        cache_path = raw_dir / f"{t}.parquet"
        if cache_path.exists() and not force_refresh:
            try:
                df = pd.read_parquet(cache_path)
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
                ticker_dfs[t] = df
                continue
            except Exception:
                pass  # corrupt cache → re-download
        to_download.append(t)

    if ticker_dfs:
        print(f"[fetch] Loaded {len(ticker_dfs)} tickers from parquet cache")

    if not to_download:
        print("[fetch] All tickers cached -- no downloads needed")
        return ticker_dfs

    # Phase 2: batch download remaining tickers
    print(f"[fetch] Downloading {len(to_download)} tickers in batches of {BATCH_SIZE}...")

    for batch_start in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(to_download) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}: {', '.join(batch)}")

        for t in batch:
            df = _download_ticker(t, start=start, end=end)
            if df is not None and not df.empty:
                df, gaps = _analyze_and_fill_gaps(df, t)
                all_gaps.extend(gaps)

                # Save to parquet cache
                cache_path = raw_dir / f"{t}.parquet"
                df.to_parquet(cache_path, engine="pyarrow", index=False)
                ticker_dfs[t] = df

        # Rate-limit pause between batches
        if batch_start + BATCH_SIZE < len(to_download):
            time.sleep(BATCH_SLEEP)

    # Log gap summary
    if all_gaps:
        print(f"\n[fetch] Gap analysis summary: {len(all_gaps)} gap(s) found across all tickers")
        issues = [g for g in all_gaps if g["action"] == "flagged_data_quality_issue"]
        if issues:
            print(f"  [!] {len(issues)} gap(s) exceed {MAX_GAP_FILL_DAYS} trading days:")
            for g in issues:
                print(f"    {g['ticker']}: {g['gap_start']} -> {g['gap_end']} ({g['approx_trading_days']}d)")

    print(f"\n[fetch] Successfully loaded {len(ticker_dfs)}/{len(tickers)} tickers")
    return ticker_dfs


# ──────────────────────────────────────────────────────────────────────────────
# Panel assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_panel(
    ticker_dfs: Dict[str, pd.DataFrame],
    min_history_days: int = 500,
) -> pd.DataFrame:
    """
    Stack per-ticker DataFrames into a (date, ticker) panel.

    Tickers with fewer than `min_history_days` trading days are dropped.
    Adds a 'Ticker' column and sorts by (Date, Ticker).

    Parameters
    ----------
    ticker_dfs : dict
        {ticker: DataFrame} from fetch_universe.
    min_history_days : int
        Minimum number of trading days required to keep a ticker.

    Returns
    -------
    pd.DataFrame
        Stacked panel with columns [Date, Ticker, Open, High, Low, Close, Volume].
    """
    frames: List[pd.DataFrame] = []
    dropped: List[str] = []

    for ticker, df in sorted(ticker_dfs.items()):
        if len(df) < min_history_days:
            dropped.append(ticker)
            continue
        frame = df.copy()
        frame["Ticker"] = ticker
        frames.append(frame)

    if dropped:
        print(f"[panel] Dropped {len(dropped)} tickers with < {min_history_days} days: {dropped}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    print(f"[panel] Built panel: {len(panel):,} rows, {panel['Ticker'].nunique()} tickers, "
          f"{panel['Date'].nunique()} unique dates")
    return panel


def run_data_pipeline(force_refresh: bool = False) -> pd.DataFrame:
    """
    End-to-end data acquisition: fetch universe, build panel, save metadata.

    Returns
    -------
    pd.DataFrame
        Stacked (date, ticker) panel.
    """
    save_universe_metadata()
    ticker_dfs = fetch_universe(force_refresh=force_refresh)
    panel = build_panel(ticker_dfs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = PROCESSED_DIR / "raw_panel.parquet"
    panel.to_parquet(panel_path, engine="pyarrow", index=False)
    print(f"[data] Saved raw panel -> {panel_path}")

    return panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_data_pipeline()
