"""
Volatility & Microstructure Feature Engineering Module
Computes advanced volatility estimators, liquidity metrics, regime indicators,
and standard technical features across the equity panel.

Feature taxonomy:
- "Denoised / regime-aware" features: Parkinson vol, Garman-Klass vol, Amihud
  illiquidity, vol regime, volume z-score, fracdiff close — these capture
  structural market characteristics that basic technicals miss.
- "Baseline technicals": SMA, RSI, MACD — standard momentum/trend indicators.
  Not wrong, just not sufficient alone.  Included for completeness.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.data.universe import get_sector_map, get_sector_codes


# ──────────────────────────────────────────────────────────────────────────────
# Advanced volatility estimators
# ──────────────────────────────────────────────────────────────────────────────

def parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int = 20
) -> pd.Series:
    """
    Parkinson (1980) range-based volatility estimator.
    More efficient than close-to-close volatility as it uses intraday range.

    Formula: σ = sqrt( (1 / 4n·ln2) · Σ [ln(H/L)]² )
    """
    log_hl = np.log(high / low.replace(0, np.nan))
    return np.sqrt(
        log_hl.pow(2).rolling(window).mean() / (4 * np.log(2))
    )


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Garman-Klass (1980) volatility estimator.
    Uses OHLC data for improved efficiency over Parkinson.

    Formula: σ = sqrt( (1/n) · Σ [0.5·(ln(H/L))² − (2ln2−1)·(ln(C/O))²] )
    """
    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))

    term = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
    return np.sqrt(term.rolling(window).mean().clip(lower=0))


def amihud_illiquidity(
    returns: pd.Series,
    dollar_volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio: average |return| / dollar volume.

    Captures price impact per unit of trading — higher values indicate
    less liquid names where trades move prices more.

    Formula: ILLIQ = (1/n) · Σ |r_i| / DollarVolume_i × 10^6
    """
    ratio = returns.abs() / dollar_volume.replace(0, np.nan) * 1e6
    return ratio.rolling(window).mean()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Average True Range (ATR): smoothed measure of daily price range.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Regime & microstructure features
# ──────────────────────────────────────────────────────────────────────────────

def vol_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_window: int = 20,
    long_window: int = 200,
) -> pd.Series:
    """
    Volatility regime indicator: short ATR / long ATR.
    Values > 1 indicate elevated volatility regime.
    """
    atr_short = atr(high, low, close, window=short_window)
    atr_long = atr(high, low, close, window=long_window)
    return atr_short / atr_long.replace(0, np.nan)


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Volume z-score: (volume − rolling mean) / rolling std.
    Detects unusual volume spikes.
    """
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std().replace(0, np.nan)
    return (volume - mean) / std


# ──────────────────────────────────────────────────────────────────────────────
# Standard technical indicators (baseline)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's Relative Strength Index.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": histogram,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Panel-level feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all cross-sectional features per ticker.

    All features are computed strictly per-ticker to prevent cross-asset leakage.
    Features are grouped into:
    1. Advanced volatility / microstructure (the upgrade)
    2. Baseline technicals (for comparison)
    3. Sector encoding (legitimate cross-sectional feature)

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with columns [Date, Ticker, Open, High, Low, Close, Volume,
        daily_return, residual_return, fracdiff_close, ...].

    Returns
    -------
    pd.DataFrame
        Panel with all engineered features added.
    """
    df = panel.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Sector encoding (integer for LightGBM categorical)
    sector_map = get_sector_map()
    sector_to_code, _ = get_sector_codes()
    df["sector"] = df["Ticker"].map(sector_map).fillna("Unknown")
    df["sector_code"] = df["sector"].map(sector_to_code).fillna(-1).astype(int)

    # Dollar volume for Amihud
    df["dollar_volume"] = df["Close"] * df["Volume"]

    # ── Per-ticker feature computation ────────────────────────────────────
    feature_frames = []

    for ticker, group in df.groupby("Ticker"):
        g = group.copy()
        close = g["Close"]
        high = g["High"]
        low = g["Low"]
        open_ = g["Open"]
        volume = g["Volume"]
        daily_ret = g["daily_return"]

        # ── Advanced volatility & microstructure ──────────────────────────
        g["parkinson_vol_10"] = parkinson_volatility(high, low, window=10)
        g["parkinson_vol_20"] = parkinson_volatility(high, low, window=20)
        g["garman_klass_vol_10"] = garman_klass_volatility(open_, high, low, close, window=10)
        g["garman_klass_vol_20"] = garman_klass_volatility(open_, high, low, close, window=20)
        g["amihud_illiq_20"] = amihud_illiquidity(daily_ret, g["dollar_volume"], window=20)
        g["atr_20"] = atr(high, low, close, window=20)
        g["vol_regime"] = vol_regime(high, low, close, short_window=20, long_window=200)
        g["volume_zscore"] = volume_zscore(volume, window=20)

        # ── Baseline technicals ──────────────────────────────────────────
        g["sma_10"] = close.rolling(10).mean()
        g["sma_20"] = close.rolling(20).mean()
        g["sma_50"] = close.rolling(50).mean()
        g["close_to_sma20"] = close / g["sma_20"].replace(0, np.nan)
        g["rsi_14"] = compute_rsi(close, period=14)

        macd_df = compute_macd(close)
        g["macd"] = macd_df["macd"].values
        g["macd_signal"] = macd_df["macd_signal"].values
        g["macd_hist"] = macd_df["macd_hist"].values

        # Bollinger bands
        bb_std = close.rolling(20).std()
        g["bb_upper"] = g["sma_20"] + 2 * bb_std
        g["bb_lower"] = g["sma_20"] - 2 * bb_std
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / g["sma_20"].replace(0, np.nan)

        # Realized volatility of returns
        g["realized_vol_10"] = daily_ret.rolling(10).std()
        g["realized_vol_20"] = daily_ret.rolling(20).std()

        # High-low spread
        g["hl_spread"] = (high - low) / close.replace(0, np.nan)

        # Volume features
        g["volume_sma_10"] = volume.rolling(10).mean()
        g["volume_change"] = volume.pct_change()

        # Return lags
        g["return_lag_1"] = daily_ret.shift(1)
        g["return_lag_2"] = daily_ret.shift(2)
        g["return_lag_3"] = daily_ret.shift(3)
        g["return_lag_5"] = daily_ret.shift(5)

        # Momentum features
        g["momentum_5"] = close.pct_change(5)
        g["momentum_10"] = close.pct_change(10)
        g["momentum_20"] = close.pct_change(20)

        feature_frames.append(g)

    result = pd.concat(feature_frames, ignore_index=True)
    result = result.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Count features added
    orig_cols = set(panel.columns)
    new_cols = set(result.columns) - orig_cols
    print(f"[features] Engineered {len(new_cols)} features across {result['Ticker'].nunique()} tickers")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Feature list definition (used by model training)
# ──────────────────────────────────────────────────────────────────────────────

# Features that the model sees — excludes identifiers, targets, raw prices
FEATURE_COLUMNS = [
    # Advanced volatility & microstructure
    "parkinson_vol_10", "parkinson_vol_20",
    "garman_klass_vol_10", "garman_klass_vol_20",
    "amihud_illiq_20",
    "atr_20",
    "vol_regime",
    "volume_zscore",
    "fracdiff_close",
    # Baseline technicals
    "close_to_sma20",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width",
    "realized_vol_10", "realized_vol_20",
    "hl_spread",
    "volume_sma_10", "volume_change",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "momentum_5", "momentum_10", "momentum_20",
    # Cross-sectional
    "sector_code",
    # Market context
    "rolling_beta",
]

# Columns to never use as features
EXCLUDE_COLUMNS = {
    "Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
    "daily_return", "market_return", "rolling_alpha", "residual_return",
    "fwd_residual_return", "fwd_raw_return",
    "log_close", "fracdiff_d", "dollar_volume", "sector",
}
