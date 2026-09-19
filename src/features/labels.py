"""
Labeling Module
Constructs prediction targets for cross-sectional equity forecasting.

Primary target: forward h-day residual return (market-beta stripped).
Optional: triple-barrier classification labels.

The residual return strips out systematic market exposure so the model
learns idiosyncratic (alpha) signal rather than just re-discovering beta:
    r_{i,t} = α_i + β_i · r_{SPY,t} + ε_{i,t}
    y_{i,t} = Σ_{k=1}^{h} ε_{i,t+k}      (h = 5 for weekly rebalance)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Rolling beta via covariance/variance (vectorized, fast)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Estimate rolling OLS beta via rolling covariance / rolling variance.

    This is equivalent to the slope from a trailing OLS regression:
        r_stock = α + β · r_market + ε

    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns for a single stock.
    market_returns : pd.Series
        Daily returns for the market proxy (SPY).
    window : int
        Trailing window length (default 252 ≈ 1 trading year).

    Returns
    -------
    pd.Series
        Rolling beta estimates (NaN for the first `window-1` observations).
    """
    cov = stock_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    beta = cov / var.replace(0, np.nan)
    return beta


def compute_rolling_alpha(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    beta: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Estimate rolling OLS alpha: α = mean(r_stock) − β · mean(r_market).

    Parameters
    ----------
    stock_returns, market_returns : pd.Series
        Daily returns.
    beta : pd.Series
        Rolling beta estimates.
    window : int
        Trailing window length.

    Returns
    -------
    pd.Series
        Rolling alpha estimates.
    """
    mean_stock = stock_returns.rolling(window).mean()
    mean_market = market_returns.rolling(window).mean()
    alpha = mean_stock - beta * mean_market
    return alpha


# ──────────────────────────────────────────────────────────────────────────────
# Residual return computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_residual_returns(
    panel: pd.DataFrame,
    market_ticker: str = "SPY",
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute daily residual returns (epsilon) for each ticker by stripping
    out rolling market beta.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns [Date, Ticker, Close].  Should include the
        market_ticker rows.
    market_ticker : str
        Ticker used as the market factor.
    window : int
        Rolling OLS window.

    Returns
    -------
    pd.DataFrame
        Panel with added columns: [daily_return, market_return, rolling_beta,
        rolling_alpha, residual_return].
    """
    df = panel.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Compute daily returns per ticker
    df["daily_return"] = df.groupby("Ticker")["Close"].pct_change()

    # Extract market returns as a date-indexed series
    market = (
        df[df["Ticker"] == market_ticker][["Date", "daily_return"]]
        .set_index("Date")["daily_return"]
        .rename("market_return")
    )

    # Merge market returns onto every row
    df = df.merge(market, left_on="Date", right_index=True, how="left")

    # Compute rolling beta and alpha per ticker
    betas = []
    alphas = []

    for ticker, group in df.groupby("Ticker"):
        if ticker == market_ticker:
            # Market's beta to itself is 1, alpha is 0
            b = pd.Series(1.0, index=group.index)
            a = pd.Series(0.0, index=group.index)
        else:
            b = compute_rolling_beta(
                group["daily_return"], group["market_return"], window=window
            )
            a = compute_rolling_alpha(
                group["daily_return"], group["market_return"], b, window=window
            )
        betas.append(b)
        alphas.append(a)

    df["rolling_beta"] = pd.concat(betas)
    df["rolling_alpha"] = pd.concat(alphas)

    # Residual: ε = r_stock − (α + β · r_market)
    df["residual_return"] = (
        df["daily_return"] - (df["rolling_alpha"] + df["rolling_beta"] * df["market_return"])
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Forward h-day residual return (primary prediction target)
# ──────────────────────────────────────────────────────────────────────────────

def compute_forward_residual_return(
    panel: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Compute the forward h-day cumulative residual return target:
        y_{i,t} = Σ_{k=1}^{h} ε_{i,t+k}

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns [Date, Ticker, residual_return].
    horizon : int
        Number of forward days to sum (default 5 = weekly).

    Returns
    -------
    pd.DataFrame
        Panel with added column 'fwd_residual_return'.
    """
    df = panel.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # For each ticker, compute the rolling sum of the NEXT h residual returns
    def _fwd_sum(group: pd.DataFrame) -> pd.Series:
        # Shift by -1 first so we don't include the current day
        shifted = group["residual_return"].shift(-1)
        fwd = shifted.rolling(window=horizon).sum().shift(-(horizon - 1))
        return fwd

    df["fwd_residual_return"] = df.groupby("Ticker", group_keys=False).apply(_fwd_sum)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Also compute forward raw return (for IC evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def compute_forward_raw_return(
    panel: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Compute the forward h-day cumulative raw return for IC evaluation:
        fwd_raw_return = Σ_{k=1}^{h} r_{i,t+k}

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns [Date, Ticker, daily_return].
    horizon : int
        Number of forward days to sum (default 5).

    Returns
    -------
    pd.DataFrame
        Panel with added column 'fwd_raw_return'.
    """
    df = panel.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def _fwd_sum(group: pd.DataFrame) -> pd.Series:
        shifted = group["daily_return"].shift(-1)
        fwd = shifted.rolling(window=horizon).sum().shift(-(horizon - 1))
        return fwd

    df["fwd_raw_return"] = df.groupby("Ticker", group_keys=False).apply(_fwd_sum)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Triple-barrier labeling (optional classification variant)
# ──────────────────────────────────────────────────────────────────────────────

def triple_barrier_label(
    prices: pd.Series,
    vol: pd.Series,
    t0_idx: int,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_hold: int = 5,
) -> Tuple[int, Optional[pd.Timestamp]]:
    """
    Apply triple-barrier labeling at a single entry point.

    Barriers scale with each stock's own volatility:
    - Upper barrier (profit take): entry × (1 + pt_mult × vol)
    - Lower barrier (stop loss): entry × (1 − sl_mult × vol)
    - Vertical barrier: max_hold days

    Parameters
    ----------
    prices : pd.Series
        Full price series for one ticker.
    vol : pd.Series
        Rolling ATR or realized volatility, aligned with prices.
    t0_idx : int
        Integer location index of the entry point.
    pt_mult, sl_mult : float
        Multipliers for profit-take / stop-loss barriers.
    max_hold : int
        Maximum holding period in trading days.

    Returns
    -------
    label : int
        +1 (profit take hit), -1 (stop loss hit), 0 (expired at vertical barrier).
    exit_date : pd.Timestamp or None
        Date the barrier was touched.
    """
    if t0_idx >= len(prices) - 1:
        return 0, None

    entry = prices.iloc[t0_idx]
    entry_vol = vol.iloc[t0_idx]

    if np.isnan(entry_vol) or entry_vol <= 0:
        return 0, None

    upper = entry * (1 + pt_mult * entry_vol)
    lower = entry * (1 - sl_mult * entry_vol)

    window = prices.iloc[t0_idx + 1 : t0_idx + 1 + max_hold]

    for date, price in window.items():
        if price >= upper:
            return 1, date
        if price <= lower:
            return -1, date

    if len(window) > 0:
        final_ret = window.iloc[-1] - entry
        label = int(np.sign(final_ret)) if final_ret != 0 else 0
        return label, window.index[-1]

    return 0, None


def run_labeling_pipeline(
    panel: pd.DataFrame,
    market_ticker: str = "SPY",
    window: int = 252,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Full labeling pipeline: residual returns + forward target.

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel with [Date, Ticker, Open, High, Low, Close, Volume].
    market_ticker : str
        Market factor ticker.
    window : int
        Rolling OLS window.
    horizon : int
        Forward return horizon.

    Returns
    -------
    pd.DataFrame
        Panel with added labeling columns.
    """
    print(f"[labels] Computing residual returns (window={window})...")
    df = compute_residual_returns(panel, market_ticker=market_ticker, window=window)

    print(f"[labels] Computing forward {horizon}-day residual return target...")
    df = compute_forward_residual_return(df, horizon=horizon)

    print(f"[labels] Computing forward {horizon}-day raw return...")
    df = compute_forward_raw_return(df, horizon=horizon)

    # Summary stats
    valid = df["fwd_residual_return"].notna().sum()
    total = len(df)
    print(f"[labels] Target coverage: {valid:,}/{total:,} rows "
          f"({100 * valid / total:.1f}%) have valid forward residual return")

    return df
