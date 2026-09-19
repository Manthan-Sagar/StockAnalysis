"""
Fractional Differentiation Module
Implements Fixed-Width Window Fractional Differentiation (FFD) to produce
stationary series that preserve maximum memory from the original price series.

Unlike integer differencing (d=1) which destroys long-range dependence,
fractional differencing at d ∈ (0, 1) achieves stationarity while retaining
useful autocorrelation structure.  The optimal d per ticker is the minimum
value that achieves ADF p-value < 0.05.

Reference: Marcos López de Prado, "Advances in Financial Machine Learning" (2018),
           Chapter 5.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ──────────────────────────────────────────────────────────────────────────────
# FFD weights and differentiation
# ──────────────────────────────────────────────────────────────────────────────

def get_weights_ffd(d: float, thresh: float = 1e-5, max_size: int = 1000) -> np.ndarray:
    """
    Compute the fixed-width window (FFD) fractional differentiation weights.

    The weights w_k are defined recursively:
        w_0 = 1
        w_k = -w_{k-1} · (d - k + 1) / k

    Weights are truncated when |w_k| < thresh.

    Parameters
    ----------
    d : float
        Fractional differencing order, typically in (0, 1).
    thresh : float
        Minimum absolute weight to include (default 1e-5).
    max_size : int
        Maximum number of weights to compute.

    Returns
    -------
    np.ndarray
        Weight vector, ordered from w_{width} to w_0 (oldest to newest).
    """
    w = [1.0]
    for k in range(1, max_size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w[::-1])


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    thresh: float = 1e-5,
) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differentiation to a series.

    Parameters
    ----------
    series : pd.Series
        Input series (typically log(Close)).
    d : float
        Fractional differencing order.
    thresh : float
        Weight truncation threshold.

    Returns
    -------
    pd.Series
        Fractionally differenced series (NaN for the warm-up period).
    """
    w = get_weights_ffd(d, thresh)
    n = len(series)

    # Truncate weights if longer than the series
    if len(w) > n:
        w = w[-(n):]

    width = len(w) - 1
    out = pd.Series(np.nan, index=series.index, dtype=float)

    for i in range(width, n):
        window = series.iloc[i - width : i + 1].values
        if len(window) == len(w):
            out.iloc[i] = np.dot(w, window)

    return out


def frac_diff_ffd_vectorized(
    series: pd.Series,
    d: float,
    thresh: float = 1e-5,
) -> pd.Series:
    """
    Vectorized FFD using np.convolve for better performance on long series.

    If the weight vector exceeds the series length, weights are truncated
    to fit (losing some of the oldest/smallest weights, which has negligible
    effect on the result due to the threshold-based truncation).

    Parameters
    ----------
    series : pd.Series
        Input series (typically log(Close)).
    d : float
        Fractional differencing order.
    thresh : float
        Weight truncation threshold.

    Returns
    -------
    pd.Series
        Fractionally differenced series (NaN for warm-up).
    """
    w = get_weights_ffd(d, thresh)
    n = len(series)

    # Truncate weights if longer than the series
    if len(w) > n:
        w = w[-(n):]  # keep the newest n weights (rightmost = newest)

    width = len(w) - 1
    values = series.values.astype(float)

    # w is ordered [w_width, ..., w_1, w_0] (oldest to newest).
    # For convolution, we reverse w to get [w_0, w_1, ..., w_width].
    # mode='valid' produces len(values) - len(w) + 1 = n - width elements.
    w_reversed = w[::-1]
    conv = np.convolve(values, w_reversed, mode="valid")

    out_values = np.full(n, np.nan)
    out_values[width : width + len(conv)] = conv

    return pd.Series(out_values, index=series.index, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Optimal d search via ADF test
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_d(
    series: pd.Series,
    d_range: Optional[List[float]] = None,
    p_threshold: float = 0.05,
    thresh: float = 1e-5,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Grid-search the minimum fractional differencing order d that achieves
    stationarity (ADF p-value < p_threshold).

    Parameters
    ----------
    series : pd.Series
        Input series (typically log(Close) for a single ticker).
    d_range : list of float, optional
        Values of d to try (default [0.1, 0.2, ..., 0.9]).
    p_threshold : float
        ADF p-value threshold for stationarity (default 0.05).
    thresh : float
        FFD weight truncation threshold.

    Returns
    -------
    optimal_d : float
        Minimum d achieving stationarity. Returns 1.0 if none found.
    optimal_p : float
        ADF p-value at the optimal d.
    adf_results : dict
        {d: p_value} for all tested values.
    """
    if d_range is None:
        d_range = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 to 0.9

    adf_results: Dict[float, float] = {}

    for d in d_range:
        diffed = frac_diff_ffd_vectorized(series, d, thresh=thresh)
        clean = diffed.dropna()

        if len(clean) < 50:
            adf_results[d] = 1.0
            continue

        try:
            adf_stat, p_value, *_ = adfuller(clean, maxlag=1, regression="c", autolag=None)
            adf_results[d] = p_value
        except Exception:
            adf_results[d] = 1.0

    # Find minimum d with p < threshold
    valid = {d: p for d, p in adf_results.items() if p < p_threshold}

    if valid:
        optimal_d = min(valid.keys())
        optimal_p = valid[optimal_d]
    else:
        optimal_d = 1.0
        optimal_p = adf_results.get(0.9, 1.0)

    return optimal_d, optimal_p, adf_results


# ──────────────────────────────────────────────────────────────────────────────
# Panel-level fractional differentiation
# ──────────────────────────────────────────────────────────────────────────────

def fracdiff_panel(
    panel: pd.DataFrame,
    p_threshold: float = 0.05,
    thresh: float = 1e-5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply per-ticker fractional differentiation to log(Close).

    For each ticker:
    1. Compute log(Close).
    2. Grid-search optimal d via ADF test.
    3. Apply FFD at optimal d.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns [Date, Ticker, Close].
    p_threshold : float
        ADF stationarity threshold.
    thresh : float
        FFD weight truncation threshold.

    Returns
    -------
    panel : pd.DataFrame
        Panel with added columns: [log_close, fracdiff_d, fracdiff_close].
    optimal_ds : dict
        {ticker: optimal_d}.
    """
    df = panel.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["log_close"] = np.log(df["Close"].clip(lower=1e-8))

    optimal_ds: Dict[str, float] = {}
    fracdiff_values = pd.Series(np.nan, index=df.index, dtype=float)

    excluded = {"SPY", "QQQ"}  # benchmarks — don't fracdiff

    for ticker, group in df.groupby("Ticker"):
        if ticker in excluded:
            optimal_ds[ticker] = 0.0
            fracdiff_values.loc[group.index] = group["log_close"].values
            continue

        log_close = group["log_close"].reset_index(drop=True)
        d_opt, p_opt, _ = find_optimal_d(log_close, p_threshold=p_threshold, thresh=thresh)
        optimal_ds[ticker] = d_opt

        # Apply FFD at optimal d
        diffed = frac_diff_ffd_vectorized(log_close, d_opt, thresh=thresh)
        fracdiff_values.iloc[group.index[0] : group.index[-1] + 1] = diffed.values

        print(f"  [{ticker}] optimal d = {d_opt:.1f}  (ADF p = {p_opt:.4f})")

    df["fracdiff_d"] = df["Ticker"].map(optimal_ds)
    df["fracdiff_close"] = fracdiff_values

    print(f"\n[fracdiff] Processed {len(optimal_ds)} tickers")
    d_values = [v for k, v in optimal_ds.items() if k not in excluded]
    if d_values:
        print(f"  Mean optimal d: {np.mean(d_values):.2f}, "
              f"Median: {np.median(d_values):.2f}, "
              f"Range: [{min(d_values):.1f}, {max(d_values):.1f}]")

    return df, optimal_ds
