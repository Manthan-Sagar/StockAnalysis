"""
Portfolio Construction & Sizing Module
Constructs a dollar-neutral long/short portfolio with inverse-volatility
(risk-parity) weighting.

At each weekly rebalance (matching h=5 label horizon):
1. Rank all names by predicted score.
2. Long top quintile, short bottom quintile.
3. Within each leg: weight inversely by trailing 20-day realized volatility
   (weight_i ∝ 1/vol_i, normalized to sum to 1 per leg).
4. Long leg sums to +1.0, short leg to −1.0 (gross 200%, net 0%).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def rank_and_assign_legs(
    scores: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """
    Rank stocks and assign to long (top quintile) or short (bottom quintile).

    Parameters
    ----------
    scores : pd.Series
        Predicted scores indexed by ticker.
    n_quantiles : int
        Number of quantiles (default 5 → top/bottom 20%).

    Returns
    -------
    pd.Series
        Values: +1 (long), -1 (short), 0 (neutral).
    """
    if len(scores) < n_quantiles:
        return pd.Series(0, index=scores.index)

    ranks = scores.rank(method="first")
    n = len(ranks)
    cutoff_low = n / n_quantiles       # bottom quintile
    cutoff_high = n * (n_quantiles - 1) / n_quantiles  # top quintile

    legs = pd.Series(0, index=scores.index)
    legs[ranks <= cutoff_low] = -1   # short bottom quintile
    legs[ranks > cutoff_high] = 1    # long top quintile

    return legs


def inverse_vol_weights(
    volatilities: pd.Series,
    leg_assignments: pd.Series,
) -> pd.Series:
    """
    Compute inverse-volatility (risk-parity) weights within each leg.

    Parameters
    ----------
    volatilities : pd.Series
        Trailing realized volatility per ticker.
    leg_assignments : pd.Series
        +1 (long), -1 (short), 0 (neutral).

    Returns
    -------
    pd.Series
        Portfolio weights. Long weights sum to +1, short to -1.
    """
    weights = pd.Series(0.0, index=volatilities.index)

    for leg_sign in [1, -1]:
        mask = leg_assignments == leg_sign
        if not mask.any():
            continue

        vols = volatilities[mask].replace(0, np.nan).dropna()
        if vols.empty:
            continue

        inv_vol = 1.0 / vols
        normalized = inv_vol / inv_vol.sum()

        # Long weights positive, short weights negative
        weights.loc[normalized.index] = normalized * leg_sign

    return weights


def construct_portfolio(
    panel: pd.DataFrame,
    score_col: str = "predicted_score",
    vol_col: str = "realized_vol_20",
    rebalance_freq: int = 5,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Construct a dollar-neutral long/short portfolio with weekly rebalancing.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [Date, Ticker, predicted_score, realized_vol_20, daily_return].
    score_col : str
        Column with predicted scores.
    vol_col : str
        Column with trailing volatility for sizing.
    rebalance_freq : int
        Rebalance every N trading days (default 5 = weekly).
    n_quantiles : int
        Number of quantiles for leg assignment.

    Returns
    -------
    pd.DataFrame
        Portfolio DataFrame with columns:
        [date, ticker, weight, leg, daily_return, next_day_return].
    """
    df = panel.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Get unique sorted dates
    dates = sorted(df["Date"].unique())

    # Select rebalance dates (every rebalance_freq days)
    rebalance_dates = [dates[i] for i in range(0, len(dates), rebalance_freq)]

    portfolio_records: List[dict] = []
    current_weights: Dict[str, float] = {}
    current_rebal_date = None

    for date in dates:
        day_data = df[df["Date"] == date].copy()

        if date in rebalance_dates:
            # Rebalance
            valid = day_data.dropna(subset=[score_col, vol_col])
            if len(valid) < n_quantiles * 2:
                continue

            scores = valid.set_index("Ticker")[score_col]
            vols = valid.set_index("Ticker")[vol_col]

            legs = rank_and_assign_legs(scores, n_quantiles=n_quantiles)
            weights = inverse_vol_weights(vols, legs)

            current_weights = weights.to_dict()
            current_rebal_date = date

        # Record daily portfolio positions
        for ticker, weight in current_weights.items():
            if weight == 0:
                continue

            row = day_data[day_data["Ticker"] == ticker]
            if row.empty:
                continue

            portfolio_records.append({
                "date": date,
                "ticker": ticker,
                "weight": weight,
                "leg": "long" if weight > 0 else "short",
                "daily_return": row["daily_return"].values[0],
                "rebalance_date": current_rebal_date,
            })

    portfolio_df = pd.DataFrame(portfolio_records)

    if not portfolio_df.empty:
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        portfolio_df = portfolio_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        # Summary
        n_rebal = len(rebalance_dates)
        n_long = (portfolio_df["leg"] == "long").sum()
        n_short = (portfolio_df["leg"] == "short").sum()
        print(f"[sizing] Portfolio constructed:")
        print(f"  Rebalance dates: {n_rebal}")
        print(f"  Long positions:  {n_long:,}")
        print(f"  Short positions: {n_short:,}")

    return portfolio_df


def verify_portfolio_weights(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify dollar-neutrality and gross exposure constraints at each date.

    Returns
    -------
    pd.DataFrame
        Per-date summary: [date, long_weight_sum, short_weight_sum,
        gross_exposure, net_exposure].
    """
    summary_records = []

    for date, group in portfolio_df.groupby("date"):
        long_wt = group[group["weight"] > 0]["weight"].sum()
        short_wt = group[group["weight"] < 0]["weight"].sum()
        gross = long_wt - short_wt  # long is positive, short is negative
        net = long_wt + short_wt

        summary_records.append({
            "date": date,
            "long_weight_sum": long_wt,
            "short_weight_sum": short_wt,
            "gross_exposure": gross,
            "net_exposure": net,
            "n_long": (group["weight"] > 0).sum(),
            "n_short": (group["weight"] < 0).sum(),
        })

    return pd.DataFrame(summary_records)
