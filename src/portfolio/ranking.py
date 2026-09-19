"""
Cross-Sectional Ranking & Information Coefficient Module
Evaluates model signal quality via rank-based metrics.

At each rebalance date t, across all tickers with valid predictions:
  IC_t = spearmanr(predicted_scores_t, realized_forward_returns_t)

Aggregate metrics:
  - Mean IC: average of IC_t across all rebalance dates
  - IC Std: standard deviation of IC_t
  - IR (Information Ratio): mean(IC) / std(IC)
  - IC t-stat: IR × √(number of rebalance periods)

A consistently near-zero-but-slightly-positive mean IC with a plausible
t-stat is a legitimate, publishable-looking result.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_cross_sectional_ic(
    panel: pd.DataFrame,
    predicted_col: str = "predicted_score",
    realized_col: str = "fwd_raw_return",
    date_col: str = "Date",
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    Compute cross-sectional Spearman rank IC at each date.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [Date, Ticker, predicted_col, realized_col].
    predicted_col : str
        Column with model predictions.
    realized_col : str
        Column with realized forward returns.
    date_col : str
        Date column.
    min_stocks : int
        Minimum number of stocks required for a valid IC observation.

    Returns
    -------
    pd.DataFrame
        Columns: [date, ic, n_stocks].
    """
    records: List[dict] = []

    for date, group in panel.groupby(date_col):
        valid = group.dropna(subset=[predicted_col, realized_col])
        if len(valid) < min_stocks:
            continue

        ic, p_value = spearmanr(valid[predicted_col], valid[realized_col])

        records.append({
            "date": date,
            "ic": ic,
            "n_stocks": len(valid),
            "p_value": p_value,
        })

    ic_df = pd.DataFrame(records)
    if not ic_df.empty:
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        ic_df = ic_df.sort_values("date").reset_index(drop=True)

    return ic_df


def compute_ic_summary(ic_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute aggregate IC statistics.

    Parameters
    ----------
    ic_df : pd.DataFrame
        Output from compute_cross_sectional_ic.

    Returns
    -------
    dict
        {mean_ic, ic_std, ir, ic_tstat, n_periods, pct_positive}.
    """
    if ic_df.empty:
        return {
            "mean_ic": 0.0,
            "ic_std": 0.0,
            "ir": 0.0,
            "ic_tstat": 0.0,
            "n_periods": 0,
            "pct_positive": 0.0,
        }

    ic_series = ic_df["ic"]
    n = len(ic_series)
    mean_ic = float(ic_series.mean())
    ic_std = float(ic_series.std())
    ir = mean_ic / ic_std if ic_std > 0 else 0.0
    tstat = ir * np.sqrt(n)
    pct_positive = float((ic_series > 0).mean() * 100)

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ir": ir,
        "ic_tstat": tstat,
        "n_periods": n,
        "pct_positive": pct_positive,
    }


def generate_predictions(
    model,
    panel: pd.DataFrame,
    feature_names: List[str],
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Generate model predictions for the full panel.

    Parameters
    ----------
    model : fitted model
        Must have a .predict() method.
    panel : pd.DataFrame
        Panel with feature columns.
    feature_names : list of str
        Feature column names.

    Returns
    -------
    pd.DataFrame
        Panel with added 'predicted_score' column.
    """
    df = panel.copy()

    # Only predict for rows with all features available
    valid_mask = df[feature_names].notna().all(axis=1)
    X = df.loc[valid_mask, feature_names].values

    df["predicted_score"] = np.nan
    df.loc[valid_mask, "predicted_score"] = model.predict(X)

    n_valid = valid_mask.sum()
    print(f"[ranking] Generated predictions for {n_valid:,}/{len(df):,} rows")

    return df


def run_ranking_evaluation(
    model,
    test_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Full ranking evaluation pipeline.

    Returns
    -------
    predictions : pd.DataFrame
        Test panel with predictions.
    ic_df : pd.DataFrame
        Per-date IC values.
    ic_summary : dict
        Aggregate IC statistics.
    """
    # Generate predictions
    predictions = generate_predictions(model, test_df, feature_names)

    # Compute IC time series
    ic_df = compute_cross_sectional_ic(
        predictions,
        predicted_col="predicted_score",
        realized_col="fwd_raw_return",
    )

    # Aggregate
    ic_summary = compute_ic_summary(ic_df)

    print(f"\n[ranking] ==========================================")
    print(f"  Cross-Sectional Rank IC Summary")
    print(f"  --------------------------------------------------")
    print(f"  Mean IC:       {ic_summary['mean_ic']:+.4f}")
    print(f"  IC Std:         {ic_summary['ic_std']:.4f}")
    print(f"  IR:             {ic_summary['ir']:+.4f}")
    print(f"  IC t-stat:      {ic_summary['ic_tstat']:+.2f}")
    print(f"  Periods:        {ic_summary['n_periods']}")
    print(f"  % Positive IC:  {ic_summary['pct_positive']:.1f}%")
    print(f"  ==================================================")

    return predictions, ic_df, ic_summary
