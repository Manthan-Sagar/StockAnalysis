"""
Vectorized Backtest Engine
Friction-adjusted, dollar-neutral long/short portfolio backtest.

A vectorized pandas backtest is the right scope here — event-driven engines
(Zipline/Backtrader) add infrastructure weight without adding signal to the
story for a solo project.

Cost model:
  net_return_t = gross_return_t - turnover_t × (total_cost_bps / 10000)
  total_cost_bps ≈ 8 (7bps slippage + ~1bp commission approximation for
  $0.005/share at ~$50 avg share price)

Metrics:
  - Sharpe = mean(daily excess return) / std(daily excess return) × √252
  - Sortino = same, using downside deviation only
  - Max Drawdown = min(cum_return / cummax - 1)
  - Calmar = annualized return / |MDD|
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Backtest core
# ──────────────────────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    portfolio_df: pd.DataFrame,
    total_cost_bps: float = 8.0,
) -> pd.DataFrame:
    """
    Compute daily portfolio returns with transaction cost adjustment.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        From portfolio.sizing.construct_portfolio.
        Columns: [date, ticker, weight, daily_return, rebalance_date].
    total_cost_bps : float
        Total round-trip transaction cost in basis points (default 8).
        Breakdown: ~7bps slippage + ~1bp commission ($0.005/share at ~$50).

    Returns
    -------
    pd.DataFrame
        Daily portfolio-level returns:
        [date, gross_return, turnover, cost, net_return, cum_return].
    """
    df = portfolio_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    dates = sorted(df["date"].unique())
    cost_multiplier = total_cost_bps / 10_000

    daily_records = []
    prev_weights: Dict[str, float] = {}

    for date in dates:
        day_data = df[df["date"] == date]

        # Gross return: weighted sum of daily returns
        gross_return = (day_data["weight"] * day_data["daily_return"]).sum()

        # Current weights
        current_weights = dict(zip(day_data["ticker"], day_data["weight"]))

        # Turnover: sum of absolute weight changes
        all_tickers = set(current_weights.keys()) | set(prev_weights.keys())
        turnover = sum(
            abs(current_weights.get(t, 0) - prev_weights.get(t, 0))
            for t in all_tickers
        )

        # Transaction cost
        cost = turnover * cost_multiplier

        # Net return
        net_return = gross_return - cost

        daily_records.append({
            "date": date,
            "gross_return": gross_return,
            "turnover": turnover,
            "cost": cost,
            "net_return": net_return,
        })

        prev_weights = current_weights

    result = pd.DataFrame(daily_records)

    if not result.empty:
        result["cum_gross"] = (1 + result["gross_return"]).cumprod()
        result["cum_net"] = (1 + result["net_return"]).cumprod()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(
    daily_returns: pd.DataFrame,
    return_col: str = "net_return",
    annual_trading_days: int = 252,
) -> Dict[str, float]:
    """
    Compute standard portfolio performance metrics.

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Must contain the specified return column.
    return_col : str
        Column name for returns to analyze.
    annual_trading_days : int
        Trading days per year for annualization.

    Returns
    -------
    dict
        Performance metrics.
    """
    returns = daily_returns[return_col].dropna()

    if len(returns) < 2:
        return {}

    # Annualized return
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / annual_trading_days
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe (excess return assumed = raw return for L/S, since it's self-financing)
    daily_mean = returns.mean()
    daily_std = returns.std()
    sharpe = (daily_mean / daily_std * np.sqrt(annual_trading_days)) if daily_std > 0 else 0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else daily_std
    sortino = (daily_mean / downside_std * np.sqrt(annual_trading_days)) if downside_std > 0 else 0

    # Max Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1
    max_dd = drawdown.min()

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Annualized turnover
    if "turnover" in daily_returns.columns:
        ann_turnover = daily_returns["turnover"].sum() / n_years if n_years > 0 else 0
    else:
        ann_turnover = 0

    # Win rate
    win_rate = (returns > 0).mean() * 100

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": daily_std * np.sqrt(annual_trading_days),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "annualized_turnover": ann_turnover,
        "win_rate_pct": win_rate,
        "n_trading_days": len(returns),
        "total_cost_bps": daily_returns.get("cost", pd.Series([0])).sum() * 10_000,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_benchmark_returns(
    panel: pd.DataFrame,
    benchmark_tickers: list = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute buy-and-hold benchmark returns for SPY and QQQ.

    Returns
    -------
    dict
        {ticker: DataFrame with [date, daily_return, cum_return]}.
    """
    if benchmark_tickers is None:
        benchmark_tickers = ["SPY", "QQQ"]

    benchmarks = {}
    for ticker in benchmark_tickers:
        bm = panel[panel["Ticker"] == ticker][["Date", "daily_return"]].copy()
        bm = bm.rename(columns={"Date": "date"})
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm.sort_values("date").reset_index(drop=True)

        if start_date:
            bm = bm[bm["date"] >= pd.to_datetime(start_date)]
        if end_date:
            bm = bm[bm["date"] <= pd.to_datetime(end_date)]

        bm = bm.dropna(subset=["daily_return"]).reset_index(drop=True)
        bm["cum_return"] = (1 + bm["daily_return"]).cumprod()
        benchmarks[ticker] = bm

    return benchmarks


# ──────────────────────────────────────────────────────────────────────────────
# Tearsheet generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_tearsheet(
    daily_returns: pd.DataFrame,
    benchmarks: Dict[str, pd.DataFrame],
    ic_df: pd.DataFrame,
    metrics: Dict[str, float],
    output_path: Path = Path("outputs/tearsheet.png"),
) -> None:
    """
    Generate a 4-panel tearsheet:
    1. Equity curve vs. SPY & QQQ buy-and-hold
    2. Drawdown chart
    3. Rolling 63-day Sharpe ratio
    4. IC time series

    Parameters
    ----------
    daily_returns : pd.DataFrame
        Portfolio daily returns with cum_net column.
    benchmarks : dict
        Benchmark DataFrames from compute_benchmark_returns.
    ic_df : pd.DataFrame
        IC time series from ranking module.
    metrics : dict
        Performance metrics dict.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={"hspace": 0.35})

    # Color palette
    strategy_color = "#2563EB"
    spy_color = "#10B981"
    qqq_color = "#F59E0B"
    dd_color = "#EF4444"
    ic_color = "#8B5CF6"

    # ── Panel 1: Equity Curve ─────────────────────────────────────────────
    ax1 = axes[0]
    dates = pd.to_datetime(daily_returns["date"])

    ax1.plot(dates, daily_returns["cum_net"], color=strategy_color,
             linewidth=2.0, label="L/S Strategy (net)", zorder=3)
    ax1.plot(dates, daily_returns["cum_gross"], color=strategy_color,
             linewidth=1.0, alpha=0.4, linestyle="--", label="L/S Strategy (gross)")

    for bm_name, bm_df in benchmarks.items():
        bm_dates = pd.to_datetime(bm_df["date"])
        # Align benchmark to strategy date range
        mask = (bm_dates >= dates.min()) & (bm_dates <= dates.max())
        bm_filtered = bm_df[mask.values].copy()
        if not bm_filtered.empty:
            # Re-base to start at 1
            bm_filtered["cum_return"] = (
                (1 + bm_filtered["daily_return"]).cumprod()
            )
            color = spy_color if bm_name == "SPY" else qqq_color
            ax1.plot(pd.to_datetime(bm_filtered["date"]),
                    bm_filtered["cum_return"],
                    color=color, linewidth=1.5, alpha=0.8,
                    label=f"{bm_name} Buy & Hold")

    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_title("Equity Curve: L/S Strategy vs. Benchmarks", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Add metrics annotation
    metrics_text = (
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
        f"Sortino: {metrics.get('sortino_ratio', 0):.2f}  |  "
        f"MDD: {metrics.get('max_drawdown', 0):.1%}  |  "
        f"Calmar: {metrics.get('calmar_ratio', 0):.2f}  |  "
        f"Ann. Turnover: {metrics.get('annualized_turnover', 0):.1f}x"
    )
    ax1.text(0.5, -0.12, metrics_text, transform=ax1.transAxes,
             ha="center", fontsize=10, color="#64748B",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9", edgecolor="#E2E8F0"))

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2 = axes[1]
    cum = daily_returns["cum_net"]
    peak = cum.cummax()
    drawdown = (cum / peak - 1) * 100

    ax2.fill_between(dates, drawdown, 0, color=dd_color, alpha=0.3)
    ax2.plot(dates, drawdown, color=dd_color, linewidth=1.0)
    ax2.set_title("Drawdown", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── Panel 3: Rolling Sharpe ───────────────────────────────────────────
    ax3 = axes[2]
    rolling_window = 63  # ~3 months
    returns = daily_returns["net_return"]
    rolling_sharpe = (
        returns.rolling(rolling_window).mean()
        / returns.rolling(rolling_window).std()
        * np.sqrt(252)
    )

    ax3.plot(dates, rolling_sharpe, color=strategy_color, linewidth=1.5)
    ax3.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax3.fill_between(dates, rolling_sharpe, 0,
                     where=rolling_sharpe >= 0, color=strategy_color, alpha=0.15)
    ax3.fill_between(dates, rolling_sharpe, 0,
                     where=rolling_sharpe < 0, color=dd_color, alpha=0.15)
    ax3.set_title(f"Rolling {rolling_window}-Day Sharpe Ratio", fontweight="bold", fontsize=13)
    ax3.set_ylabel("Sharpe Ratio")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── Panel 4: IC Time Series ───────────────────────────────────────────
    ax4 = axes[3]
    if not ic_df.empty:
        ic_dates = pd.to_datetime(ic_df["date"])
        ic_vals = ic_df["ic"]

        ax4.bar(ic_dates, ic_vals, width=2, color=ic_color, alpha=0.6)
        ax4.axhline(ic_vals.mean(), color=ic_color, linestyle="--",
                    linewidth=1.5, label=f"Mean IC = {ic_vals.mean():.4f}")
        ax4.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax4.legend(loc="upper right", frameon=True, facecolor="white")

    ax4.set_title("Cross-Sectional Rank IC Time Series", fontweight="bold", fontsize=13)
    ax4.set_ylabel("Spearman Rank IC")
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[backtest] Saved tearsheet -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Full backtest pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(
    portfolio_df: pd.DataFrame,
    panel: pd.DataFrame,
    ic_df: pd.DataFrame,
    total_cost_bps: float = 8.0,
    output_dir: Path = Path("outputs"),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    End-to-end backtest: compute returns, metrics, benchmarks, and tearsheet.

    Returns
    -------
    daily_returns : pd.DataFrame
        Daily portfolio returns.
    metrics : dict
        Performance metrics.
    """
    print("\n[backtest] ==========================================")
    print(f"  Running vectorized backtest (cost = {total_cost_bps} bps)")
    print(f"  ==================================================")

    # Compute portfolio returns
    daily_returns = compute_portfolio_returns(portfolio_df, total_cost_bps=total_cost_bps)

    if daily_returns.empty:
        print("[backtest] No portfolio returns generated!")
        return daily_returns, {}

    # Performance metrics
    metrics_gross = compute_performance_metrics(daily_returns, return_col="gross_return")
    metrics_net = compute_performance_metrics(daily_returns, return_col="net_return")

    print(f"\n  Gross Performance:")
    print(f"    Sharpe:           {metrics_gross.get('sharpe_ratio', 0):+.3f}")
    print(f"    Ann. Return:      {metrics_gross.get('annualized_return', 0):+.2%}")
    print(f"    Max Drawdown:     {metrics_gross.get('max_drawdown', 0):.2%}")

    print(f"\n  Net Performance (after {total_cost_bps}bps costs):")
    print(f"    Sharpe:           {metrics_net.get('sharpe_ratio', 0):+.3f}")
    print(f"    Sortino:          {metrics_net.get('sortino_ratio', 0):+.3f}")
    print(f"    Ann. Return:      {metrics_net.get('annualized_return', 0):+.2%}")
    print(f"    Ann. Volatility:  {metrics_net.get('annualized_volatility', 0):.2%}")
    print(f"    Max Drawdown:     {metrics_net.get('max_drawdown', 0):.2%}")
    print(f"    Calmar:           {metrics_net.get('calmar_ratio', 0):+.3f}")
    print(f"    Ann. Turnover:    {metrics_net.get('annualized_turnover', 0):.1f}x")
    print(f"    Win Rate:         {metrics_net.get('win_rate_pct', 0):.1f}%")
    print(f"    Trading Days:     {metrics_net.get('n_trading_days', 0)}")

    # Benchmark returns
    test_start = str(daily_returns["date"].min().date())
    test_end = str(daily_returns["date"].max().date())
    benchmarks = compute_benchmark_returns(
        panel, benchmark_tickers=["SPY", "QQQ"],
        start_date=test_start, end_date=test_end,
    )

    for bm_name, bm_df in benchmarks.items():
        bm_metrics = compute_performance_metrics(
            bm_df.rename(columns={"daily_return": "net_return"}),
            return_col="net_return",
        )
        print(f"\n  {bm_name} Buy & Hold:")
        print(f"    Sharpe:           {bm_metrics.get('sharpe_ratio', 0):+.3f}")
        print(f"    Ann. Return:      {bm_metrics.get('annualized_return', 0):+.2%}")
        print(f"    Max Drawdown:     {bm_metrics.get('max_drawdown', 0):.2%}")

    # Generate tearsheet
    generate_tearsheet(
        daily_returns, benchmarks, ic_df, metrics_net,
        output_path=output_dir / "tearsheet.png",
    )

    # Save daily returns
    returns_path = output_dir / "daily_returns.csv"
    daily_returns.to_csv(returns_path, index=False)
    print(f"[backtest] Saved daily returns -> {returns_path}")

    return daily_returns, metrics_net
