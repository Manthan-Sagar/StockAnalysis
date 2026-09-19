"""
Cross-Sectional Equity Forecasting & Friction-Adjusted Long/Short Backtest
End-to-end pipeline orchestrator.

Executes:
  1. Data acquisition (cached parquet download)
  2. Labeling (residual returns via rolling market beta)
  3. Feature engineering (fracdiff, volatility, technicals)
  4. Model training (LightGBM with purged CV)
  5. Ranking evaluation (cross-sectional IC/IR)
  6. Portfolio construction (dollar-neutral, vol-parity)
  7. Backtest (vectorized, friction-adjusted)
  8. Tearsheet generation

Usage:
    python -m src.run_pipeline
"""

import json
import sys
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch import run_data_pipeline
from src.features.labels import run_labeling_pipeline
from src.features.fracdiff import fracdiff_panel
from src.features.volatility import engineer_features
from src.models.train import run_training_pipeline, load_model
from src.portfolio.ranking import run_ranking_evaluation
from src.portfolio.sizing import construct_portfolio, verify_portfolio_weights
from src.backtest.engine import run_backtest

# Configuration
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
LABEL_HORIZON = 5
REBALANCE_FREQ = 5
COST_BPS = 8.0


def main(force_refresh: bool = False):
    """Run the complete research pipeline end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 70)
    print("CROSS-SECTIONAL EQUITY FORECASTING & L/S BACKTEST PIPELINE")
    print("=" * 70)
    start_total = time.time()

    # ── Phase 1: Data Acquisition ─────────────────────────────────────────
    print("\n>>> PHASE 1: Data Acquisition")
    t0 = time.time()
    panel = run_data_pipeline(force_refresh=force_refresh)
    print(f"Phase 1 completed in {time.time() - t0:.1f}s\n")

    # ── Phase 2: Labeling ─────────────────────────────────────────────────
    print(">>> PHASE 2: Labeling (Residual Returns)")
    t0 = time.time()
    panel = run_labeling_pipeline(
        panel, market_ticker="SPY", window=252, horizon=LABEL_HORIZON
    )
    print(f"Phase 2 completed in {time.time() - t0:.1f}s\n")

    # ── Phase 3a: Fractional Differentiation ──────────────────────────────
    print(">>> PHASE 3a: Fractional Differentiation")
    t0 = time.time()
    panel, optimal_ds = fracdiff_panel(panel)
    print(f"Phase 3a completed in {time.time() - t0:.1f}s\n")

    # ── Phase 3b: Feature Engineering ─────────────────────────────────────
    print(">>> PHASE 3b: Feature Engineering")
    t0 = time.time()
    panel = engineer_features(panel)
    print(f"Phase 3b completed in {time.time() - t0:.1f}s\n")

    # Save the processed panel
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    panel_path = processed_dir / "panel.parquet"
    panel.to_parquet(panel_path, engine="pyarrow", index=False)
    print(f"Saved processed panel -> {panel_path} ({len(panel):,} rows)")

    # ── Phase 5: Model Training ───────────────────────────────────────────
    print("\n>>> PHASE 5: Model Training (LightGBM + Purged CV)")
    t0 = time.time()
    model, train_df, test_df, feature_names = run_training_pipeline(
        panel,
        train_end=TRAIN_END,
        test_start=TEST_START,
        n_iter=30,
        label_horizon=LABEL_HORIZON,
    )
    print(f"Phase 5 completed in {time.time() - t0:.1f}s\n")

    # ── Phase 6: Ranking Evaluation ───────────────────────────────────────
    print(">>> PHASE 6: Ranking Evaluation (Cross-Sectional IC/IR)")
    t0 = time.time()
    predictions, ic_df, ic_summary = run_ranking_evaluation(
        model, test_df, feature_names
    )
    print(f"Phase 6 completed in {time.time() - t0:.1f}s\n")

    # ── Phase 7: Portfolio Construction ───────────────────────────────────
    print(">>> PHASE 7: Portfolio Construction (Dollar-Neutral L/S)")
    t0 = time.time()
    portfolio_df = construct_portfolio(
        predictions,
        score_col="predicted_score",
        vol_col="realized_vol_20",
        rebalance_freq=REBALANCE_FREQ,
    )

    # Verify weights
    weight_summary = verify_portfolio_weights(portfolio_df)
    if not weight_summary.empty:
        mean_net = weight_summary["net_exposure"].abs().mean()
        mean_gross = weight_summary["gross_exposure"].mean()
        print(f"  Mean gross exposure: {mean_gross:.2f}")
        print(f"  Mean abs net exposure: {mean_net:.4f}")
    print(f"Phase 7 completed in {time.time() - t0:.1f}s\n")

    # ── Phase 8 & 9: Backtest & Tearsheet ─────────────────────────────────
    print(">>> PHASE 8-9: Backtest & Performance Reporting")
    t0 = time.time()
    daily_returns, metrics = run_backtest(
        portfolio_df, panel, ic_df,
        total_cost_bps=COST_BPS,
    )
    print(f"Phase 8-9 completed in {time.time() - t0:.1f}s\n")

    # ── Save results summary ──────────────────────────────────────────────
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "ic_summary": ic_summary,
        "performance_metrics": {k: float(v) if not isinstance(v, (str, bool)) else v
                                for k, v in metrics.items()},
        "optimal_fracdiff_d": {k: float(v) for k, v in optimal_ds.items()},
        "config": {
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "label_horizon": LABEL_HORIZON,
            "rebalance_freq": REBALANCE_FREQ,
            "cost_bps": COST_BPS,
        },
    }

    results_path = outputs_dir / "results_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results summary -> {results_path}")

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s")
    print("=" * 70)

    return panel, model, daily_returns, metrics, ic_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full research pipeline.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download of all market data.")
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
