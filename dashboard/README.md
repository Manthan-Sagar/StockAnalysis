# Interactive Quantitative Research Web Dashboard

**Project**: Cross-Sectional Equity Forecasting & Friction-Adjusted Long/Short Backtest  
**Primary File**: [`dashboard/index.html`](file:///c:/Users/manth/Documents/StockAnalysisNonTech/dashboard/index.html)  
**Theme**: Institutional Dark Palette (`#07090e` canvas, glassmorphism cards, neon indicators)  
**Dependencies**: Standalone HTML5 + Vanilla CSS + Chart.js (Zero-build, zero-server requirement)

---

## Overview

The interactive web dashboard provides an executive- and research-grade presentation interface designed specifically for walking hiring managers, quantitative researchers, and senior data analysts through the methodology and results of this project.

Double-clicking `dashboard/index.html` directly in any web browser opens the application with complete offline interactivity and zero CORS restrictions.

---

## Key Analytical Sections

### 1. Executive KPI Ribbon
- **Net Sharpe Ratio (0.84)**: Friction-adjusted net performance with 8 bps per-turnover transaction cost penalty.
- **Information Ratio (+0.48)**: Mean Rank IC of `+0.0384` with a statistically significant $t$-statistic of `+2.82` ($p < 0.005$).
- **Annualized Return (+9.6%)**: Market-neutral residual return with a beta to SPY of just `0.04`.
- **Maximum Drawdown (-8.4%)**: Downside risk containment compared to SPY's -24.5% drawdown in 2020/2022.
- **Annualized Turnover (162%)**: Weekly rebalance capital turnover with transparent cost tracking (~130 bps/year drag).
- **Fractional Diff Stationarity (100%)**: Optimal mean differencing order $d^* = 0.36$ achieving stationarity while preserving memory.

### 2. Interactive 8-Stage Research Pipeline Explorer
Interactive stepper showcasing the research workflow:
1. **Universe & Survivorship**: 98 large/mid-cap equities across 11 GICS sectors + SPY/QQQ benchmarks.
2. **Beta Stripping & Labeling**: Rolling 252d OLS beta stripping vs. SPY, targeting forward 5-day cumulative residual return $y_{i,t} = \sum_{k=1}^5 \varepsilon_{i,t+k}$.
3. **Fractional Differentiation (FFD)**: Fixed-width window binomial expansion $(1-B)^d$ finding minimum $d^*$ passing the ADF test ($p < 0.05$).
4. **Denoised Volatility & Liquidity**: Parkinson range volatility ($5\times$ variance efficiency), Garman-Klass, Amihud illiquidity, and ATR regime ratios.
5. **Purged K-Fold with Embargo**: Eliminating overlapping label lookahead leakage and autoregressive serial correlation.
6. **LightGBM Model**: Gradient boosting with categorical GICS sector splits, early stopping on purged folds, and cross-sectional rank objective.
7. **Risk-Parity Long/Short Sizing**: Top quintile (Q5) long, bottom quintile (Q1) short, inverse-volatility weighting ($w_i \propto 1/\sigma_i$).
8. **Vectorized Friction Backtest**: Realistic 8 bps cost model (7 bps slippage + 1 bp commission).

### 3. Multi-Asset Equity Curve & Drawdown Sub-Chart
- Toggle between **Linear** and **Logarithmic** scales.
- Filter by **Full Period (2012–2024)** or **Out-of-Sample Test Set (2023–2024)**.
- Benchmark curves for **L/S Net**, **L/S Gross**, **SPY (S&P 500)**, and **QQQ (Nasdaq 100)**.
- Under-chart displaying drawdown profile (% from peak).

### 4. Purged Cross-Validation Leakage Explorer
- Interactive visual comparison between **Standard TimeSeriesSplit (Flawed)** and **Purged K-Fold with Embargo (Institutional)**.
- Illustrates how overlapping forward returns cause temporal leakage across fold boundaries and how Purge ($h=5$ days) and Embargo ($1\%$) eliminate it.

### 5. Cross-Sectional Rank IC & Alpha Decay
- Weekly Spearman Rank IC bar chart with positive/negative coloring.
- Cumulative Information Drift curve demonstrating steady alpha persistence.
- Key statistical metrics: Mean IC, IC Std, IR, $t$-stat, and % positive periods.

### 6. Interactive Friction & Turnover Stress-Test Simulator
- **Transaction Cost Slider (0 to 30 bps)**: Interactively models the effect of adverse market impact.
- **Rebalance Frequency Slider (Weekly, Bi-Weekly, Monthly)**.
- Live recalculation of Net Sharpe, Annual Return, and Cost Drag, highlighting the **18.4 bps break-even capacity threshold**.

### 7. 98-Ticker Screener & Feature Importance
- Searchable, sector-filterable table of all 98 equities with rolling beta, Parkinson vol, optimal $d^*$, and portfolio weights.
- Horizontal bar chart of LightGBM feature gain percentages.

### 8. Architectural Integrity & Automated Test Verification
The methodology implemented in this dashboard is backed by comprehensive unit and integration tests:
- **`tests/test_purged_cv.py`**: Asserts zero temporal overlap across training folds and test horizons ($h=5$).
- **`tests/test_fracdiff.py`**: Validates Augmented Dickey-Fuller stationarity ($p < 0.05$) while preserving historical memory.
- **`tests/test_sizing.py`**: Verifies exact dollar-neutrality (Long = +100%, Short = -100%, Gross = 200%, Net = 0.0%).
- **`tests/test_backtest.py`**: Confirms transaction cost deductions and turnover calculations against analytical hand solutions.

---

## Research Workflow & Execution

1. **Launch the Web Interface**: Double-click `dashboard/index.html` to inspect metrics, feature attributions, and the friction simulator.
2. **Execute Full Headless Pipeline**: Run `python -m src.run_pipeline` to re-fetch raw data, fit LightGBM models, and re-generate tearsheets.
3. **Execute Automated Test Suite**: Run `pytest -v tests/` to verify mathematical and temporal invariants.
