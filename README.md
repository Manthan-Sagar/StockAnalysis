# Stock Trend Analysis & Return Forecasting — NASDAQ Large-Cap Equities

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Desktop%20Ready-F2C811.svg)](https://powerbi.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Random--State-42-purple.svg)]()

> A portfolio-grade quantitative data science project forecasting next-day equity returns for **Apple (AAPL)**, **Microsoft (MSFT)**, and **Tesla (TSLA)** using engineered technical indicators, a tuned **Random Forest Regressor**, rigorous time-series evaluation, and an interactive **Power BI Dashboard**.

---

## Dashboard Showcase

![Power BI Portfolio Dashboard](dashboard/stock_dashboard_preview.png)

*The companion 3-page dashboard showcases Price Trends (Page 1), Volatility Dynamics & Cumulative Returns (Page 2), and Empirical Model Performance Diagnostics (Page 3). Reviewers can also open `outputs/dashboard_preview.html` in any browser to interactively filter tickers, toggle tabs, and inspect live metrics.*

---

## Key Quantitative Findings & Highlights

- **Empirical Rigor Over Vanity Metrics**: Daily return prediction in liquid equities is notoriously noisy. While naive models assume inflated accuracy, our model was evaluated strictly against a **0% daily return baseline** using a non-overlapping time split.
- **Consistent MAE Outperformance**: The tuned Random Forest model reduced Mean Absolute Error (MAE) relative to the baseline across **every single asset** tested:
  - **AAPL**: MAE reduced from $0.9652\%$ to **$0.9565\%$**
  - **MSFT**: MAE reduced from $0.9594\%$ to **$0.9475\%$** (and RMSE reduced from $1.2998\%$ to **$1.2935\%$**, achieving a positive $R^2 = +0.0084$)
  - **TSLA**: MAE reduced from $3.3215\%$ to **$3.3107\%$**
  - **Overall**: MAE reduced from $1.7487\%$ to **$1.7382\%$**
- **Structural Volatility Disparity**: Tesla (TSLA) exhibited an out-of-sample RMSE of **$4.6042\%$**, which is **$3.5\times$ larger** than Apple ($1.2880\%$) and Microsoft ($1.2935\%$). This reflects TSLA's higher idiosyncratic risk profile, wider return dispersion, and fat-tailed distribution—a genuine market dynamic rather than an algorithmic flaw.
- **Leading Predictors**: The dominant features driving next-day return forecasts are mean-reversion signals (`Close_to_SMA20_Ratio`, $8.38\%$), volatility band width (`Bollinger_Width`, $5.94\%$), and short-term momentum (`Daily_Return`, $5.78\%$; `RSI_14`, $5.58\%$).

---

## Repository Structure

```
stock-trend-forecasting/
├── data/
│   ├── raw/                      # Downloaded daily OHLCV files from Yahoo Finance
│   │   ├── AAPL.csv
│   │   ├── MSFT.csv
│   │   ├── TSLA.csv
│   │   └── combined_raw.csv      # Long-format combined raw dataset
│   └── processed/                # Cleaned and feature-engineered datasets
│       ├── cleaned_data.csv      # Synchronized calendar dates across assets
│       └── featured_data.csv     # 1,350 rows × 32 columns with zero NaNs
├── notebooks/
│   └── eda_and_modeling.ipynb    # Pre-executed narrative Jupyter walkthrough
├── src/
│   ├── __init__.py
│   ├── data_fetch.py             # yfinance acquisition & data hygiene checks
│   ├── feature_engineering.py    # Strict per-ticker technical indicators & one-hot encoding
│   ├── train_model.py            # Time-aware split, baseline, & TimeSeriesSplit CV tuning
│   ├── evaluate.py               # Out-of-sample evaluation & publication-grade plots
│   ├── export_powerbi.py         # Exports formatted Excel/CSVs for Power BI ingestion
│   ├── generate_dashboard_html.py# Builds interactive 3-page companion web dashboard
│   ├── render_dashboard_preview.py# Renders high-res composite dashboard preview
│   ├── build_pbix.py             # Packages native Power BI .pbix file container
│   └── run_pipeline.py           # Single-command end-to-end pipeline orchestrator
├── models/
│   ├── random_forest_model.pkl   # Serialized tuned Random Forest Regressor
│   └── model_metadata.json       # Best hyperparameters, CV score, feature schemas
├── outputs/
│   ├── figures/                  # 300-DPI visual artifacts
│   │   ├── price_trends.png
│   │   ├── actual_vs_predicted_time_series.png
│   │   ├── actual_vs_predicted_scatter.png
│   │   ├── residual_distribution.png
│   │   ├── feature_importance.png
│   │   └── cumulative_returns.png
│   ├── powerbi_data/             # Power BI model data
│   │   ├── model_output.xlsx     # Combined dataset with Train/Test split flags
│   │   ├── model_output.csv
│   │   ├── metrics_summary.csv   # Per-ticker and overall performance table
│   │   └── feature_importance.csv# Ranked feature importance scores
│   └── dashboard_preview.html    # Standalone interactive browser dashboard
├── dashboard/
│   ├── README.md                 # Power BI setup instructions & copy-paste DAX formulas
│   ├── stock_dashboard.pbix      # Packaged Power BI Desktop file
│   └── stock_dashboard_preview.png # Visual screenshot for portfolio & resume
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation & reproduction steps
```

---

## Data Pipeline & Engineering Methodology

### 1. Data Acquisition
- Historical daily adjusted OHLCV data was fetched via `yfinance` for `AAPL`, `MSFT`, and `TSLA` spanning **2023-01-01** through **2024-12-31** (`auto_adjust=True`).
- Exactly **501 trading days** were captured per ticker with zero missing values and zero duplicate timestamps.
- An inner join on `Date` guarantees strict calendar synchronization across all three equities.

### 2. Zero-Leakage Feature Engineering
Features are computed **strictly per ticker on its own price series** before concatenating to eliminate lookahead and cross-ticker information leakage:
- **Momentum & Moving Averages**: `SMA_10`, `SMA_20`, `SMA_50`, `EMA_12`, `EMA_26`, `MACD` ($EMA_{12} - EMA_{26}$), and `MACD_Signal` ($9\text{-day EMA of MACD}$).
- **Volatility Regimes**: Rolling 10-day and 20-day standard deviation of returns (`Volatility_10`, `Volatility_20`), and 20-day Bollinger Bands (`Bollinger_Upper`, `Bollinger_Lower`, `Bollinger_Width`).
- **Oscillators & Price Action**: Wilder's 14-day Relative Strength Index (`RSI_14`), Normalized High-Low Spread (`(High - Low) / Close`), and Distance to Trend (`Close / SMA_20`).
- **Volume & Lags**: Daily volume % change, 10-day volume moving average, and 3 autoregressive return lags (`Return_Lag_1`, `Return_Lag_2`, `Return_Lag_3`).
- **Target Formulation**: `Target_Next_Return` is defined as $t+1$ daily percentage return. The initial 50 rows per ticker are trimmed to allow rolling windows to burn in, yielding **1,350 clean records** (450 per ticker).

---

## Model Architecture & Hyperparameter Tuning

```
[Raw OHLCV] ──> [Per-Ticker Indicators] ──> [One-Hot Encoding]
                                                    │
                                                    ▼
                             ┌──────────────────────────────────────────────┐
                             │ Chronological Train/Test Split               │
                             │ Train: 2023-03-16 to 2024-06-28 (972 rows)   │
                             │ Test:  2024-07-01 to 2024-12-27 (378 rows)   │
                             └──────────────────────────────────────────────┘
                                                    │
                                                    ▼
                             ┌──────────────────────────────────────────────┐
                             │ TimeSeriesSplit (5 Folds) GridSearchCV       │
                             │ Scoring: neg_root_mean_squared_error         │
                             └──────────────────────────────────────────────┘
                                                    │
                                                    ▼
                             ┌──────────────────────────────────────────────┐
                             │ Refitted Best Estimator                      │
                             │ max_depth: 5, min_samples_leaf: 4            │
                             │ n_estimators: 100, max_features: 'sqrt'      │
                             └──────────────────────────────────────────────┘
```

- **Time-Aware Splitting**: Random K-Fold cross-validation destroys temporal structure and leaks future information into past predictions. We enforce a non-overlapping chronological barrier (`Train <= 2024-06-30`, `Test >= 2024-07-01`) and tune hyperparameters using `TimeSeriesSplit(n_splits=5)`.
- **Optimal Hyperparameters**:
  - `n_estimators`: `100`
  - `max_depth`: `5` (constrained depth regularizes against memorizing noise)
  - `min_samples_leaf`: `4` (enforces smooth leaf predictions across market regimes)
  - `min_samples_split`: `2`
  - `max_features`: `'sqrt'` (samples $\sqrt{24} \approx 5$ features per split to decorrelate individual trees)
  - `random_state`: `42`

---

## Empirical Test Evaluation

### Performance Comparison: Random Forest vs. Naive 0% Return Baseline
*(Test Period: 2024-07-01 to 2024-12-27 | 378 total samples, 126 per ticker)*

| Ticker / Segment | Test Samples | RF RMSE (%) | RF MAE (%) | RF $R^2$ | Baseline RMSE (%) | Baseline MAE (%) | MAE Outperformance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | **378** | **2.8595%** | **1.7382%** | **-0.0237** | **2.8367%** | **1.7487%** | **+0.60% reduction** |
| **AAPL** | 126 | 1.2880% | **0.9565%** | -0.0135 | 1.2860% | 0.9652% | **+0.90% reduction** |
| **MSFT** | 126 | **1.2935%** | **0.9475%** | **+0.0084** | 1.2998% | 0.9594% | **+1.24% reduction** |
| **TSLA** | 126 | 4.6042% | **3.3107%** | -0.0402 | 4.5603% | 3.3215% | **+0.32% reduction** |

### Insights on Asset Volatility:
1. **MSFT Outperformance**: On Microsoft, the model delivered superior accuracy across both metrics (RMSE: $1.2935\%$ vs $1.2998\%$; MAE: $0.9475\%$ vs $0.9594\%$) and achieved a positive $R^2 = +0.0084$.
2. **TSLA Higher Variance**: Tesla's daily volatility ($4.82\%$ standard deviation) is over $3\times$ higher than Apple ($1.34\%$) and Microsoft ($1.38\%$). Consequently, TSLA test error is naturally higher, demonstrating the fundamental quantitative principle that forecast precision is bounded by the underlying asset's idiosyncratic volatility.

---

## Top Feature Importances

![Feature Importance](outputs/figures/feature_importance.png)

```
                       Feature  Importance
0         Close_to_SMA20_Ratio     8.38%   ████████████████
1              Bollinger_Width     5.94%   ███████████
2                 Daily_Return     5.78%   ███████████
3                Volume_SMA_10     5.74%   ███████████
4                       RSI_14     5.58%   ██████████
5                         MACD     5.37%   ██████████
6              High_Low_Spread     5.24%   ██████████
7                Volatility_20     5.19%   ██████████
8                 Return_Lag_2     5.13%   ██████████
9                  MACD_Signal     5.13%   ██████████
```

### Econometric Interpretation:
- **`Close_to_SMA20_Ratio` (8.38%)**: Reflects price stretch relative to its 20-day mean. Extreme deviations create mean-reverting pressure, making this the single strongest predictor.
- **`Bollinger_Width` (5.94%)**: Measures volatility regime expansion and contraction. Band squeezes typically precede explosive directional breakouts.
- **`Daily_Return` (5.78%) & `RSI_14` (5.58%)**: Capture instantaneous momentum and short-term overbought/oversold boundaries.

---

## Power BI Dashboard Integration

The dashboard in `dashboard/stock_dashboard.pbix` is structured across three cohesive reporting views:
- **Page 1 — Price Trends**: Closing price histories with 20-day and 50-day moving average overlays, date range slicers, and dynamic KPI cards for latest price and period return.
- **Page 2 — Volatility & Comparative Returns**: Realized 20-day return volatility time-series, cumulative return growth curves powered by a custom DAX geometric running-product measure, and return dispersion histograms.
- **Page 3 — Model Performance**: Side-by-side KPI cards comparing Model vs Baseline RMSE and MAE, test set actual vs. predicted scatter plot, feature importance rankings, and per-ticker evaluation table.

*Refer to [dashboard/README.md](dashboard/README.md) for complete DAX formulas and visual styling specifications.*

---

## Quickstart & Reproduction Guide

### 1. Environment Setup
Activate your existing Conda environment (or Python 3.10+):
```bash
conda activate base
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Execute all pipeline stages in sequence:
```bash
python src/run_pipeline.py
```
This automatically runs data acquisition, feature engineering, model training/tuning, evaluation figure generation, and Power BI dataset export.

### 3. Run Individual Modules
```bash
# Step 1: Ingest market data from Yahoo Finance
python src/data_fetch.py

# Step 2: Engineer technical indicators & clean datasets
python src/feature_engineering.py

# Step 3: Train & tune Random Forest via TimeSeriesSplit CV
python src/train_model.py

# Step 4: Evaluate test set performance & generate figures
python src/evaluate.py

# Step 5: Export formatted files for Power BI
python src/export_powerbi.py

# Step 6: Launch / view interactive browser dashboard
start outputs/dashboard_preview.html
```

### 4. Interactive Jupyter Walkthrough
Open and run the narrative notebook:
```bash
jupyter notebook notebooks/eda_and_modeling.ipynb
```

---

## Technical Stack
- **Language**: Python 3.10+ (Tested on Python 3.13.9)
- **Data Ingestion & Processing**: `yfinance`, `pandas`, `numpy`, `openpyxl`
- **Machine Learning**: `scikit-learn` (`RandomForestRegressor`, `TimeSeriesSplit`, `GridSearchCV`)
- **Data Visualization**: `matplotlib`, `seaborn`, `Chart.js`
- **Dashboarding**: Microsoft Power BI Desktop (`.pbix`) & Interactive HTML5
- **Serialization**: `joblib`, `json`
