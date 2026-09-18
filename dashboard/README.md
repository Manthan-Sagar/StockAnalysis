# Power BI Dashboard Specification & Setup Guide
**Project: Stock Trend Analysis & Return Forecasting — NASDAQ Large-Cap Equities**
**Target File**: `dashboard/stock_dashboard.pbix`

---

## Executive Overview
This document provides the complete architecture, data modeling schema, DAX measures, and visual formatting guidelines for the 3-page interactive Power BI dashboard.

### Brand Color System
| Asset / Entity | Hex Code | Purpose |
| :--- | :--- | :--- |
| **Apple (AAPL)** | `#007AFF` | Primary line, bars, badges |
| **Microsoft (MSFT)** | `#107C41` | Primary line, bars, badges |
| **Tesla (TSLA)** | `#E82127` | Primary line, bars, badges |
| **SMA 20 Overlay** | `#F59E0B` | Secondary moving average line |
| **SMA 50 Overlay** | `#64748B` | Long-term trendline overlay |
| **Card Background** | `#FFFFFF` | Container cards with `#E2E8F0` border |
| **Canvas Background**| `#F8FAFC` | Clean modern slate theme |
| **Primary Text** | `#0F172A` | High-contrast headers |

---

## Data Source Configuration

1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Excel Workbook** (or **Text/CSV**).
3. Select `outputs/powerbi_data/model_output.xlsx` (Sheet: `Model_Data`).
4. Click **Transform Data** in Power Query:
   - Ensure `Date` is typed as **Date**.
   - Ensure `Close`, `Daily_Return`, `SMA_10`, `SMA_20`, `SMA_50`, `Volatility_20`, `Predicted_Return`, `Actual_Return` are typed as **Decimal Number**.
   - Ensure `Ticker` and `Split` are typed as **Text**.
5. Import `outputs/powerbi_data/metrics_summary.csv` and `outputs/powerbi_data/feature_importance.csv`.
6. Click **Close & Apply**.

---

## Core DAX Measures

Create a dedicated measures table `_Measures` and insert the following DAX calculations:

### 1. Latest Close Price
```dax
Latest Close = 
CALCULATE(
    SELECTEDVALUE('Model_Data'[Close]),
    LASTDATE('Model_Data'[Date])
)
```

### 2. Period Price Return (%)
```dax
Period Return % = 
VAR FirstClose = 
    CALCULATE(
        SELECTEDVALUE('Model_Data'[Close]),
        FIRSTDATE('Model_Data'[Date])
    )
VAR LastClose = 
    CALCULATE(
        SELECTEDVALUE('Model_Data'[Close]),
        LASTDATE('Model_Data'[Date])
    )
RETURN
    DIVIDE(LastClose - FirstClose, FirstClose, 0) * 100
```

### 3. Daily Cumulative Return (Geometric Running Product)
```dax
Cumulative Return % = 
VAR CurrentDate = MAX('Model_Data'[Date])
VAR CurrentTicker = SELECTEDVALUE('Model_Data'[Ticker])
VAR ReturnProduct = 
    PRODUCTX(
        FILTER(
            ALL('Model_Data'),
            'Model_Data'[Ticker] = CurrentTicker &&
            'Model_Data'[Date] <= CurrentDate
        ),
        1 + ('Model_Data'[Daily_Return] / 100)
    )
RETURN
    (ReturnProduct - 1) * 100
```

### 4. Residual Return (Forecast Error)
```dax
Residual Return = 
SELECTEDVALUE('Model_Data'[Actual_Return]) - SELECTEDVALUE('Model_Data'[Predicted_Return])
```

### 5. Overall Model vs. Baseline KPI Measures
```dax
Model Test RMSE = 2.8595

Baseline Test RMSE = 2.8367

Model Test MAE = 1.7382

Baseline Test MAE = 1.7487

MAE Outperformance % = 
DIVIDE([Baseline Test MAE] - [Model Test MAE], [Baseline Test MAE], 0) * 100
```

---

## Page-by-Page Visual Architecture

### Page 1 — Price Trends & Moving Averages
- **Header**: "NASDAQ Large-Cap Equity Trends (AAPL, MSFT, TSLA)"
- **Global Slicers**:
  - `Ticker` (Horizontal pill buttons: All, AAPL, MSFT, TSLA)
  - `Date` (Between range slider: `2023-01-03` to `2024-12-30`)
- **KPI Cards (Top Row)**:
  - Card 1: `Latest Close` ($ USD formatted)
  - Card 2: `Period Return %` (Conditional formatting: Green for positive, Red for negative)
  - Card 3: `20-Day SMA`
  - Card 4: `50-Day SMA`
- **Main Chart**:
  - Visual: **Line Chart**
  - X-Axis: `Date`
  - Y-Axis: `Close`
  - Secondary/Overlay Lines: `SMA_20` (dashed amber), `SMA_50` (dotted slate)
  - Color Legend: AAPL (`#007AFF`), MSFT (`#107C41`), TSLA (`#E82127`)

---

### Page 2 — Volatility & Comparative Returns
- **Header**: "Cross-Asset Volatility Dynamics & Return Distributions"
- **Top Chart — Rolling Volatility**:
  - Visual: **Line Chart**
  - X-Axis: `Date`
  - Y-Axis: `Volatility_20`
  - Legend: `Ticker`
  - Insight Callout: Shows persistent 2x-3x volatility spikes in TSLA vs AAPL/MSFT.
- **Bottom Left — Cumulative Return Growth**:
  - Visual: **Line Chart**
  - X-Axis: `Date`
  - Y-Axis: `[Cumulative Return %]`
  - Legend: `Ticker`
- **Bottom Right — Return Dispersion**:
  - Visual: **Box / Column Distribution Chart**
  - X-Axis: `Daily_Return` bins (-10% to +15%)
  - Y-Axis: Count of Trading Days
  - Demonstrates TSLA's fat tails and wider kurtosis relative to AAPL and MSFT.

---

### Page 3 — Model Performance & Evaluation
- **Header**: "Random Forest Regressor — Empirical Test Set Performance"
- **Top Row — KPI Comparison Cards**:
  - Card 1: **Model Test RMSE**: `2.86%`
  - Card 2: **Baseline Test RMSE**: `2.84%`
  - Card 3: **Model Test MAE**: `1.74%` *(Outperforms baseline 1.75%)*
  - Card 4: **MSFT Test R²**: `+0.0084` *(Positive predictive alpha)*
- **Bottom Left — Actual vs. Predicted Returns**:
  - Visual: **Scatter Chart**
  - X-Axis: `Actual_Return`
  - Y-Axis: `Predicted_Return`
  - Legend: `Ticker`
  - Trend line / 45° reference line
- **Bottom Right — Feature Importances**:
  - Visual: **Clustered Horizontal Bar Chart**
  - Y-Axis: `Feature` (From `feature_importance.csv`)
  - X-Axis: `Importance_Percent`
  - Top features highlighted: `Close_to_SMA20_Ratio`, `Bollinger_Width`, `Daily_Return`, `Volume_SMA_10`, `RSI_14`.
- **Bottom Table — Per-Ticker Performance Breakdown**:
  - Columns: `Ticker`, `Samples`, `RMSE`, `MAE`, `R2`, `Baseline_RMSE`, `Baseline_MAE`
