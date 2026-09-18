"""
Power BI Data Export Module
Generates structured Excel and CSV files optimized for Power BI Desktop ingestion,
including price metrics, model predictions across train/test splits, evaluation summaries,
and feature importance rankings.
"""

from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths & Constants
MODELS_DIR: Path = Path("models")
MODEL_PATH: Path = MODELS_DIR / "random_forest_model.pkl"
FEATURED_DATA_PATH: Path = Path("data/processed/featured_data.csv")
POWERBI_DATA_DIR: Path = Path("outputs/powerbi_data")

TRAIN_END_DATE: str = "2024-06-30"
EXCLUDE_FROM_X: List[str] = [
    "Date",
    "Ticker",
    "Target_Next_Return",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


def export_powerbi_datasets() -> Tuple[Path, Path, Path]:
    """
    Export model outputs, evaluation metrics, and feature importances for Power BI.
    """
    POWERBI_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Run train_model.py first.")
    if not FEATURED_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file {FEATURED_DATA_PATH} not found. Run feature_engineering.py first.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURED_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]

    # Generate predictions for all rows (both Train and Test)
    X_all = df[feature_cols]
    df["Predicted_Return"] = model.predict(X_all)
    df["Actual_Return"] = df["Target_Next_Return"]

    # Assign Split
    df["Split"] = np.where(df["Date"] <= TRAIN_END_DATE, "Train", "Test")

    # 1. Export model_output.xlsx & model_output.csv
    # Required columns: Date, Ticker, Close, Daily_Return, SMA_10, SMA_20, SMA_50, Volatility_20, Predicted_Return, Actual_Return, Split
    export_cols = [
        "Date",
        "Ticker",
        "Close",
        "Daily_Return",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "Volatility_20",
        "Predicted_Return",
        "Actual_Return",
        "Split"
    ]
    model_output_df = df[export_cols].copy()
    # Format Date as string YYYY-MM-DD for clean Excel and CSV parsing
    model_output_df["Date"] = model_output_df["Date"].dt.strftime("%Y-%m-%d")

    excel_path = POWERBI_DATA_DIR / "model_output.xlsx"
    csv_model_output_path = POWERBI_DATA_DIR / "model_output.csv"

    # Write to Excel with openpyxl
    model_output_df.to_excel(excel_path, index=False, sheet_name="Model_Data")
    model_output_df.to_csv(csv_model_output_path, index=False)
    print(f"Exported {len(model_output_df)} records -> {excel_path} and {csv_model_output_path}")

    # 2. Export metrics_summary.csv
    test_df = df[df["Split"] == "Test"].copy()
    metric_rows = []

    # Overall Test
    y_test_all = test_df["Actual_Return"]
    y_pred_all = test_df["Predicted_Return"].values
    baseline_all = np.zeros_like(y_test_all)

    metric_rows.append({
        "Ticker": "Overall",
        "Split": "Test",
        "Samples": len(test_df),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_all, y_pred_all))),
        "MAE": float(mean_absolute_error(y_test_all, y_pred_all)),
        "R2": float(r2_score(y_test_all, y_pred_all)),
        "Baseline_RMSE": float(np.sqrt(mean_squared_error(y_test_all, baseline_all))),
        "Baseline_MAE": float(mean_absolute_error(y_test_all, baseline_all)),
    })

    # Per Ticker Test
    for ticker in ["AAPL", "MSFT", "TSLA"]:
        sub = test_df[test_df["Ticker"] == ticker]
        y_true = sub["Actual_Return"]
        y_pred = sub["Predicted_Return"].values
        base = np.zeros_like(y_true)

        metric_rows.append({
            "Ticker": ticker,
            "Split": "Test",
            "Samples": len(sub),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "Baseline_RMSE": float(np.sqrt(mean_squared_error(y_true, base))),
            "Baseline_MAE": float(mean_absolute_error(y_true, base)),
        })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = POWERBI_DATA_DIR / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Exported metrics summary -> {metrics_path}")

    # 3. Export feature_importance.csv
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
        "Importance_Percent": importances * 100.0
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    feat_imp_df["Rank"] = feat_imp_df.index + 1

    feat_imp_path = POWERBI_DATA_DIR / "feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_path, index=False)
    print(f"Exported feature importances -> {feat_imp_path}")

    return excel_path, metrics_path, feat_imp_path


if __name__ == "__main__":
    export_powerbi_datasets()
