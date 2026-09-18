"""
Comprehensive Automated Test Suite for Stock Trend Analysis & Return Forecasting
Validates data hygiene, feature calculations, time-series split barriers,
model integrity, Power BI exports, and visual artifacts.
"""

import json
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def raw_combined():
    path = PROJECT_ROOT / "data/raw/combined_raw.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture(scope="session")
def cleaned_df():
    path = PROJECT_ROOT / "data/processed/cleaned_data.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture(scope="session")
def featured_df():
    path = PROJECT_ROOT / "data/processed/featured_data.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture(scope="session")
def trained_model():
    path = PROJECT_ROOT / "models/random_forest_model.pkl"
    assert path.exists(), f"Missing {path}"
    model = joblib.load(path)
    return model


# ============================================================================
# 1. Raw Data Integrity Tests
# ============================================================================
def test_raw_data_files_exist():
    """Verify raw files exist for all tickers and combined."""
    for ticker in ["AAPL", "MSFT", "TSLA"]:
        path = PROJECT_ROOT / f"data/raw/{ticker}.csv"
        assert path.exists(), f"Raw file for {ticker} missing!"
        df = pd.read_csv(path)
        assert len(df) >= 450, f"Expected ~500 trading days for {ticker}, got {len(df)}"
        assert df["Date"].duplicated().sum() == 0, f"Duplicate dates in {ticker}!"
        for col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns, f"Missing column {col} in {ticker}"
            assert not df[col].isnull().any(), f"Null values in {ticker} {col}"


def test_raw_combined_dates(raw_combined):
    """Verify combined raw data spans 2023 through 2024 without duplicates."""
    assert raw_combined["Date"].min() >= pd.to_datetime("2023-01-01")
    assert raw_combined["Date"].max() <= pd.to_datetime("2024-12-31")
    assert set(raw_combined["Ticker"].unique()) == {"AAPL", "MSFT", "TSLA"}


# ============================================================================
# 2. Data Cleaning & Synchronization Tests
# ============================================================================
def test_cleaned_calendar_synchronization(cleaned_df):
    """Verify inner join on Date across all 3 tickers keeps identical dates."""
    dates_per_ticker = cleaned_df.groupby("Ticker")["Date"].apply(set)
    aapl_dates = dates_per_ticker["AAPL"]
    msft_dates = dates_per_ticker["MSFT"]
    tsla_dates = dates_per_ticker["TSLA"]

    assert aapl_dates == msft_dates, "AAPL and MSFT dates do not match!"
    assert aapl_dates == tsla_dates, "AAPL and TSLA dates do not match!"
    assert len(aapl_dates) == 501, f"Expected 501 common days, got {len(aapl_dates)}"
    assert not cleaned_df.isnull().any().any(), "Cleaned dataset contains null values!"


# ============================================================================
# 3. Feature Engineering Tests
# ============================================================================
def test_featured_dataset_properties(featured_df):
    """Verify engineered indicators, row counts, and no residual NaNs."""
    # 450 rows per ticker (501 - 50 warm-up - 1 target shift = 450)
    counts = featured_df.groupby("Ticker").size().to_dict()
    assert counts == {"AAPL": 450, "MSFT": 450, "TSLA": 450}, f"Unexpected row distribution: {counts}"
    assert len(featured_df) == 1350

    # Required indicators
    required_features = [
        "Daily_Return", "Target_Next_Return",
        "SMA_10", "SMA_20", "SMA_50",
        "EMA_12", "EMA_26", "MACD", "MACD_Signal",
        "Volatility_10", "Volatility_20",
        "Bollinger_Upper", "Bollinger_Lower", "Bollinger_Width",
        "RSI_14", "Volume_Change", "Volume_SMA_10",
        "High_Low_Spread", "Close_to_SMA20_Ratio",
        "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
        "Ticker_AAPL", "Ticker_MSFT", "Ticker_TSLA"
    ]
    for feat in required_features:
        assert feat in featured_df.columns, f"Missing engineered feature: {feat}"

    # Zero NaNs anywhere in featured dataset
    assert not featured_df.isnull().any().any(), "Featured dataset has null values!"

    # Indicator mathematical bounds
    assert (featured_df["RSI_14"] >= 0).all() and (featured_df["RSI_14"] <= 100).all(), "RSI out of [0, 100]!"
    assert (featured_df["Bollinger_Upper"] >= featured_df["Bollinger_Lower"]).all(), "Upper Bollinger Band < Lower!"
    assert (featured_df["Volatility_20"] >= 0).all(), "Volatility must be non-negative!"


# ============================================================================
# 4. Time-Aware Split & No-Leakage Tests
# ============================================================================
def test_time_aware_split_no_leakage(featured_df):
    """Verify strict chronological barrier between train and test."""
    train_df = featured_df[featured_df["Date"] <= "2024-06-30"]
    test_df = featured_df[featured_df["Date"] >= "2024-07-01"]

    assert len(train_df) == 972, f"Expected 972 train rows, got {len(train_df)}"
    assert len(test_df) == 378, f"Expected 378 test rows, got {len(test_df)}"

    # Exactly 126 test days per ticker
    test_counts = test_df.groupby("Ticker").size().to_dict()
    assert test_counts == {"AAPL": 126, "MSFT": 126, "TSLA": 126}

    # Strict temporal ordering: train max date < test min date
    assert train_df["Date"].max() < test_df["Date"].min(), "Temporal lookahead leakage detected!"


# ============================================================================
# 5. Model & Prediction Integrity Tests
# ============================================================================
def test_trained_model_inference(trained_model, featured_df):
    """Verify model type, feature names, and inference capability."""
    assert isinstance(trained_model, RandomForestRegressor), "Model is not a RandomForestRegressor!"
    assert hasattr(trained_model, "feature_importances_"), "Model missing feature importances!"
    assert len(trained_model.feature_importances_) == 24, "Expected 24 feature importances!"

    # Load metadata
    meta_path = PROJECT_ROOT / "models/model_metadata.json"
    assert meta_path.exists(), "Model metadata JSON missing!"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    assert len(feature_names) == 24

    # Run inference test
    X = featured_df[feature_names].head(10)
    preds = trained_model.predict(X)
    assert len(preds) == 10
    assert not np.isnan(preds).any(), "Model predictions contain NaNs!"


# ============================================================================
# 6. Power BI Data Export Tests
# ============================================================================
def test_powerbi_data_exports():
    """Verify model_output.xlsx, metrics_summary.csv, and feature_importance.csv."""
    excel_path = PROJECT_ROOT / "outputs/powerbi_data/model_output.xlsx"
    csv_path = PROJECT_ROOT / "outputs/powerbi_data/model_output.csv"
    metrics_path = PROJECT_ROOT / "outputs/powerbi_data/metrics_summary.csv"
    feat_path = PROJECT_ROOT / "outputs/powerbi_data/feature_importance.csv"

    assert excel_path.exists(), "Missing model_output.xlsx!"
    assert csv_path.exists(), "Missing model_output.csv!"
    assert metrics_path.exists(), "Missing metrics_summary.csv!"
    assert feat_path.exists(), "Missing feature_importance.csv!"

    # Read and inspect excel
    df_excel = pd.read_excel(excel_path, sheet_name="Model_Data")
    assert len(df_excel) == 1350, f"Expected 1,350 rows in model_output.xlsx, got {len(df_excel)}"

    required_excel_cols = [
        "Date", "Ticker", "Close", "Daily_Return",
        "SMA_10", "SMA_20", "SMA_50", "Volatility_20",
        "Predicted_Return", "Actual_Return", "Split"
    ]
    for col in required_excel_cols:
        assert col in df_excel.columns, f"Missing {col} in model_output.xlsx"

    # Verify metrics summary
    df_metrics = pd.read_csv(metrics_path)
    assert set(df_metrics["Ticker"].unique()) == {"Overall", "AAPL", "MSFT", "TSLA"}
    assert (df_metrics["RMSE"] > 0).all()
    assert (df_metrics["MAE"] > 0).all()

    # Verify PBIX container
    pbix_path = PROJECT_ROOT / "dashboard/stock_dashboard.pbix"
    assert pbix_path.exists(), "Missing stock_dashboard.pbix!"
    assert zipfile.is_zipfile(pbix_path), "stock_dashboard.pbix is not a valid zip archive!"


# ============================================================================
# 7. Visual Artifacts & Documentation Tests
# ============================================================================
def test_visual_figures_and_previews():
    """Verify all 300-DPI publication figures and interactive HTML exist."""
    figure_names = [
        "price_trends.png",
        "actual_vs_predicted_time_series.png",
        "actual_vs_predicted_scatter.png",
        "residual_distribution.png",
        "feature_importance.png",
        "cumulative_returns.png"
    ]
    for fig_name in figure_names:
        fig_path = PROJECT_ROOT / f"outputs/figures/{fig_name}"
        assert fig_path.exists(), f"Missing figure {fig_name}!"
        assert fig_path.stat().st_size > 10_000, f"Figure {fig_name} is too small / empty!"

    preview_path = PROJECT_ROOT / "dashboard/stock_dashboard_preview.png"
    assert preview_path.exists(), "Missing stock_dashboard_preview.png!"
    assert preview_path.stat().st_size > 50_000, "Dashboard preview image too small!"

    # Interactive HTML dashboards
    html_out = PROJECT_ROOT / "outputs/dashboard_preview.html"
    html_dash = PROJECT_ROOT / "dashboard/index.html"
    assert html_out.exists(), "Missing outputs/dashboard_preview.html!"
    assert html_dash.exists(), "Missing dashboard/index.html!"
    assert "chart.js" in html_out.read_text(encoding="utf-8").lower()


def test_notebook_executed_outputs():
    """Verify notebooks/eda_and_modeling.ipynb is valid JSON and has executed outputs."""
    nb_path = PROJECT_ROOT / "notebooks/eda_and_modeling.ipynb"
    assert nb_path.exists(), "Missing eda_and_modeling.ipynb!"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    code_cells = [c for c in nb_data["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 8, f"Expected at least 8 code cells, got {len(code_cells)}"

    # Verify at least several code cells have pre-rendered outputs
    cells_with_outputs = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    assert len(cells_with_outputs) >= 6, "Notebook has not been pre-executed with outputs!"
