"""
Model Evaluation & Visual Diagnostics Module
Calculates test set metrics (overall & per-ticker), compares against naive baseline,
generates publication-ready figures, and outputs feature importances.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths & Styling
MODELS_DIR: Path = Path("models")
MODEL_PATH: Path = MODELS_DIR / "random_forest_model.pkl"
FEATURED_DATA_PATH: Path = Path("data/processed/featured_data.csv")
FIGURES_DIR: Path = Path("outputs/figures")

TRAIN_END_DATE: str = "2024-06-30"
TEST_START_DATE: str = "2024-07-01"
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

# Color Palette
TICKER_COLORS = {
    "AAPL": "#007AFF",  # Apple Blue
    "MSFT": "#107C41",  # Microsoft Green
    "TSLA": "#E82127",  # Tesla Red
}

# Matplotlib styling for portfolio-grade figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True
})


def load_model_and_datasets() -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load saved model and featured dataset, splitting into train and test.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train the model first.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURED_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    train_df = df[df["Date"] <= TRAIN_END_DATE].copy().reset_index(drop=True)
    test_df = df[df["Date"] >= TEST_START_DATE].copy().reset_index(drop=True)

    return model, df, train_df, test_df, feature_cols


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, R², and Baseline RMSE for any given series.
    """
    baseline_pred = np.zeros_like(y_true)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
    baseline_mae = float(mean_absolute_error(y_true, baseline_pred))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_MAE": baseline_mae
    }


def evaluate_test_performance(
    model: Any,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate model on test dataset overall and per ticker.
    """
    X_test = test_df[feature_cols]
    y_test = test_df["Target_Next_Return"]

    test_df["Predicted_Return"] = model.predict(X_test)
    test_df["Actual_Return"] = y_test
    test_df["Residual"] = test_df["Actual_Return"] - test_df["Predicted_Return"]

    # Overall metrics
    overall_metrics = compute_metrics(y_test, test_df["Predicted_Return"].values)
    overall_metrics["Ticker"] = "Overall"
    overall_metrics["Samples"] = len(test_df)

    rows = [overall_metrics]

    # Per-ticker metrics
    for ticker in ["AAPL", "MSFT", "TSLA"]:
        sub = test_df[test_df["Ticker"] == ticker]
        if not sub.empty:
            m = compute_metrics(sub["Actual_Return"], sub["Predicted_Return"].values)
            m["Ticker"] = ticker
            m["Samples"] = len(sub)
            rows.append(m)

    metrics_df = pd.DataFrame(rows)
    cols = ["Ticker", "Samples", "RMSE", "MAE", "R2", "Baseline_RMSE", "Baseline_MAE"]
    metrics_df = metrics_df[cols]

    print("\n" + "=" * 65)
    print("TEST SET PERFORMANCE EVALUATION (2024-07-01 to 2024-12-31)")
    print("=" * 65)
    print(metrics_df.to_string(index=False))

    return test_df, metrics_df


def plot_price_trends(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """
    Plot historical Close prices and overlay moving averages for each ticker.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    tickers = ["AAPL", "MSFT", "TSLA"]

    for ax, ticker in zip(axes, tickers):
        sub = df[df["Ticker"] == ticker].sort_values("Date")
        color = TICKER_COLORS.get(ticker, "#333333")

        ax.plot(sub["Date"], sub["Close"], label=f"{ticker} Close", color=color, linewidth=2.0)
        ax.plot(sub["Date"], sub["SMA_20"], label="20-Day SMA", color="#F59E0B", linestyle="--", alpha=0.85, linewidth=1.4)
        ax.plot(sub["Date"], sub["SMA_50"], label="50-Day SMA", color="#6B7280", linestyle=":", alpha=0.85, linewidth=1.4)

        ax.set_title(f"{ticker} — Price Trend & Moving Averages (2023 - 2024)", fontweight="bold", loc="left")
        ax.set_ylabel("Price ($ USD)")
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0")
        ax.grid(True)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    output_path = output_dir / "price_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def plot_actual_vs_predicted_time_series(test_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """
    Plot actual vs predicted daily returns across the test period per ticker.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    tickers = ["AAPL", "MSFT", "TSLA"]

    for ax, ticker in zip(axes, tickers):
        sub = test_df[test_df["Ticker"] == ticker].sort_values("Date")
        color = TICKER_COLORS.get(ticker, "#333333")

        ax.plot(sub["Date"], sub["Actual_Return"], label="Actual Return (%)", color=color, alpha=0.7, linewidth=1.4)
        ax.plot(sub["Date"], sub["Predicted_Return"], label="Predicted Return (%)", color="#1E293B", linestyle="--", linewidth=1.8)

        ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{ticker} — Actual vs. Predicted Daily Return (Test Period: H2 2024)", fontweight="bold", loc="left")
        ax.set_ylabel("Daily Return (%)")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E2E8F0")
        ax.grid(True)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    output_path = output_dir / "actual_vs_predicted_time_series.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def plot_actual_vs_predicted_scatter(test_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """
    Plot actual vs predicted returns scatter with 45-degree reference line.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    for ticker, group in test_df.groupby("Ticker"):
        ax.scatter(
            group["Actual_Return"],
            group["Predicted_Return"],
            label=ticker,
            color=TICKER_COLORS.get(ticker, "#333333"),
            alpha=0.65,
            edgecolors="none",
            s=40
        )

    # 45-degree reference line
    lims = [
        min(test_df["Actual_Return"].min(), test_df["Predicted_Return"].min()) - 1,
        max(test_df["Actual_Return"].max(), test_df["Predicted_Return"].max()) + 1
    ]
    ax.plot(lims, lims, "k--", alpha=0.6, linewidth=1.5, label="Perfect Prediction (45° Line)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Daily Return (%)", fontweight="bold")
    ax.set_ylabel("Predicted Daily Return (%)", fontweight="bold")
    ax.set_title("Actual vs. Predicted Daily Returns — Test Set Evaluation", fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax.grid(True)

    plt.tight_layout()
    output_path = output_dir / "actual_vs_predicted_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def plot_residual_distribution(test_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """
    Plot error distribution (residuals) for each ticker.
    Highlights TSLA's higher volatility and wider error spread.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for ticker, group in test_df.groupby("Ticker"):
        sns.kdeplot(
            group["Residual"],
            label=f"{ticker} (std: {group['Residual'].std():.2f}%)",
            color=TICKER_COLORS.get(ticker, "#333333"),
            linewidth=2.2,
            fill=True,
            alpha=0.15,
            ax=ax
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Zero Error")
    ax.set_xlabel("Residual (Actual Return − Predicted Return) (%)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Residual Error Distribution by Ticker (TSLA Exhibits Wider Variance)", fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax.grid(True)

    plt.tight_layout()
    output_path = output_dir / "residual_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    output_dir: Path = FIGURES_DIR
) -> pd.DataFrame:
    """
    Extract and plot model feature importances as a horizontal bar chart.
    """
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(feat_imp["Feature"], feat_imp["Importance"] * 100, color="#2563EB", edgecolor="#1D4ED8", height=0.65)

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%",
                va="center", ha="left", fontsize=9, color="#1E293B")

    ax.set_xlabel("Relative Importance (%)", fontweight="bold")
    ax.set_title("Random Forest Regressor — Feature Importances", fontweight="bold", loc="left", pad=12)
    ax.grid(axis="x")
    ax.set_xlim(0, max(feat_imp["Importance"] * 100) + 2.5)

    plt.tight_layout()
    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")

    # Return descending table
    return feat_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)


def plot_cumulative_returns(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """
    Plot cumulative return trajectories of AAPL, MSFT, and TSLA from 2023 through 2024.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for ticker, group in df.groupby("Ticker"):
        sub = group.sort_values("Date").copy()
        sub["Cumulative_Return"] = (1.0 + sub["Daily_Return"] / 100.0).cumprod() - 1.0
        ax.plot(
            sub["Date"],
            sub["Cumulative_Return"] * 100.0,
            label=f"{ticker} (Final: {sub['Cumulative_Return'].iloc[-1]*100:.1f}%)",
            color=TICKER_COLORS.get(ticker, "#333333"),
            linewidth=2.2
        )

    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontweight="bold")
    ax.set_title("Cumulative Equity Returns: AAPL vs. MSFT vs. TSLA (2023 - 2024)", fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    output_path = output_dir / "cumulative_returns.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {output_path}")


def run_evaluation_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run full model evaluation and visual generation suite.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model, full_df, train_df, test_df, feature_cols = load_model_and_datasets()

    test_results, metrics_df = evaluate_test_performance(model, test_df, feature_cols)
    feat_imp_df = plot_feature_importance(model, feature_cols)
    plot_price_trends(full_df)
    plot_actual_vs_predicted_time_series(test_results)
    plot_actual_vs_predicted_scatter(test_results)
    plot_residual_distribution(test_results)
    plot_cumulative_returns(full_df)

    return test_results, metrics_df, feat_imp_df


if __name__ == "__main__":
    run_evaluation_pipeline()
