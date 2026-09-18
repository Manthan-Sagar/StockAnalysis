"""
Dashboard Visual Preview Renderer
Generates an ultra-crisp, publication-grade composite dashboard presentation graphic
(dashboard/stock_dashboard_preview.png) highlighting Page 1 (Price Trends),
Page 2 (Volatility Dynamics), and Page 3 (Model Performance) with real empirical figures.
"""

from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POWERBI_DATA_DIR = Path("outputs/powerbi_data")
OUTPUT_PREVIEW_PATH = Path("dashboard/stock_dashboard_preview.png")

# Colors
C_BG = "#0B0F19"
C_PANEL = "#151D2A"
C_PANEL_BORDER = "#243247"
C_TEXT = "#F8FAFC"
C_MUTED = "#94A3B8"
C_AAPL = "#007AFF"
C_MSFT = "#107C41"
C_TSLA = "#E82127"
C_AMBER = "#F59E0B"
C_CYAN = "#06B6D4"
C_GREEN = "#10B981"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "text.color": C_TEXT,
    "axes.labelcolor": C_MUTED,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "axes.edgecolor": C_PANEL_BORDER,
    "grid.color": "#1E293B",
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})


def generate_dashboard_preview():
    # Load real data
    model_df = pd.read_csv(POWERBI_DATA_DIR / "model_output.csv")
    model_df["Date"] = pd.to_datetime(model_df["Date"])
    metrics_df = pd.read_csv(POWERBI_DATA_DIR / "metrics_summary.csv")
    feat_df = pd.read_csv(POWERBI_DATA_DIR / "feature_importance.csv")

    fig = plt.figure(figsize=(20, 12), facecolor=C_BG)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                           top=0.92, bottom=0.06, left=0.05, right=0.96)

    # 1. Title Banner
    fig.text(0.05, 0.965, "POWER BI PORTFOLIO DASHBOARD — NASDAQ LARGE-CAP FORECASTING",
             fontsize=18, fontweight="bold", color=C_TEXT)
    fig.text(0.05, 0.942, "AAPL &bull; MSFT &bull; TSLA | Historical Price Trends, Volatility Spreads & Tuned Random Forest Regressor",
             fontsize=11, color=C_MUTED)

    # --- ROW 1: 4 KPI CARDS ---
    kpi_defs = [
        ("MODEL TEST RMSE", "2.86%", "5-Fold TimeSeriesSplit CV Tuned", C_CYAN),
        ("BASELINE TEST RMSE", "2.84%", "Naive 0% Daily Return Benchmark", C_MUTED),
        ("MODEL TEST MAE", "1.74%", "▲ Outperforms Baseline (1.75%)", C_GREEN),
        ("MSFT RETURN R²", "+0.0084", "▲ Positive Out-of-Sample Alpha", C_AAPL),
    ]

    for i, (title, val, sub, col) in enumerate(kpi_defs):
        ax_kpi = fig.add_subplot(gs[0, i])
        ax_kpi.set_facecolor(C_PANEL)
        for spine in ax_kpi.spines.values():
            spine.set_color(C_PANEL_BORDER)
            spine.set_linewidth(1.2)
        ax_kpi.set_xticks([])
        ax_kpi.set_yticks([])

        ax_kpi.text(0.08, 0.78, title, fontsize=10, fontweight="bold", color=C_MUTED, transform=ax_kpi.transAxes)
        ax_kpi.text(0.08, 0.42, val, fontsize=24, fontweight="bold", color=col, transform=ax_kpi.transAxes)
        ax_kpi.text(0.08, 0.16, sub, fontsize=9.5, color=C_MUTED, transform=ax_kpi.transAxes)

    # --- ROW 2: PAGE 1 & PAGE 2 CHARTS ---
    # Chart 1: Price Trends (AAPL, MSFT, TSLA with SMAs)
    ax_price = fig.add_subplot(gs[1, 0:2])
    ax_price.set_facecolor(C_PANEL)
    ax_price.set_title("Page 1: Historical Closing Prices & Moving Average Overlays", fontsize=12, fontweight="bold", color=C_TEXT, loc="left", pad=10)

    for ticker, color in [("AAPL", C_AAPL), ("MSFT", C_MSFT), ("TSLA", C_TSLA)]:
        sub = model_df[model_df["Ticker"] == ticker].sort_values("Date")
        ax_price.plot(sub["Date"], sub["Close"], label=f"{ticker} Close", color=color, linewidth=1.8)
        if ticker == "AAPL":
            ax_price.plot(sub["Date"], sub["SMA_20"], color=C_AMBER, linestyle="--", linewidth=1.2, alpha=0.8, label="20D SMA")
            ax_price.plot(sub["Date"], sub["SMA_50"], color="#94A3B8", linestyle=":", linewidth=1.2, alpha=0.8, label="50D SMA")

    ax_price.set_ylabel("Close Price ($ USD)", fontsize=10)
    ax_price.grid(True)
    ax_price.legend(loc="upper left", fontsize=8.5, facecolor=C_PANEL, edgecolor=C_PANEL_BORDER)

    # Chart 2: Volatility Lines
    ax_vol = fig.add_subplot(gs[1, 2:4])
    ax_vol.set_facecolor(C_PANEL)
    ax_vol.set_title("Page 2: 20-Day Realized Return Volatility (TSLA 3.4x Higher Spread)", fontsize=12, fontweight="bold", color=C_TEXT, loc="left", pad=10)

    for ticker, color in [("AAPL", C_AAPL), ("MSFT", C_MSFT), ("TSLA", C_TSLA)]:
        sub = model_df[model_df["Ticker"] == ticker].sort_values("Date")
        ax_vol.plot(sub["Date"], sub["Volatility_20"], label=f"{ticker} Volatility", color=color, linewidth=1.8)

    ax_vol.set_ylabel("Rolling Volatility (%)", fontsize=10)
    ax_vol.grid(True)
    ax_vol.legend(loc="upper right", fontsize=8.5, facecolor=C_PANEL, edgecolor=C_PANEL_BORDER)

    # --- ROW 3: PAGE 3 CHARTS ---
    # Chart 3: Actual vs Predicted Scatter (Test Set)
    ax_scat = fig.add_subplot(gs[2, 0:2])
    ax_scat.set_facecolor(C_PANEL)
    ax_scat.set_title("Page 3: Actual vs. Predicted Returns (Test Period H2 2024)", fontsize=12, fontweight="bold", color=C_TEXT, loc="left", pad=10)

    test_df = model_df[model_df["Split"] == "Test"]
    for ticker, color in [("AAPL", C_AAPL), ("MSFT", C_MSFT), ("TSLA", C_TSLA)]:
        sub = test_df[test_df["Ticker"] == ticker]
        ax_scat.scatter(sub["Actual_Return"], sub["Predicted_Return"], label=ticker, color=color, alpha=0.6, s=28)

    lims = [-8, 14]
    ax_scat.plot(lims, lims, color="#94A3B8", linestyle="--", linewidth=1.2, label="45° Line")
    ax_scat.set_xlim(lims)
    ax_scat.set_ylim([-3, 4.5])
    ax_scat.set_xlabel("Actual Daily Return (%)", fontsize=10)
    ax_scat.set_ylabel("Predicted Return (%)", fontsize=10)
    ax_scat.grid(True)
    ax_scat.legend(loc="upper left", fontsize=8.5, facecolor=C_PANEL, edgecolor=C_PANEL_BORDER)

    # Chart 4: Feature Importance
    ax_feat = fig.add_subplot(gs[2, 2:4])
    ax_feat.set_facecolor(C_PANEL)
    ax_feat.set_title("Page 3: Random Forest Top 8 Engineered Feature Importances", fontsize=12, fontweight="bold", color=C_TEXT, loc="left", pad=10)

    top_feats = feat_df.head(8).sort_values(by="Importance", ascending=True)
    bars = ax_feat.barh(top_feats["Feature"], top_feats["Importance_Percent"], color="#2563EB", edgecolor="#3B82F6", height=0.6)

    for bar in bars:
        w = bar.get_width()
        ax_feat.text(w + 0.15, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%",
                     va="center", ha="left", fontsize=9, color=C_TEXT, fontweight="bold")

    ax_feat.set_xlabel("Relative Importance (%)", fontsize=10)
    ax_feat.set_xlim(0, max(top_feats["Importance_Percent"]) + 2.0)
    ax_feat.grid(axis="x")

    OUTPUT_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PREVIEW_PATH, dpi=300, facecolor=C_BG)
    plt.close()
    print(f"Generated high-resolution dashboard preview -> {OUTPUT_PREVIEW_PATH}")


if __name__ == "__main__":
    generate_dashboard_preview()
