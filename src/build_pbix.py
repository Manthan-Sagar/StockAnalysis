"""
Power BI PBIX File Builder
Constructs the dashboard/stock_dashboard.pbix file container
with complete 3-page layout schemas, visual metadata, and data references.
"""

import io
import json
import zipfile
from pathlib import Path

DASHBOARD_DIR = Path("dashboard")
PBIX_PATH = DASHBOARD_DIR / "stock_dashboard.pbix"


def create_pbix_package():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Version stream
    version_content = "1.28\n"

    # 2. [Content_Types].xml
    content_types_xml = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="json" ContentType="application/json" />
  <Default Extension="xml" ContentType="application/xml" />
  <Default Extension="txt" ContentType="text/plain" />
  <Override PartName="/Report/Layout" ContentType="application/json" />
  <Override PartName="/Version" ContentType="text/plain" />
  <Override PartName="/Settings" ContentType="application/json" />
</Types>"""

    # 3. Settings
    settings_json = {
        "version": "2.0",
        "settings": {
            "useNewFilterPaneExperience": True,
            "allowChangeFilterTypes": True,
            "useStylableVisualContainerHeader": True,
            "queryLimitOption": 6
        }
    }

    # 4. Report Layout JSON (UTF-16LE encoded in native PBIX)
    layout_json = {
        "id": 0,
        "theme": "ModernDark",
        "resourcePackages": [],
        "sections": [
            {
                "id": 0,
                "name": "Section1_PriceTrends",
                "displayName": "1 — Price Trends",
                "filters": "[]",
                "ordinal": 0,
                "visualContainers": [
                    {
                        "x": 20,
                        "y": 20,
                        "z": 1000,
                        "width": 1240,
                        "height": 80,
                        "config": json.dumps({
                            "name": "HeaderCard",
                            "singleVisual": {
                                "visualType": "textbox",
                                "title": "NASDAQ Large-Cap Equity Trends (AAPL, MSFT, TSLA)",
                                "subtitle": "Daily OHLCV, 20-Day SMA, and 50-Day SMA Overlays"
                            }
                        })
                    },
                    {
                        "x": 20,
                        "y": 120,
                        "z": 2000,
                        "width": 1240,
                        "height": 560,
                        "config": json.dumps({
                            "name": "PriceTrendLineChart",
                            "singleVisual": {
                                "visualType": "lineChart",
                                "projections": {
                                    "Category": [{"queryRef": "Model_Data.Date"}],
                                    "Y": [
                                        {"queryRef": "Model_Data.Close"},
                                        {"queryRef": "Model_Data.SMA_20"},
                                        {"queryRef": "Model_Data.SMA_50"}
                                    ],
                                    "Series": [{"queryRef": "Model_Data.Ticker"}]
                                }
                            }
                        })
                    }
                ],
                "config": json.dumps({"pageSize": {"type": "Custom", "width": 1280, "height": 720}})
            },
            {
                "id": 1,
                "name": "Section2_VolatilityReturns",
                "displayName": "2 — Volatility & Comparative Returns",
                "filters": "[]",
                "ordinal": 1,
                "visualContainers": [
                    {
                        "x": 20,
                        "y": 20,
                        "z": 1000,
                        "width": 600,
                        "height": 660,
                        "config": json.dumps({
                            "name": "VolatilityLineChart",
                            "singleVisual": {
                                "visualType": "lineChart",
                                "projections": {
                                    "Category": [{"queryRef": "Model_Data.Date"}],
                                    "Y": [{"queryRef": "Model_Data.Volatility_20"}],
                                    "Series": [{"queryRef": "Model_Data.Ticker"}]
                                }
                            }
                        })
                    },
                    {
                        "x": 640,
                        "y": 20,
                        "z": 2000,
                        "width": 620,
                        "height": 660,
                        "config": json.dumps({
                            "name": "CumulativeReturnChart",
                            "singleVisual": {
                                "visualType": "lineChart",
                                "projections": {
                                    "Category": [{"queryRef": "Model_Data.Date"}],
                                    "Y": [{"queryRef": "_Measures.Cumulative Return %"}],
                                    "Series": [{"queryRef": "Model_Data.Ticker"}]
                                }
                            }
                        })
                    }
                ],
                "config": json.dumps({"pageSize": {"type": "Custom", "width": 1280, "height": 720}})
            },
            {
                "id": 2,
                "name": "Section3_ModelPerformance",
                "displayName": "3 — Model Performance",
                "filters": "[]",
                "ordinal": 2,
                "visualContainers": [
                    {
                        "x": 20,
                        "y": 20,
                        "z": 1000,
                        "width": 600,
                        "height": 380,
                        "config": json.dumps({
                            "name": "ScatterPlotActualVsPredicted",
                            "singleVisual": {
                                "visualType": "scatterChart",
                                "projections": {
                                    "X": [{"queryRef": "Model_Data.Actual_Return"}],
                                    "Y": [{"queryRef": "Model_Data.Predicted_Return"}],
                                    "Details": [{"queryRef": "Model_Data.Date"}],
                                    "Legend": [{"queryRef": "Model_Data.Ticker"}]
                                }
                            }
                        })
                    },
                    {
                        "x": 640,
                        "y": 20,
                        "z": 2000,
                        "width": 620,
                        "height": 380,
                        "config": json.dumps({
                            "name": "FeatureImportanceBarChart",
                            "singleVisual": {
                                "visualType": "barChart",
                                "projections": {
                                    "Category": [{"queryRef": "feature_importance.Feature"}],
                                    "Y": [{"queryRef": "feature_importance.Importance_Percent"}]
                                }
                            }
                        })
                    },
                    {
                        "x": 20,
                        "y": 420,
                        "z": 3000,
                        "width": 1240,
                        "height": 260,
                        "config": json.dumps({
                            "name": "MetricsSummaryTable",
                            "singleVisual": {
                                "visualType": "tableEx",
                                "projections": {
                                    "Values": [
                                        {"queryRef": "metrics_summary.Ticker"},
                                        {"queryRef": "metrics_summary.Samples"},
                                        {"queryRef": "metrics_summary.RMSE"},
                                        {"queryRef": "metrics_summary.MAE"},
                                        {"queryRef": "metrics_summary.R2"},
                                        {"queryRef": "metrics_summary.Baseline_RMSE"},
                                        {"queryRef": "metrics_summary.Baseline_MAE"}
                                    ]
                                }
                            }
                        })
                    }
                ],
                "config": json.dumps({"pageSize": {"type": "Custom", "width": 1280, "height": 720}})
            }
        ]
    }

    # Write into zip container
    with zipfile.ZipFile(PBIX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml.encode("utf-8"))
        zf.writestr("Version", version_content.encode("utf-8"))
        zf.writestr("Settings", json.dumps(settings_json, indent=2).encode("utf-8"))
        # Layout in PBIX files uses UTF-16LE encoding
        zf.writestr("Report/Layout", json.dumps(layout_json, indent=2).encode("utf-16le"))

    print(f"Generated Power BI dashboard file -> {PBIX_PATH} ({PBIX_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    create_pbix_package()
