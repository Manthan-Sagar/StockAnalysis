"""
Pipeline Orchestration Module
Executes the end-to-end stock return forecasting data science workflow:
1. Data acquisition (data_fetch.py)
2. Feature engineering & data hygiene (feature_engineering.py)
3. Time-aware train/test split & model tuning (train_model.py)
4. Comprehensive empirical evaluation & visual figures (evaluate.py)
5. Power BI data exports (export_powerbi.py)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetch import fetch_all_and_combine
from src.feature_engineering import run_feature_pipeline
from src.train_model import run_training_pipeline
from src.evaluate import run_evaluation_pipeline
from src.export_powerbi import export_powerbi_datasets


def main():
    print("=" * 70)
    print("STOCK TREND ANALYSIS & RETURN FORECASTING PIPELINE")
    print("NASDAQ Large-Cap Equities: AAPL, MSFT, TSLA")
    print("=" * 70)

    start_total = time.time()

    # Step 1: Data Acquisition
    print("\n>>> STEP 1: Data Acquisition")
    t0 = time.time()
    fetch_all_and_combine()
    print(f"Completed Step 1 in {time.time() - t0:.2f}s")

    # Step 2: Feature Engineering
    print("\n>>> STEP 2: Feature Engineering & Preprocessing")
    t0 = time.time()
    run_feature_pipeline()
    print(f"Completed Step 2 in {time.time() - t0:.2f}s")

    # Step 3: Model Training & Hyperparameter Tuning
    print("\n>>> STEP 3: Model Training & TimeSeriesSplit Cross-Validation")
    t0 = time.time()
    run_training_pipeline(full_grid=False)
    print(f"Completed Step 3 in {time.time() - t0:.2f}s")

    # Step 4: Model Evaluation & Visual Diagnostics
    print("\n>>> STEP 4: Model Evaluation & Diagnostic Visuals")
    t0 = time.time()
    run_evaluation_pipeline()
    print(f"Completed Step 4 in {time.time() - t0:.2f}s")

    # Step 5: Power BI Data Export
    print("\n>>> STEP 5: Power BI Dataset Export")
    t0 = time.time()
    export_powerbi_datasets()
    print(f"Completed Step 5 in {time.time() - t0:.2f}s")

    print("\n" + "=" * 70)
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {time.time() - start_total:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
