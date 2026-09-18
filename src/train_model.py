"""
Model Training & Hyperparameter Tuning Module
Executes chronological time-series train/test split, calculates baseline benchmark,
tunes RandomForestRegressor via TimeSeriesSplit GridSearchCV, and persists the model.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Paths & Constants
FEATURED_DATA_PATH: Path = Path("data/processed/featured_data.csv")
MODELS_DIR: Path = Path("models")
MODEL_OUTPUT_PATH: Path = MODELS_DIR / "random_forest_model.pkl"
METADATA_OUTPUT_PATH: Path = MODELS_DIR / "model_metadata.json"

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


def load_and_split_data(
    filepath: Path = FEATURED_DATA_PATH,
    train_end: str = TRAIN_END_DATE,
    test_start: str = TEST_START_DATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Loads featured dataset, sorts chronologically, and applies a time-aware train/test split.

    Parameters
    ----------
    filepath : Path
        Path to featured_data.csv.
    train_end : str
        Cutoff date for training set (inclusive).
    test_start : str
        Start date for testing set (inclusive).

    Returns
    -------
    train_df : pd.DataFrame
        Full training DataFrame.
    test_df : pd.DataFrame
        Full testing DataFrame.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target series.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.Series
        Testing target series.
    feature_names : List[str]
        List of feature column names.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist. Run feature_engineering.py first.")

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    # Sort strictly by Date to preserve temporal ordering
    df = df.sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    train_mask = df["Date"] <= train_end
    test_mask = df["Date"] >= test_start

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["Target_Next_Return"].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df["Target_Next_Return"].copy()

    # Sanity assertions
    assert not X_train.isnull().any().any(), "Error: NaNs detected in X_train!"
    assert not X_test.isnull().any().any(), "Error: NaNs detected in X_test!"
    assert not y_train.isnull().any(), "Error: NaNs detected in y_train!"
    assert not y_test.isnull().any(), "Error: NaNs detected in y_test!"
    assert train_df["Date"].max() < test_df["Date"].min(), (
        f"Data leakage warning: Train max date {train_df['Date'].max()} is not before Test min date {test_df['Date'].min()}"
    )
    assert list(X_train.columns) == list(X_test.columns), "Feature mismatch between train and test sets!"

    print(f"Data Split Summary:")
    print(f"  Training set: {len(train_df)} rows ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  Testing set:  {len(test_df)} rows ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    return train_df, test_df, X_train, y_train, X_test, y_test, feature_cols


def compute_baseline_metrics(y_test: pd.Series) -> Dict[str, float]:
    """
    Compute naive baseline metrics (predicting constant 0% return daily).

    Parameters
    ----------
    y_test : pd.Series
        Actual test set returns.

    Returns
    -------
    Dict[str, float]
        Baseline RMSE and MAE.
    """
    y_pred_baseline = np.zeros_like(y_test)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    baseline_mae = float(mean_absolute_error(y_test, y_pred_baseline))
    baseline_r2 = float(r2_score(y_test, y_pred_baseline))

    print(f"\n--- NAIVE BASELINE (0% Return Prediction) ---")
    print(f"  Baseline RMSE: {baseline_rmse:.4f}%")
    print(f"  Baseline MAE:  {baseline_mae:.4f}%")
    print(f"  Baseline R²:   {baseline_r2:.4f}")

    return {
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2
    }


def train_and_tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    full_grid: bool = False
) -> Tuple[RandomForestRegressor, Dict[str, Any], float]:
    """
    Tune RandomForestRegressor with TimeSeriesSplit cross-validation,
    and refit best estimator on full training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    full_grid : bool
        Whether to run the exhaustive grid (360 combinations) or the high-efficiency grid.

    Returns
    -------
    best_estimator : RandomForestRegressor
        Fitted best model.
    best_params : Dict[str, Any]
        Optimal hyperparameters.
    best_cv_rmse : float
        Best TimeSeriesSplit CV RMSE.
    """
    tscv = TimeSeriesSplit(n_splits=5)

    if full_grid:
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
    else:
        # High-efficiency grid spanning all dimensions with fast execution
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

    total_candidates = np.prod([len(v) for v in param_grid.values()])
    print(f"\nInitiating TimeSeriesSplit GridSearchCV across {total_candidates} candidates (5 folds = {total_candidates * 5} fits)...")

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_estimator: RandomForestRegressor = grid_search.best_estimator_
    best_params: Dict[str, Any] = grid_search.best_params_
    best_cv_rmse = float(-grid_search.best_score_)

    print(f"\n--- GRIDSEARCHCV RESULTS ---")
    print(f"  Best CV RMSE: {best_cv_rmse:.4f}%")
    print(f"  Best Parameters: {best_params}")

    return best_estimator, best_params, best_cv_rmse


def save_model_and_artifacts(
    model: RandomForestRegressor,
    best_params: Dict[str, Any],
    best_cv_rmse: float,
    baseline_metrics: Dict[str, float],
    feature_names: List[str]
) -> None:
    """
    Persist trained model and metadata to disk.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Saved trained Random Forest model -> {MODEL_OUTPUT_PATH}")

    metadata = {
        "model_type": "RandomForestRegressor",
        "random_state": 42,
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "baseline_metrics": baseline_metrics,
        "feature_names": feature_names,
        "train_end_date": TRAIN_END_DATE,
        "test_start_date": TEST_START_DATE
    }
    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata -> {METADATA_OUTPUT_PATH}")


def run_training_pipeline(full_grid: bool = False) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Main training execution function.
    """
    train_df, test_df, X_train, y_train, X_test, y_test, feature_names = load_and_split_data()
    baseline_metrics = compute_baseline_metrics(y_test)
    best_model, best_params, best_cv_rmse = train_and_tune_model(X_train, y_train, full_grid=full_grid)
    save_model_and_artifacts(best_model, best_params, best_cv_rmse, baseline_metrics, feature_names)
    return best_model, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune Random Forest model.")
    parser.add_argument("--full-grid", action="store_true", help="Run exhaustive 360-candidate grid search.")
    args = parser.parse_args()
    run_training_pipeline(full_grid=args.full_grid)
