"""
Model Training Module
LightGBM regression model for cross-sectional residual return prediction.

Uses PurgedKFoldEmbargo for leakage-safe hyperparameter search.
No ticker-specific target encoding — sector is the only categorical feature,
and it's a legitimate cross-sectional characteristic.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from src.features.volatility import FEATURE_COLUMNS
from src.validation.purged_cv import PurgedKFoldEmbargo


# Configuration
MODELS_DIR = Path("models")
TARGET_COL = "fwd_residual_return"


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_training_data(
    panel: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    exclude_tickers: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare train/test splits from the feature panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel with features and target.
    feature_cols : list of str, optional
        Feature column names (default: FEATURE_COLUMNS).
    target_col : str
        Target column name.
    train_end : str
        Last date of training period (inclusive).
    test_start : str
        First date of test period (inclusive).
    exclude_tickers : list of str, optional
        Tickers to exclude (e.g., benchmarks).

    Returns
    -------
    train_df : pd.DataFrame
        Training data (features + target + Date + Ticker).
    test_df : pd.DataFrame
        Test data.
    feature_names : list of str
        Columns used as features.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    df = panel.copy()

    # Exclude benchmarks
    if exclude_tickers:
        df = df[~df["Ticker"].isin(exclude_tickers)].copy()

    # Only keep rows with valid target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Only keep feature columns that actually exist in the panel
    available_features = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"[train] Warning: missing features: {missing}")

    # Drop rows where any feature is NaN
    df = df.dropna(subset=available_features).reset_index(drop=True)

    # Time-based split
    df["Date"] = pd.to_datetime(df["Date"])
    train_mask = df["Date"] <= pd.to_datetime(train_end)
    test_mask = df["Date"] >= pd.to_datetime(test_start)

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    # Verify no temporal overlap
    if len(train_df) > 0 and len(test_df) > 0:
        assert train_df["Date"].max() < test_df["Date"].min(), (
            f"Temporal leakage! Train max: {train_df['Date'].max()}, "
            f"Test min: {test_df['Date'].min()}"
        )

    print(f"[train] Data split:")
    print(f"  Training: {len(train_df):,} rows "
          f"({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  Testing:  {len(test_df):,} rows "
          f"({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    print(f"  Features: {len(available_features)}")

    return train_df, test_df, available_features


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM training with purged CV
# ──────────────────────────────────────────────────────────────────────────────

def train_lightgbm(
    train_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = TARGET_COL,
    n_splits: int = 5,
    label_horizon: int = 5,
    embargo_pct: float = 0.01,
    n_iter: int = 30,
    random_state: int = 42,
) -> Tuple[lgb.LGBMRegressor, Dict[str, Any], float]:
    """
    Train LightGBM regressor with RandomizedSearchCV using PurgedKFoldEmbargo.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    feature_names : list of str
        Feature column names.
    target_col : str
        Target column name.
    n_splits : int
        Number of CV folds.
    label_horizon : int
        Label lookahead horizon (days).
    embargo_pct : float
        Embargo percentage for purged CV.
    n_iter : int
        Number of random search iterations.
    random_state : int
        Random seed.

    Returns
    -------
    best_model : lgb.LGBMRegressor
        Best fitted model.
    best_params : dict
        Best hyperparameters.
    best_score : float
        Best CV score (negative RMSE).
    """
    X_train = train_df[feature_names].values
    y_train = train_df[target_col].values

    # Purged CV splitter
    cv = PurgedKFoldEmbargo(
        n_splits=n_splits,
        label_horizon=label_horizon,
        embargo_pct=embargo_pct,
    )

    # Hyperparameter search space
    param_distributions = {
        "num_leaves": [15, 31, 63],
        "max_depth": [4, 6, 8, -1],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 500, 1000],
        "feature_fraction": [0.7, 0.8, 0.9],
        "bagging_fraction": [0.7, 0.8, 0.9],
        "bagging_freq": [1],
        "min_child_samples": [20, 50, 100],
    }

    base_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
        categorical_feature=[feature_names.index("sector_code")]
        if "sector_code" in feature_names else [],
    )

    print(f"\n[train] Starting RandomizedSearchCV ({n_iter} iterations, "
          f"{n_splits} purged folds)...")
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = -search.best_score_  # Convert neg RMSE to positive

    print(f"\n[train] Search completed in {elapsed:.1f}s")
    print(f"  Best CV RMSE: {best_score:.6f}")
    print(f"  Best params: {best_params}")

    return best_model, best_params, best_score


# ──────────────────────────────────────────────────────────────────────────────
# Model persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_model(
    model: lgb.LGBMRegressor,
    best_params: Dict[str, Any],
    best_cv_rmse: float,
    feature_names: List[str],
    output_dir: Path = MODELS_DIR,
) -> None:
    """
    Save trained model and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "lightgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"[train] Saved model -> {model_path}")

    # Save metadata
    metadata = {
        "model_type": "LGBMRegressor",
        "best_params": {k: int(v) if isinstance(v, (np.integer,)) else v
                       for k, v in best_params.items()},
        "best_cv_rmse": float(best_cv_rmse),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "target": TARGET_COL,
    }

    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] Saved metadata -> {meta_path}")

    # Feature importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "importance_pct": 100 * importances / importances.sum(),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    imp_path = output_dir / "feature_importances.csv"
    feat_imp.to_csv(imp_path, index=False)
    print(f"[train] Saved feature importances -> {imp_path}")
    print(f"\n  Top 10 features:")
    for _, row in feat_imp.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance_pct']:6.2f}%")


def load_model(
    model_dir: Path = MODELS_DIR,
) -> Tuple[lgb.LGBMRegressor, Dict[str, Any], List[str]]:
    """
    Load saved model, metadata, and feature names.
    """
    model = joblib.load(model_dir / "lightgbm_model.pkl")
    with open(model_dir / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, metadata, metadata["feature_names"]


def run_training_pipeline(
    panel: pd.DataFrame,
    train_end: str = "2022-12-31",
    test_start: str = "2023-01-01",
    n_iter: int = 30,
    label_horizon: int = 5,
) -> Tuple[lgb.LGBMRegressor, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    End-to-end training pipeline.

    Returns
    -------
    model : LGBMRegressor
        Trained model.
    train_df, test_df : pd.DataFrame
        Train and test splits.
    feature_names : list of str
        Features used.
    """
    train_df, test_df, feature_names = prepare_training_data(
        panel,
        train_end=train_end,
        test_start=test_start,
        exclude_tickers=["SPY", "QQQ"],
    )

    model, best_params, best_cv_rmse = train_lightgbm(
        train_df,
        feature_names,
        n_iter=n_iter,
        label_horizon=label_horizon,
    )

    save_model(model, best_params, best_cv_rmse, feature_names)

    return model, train_df, test_df, feature_names
