"""
Purged K-Fold Cross-Validation with Embargo
Prevents information leakage from overlapping label windows in time-series
financial data.

Standard TimeSeriesSplit still leaks: when labels span h future days, a training
sample near a test fold's boundary can overlap the test label's window.
This implementation adds:
  - Purging: removes training samples whose label window overlaps the test set
  - Embargo: removes an additional buffer after the test set to guard against
    serial correlation

This is sklearn-cv-compatible — pass directly to GridSearchCV(cv=...).

Reference: Marcos López de Prado, "Advances in Financial Machine Learning" (2018),
           Chapter 7.
"""

from typing import Generator, Optional, Tuple

import numpy as np


class PurgedKFoldEmbargo:
    """
    Purged K-Fold cross-validator with embargo period.

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    label_horizon : int
        Number of forward days in the label window (default 5 for weekly
        rebalance).  Training samples within `label_horizon` days before
        a test fold's start are purged.
    embargo_pct : float
        Fraction of total samples to embargo after each test fold's end
        (default 0.01 ≈ 1% of the dataset).

    Notes
    -----
    Unlike standard KFold, this splitter:
    1. Assigns contiguous blocks to each fold (respects temporal ordering).
    2. Purges training samples within `h` rows before each test fold.
    3. Embargoes training samples within `embargo` rows after each test fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        label_horizon: int = 5,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.h = label_horizon
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data — only shape is used.
        y : ignored
        groups : ignored

        Yields
        ------
        train_idx : np.ndarray
            Indices of training samples (purged + embargoed).
        test_idx : np.ndarray
            Indices of test samples.
        """
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        embargo = int(n * self.embargo_pct)

        # Contiguous fold boundaries
        fold_bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            test_start = int(fold_bounds[i])
            test_end = int(fold_bounds[i + 1])

            test_idx = np.arange(test_start, test_end)

            # Purge: remove training samples within h rows before test start
            purge_start = max(0, test_start - self.h)

            # Embargo: remove training samples within embargo rows after test end
            embargo_end = min(n, test_end + embargo)

            # Training set: everything outside [purge_start, embargo_end)
            train_idx = np.array(
                [j for j in range(n) if j < purge_start or j >= embargo_end]
            )

            yield train_idx, test_idx

    def __repr__(self) -> str:
        return (
            f"PurgedKFoldEmbargo(n_splits={self.n_splits}, "
            f"label_horizon={self.h}, embargo_pct={self.embargo_pct})"
        )
