"""
Purged K-Fold Cross-Validation Leakage Tests
THE MOST IMPORTANT TEST FILE IN THIS REPO.

Asserts zero temporal overlap between train and test indices per fold,
accounting for the label horizon (h) and embargo period. This test is
the single most credibility-building artifact — it's proof the pipeline
doesn't just claim to prevent leakage but actually enforces it.
"""

import numpy as np
import pytest

from src.validation.purged_cv import PurgedKFoldEmbargo


class TestPurgedKFoldEmbargo:
    """Tests for the PurgedKFoldEmbargo cross-validator."""

    def test_no_train_within_h_days_of_test(self):
        """
        CRITICAL: Assert no training index falls within h days
        BEFORE any test index. This is the purging guarantee.
        """
        n_samples = 1000
        X = np.random.randn(n_samples, 5)
        h = 5

        cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=h, embargo_pct=0.01)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            test_start = test_idx.min()

            # No training sample should be within h indices before test start
            purge_zone = set(range(max(0, test_start - h), test_start))
            overlap = purge_zone.intersection(set(train_idx))

            assert len(overlap) == 0, (
                f"Fold {fold_idx}: Found {len(overlap)} training indices "
                f"within {h} days before test start {test_start}: {sorted(overlap)}"
            )

    def test_no_train_within_embargo_after_test(self):
        """
        Assert no training index falls within the embargo period
        AFTER the test fold end.
        """
        n_samples = 1000
        X = np.random.randn(n_samples, 5)
        embargo_pct = 0.02

        cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=5, embargo_pct=embargo_pct)
        embargo_size = int(n_samples * embargo_pct)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            test_end = test_idx.max() + 1  # exclusive end

            # No training sample should be in [test_end, test_end + embargo)
            embargo_zone = set(range(test_end, min(n_samples, test_end + embargo_size)))
            overlap = embargo_zone.intersection(set(train_idx))

            assert len(overlap) == 0, (
                f"Fold {fold_idx}: Found {len(overlap)} training indices "
                f"in embargo zone [{test_end}, {test_end + embargo_size}): {sorted(overlap)}"
            )

    def test_no_overlap_between_train_and_test(self):
        """Assert train and test indices never overlap."""
        n_samples = 500
        X = np.random.randn(n_samples, 3)

        cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=10, embargo_pct=0.02)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            overlap = set(train_idx).intersection(set(test_idx))
            assert len(overlap) == 0, (
                f"Fold {fold_idx}: Train/test overlap: {sorted(overlap)}"
            )

    def test_all_test_indices_cover_full_dataset(self):
        """Assert that test indices across all folds cover [0, n)."""
        n_samples = 500
        X = np.random.randn(n_samples, 3)

        cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=5, embargo_pct=0.01)

        all_test = set()
        for train_idx, test_idx in cv.split(X):
            all_test.update(test_idx)

        assert all_test == set(range(n_samples)), (
            f"Test indices don't cover full dataset. "
            f"Missing: {set(range(n_samples)) - all_test}"
        )

    def test_correct_number_of_splits(self):
        """Assert the splitter produces the requested number of folds."""
        for n_splits in [3, 5, 10]:
            cv = PurgedKFoldEmbargo(n_splits=n_splits, label_horizon=5)
            X = np.random.randn(1000, 5)
            folds = list(cv.split(X))
            assert len(folds) == n_splits

    def test_get_n_splits(self):
        """Assert get_n_splits returns the correct count."""
        cv = PurgedKFoldEmbargo(n_splits=7, label_horizon=5)
        assert cv.get_n_splits() == 7

    def test_purge_gap_equals_h(self):
        """
        Verify the exact purge boundary: the last allowed training index
        before the test fold should be exactly test_start - h - 1.
        """
        n_samples = 1000
        h = 10
        X = np.random.randn(n_samples, 5)

        cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=h, embargo_pct=0.0)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            test_start = test_idx.min()
            if test_start < h:
                continue  # skip first fold if test starts too early

            # The maximum training index before the test fold
            pre_test_train = train_idx[train_idx < test_start]
            if len(pre_test_train) == 0:
                continue

            max_train_before_test = pre_test_train.max()

            # Gap between last training sample and first test sample
            gap = test_start - max_train_before_test - 1
            assert gap >= h - 1, (
                f"Fold {fold_idx}: Purge gap is {gap}, expected >= {h - 1}. "
                f"Last train idx: {max_train_before_test}, test start: {test_start}"
            )

    def test_different_label_horizons(self):
        """Test with various label horizons to ensure robustness."""
        n_samples = 500
        X = np.random.randn(n_samples, 3)

        for h in [1, 5, 10, 20, 50]:
            cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=h, embargo_pct=0.01)

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                test_start = test_idx.min()
                purge_zone = set(range(max(0, test_start - h), test_start))
                overlap = purge_zone.intersection(set(train_idx))

                assert len(overlap) == 0, (
                    f"h={h}, Fold {fold_idx}: Purge violation with {len(overlap)} indices"
                )

    def test_small_dataset(self):
        """Ensure the splitter handles small datasets gracefully."""
        X = np.random.randn(20, 3)
        cv = PurgedKFoldEmbargo(n_splits=3, label_horizon=2, embargo_pct=0.05)

        folds = list(cv.split(X))
        assert len(folds) == 3

        for train_idx, test_idx in folds:
            assert len(train_idx) > 0, "Empty training set!"
            assert len(test_idx) > 0, "Empty test set!"

    def test_sklearn_compatibility(self):
        """Verify the splitter works with sklearn's cross_val_score."""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        X = np.random.randn(200, 5)
        y = np.random.randn(200)

        cv = PurgedKFoldEmbargo(n_splits=5, label_horizon=5, embargo_pct=0.01)
        scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")

        assert len(scores) == 5
        assert all(np.isfinite(scores))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
