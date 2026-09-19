"""
Fractional Differentiation Tests
Validates that FFD produces stationary series at the chosen d value
while preserving maximum memory from the original series.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from src.features.fracdiff import (
    frac_diff_ffd,
    frac_diff_ffd_vectorized,
    find_optimal_d,
    get_weights_ffd,
)


class TestGetWeightsFFD:
    """Tests for the FFD weight computation."""

    def test_first_weight_is_one(self):
        """The newest weight (last element) should be 1.0."""
        for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
            w = get_weights_ffd(d)
            assert w[-1] == 1.0, f"d={d}: last weight should be 1.0, got {w[-1]}"

    def test_weights_decay(self):
        """Absolute weights should generally decrease for d < 1."""
        w = get_weights_ffd(0.5)
        # The weights (newest to oldest) should decay in absolute value
        abs_w = np.abs(w[::-1])  # reverse to newest-first
        for i in range(1, min(5, len(abs_w))):
            assert abs_w[i] <= abs_w[i - 1] + 1e-10, (
                f"Weight {i} ({abs_w[i]}) > weight {i-1} ({abs_w[i-1]})"
            )

    def test_higher_d_produces_multiple_weights(self):
        """Any d > 0 should produce at least 2 weights (non-trivial differencing)."""
        for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
            w = get_weights_ffd(d)
            assert len(w) >= 2, (
                f"d={d} produced only {len(w)} weight(s), expected >= 2"
            )

    def test_d_zero_returns_identity(self):
        """d=0 should produce a single weight [1.0] (no differencing)."""
        w = get_weights_ffd(0.0)
        assert len(w) == 1
        assert w[0] == 1.0

    def test_d_one_approximates_first_difference(self):
        """d=1 should produce weights [−1, 1] (first difference)."""
        w = get_weights_ffd(1.0, thresh=1e-10)
        assert len(w) == 2
        np.testing.assert_allclose(w, [-1.0, 1.0])


class TestFracDiffFFD:
    """Tests for FFD application to series."""

    @pytest.fixture
    def random_walk(self):
        """Generate a random walk (non-stationary) series."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(np.log(prices))

    def test_output_length_matches_input(self, random_walk):
        """Output series should have the same length as input."""
        result = frac_diff_ffd(random_walk, d=0.5)
        assert len(result) == len(random_walk)

    def test_vectorized_matches_loop(self, random_walk):
        """Vectorized and loop implementations should produce identical results."""
        d = 0.4
        result_loop = frac_diff_ffd(random_walk, d=d)
        result_vec = frac_diff_ffd_vectorized(random_walk, d=d)

        # Compare non-NaN values
        valid = result_loop.notna() & result_vec.notna()
        np.testing.assert_allclose(
            result_loop[valid].values,
            result_vec[valid].values,
            rtol=1e-10,
            err_msg="Loop and vectorized FFD produce different results",
        )

    def test_stationarity_at_chosen_d(self):
        """Fractionally differenced series should be stationary (ADF p < 0.05)."""
        # Use a longer series and lower d to ensure enough non-NaN output
        np.random.seed(42)
        returns = np.random.randn(2000) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))
        long_walk = pd.Series(np.log(prices))

        result = frac_diff_ffd_vectorized(long_walk, d=0.3)
        clean = result.dropna()

        assert len(clean) > 50, f"Not enough clean values for ADF test (got {len(clean)})"

        adf_stat, p_value, *_ = adfuller(clean, maxlag=1, regression="c", autolag=None)
        assert p_value < 0.05, (
            f"ADF p-value {p_value:.4f} >= 0.05 at d=0.3 — series not stationary"
        )

    def test_original_series_is_nonstationary(self, random_walk):
        """Confirm the original log-price series is non-stationary."""
        adf_stat, p_value, *_ = adfuller(random_walk, maxlag=1, regression="c", autolag=None)
        assert p_value > 0.05, (
            f"Original series is already stationary (p={p_value:.4f}). "
            f"Test setup issue — expected non-stationary random walk."
        )


class TestFindOptimalD:
    """Tests for the optimal d grid search."""

    @pytest.fixture
    def random_walk(self):
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(np.log(prices))

    def test_optimal_d_achieves_stationarity(self, random_walk):
        """The chosen optimal d should produce a stationary series."""
        d_opt, p_opt, adf_results = find_optimal_d(random_walk)

        if d_opt < 1.0:
            assert p_opt < 0.05, (
                f"Optimal d={d_opt} has ADF p={p_opt:.4f} >= 0.05"
            )

    def test_optimal_d_is_minimum(self, random_walk):
        """Optimal d should be the smallest d that achieves stationarity."""
        d_opt, _, adf_results = find_optimal_d(random_walk)

        for d, p in sorted(adf_results.items()):
            if d < d_opt and p < 0.05:
                pytest.fail(
                    f"d={d} also achieves stationarity (p={p:.4f}) "
                    f"but optimal was d={d_opt}"
                )

    def test_returns_all_adf_results(self, random_walk):
        """Should return ADF p-values for all tested d values."""
        d_range = [0.1, 0.2, 0.3, 0.4, 0.5]
        _, _, adf_results = find_optimal_d(random_walk, d_range=d_range)

        assert len(adf_results) == len(d_range)
        for d in d_range:
            assert d in adf_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
