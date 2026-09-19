"""
Portfolio Sizing Tests
Validates dollar-neutrality, gross exposure constraints, and inverse-volatility
weighting correctness.
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.sizing import (
    inverse_vol_weights,
    rank_and_assign_legs,
    verify_portfolio_weights,
)


class TestRankAndAssignLegs:
    """Tests for quintile-based leg assignment."""

    def test_top_bottom_quintile_assignment(self):
        """Top 20% should be long, bottom 20% should be short."""
        scores = pd.Series(range(100), index=[f"T{i}" for i in range(100)])
        legs = rank_and_assign_legs(scores, n_quantiles=5)

        n_long = (legs == 1).sum()
        n_short = (legs == -1).sum()
        n_neutral = (legs == 0).sum()

        assert n_long == 20, f"Expected 20 long positions, got {n_long}"
        assert n_short == 20, f"Expected 20 short positions, got {n_short}"
        assert n_neutral == 60, f"Expected 60 neutral positions, got {n_neutral}"

    def test_highest_scores_are_long(self):
        """Stocks with highest scores should be assigned to the long leg."""
        scores = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                          index=[f"T{i}" for i in range(10)])
        legs = rank_and_assign_legs(scores, n_quantiles=5)

        # Top 2 should be long
        assert legs["T8"] == 1 or legs["T9"] == 1
        # Bottom 2 should be short
        assert legs["T0"] == -1 or legs["T1"] == -1

    def test_too_few_stocks_returns_zeros(self):
        """With fewer stocks than quantiles, should return all zeros."""
        scores = pd.Series([1, 2, 3], index=["A", "B", "C"])
        legs = rank_and_assign_legs(scores, n_quantiles=5)
        assert (legs == 0).all()


class TestInverseVolWeights:
    """Tests for inverse-volatility (risk-parity) weighting."""

    def test_long_weights_sum_to_plus_one(self):
        """Long leg weights must sum to exactly +1.0."""
        vols = pd.Series([0.02, 0.03, 0.04, 0.01, 0.05],
                        index=["A", "B", "C", "D", "E"])
        legs = pd.Series([1, 1, 0, -1, -1],
                        index=["A", "B", "C", "D", "E"])

        weights = inverse_vol_weights(vols, legs)
        long_sum = weights[weights > 0].sum()

        np.testing.assert_almost_equal(
            long_sum, 1.0, decimal=10,
            err_msg=f"Long weights sum to {long_sum}, expected 1.0"
        )

    def test_short_weights_sum_to_minus_one(self):
        """Short leg weights must sum to exactly -1.0."""
        vols = pd.Series([0.02, 0.03, 0.04, 0.01, 0.05],
                        index=["A", "B", "C", "D", "E"])
        legs = pd.Series([1, 1, 0, -1, -1],
                        index=["A", "B", "C", "D", "E"])

        weights = inverse_vol_weights(vols, legs)
        short_sum = weights[weights < 0].sum()

        np.testing.assert_almost_equal(
            short_sum, -1.0, decimal=10,
            err_msg=f"Short weights sum to {short_sum}, expected -1.0"
        )

    def test_gross_exposure_equals_two(self):
        """Gross exposure (|long| + |short|) must equal 2.0."""
        vols = pd.Series([0.02, 0.03, 0.04, 0.01, 0.05],
                        index=["A", "B", "C", "D", "E"])
        legs = pd.Series([1, 1, 0, -1, -1],
                        index=["A", "B", "C", "D", "E"])

        weights = inverse_vol_weights(vols, legs)
        gross = weights.abs().sum()

        np.testing.assert_almost_equal(
            gross, 2.0, decimal=10,
            err_msg=f"Gross exposure is {gross}, expected 2.0"
        )

    def test_net_exposure_is_zero(self):
        """Dollar-neutral: net exposure must be zero."""
        vols = pd.Series([0.02, 0.03, 0.04, 0.01, 0.05],
                        index=["A", "B", "C", "D", "E"])
        legs = pd.Series([1, 1, 0, -1, -1],
                        index=["A", "B", "C", "D", "E"])

        weights = inverse_vol_weights(vols, legs)
        net = weights.sum()

        np.testing.assert_almost_equal(
            net, 0.0, decimal=10,
            err_msg=f"Net exposure is {net}, expected 0.0"
        )

    def test_lower_vol_gets_higher_weight(self):
        """Within a leg, lower-volatility stocks should get higher weight."""
        vols = pd.Series([0.01, 0.05], index=["LowVol", "HighVol"])
        legs = pd.Series([1, 1], index=["LowVol", "HighVol"])

        weights = inverse_vol_weights(vols, legs)

        assert weights["LowVol"] > weights["HighVol"], (
            f"LowVol weight ({weights['LowVol']:.4f}) should be > "
            f"HighVol weight ({weights['HighVol']:.4f})"
        )

    def test_neutral_positions_get_zero_weight(self):
        """Stocks assigned to neutral (leg=0) should have zero weight."""
        vols = pd.Series([0.02, 0.03, 0.04],
                        index=["A", "B", "C"])
        legs = pd.Series([1, 0, -1],
                        index=["A", "B", "C"])

        weights = inverse_vol_weights(vols, legs)
        assert weights["B"] == 0.0

    def test_equal_vols_produce_equal_weights(self):
        """Equal volatilities within a leg should produce equal weights."""
        vols = pd.Series([0.03, 0.03, 0.03],
                        index=["A", "B", "C"])
        legs = pd.Series([1, 1, 1],
                        index=["A", "B", "C"])

        weights = inverse_vol_weights(vols, legs)

        np.testing.assert_almost_equal(
            weights["A"], weights["B"], decimal=10
        )
        np.testing.assert_almost_equal(
            weights["B"], weights["C"], decimal=10
        )
        np.testing.assert_almost_equal(
            weights["A"], 1.0 / 3, decimal=10
        )

    def test_many_stocks(self):
        """Test with a realistic number of stocks (100)."""
        n = 100
        np.random.seed(42)
        tickers = [f"T{i:03d}" for i in range(n)]
        vols = pd.Series(np.random.uniform(0.01, 0.05, n), index=tickers)

        # Top 20 long, bottom 20 short
        scores = pd.Series(np.random.randn(n), index=tickers)
        legs = rank_and_assign_legs(scores, n_quantiles=5)
        weights = inverse_vol_weights(vols, legs)

        long_sum = weights[weights > 0].sum()
        short_sum = weights[weights < 0].sum()
        gross = weights.abs().sum()

        np.testing.assert_almost_equal(long_sum, 1.0, decimal=10)
        np.testing.assert_almost_equal(short_sum, -1.0, decimal=10)
        np.testing.assert_almost_equal(gross, 2.0, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
