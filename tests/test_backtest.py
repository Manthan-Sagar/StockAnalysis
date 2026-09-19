"""
Backtest Engine Tests
Uses a synthetic 2-asset toy case with known returns and costs to verify
the backtest engine produces hand-computable expected results.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    compute_performance_metrics,
    compute_portfolio_returns,
)


class TestComputePortfolioReturns:
    """Tests for the vectorized portfolio return computation."""

    @pytest.fixture
    def synthetic_portfolio(self):
        """
        Synthetic 2-asset portfolio over 5 days with known returns.

        Day 1: Rebalance — Asset A weight +0.6, Asset B weight -0.6
        Day 2-5: Hold (same weights, no turnover after day 1)

        Asset A daily returns: [0.01, 0.02, -0.01, 0.005, 0.01]
        Asset B daily returns: [-0.005, 0.01, 0.02, -0.01, 0.005]
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        records = []

        a_returns = [0.01, 0.02, -0.01, 0.005, 0.01]
        b_returns = [-0.005, 0.01, 0.02, -0.01, 0.005]

        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "ticker": "A",
                "weight": 0.6,
                "daily_return": a_returns[i],
                "rebalance_date": dates[0],
                "leg": "long",
            })
            records.append({
                "date": date,
                "ticker": "B",
                "weight": -0.6,
                "daily_return": b_returns[i],
                "rebalance_date": dates[0],
                "leg": "short",
            })

        return pd.DataFrame(records)

    def test_gross_return_computation(self, synthetic_portfolio):
        """Verify gross return = weighted sum of daily returns."""
        result = compute_portfolio_returns(synthetic_portfolio, total_cost_bps=0)

        # Day 1: 0.6 * 0.01 + (-0.6) * (-0.005) = 0.006 + 0.003 = 0.009
        expected_day1 = 0.6 * 0.01 + (-0.6) * (-0.005)
        np.testing.assert_almost_equal(
            result["gross_return"].iloc[0], expected_day1, decimal=10,
            err_msg=f"Day 1 gross return: expected {expected_day1}"
        )

        # Day 2: 0.6 * 0.02 + (-0.6) * 0.01 = 0.012 - 0.006 = 0.006
        expected_day2 = 0.6 * 0.02 + (-0.6) * 0.01
        np.testing.assert_almost_equal(
            result["gross_return"].iloc[1], expected_day2, decimal=10,
            err_msg=f"Day 2 gross return: expected {expected_day2}"
        )

    def test_turnover_on_first_day(self, synthetic_portfolio):
        """First day turnover should equal sum of |initial weights|."""
        result = compute_portfolio_returns(synthetic_portfolio, total_cost_bps=8)

        # Day 1: initial weights from zero → |0.6| + |-0.6| = 1.2
        expected_turnover = abs(0.6) + abs(-0.6)
        np.testing.assert_almost_equal(
            result["turnover"].iloc[0], expected_turnover, decimal=10
        )

    def test_zero_turnover_on_hold_days(self, synthetic_portfolio):
        """Turnover should be zero on non-rebalance days (weights unchanged)."""
        result = compute_portfolio_returns(synthetic_portfolio, total_cost_bps=8)

        # Days 2-5 should have zero turnover (same weights)
        for i in range(1, 5):
            np.testing.assert_almost_equal(
                result["turnover"].iloc[i], 0.0, decimal=10,
                err_msg=f"Day {i+1} should have zero turnover"
            )

    def test_cost_deducted_correctly(self, synthetic_portfolio):
        """Net return should equal gross return minus turnover × cost rate."""
        cost_bps = 10.0
        result = compute_portfolio_returns(synthetic_portfolio, total_cost_bps=cost_bps)

        for _, row in result.iterrows():
            expected_cost = row["turnover"] * cost_bps / 10_000
            expected_net = row["gross_return"] - expected_cost

            np.testing.assert_almost_equal(
                row["cost"], expected_cost, decimal=10
            )
            np.testing.assert_almost_equal(
                row["net_return"], expected_net, decimal=10
            )

    def test_zero_cost_means_net_equals_gross(self, synthetic_portfolio):
        """With zero costs, net return should equal gross return."""
        result = compute_portfolio_returns(synthetic_portfolio, total_cost_bps=0)

        np.testing.assert_array_almost_equal(
            result["gross_return"].values,
            result["net_return"].values,
            decimal=10,
        )

    def test_cumulative_return_consistency(self, synthetic_portfolio):
        """Cumulative return should equal product of (1 + daily_return)."""
        result = compute_portfolio_returns(synthetic_portfolio, total_cost_bps=0)

        expected_cum = (1 + result["net_return"]).cumprod()
        np.testing.assert_array_almost_equal(
            result["cum_net"].values,
            expected_cum.values,
            decimal=10,
        )


class TestPerformanceMetrics:
    """Tests for performance metric computations."""

    @pytest.fixture
    def constant_positive_returns(self):
        """Create a DataFrame with constant positive daily returns."""
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        daily_ret = 0.001  # 10bps per day

        return pd.DataFrame({
            "date": dates,
            "net_return": daily_ret,
            "gross_return": daily_ret,
            "turnover": 0.0,
        })

    def test_sharpe_positive_for_positive_returns(self, constant_positive_returns):
        """Sharpe ratio should be positive for consistently positive returns."""
        metrics = compute_performance_metrics(constant_positive_returns)
        assert metrics["sharpe_ratio"] > 0

    def test_max_drawdown_zero_for_monotonic(self, constant_positive_returns):
        """MDD should be 0 for monotonically increasing returns."""
        metrics = compute_performance_metrics(constant_positive_returns)
        assert metrics["max_drawdown"] == 0.0

    def test_sortino_higher_than_sharpe_for_positive_skew(self):
        """Sortino should be >= Sharpe when returns are positively skewed."""
        np.random.seed(42)
        # Create positively skewed returns
        returns = np.abs(np.random.randn(252)) * 0.001
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=252, freq="B"),
            "net_return": returns,
        })

        metrics = compute_performance_metrics(df)
        assert metrics["sortino_ratio"] >= metrics["sharpe_ratio"]

    def test_max_drawdown_negative(self):
        """MDD should be negative (representing a loss)."""
        # Create returns with a drawdown
        returns = [0.01] * 50 + [-0.05] * 10 + [0.01] * 50
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(returns), freq="B"),
            "net_return": returns,
        })

        metrics = compute_performance_metrics(df)
        assert metrics["max_drawdown"] < 0

    def test_empty_returns(self):
        """Should handle empty/short return series gracefully."""
        df = pd.DataFrame({"net_return": [0.01]})
        metrics = compute_performance_metrics(df)
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
