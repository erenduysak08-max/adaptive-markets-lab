import unittest

import numpy as np
import pandas as pd

from adaptive_markets_lab import BacktestConfig, MomentumConfig, TradingMode, run_backtest


def prices(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


class BacktestTests(unittest.TestCase):
    def test_signal_is_delayed_one_period(self) -> None:
        result = run_backtest(prices([100, 110, 99]), MomentumConfig(half_life=1))

        self.assertEqual(result.frame["target_position"].iloc[1], 1.0)
        self.assertEqual(result.frame["position"].iloc[1], 0.0)
        self.assertEqual(result.frame["position"].iloc[2], 1.0)

    def test_spot_mode_never_shorts(self) -> None:
        result = run_backtest(
            prices([100, 90, 80, 70]),
            MomentumConfig(half_life=1),
            BacktestConfig(trading_mode=TradingMode.SPOT_LONG_ONLY),
        )

        self.assertTrue((result.frame["position"] >= 0).all())
        self.assertLessEqual(result.frame["position"].max(), 1.0)

    def test_long_short_mode_can_short_with_leverage(self) -> None:
        result = run_backtest(
            prices([100, 90, 80, 70]),
            MomentumConfig(half_life=1),
            BacktestConfig(trading_mode=TradingMode.LONG_SHORT, gross_leverage=2.0),
        )

        self.assertIn(-2.0, result.frame["position"].values)

    def test_transaction_cost_equals_turnover_times_rate(self) -> None:
        result = run_backtest(
            prices([100, 110, 99, 110]),
            MomentumConfig(half_life=1),
            BacktestConfig(transaction_cost_bps=10),
        )

        expected = result.frame["turnover"] * 0.001
        pd.testing.assert_series_equal(
            result.frame["transaction_cost"], expected.rename("transaction_cost")
        )

    def test_spot_mode_rejects_leverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "spot_long_only"):
            BacktestConfig(
                trading_mode=TradingMode.SPOT_LONG_ONLY,
                gross_leverage=1.5,
            )

    def test_input_is_sorted_and_metrics_are_returned(self) -> None:
        unordered = prices([100, 101, 102]).sort_index(ascending=False)
        result = run_backtest(unordered)

        self.assertTrue(result.frame.index.is_monotonic_increasing)
        self.assertTrue(np.isfinite(result.metrics["total_return"]))


if __name__ == "__main__":
    unittest.main()
