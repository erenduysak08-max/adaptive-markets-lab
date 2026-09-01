import numpy as np
import pandas as pd

from adaptive_markets_lab import BacktestConfig, PairsConfig, TradingMode
from adaptive_markets_lab.data import generate_cointegrated_pair
from adaptive_markets_lab.pairs import pairs_target_state, run_pairs_backtest
from adaptive_markets_lab.research import run_pairs_research


def test_pair_state_enters_holds_and_exits() -> None:
    z_score = pd.Series([0.0, 2.2, 1.0, 0.4, -2.3, -0.3])

    state = pairs_target_state(z_score, entry_z=2.0, exit_z=0.5)

    assert state.tolist() == [0.0, -1.0, -1.0, 0.0, 1.0, 0.0]


def test_pair_positions_are_delayed_one_observation() -> None:
    result = run_pairs_backtest(
        generate_cointegrated_pair(400),
        PairsConfig(lookback=40, entry_z=1.5, exit_z=0.5),
    )

    expected_a = result.frame["target_weight_a"].shift(1).fillna(0.0)
    expected_b = result.frame["target_weight_b"].shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        result.frame["weight_a"], expected_a, check_names=False
    )
    pd.testing.assert_series_equal(
        result.frame["weight_b"], expected_b, check_names=False
    )


def test_long_short_pair_respects_gross_leverage() -> None:
    result = run_pairs_backtest(
        generate_cointegrated_pair(500),
        PairsConfig(lookback=40, entry_z=1.5, exit_z=0.5),
        BacktestConfig(
            trading_mode=TradingMode.LONG_SHORT,
            gross_leverage=2.0,
        ),
    )

    assert result.frame["gross_exposure"].max() <= 2.0 + 1e-12
    assert (result.frame[["weight_a", "weight_b"]] < 0).any().any()


def test_spot_pair_rotation_never_shorts_or_leverages() -> None:
    result = run_pairs_backtest(
        generate_cointegrated_pair(500),
        PairsConfig(lookback=40, entry_z=1.5, exit_z=0.5),
        BacktestConfig(trading_mode=TradingMode.SPOT_LONG_ONLY),
    )

    assert (result.frame[["weight_a", "weight_b"]] >= 0).all().all()
    assert result.frame["gross_exposure"].max() <= 1.0


def test_pair_cost_equals_two_leg_turnover_times_rate() -> None:
    result = run_pairs_backtest(
        generate_cointegrated_pair(400),
        PairsConfig(lookback=40, entry_z=1.5, exit_z=0.5),
        BacktestConfig(
            trading_mode=TradingMode.LONG_SHORT,
            transaction_cost_bps=10.0,
        ),
    )

    expected = result.frame["turnover"] * 0.001
    pd.testing.assert_series_equal(
        result.frame["transaction_cost"], expected, check_names=False
    )


def test_future_pair_prices_cannot_change_earlier_results() -> None:
    original = generate_cointegrated_pair(400)
    changed = original.copy()
    cutoff = original.index[300]
    changed.loc[cutoff:, changed.columns[0]] *= np.linspace(
        1.1, 1.5, len(changed.loc[cutoff:])
    )

    first = run_pairs_backtest(original, PairsConfig(lookback=40))
    second = run_pairs_backtest(changed, PairsConfig(lookback=40))

    pd.testing.assert_frame_equal(
        first.frame.loc[first.frame.index < cutoff],
        second.frame.loc[second.frame.index < cutoff],
    )


def test_pair_research_uses_same_window_for_benchmark() -> None:
    research = run_pairs_research(
        generate_cointegrated_pair(400),
        PairsConfig(lookback=40, entry_z=1.5, exit_z=0.5),
    )

    assert research.equity_curves.shape == (360, 2)
    assert list(research.comparison.index) == [
        "Long-only pair rotation",
        "50/50 buy and hold",
    ]
