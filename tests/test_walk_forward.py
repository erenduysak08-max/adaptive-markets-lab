import numpy as np
import pandas as pd

from adaptive_markets_lab import (
    BacktestConfig,
    WalkForwardConfig,
    run_walk_forward,
)


def changing_prices(periods: int = 80) -> pd.Series:
    first = np.linspace(100.0, 140.0, periods // 2)
    second = np.linspace(140.0, 90.0, periods - periods // 2)
    return pd.Series(
        np.concatenate([first, second]),
        index=pd.bdate_range("2022-01-03", periods=periods),
    )


def walk_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        candidate_half_lives=(2.0, 5.0, 10.0),
        train_periods=20,
        test_periods=10,
    )


def test_every_reported_observation_is_out_of_sample() -> None:
    prices = changing_prices()
    result = run_walk_forward(prices, walk_config())

    assert len(result.frame) == 60
    assert result.frame.index[0] == prices.index[20]
    assert not result.frame.isna().any().any()
    assert (result.selections["train_end"] < result.selections["test_start"]).all()


def test_future_prices_cannot_change_earlier_results() -> None:
    original = changing_prices()
    changed_future = original.copy()
    cutoff = original.index[50]
    changed_future.loc[cutoff:] *= np.linspace(
        1.2, 2.0, len(changed_future.loc[cutoff:])
    )

    first = run_walk_forward(original, walk_config())
    second = run_walk_forward(changed_future, walk_config())

    pd.testing.assert_frame_equal(
        first.frame.loc[first.frame.index < cutoff],
        second.frame.loc[second.frame.index < cutoff],
    )
    first_choices = first.selections.loc[
        first.selections["test_start"] < cutoff, "selected_half_life"
    ].reset_index(drop=True)
    second_choices = second.selections.loc[
        second.selections["test_start"] < cutoff, "selected_half_life"
    ].reset_index(drop=True)
    pd.testing.assert_series_equal(first_choices, second_choices)


def test_transaction_costs_reduce_walk_forward_returns() -> None:
    prices = changing_prices()
    free = run_walk_forward(
        prices, walk_config(), BacktestConfig(transaction_cost_bps=0.0)
    )
    costly = run_walk_forward(
        prices, walk_config(), BacktestConfig(transaction_cost_bps=25.0)
    )

    assert (
        costly.frame["net_strategy_return"].sum()
        <= free.frame["net_strategy_return"].sum()
    )


def test_too_little_data_is_rejected() -> None:
    prices = changing_prices(20)
    try:
        run_walk_forward(prices, walk_config())
    except ValueError as error:
        assert "train_periods" in str(error)
    else:
        raise AssertionError("expected short data to be rejected")
