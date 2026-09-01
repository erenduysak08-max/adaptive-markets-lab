import numpy as np

from adaptive_markets_lab import (
    BacktestConfig,
    MomentumConfig,
    PairsConfig,
    TradingMode,
)
from adaptive_markets_lab.data import (
    generate_cointegrated_pair,
    generate_regime_prices,
)
from adaptive_markets_lab.sensitivity import (
    momentum_parameter_surface,
    pairs_parameter_surface,
)


def test_momentum_surface_combines_both_axes() -> None:
    prices = generate_regime_prices(300)
    evaluation_index = prices.index[100:]
    surface = momentum_parameter_surface(
        prices,
        evaluation_index,
        "Half-life",
        (5.0, 20.0),
        "Signal threshold",
        (0.0, 0.001),
        MomentumConfig(),
        BacktestConfig(transaction_cost_bps=5.0),
    )

    assert surface.shape == (2, 2)
    assert surface.index.name == "Signal threshold"
    assert surface.columns.name == "Half-life"
    assert np.isfinite(surface.to_numpy()).all()


def test_pairs_surface_uses_a_common_evaluation_window() -> None:
    surface = pairs_parameter_surface(
        generate_cointegrated_pair(350),
        "Regression lookback",
        (30.0, 60.0),
        "Entry z-score",
        (1.5, 2.0),
        PairsConfig(lookback=40, entry_z=2.0, exit_z=0.5),
        BacktestConfig(trading_mode=TradingMode.LONG_SHORT),
    )

    assert surface.shape == (2, 2)
    assert np.isfinite(surface.to_numpy()).all()


def test_pairs_surface_marks_invalid_entry_exit_combinations_blank() -> None:
    surface = pairs_parameter_surface(
        generate_cointegrated_pair(250),
        "Entry z-score",
        (1.0, 2.0),
        "Exit z-score",
        (0.5, 1.5),
        PairsConfig(),
        BacktestConfig(trading_mode=TradingMode.LONG_SHORT),
    )

    assert np.isnan(surface.loc[1.5, 1.0])
    assert np.isfinite(surface.loc[0.5, 2.0])
