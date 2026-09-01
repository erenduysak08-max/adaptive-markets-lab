import numpy as np

from adaptive_markets_lab import BacktestConfig, WalkForwardConfig
from adaptive_markets_lab.data import generate_regime_prices
from adaptive_markets_lab.research import run_research


def research_result():
    return run_research(
        generate_regime_prices(700, seed=11),
        WalkForwardConfig(
            candidate_half_lives=(5.0, 20.0, 60.0),
            train_periods=252,
            test_periods=63,
        ),
        BacktestConfig(transaction_cost_bps=5.0),
    )


def test_comparison_uses_three_clear_baselines() -> None:
    result = research_result()

    assert list(result.comparison.index) == [
        "Adaptive momentum",
        "Fixed momentum (20d half-life)",
        "Buy and hold",
    ]
    assert result.equity_curves.shape == (448, 3)
    assert result.equity_curves.index.equals(result.adaptive.frame.index)


def test_sensitivity_contains_every_candidate() -> None:
    result = research_result()

    assert list(result.sensitivity.index) == [5.0, 20.0, 60.0]
    assert np.isfinite(result.sensitivity["total_return"]).all()


def test_regimes_are_lagged_and_reported_for_each_model() -> None:
    result = research_result()

    assert set(result.regime_metrics["model"]) == set(result.comparison.index)
    assert set(result.regime_metrics["regime"]) == {
        "positive trailing market",
        "down/flat trailing market",
    }


def test_uncertainty_compares_adaptive_with_both_baselines() -> None:
    result = research_result()

    assert len(result.uncertainty) == 2
    assert (result.uncertainty["ci_lower"] <= result.uncertainty["ci_upper"]).all()
