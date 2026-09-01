import pandas as pd
import pytest

from adaptive_markets_lab.metrics import (
    block_bootstrap_mean_ci,
    metrics_from_frame,
    performance_metrics,
)


def test_drawdown_uses_running_peak() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])
    metrics = performance_metrics(returns, pd.Series([0.0, 0.0, 0.0]))

    assert metrics["max_drawdown"] == pytest.approx(-0.20)


def test_metrics_from_frame_validates_columns() -> None:
    with pytest.raises(ValueError, match="required columns"):
        metrics_from_frame(pd.DataFrame({"other": [1.0]}))


def test_block_bootstrap_is_deterministic_and_ordered() -> None:
    differences = pd.Series([0.01, -0.02, 0.03, 0.0] * 20)

    first = block_bootstrap_mean_ci(differences, block_size=4, samples=200, seed=5)
    second = block_bootstrap_mean_ci(differences, block_size=4, samples=200, seed=5)

    assert first == second
    observed = first["observed_annual_mean_difference"]
    assert first["ci_lower"] <= observed + 1e-12
    assert observed <= first["ci_upper"] + 1e-12
