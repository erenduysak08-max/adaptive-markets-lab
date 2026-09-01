import numpy as np
import pytest

from adaptive_markets_lab.data import generate_cointegrated_pair, generate_regime_prices


def test_demo_data_is_deterministic_and_positive() -> None:
    first = generate_regime_prices(100, seed=3)
    second = generate_regime_prices(100, seed=3)

    assert np.array_equal(first.values, second.values)
    assert (first > 0).all()
    assert first.index.is_monotonic_increasing


def test_demo_data_rejects_tiny_sample() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        generate_regime_prices(10)


def test_synthetic_pair_is_deterministic_positive_and_correlated() -> None:
    first = generate_cointegrated_pair(300, seed=4)
    second = generate_cointegrated_pair(300, seed=4)

    assert np.array_equal(first.values, second.values)
    assert (first > 0).all().all()
    assert first.pct_change().corr().iloc[0, 1] > 0.4
