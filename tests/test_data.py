import json

import numpy as np
import pytest

from adaptive_markets_lab.data import (
    download_adjusted_close,
    generate_cointegrated_pair,
    generate_regime_prices,
)


class _YahooResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_yahoo_chart_download_returns_adjusted_close(monkeypatch) -> None:
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [1704205800, 1704292200],
                    "indicators": {
                        "adjclose": [{"adjclose": [100.0, 102.5]}],
                        "quote": [{"close": [101.0, 103.0]}],
                    },
                }
            ],
        }
    }
    monkeypatch.setattr(
        "adaptive_markets_lab.data.urlopen",
        lambda request, timeout: _YahooResponse(payload),
    )

    prices = download_adjusted_close(" spy ", "2024-01-01", "2024-02-01")

    assert prices.name == "SPY"
    assert prices.tolist() == [100.0, 102.5]
    assert prices.index.tolist() == [
        np.datetime64("2024-01-02"),
        np.datetime64("2024-01-03"),
    ]


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
