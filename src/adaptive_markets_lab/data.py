import numpy as np
import pandas as pd


def download_adjusted_close(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices with a stable one-dimensional result."""
    import yfinance as yf

    if not ticker.strip():
        raise ValueError("ticker cannot be empty")
    data = yf.download(
        ticker.strip().upper(),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if data.empty or "Adj Close" not in data:
        raise ValueError(f"no adjusted-close data returned for {ticker!r}")
    return data["Adj Close"].rename(ticker.strip().upper())


def download_adjusted_pair(
    ticker_a: str, ticker_b: str, start: str, end: str
) -> pd.DataFrame:
    """Download and align adjusted closes for two distinct assets."""
    first = ticker_a.strip().upper()
    second = ticker_b.strip().upper()
    if not first or not second:
        raise ValueError("both pair tickers are required")
    if first == second:
        raise ValueError("pair tickers must be different")
    prices = pd.concat(
        [
            download_adjusted_close(first, start, end),
            download_adjusted_close(second, start, end),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if prices.empty:
        raise ValueError("the pair has no overlapping adjusted-close data")
    prices.columns = [first, second]
    return prices


def generate_regime_prices(periods: int = 1_500, seed: int = 7) -> pd.Series:
    """Create deterministic synthetic prices for an offline smoke test.

    The changing drift and volatility make this useful for exercising the
    adaptive pipeline. It is deliberately not presented as market evidence.
    """
    if periods < 20:
        raise ValueError("periods must be at least 20")

    generator = np.random.default_rng(seed)
    boundaries = np.linspace(0, periods, 5, dtype=int)
    regimes = (
        (0.0005, 0.008),
        (-0.0004, 0.018),
        (0.0001, 0.006),
        (0.0007, 0.014),
    )
    log_returns = np.empty(periods)
    for (start, stop), (drift, volatility) in zip(
        zip(boundaries[:-1], boundaries[1:], strict=True), regimes, strict=True
    ):
        log_returns[start:stop] = generator.normal(drift, volatility, stop - start)

    index = pd.bdate_range("2018-01-02", periods=periods)
    values = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(values, index=index, name="SYNTHETIC")


def generate_cointegrated_pair(periods: int = 1_500, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic pair with a noisy mean-reverting log spread."""
    if periods < 20:
        raise ValueError("periods must be at least 20")

    generator = np.random.default_rng(seed)
    common = np.cumsum(generator.normal(0.0002, 0.009, periods))
    spread = np.empty(periods)
    spread[0] = 0.0
    for index in range(1, periods):
        spread[index] = 0.94 * spread[index - 1] + generator.normal(0.0, 0.012)

    beta = np.where(np.arange(periods) < periods // 2, 1.05, 0.85)
    log_b = np.log(100.0) + common
    log_a = np.log(95.0) + beta * (log_b - np.log(100.0)) + spread
    dates = pd.bdate_range("2018-01-02", periods=periods)
    return pd.DataFrame(
        {"SYNTH_A": np.exp(log_a), "SYNTH_B": np.exp(log_b)}, index=dates
    )
