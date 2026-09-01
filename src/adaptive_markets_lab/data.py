import json
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def _unix_timestamp(value: str) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())


def _download_yahoo_chart(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted closes from Yahoo's JSON chart response.

    This endpoint does not require the cookie/crumb handshake that is often
    rate-limited on shared cloud hosts. ``yfinance`` remains a fallback in the
    public downloader because Yahoo can change either interface independently.
    """
    period_start = _unix_timestamp(start)
    period_end = _unix_timestamp(end)
    if period_end <= period_start:
        raise ValueError("end date must be later than start date")

    query = urlencode(
        {
            "period1": period_start,
            "period2": period_end,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker, safe='')}?{query}"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())

    chart = payload.get("chart", {})
    if chart.get("error"):
        description = chart["error"].get("description", "Yahoo request failed")
        raise ValueError(description)
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo returned no chart result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    adjusted = indicators.get("adjclose") or []
    values = adjusted[0].get("adjclose", []) if adjusted else []
    if not values:
        quotes = indicators.get("quote") or []
        values = quotes[0].get("close", []) if quotes else []
    if not timestamps or len(timestamps) != len(values):
        raise ValueError("Yahoo returned incomplete daily prices")

    index = pd.to_datetime(timestamps, unit="s", utc=True)
    timezone = result.get("meta", {}).get("exchangeTimezoneName", "UTC")
    try:
        index = index.tz_convert(timezone)
    except (KeyError, TypeError, ValueError):
        pass
    index = index.normalize().tz_localize(None)
    series = pd.Series(values, index=index, dtype=float, name=ticker).dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError("Yahoo returned no usable adjusted-close data")
    return series


def download_adjusted_close(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices with a stable one-dimensional result."""
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("ticker cannot be empty")
    try:
        return _download_yahoo_chart(symbol, start, end)
    except Exception as chart_error:
        import yfinance as yf

        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
                threads=False,
                timeout=10,
            )
        except Exception as yfinance_error:
            raise ValueError(
                f"Yahoo data is temporarily unavailable for {symbol}. "
                "Try again later or use the offline demo."
            ) from yfinance_error
        price_column = "Adj Close" if "Adj Close" in data else "Close"
        if data.empty or price_column not in data:
            raise ValueError(
                f"no adjusted-close data returned for {symbol}; "
                f"Yahoo chart request failed: {chart_error}"
            ) from chart_error
        return data[price_column].rename(symbol).dropna()


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
