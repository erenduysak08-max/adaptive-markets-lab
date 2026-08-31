import pandas as pd
import yfinance as yf


def download_adjusted_close(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices with a stable one-dimensional result."""
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

