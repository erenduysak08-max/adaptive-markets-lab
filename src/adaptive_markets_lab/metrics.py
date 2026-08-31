import math

import numpy as np
import pandas as pd


def performance_metrics(
    net_returns: pd.Series,
    turnover: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    returns = net_returns.fillna(0.0).astype(float)
    growth = (1.0 + returns).cumprod()
    observations = len(returns)

    total_return = float(growth.iloc[-1] - 1.0) if observations else 0.0
    annual_return = (
        float(growth.iloc[-1] ** (periods_per_year / observations) - 1.0)
        if observations and growth.iloc[-1] > 0
        else float("nan")
    )
    annual_volatility = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year))
        if observations > 1 and returns.std(ddof=1) > 0
        else float("nan")
    )
    drawdown = growth.div(growth.cummax()).sub(1.0)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()) if observations else 0.0,
        "annual_turnover": float(turnover.fillna(0.0).mean() * periods_per_year),
    }


def finite_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Make metrics safe for JSON and UI display."""
    return {key: value if np.isfinite(value) else None for key, value in metrics.items()}

