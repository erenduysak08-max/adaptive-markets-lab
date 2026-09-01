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


def metrics_from_frame(
    frame: pd.DataFrame, periods_per_year: int = 252
) -> dict[str, float]:
    """Recalculate metrics for a selected evaluation window."""
    required = {"net_strategy_return", "turnover"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
    return performance_metrics(
        frame["net_strategy_return"], frame["turnover"], periods_per_year
    )


def block_bootstrap_mean_ci(
    differences: pd.Series,
    block_size: int = 20,
    samples: int = 2_000,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    seed: int = 0,
) -> dict[str, float | int]:
    """Estimate uncertainty in a mean return difference with circular blocks.

    Resampling contiguous blocks retains short-range serial dependence that an
    independent daily bootstrap would destroy. The interval concerns the
    annualised arithmetic mean difference, not a Sharpe ratio or future profit.
    """
    clean = differences.dropna().astype(float).to_numpy()
    if len(clean) < 2:
        raise ValueError("differences must contain at least two observations")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    block_size = min(block_size, len(clean))
    blocks_per_sample = math.ceil(len(clean) / block_size)
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples)
    offsets = np.arange(block_size)
    for draw in range(samples):
        starts = generator.integers(0, len(clean), size=blocks_per_sample)
        indices = (starts[:, None] + offsets) % len(clean)
        bootstrap_means[draw] = clean[indices.ravel()[: len(clean)]].mean()

    tail = (1.0 - confidence) / 2.0
    scale = periods_per_year
    return {
        "observed_annual_mean_difference": float(clean.mean() * scale),
        "ci_lower": float(np.quantile(bootstrap_means, tail) * scale),
        "ci_upper": float(np.quantile(bootstrap_means, 1.0 - tail) * scale),
        "confidence": confidence,
        "block_size": block_size,
        "bootstrap_samples": samples,
    }


def finite_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Make metrics safe for JSON and UI display."""
    return {
        key: value if np.isfinite(value) else None for key, value in metrics.items()
    }
