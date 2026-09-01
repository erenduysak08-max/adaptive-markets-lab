"""Two-parameter performance surfaces for dashboard diagnostics."""

from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import BacktestConfig, MomentumConfig, PairsConfig
from .metrics import performance_metrics
from .pairs import run_pairs_backtest

SUPPORTED_METRICS = {
    "Total return": "total_return",
    "Annual return": "annual_return",
    "Sharpe ratio": "sharpe_ratio",
    "Maximum drawdown": "max_drawdown",
}

MOMENTUM_PARAMETER_VALUES: dict[str, tuple[float, ...]] = {
    "Half-life": (3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0),
    "Signal threshold": (0.0, 0.00025, 0.0005, 0.001, 0.002),
    "Transaction cost (bps)": (0.0, 2.5, 5.0, 10.0, 20.0),
}

PAIRS_PARAMETER_VALUES: dict[str, tuple[float, ...]] = {
    "Regression lookback": (20.0, 40.0, 60.0, 90.0, 120.0),
    "Entry z-score": (1.0, 1.5, 2.0, 2.5, 3.0),
    "Exit z-score": (0.0, 0.25, 0.5, 0.75, 1.0),
    "Transaction cost (bps)": (0.0, 2.5, 5.0, 10.0, 20.0),
}

_MOMENTUM_FIELDS = {
    "Half-life": "half_life",
    "Signal threshold": "threshold",
}
_PAIRS_FIELDS = {
    "Regression lookback": "lookback",
    "Entry z-score": "entry_z",
    "Exit z-score": "exit_z",
}


def _performance_value(
    frame: pd.DataFrame, metric: str, periods_per_year: int
) -> float:
    metrics = performance_metrics(
        frame["net_strategy_return"], frame["turnover"], periods_per_year
    )
    return metrics[metric]


def momentum_parameter_surface(
    prices: pd.Series,
    evaluation_index: pd.Index,
    x_parameter: str,
    x_values: tuple[float, ...],
    y_parameter: str,
    y_values: tuple[float, ...],
    model: MomentumConfig,
    backtest: BacktestConfig,
    metric: str = "total_return",
) -> pd.DataFrame:
    """Evaluate a fixed momentum model over a two-dimensional parameter grid."""
    if x_parameter == y_parameter:
        raise ValueError("heatmap axes must use different parameters")
    if metric not in SUPPORTED_METRICS.values():
        raise ValueError("unsupported heatmap metric")

    surface = pd.DataFrame(index=y_values, columns=x_values, dtype=float)
    for y_value in y_values:
        for x_value in x_values:
            current_model = model
            current_backtest = backtest
            for parameter, value in (
                (x_parameter, x_value),
                (y_parameter, y_value),
            ):
                if parameter == "Transaction cost (bps)":
                    current_backtest = replace(
                        current_backtest, transaction_cost_bps=float(value)
                    )
                else:
                    field = _MOMENTUM_FIELDS.get(parameter)
                    if field is None:
                        raise ValueError(f"unsupported momentum parameter: {parameter}")
                    current_model = replace(current_model, **{field: float(value)})
            result = run_backtest(prices, current_model, current_backtest)
            evaluation = result.frame.loc[evaluation_index]
            surface.loc[y_value, x_value] = _performance_value(
                evaluation, metric, current_backtest.periods_per_year
            )

    surface.index.name = y_parameter
    surface.columns.name = x_parameter
    return surface


def pairs_parameter_surface(
    prices: pd.DataFrame,
    x_parameter: str,
    x_values: tuple[float, ...],
    y_parameter: str,
    y_values: tuple[float, ...],
    model: PairsConfig,
    backtest: BacktestConfig,
    metric: str = "total_return",
) -> pd.DataFrame:
    """Evaluate pairs parameters on one common post-warm-up date range."""
    if x_parameter == y_parameter:
        raise ValueError("heatmap axes must use different parameters")
    if metric not in SUPPORTED_METRICS.values():
        raise ValueError("unsupported heatmap metric")

    lookbacks = [model.lookback]
    if x_parameter == "Regression lookback":
        lookbacks.extend(int(value) for value in x_values)
    if y_parameter == "Regression lookback":
        lookbacks.extend(int(value) for value in y_values)
    common_start = max(lookbacks)
    if len(prices) <= common_start:
        raise ValueError("not enough observations for the largest lookback")
    evaluation_index = prices.dropna().sort_index().index[common_start:]

    surface = pd.DataFrame(index=y_values, columns=x_values, dtype=float)
    for y_value in y_values:
        for x_value in x_values:
            current_model = model
            current_backtest = backtest
            valid = True
            for parameter, value in (
                (x_parameter, x_value),
                (y_parameter, y_value),
            ):
                if parameter == "Transaction cost (bps)":
                    current_backtest = replace(
                        current_backtest, transaction_cost_bps=float(value)
                    )
                else:
                    field = _PAIRS_FIELDS.get(parameter)
                    if field is None:
                        raise ValueError(f"unsupported pairs parameter: {parameter}")
                    cast_value = int(value) if field == "lookback" else float(value)
                    try:
                        current_model = replace(current_model, **{field: cast_value})
                    except ValueError:
                        valid = False
                        break
            if not valid:
                surface.loc[y_value, x_value] = np.nan
                continue
            result = run_pairs_backtest(prices, current_model, current_backtest)
            evaluation = result.frame.loc[evaluation_index]
            surface.loc[y_value, x_value] = _performance_value(
                evaluation, metric, current_backtest.periods_per_year
            )

    surface.index.name = y_parameter
    surface.columns.name = x_parameter
    return surface
