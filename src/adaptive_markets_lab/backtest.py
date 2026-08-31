from dataclasses import dataclass

import pandas as pd

from .config import BacktestConfig, MomentumConfig
from .metrics import performance_metrics
from .momentum import momentum_score, momentum_signal


@dataclass(frozen=True)
class BacktestResult:
    frame: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    prices: pd.Series,
    model: MomentumConfig | None = None,
    backtest: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a close-to-close backtest without using future information.

    The signal observed at close t is executed for the return from t to t+1,
    represented by shifting the target position by one row.
    """
    model = model or MomentumConfig()
    backtest = backtest or BacktestConfig()

    clean_prices = prices.dropna().astype(float).sort_index()
    if len(clean_prices) < 2:
        raise ValueError("prices must contain at least two observations")
    if not clean_prices.index.is_unique:
        raise ValueError("prices index must be unique")
    if (clean_prices <= 0).any():
        raise ValueError("prices must be positive")

    asset_returns = clean_prices.pct_change().fillna(0.0).rename("asset_return")
    score = momentum_score(asset_returns, model).rename("momentum_score")
    target = momentum_signal(asset_returns, model, backtest.trading_mode)
    target = (target * backtest.gross_leverage).rename("target_position")
    position = target.shift(1).fillna(0.0).rename("position")
    turnover = position.diff().abs().fillna(position.abs()).rename("turnover")
    costs = (turnover * backtest.transaction_cost_bps / 10_000.0).rename(
        "transaction_cost"
    )
    gross_returns = (position * asset_returns).rename("gross_strategy_return")
    net_returns = (gross_returns - costs).rename("net_strategy_return")
    strategy_growth = (1.0 + net_returns).cumprod().rename("strategy_growth")
    benchmark_growth = (1.0 + asset_returns).cumprod().rename("benchmark_growth")

    frame = pd.concat(
        [
            clean_prices.rename("price"),
            asset_returns,
            score,
            target,
            position,
            turnover,
            costs,
            gross_returns,
            net_returns,
            strategy_growth,
            benchmark_growth,
        ],
        axis=1,
    )
    metrics = performance_metrics(net_returns, turnover, backtest.periods_per_year)
    return BacktestResult(frame=frame, metrics=metrics)

