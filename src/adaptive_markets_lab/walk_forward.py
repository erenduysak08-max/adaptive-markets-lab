"""Point-in-time walk-forward model selection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest, validate_prices
from .config import BacktestConfig, MomentumConfig, WalkForwardConfig
from .metrics import performance_metrics
from .momentum import momentum_score, momentum_signal


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample returns plus an audit trail for every selection."""

    frame: pd.DataFrame
    selections: pd.DataFrame
    candidate_scores: pd.DataFrame
    metrics: dict[str, float]


def _selection_value(result: BacktestResult, metric: str) -> float:
    value = result.metrics[metric]
    return value if np.isfinite(value) else float("-inf")


def run_walk_forward(
    prices: pd.Series,
    walk_forward: WalkForwardConfig | None = None,
    backtest: BacktestConfig | None = None,
    threshold: float = 0.0,
) -> WalkForwardResult:
    """Select a momentum half-life on past data, then test it unseen.

    At each boundary every candidate is scored on an expanding training window.
    The winner is frozen for the next test block. The first position in that
    block uses the signal available at the preceding close, and changes between
    selected models are included in turnover.
    """
    clean_prices = validate_prices(prices)
    walk_forward = walk_forward or WalkForwardConfig()
    backtest = backtest or BacktestConfig()
    if len(clean_prices) <= walk_forward.train_periods:
        raise ValueError("prices must contain more observations than train_periods")

    asset_returns = clean_prices.pct_change().fillna(0.0).rename("asset_return")
    oos_index = clean_prices.index[walk_forward.train_periods :]
    selected_half_life = pd.Series(index=oos_index, dtype=float)
    selected_score = pd.Series(index=oos_index, dtype=float)
    target = pd.Series(index=oos_index, dtype=float)
    position = pd.Series(index=oos_index, dtype=float)
    selections: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for fold, test_start in enumerate(
        range(
            walk_forward.train_periods,
            len(clean_prices),
            walk_forward.test_periods,
        )
    ):
        test_stop = min(test_start + walk_forward.test_periods, len(clean_prices))
        train_start = (
            0 if walk_forward.expanding else test_start - walk_forward.train_periods
        )
        training_prices = clean_prices.iloc[train_start:test_start]
        candidates: list[tuple[float, float]] = []

        for half_life in walk_forward.candidate_half_lives:
            candidate = run_backtest(
                training_prices,
                MomentumConfig(half_life=half_life, threshold=threshold),
                backtest,
            )
            value = _selection_value(candidate, walk_forward.selection_metric)
            candidates.append((half_life, value))
            score_rows.append(
                {
                    "fold": fold,
                    "train_end": training_prices.index[-1],
                    "half_life": half_life,
                    walk_forward.selection_metric: value,
                }
            )

        # Prefer the shorter memory when scores are exactly equal.
        chosen_half_life, chosen_value = max(
            candidates, key=lambda item: (item[1], -item[0])
        )
        available_returns = asset_returns.iloc[train_start:test_stop]
        model = MomentumConfig(half_life=chosen_half_life, threshold=threshold)
        model_score = momentum_score(available_returns, model)
        model_target = (
            momentum_signal(available_returns, model, backtest.trading_mode)
            * backtest.gross_leverage
        )
        block_index = clean_prices.index[test_start:test_stop]

        selected_half_life.loc[block_index] = chosen_half_life
        selected_score.loc[block_index] = model_score.loc[block_index]
        target.loc[block_index] = model_target.loc[block_index]
        position.loc[block_index] = model_target.shift(1).loc[block_index]
        selections.append(
            {
                "fold": fold,
                "train_start": training_prices.index[0],
                "train_end": training_prices.index[-1],
                "test_start": block_index[0],
                "test_end": block_index[-1],
                "selected_half_life": chosen_half_life,
                f"training_{walk_forward.selection_metric}": chosen_value,
            }
        )

    position = position.fillna(0.0).rename("position")
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    turnover = turnover.rename("turnover")
    costs = (turnover * backtest.transaction_cost_bps / 10_000.0).rename(
        "transaction_cost"
    )
    oos_returns = asset_returns.loc[oos_index]
    gross_returns = (position * oos_returns).rename("gross_strategy_return")
    net_returns = (gross_returns - costs).rename("net_strategy_return")

    frame = pd.concat(
        [
            clean_prices.loc[oos_index].rename("price"),
            oos_returns,
            selected_score.rename("momentum_score"),
            target.rename("target_position"),
            position,
            selected_half_life.rename("selected_half_life"),
            turnover,
            costs,
            gross_returns,
            net_returns,
            (1.0 + net_returns).cumprod().rename("strategy_growth"),
            (1.0 + oos_returns).cumprod().rename("benchmark_growth"),
        ],
        axis=1,
    )
    metrics = performance_metrics(net_returns, turnover, backtest.periods_per_year)
    return WalkForwardResult(
        frame=frame,
        selections=pd.DataFrame(selections),
        candidate_scores=pd.DataFrame(score_rows),
        metrics=metrics,
    )
