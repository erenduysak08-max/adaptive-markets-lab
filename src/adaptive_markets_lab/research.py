"""Fixed-versus-adaptive research comparisons."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import BacktestConfig, MomentumConfig, PairsConfig, WalkForwardConfig
from .metrics import block_bootstrap_mean_ci, performance_metrics
from .pairs import PairBacktestResult, run_pairs_backtest
from .walk_forward import WalkForwardResult, run_walk_forward


@dataclass(frozen=True)
class ResearchResult:
    """Tables required to inspect and reproduce one experiment."""

    adaptive: WalkForwardResult
    comparison: pd.DataFrame
    equity_curves: pd.DataFrame
    sensitivity: pd.DataFrame
    regime_metrics: pd.DataFrame
    uncertainty: pd.DataFrame


@dataclass(frozen=True)
class PairResearchResult:
    """Pairs result with a same-window passive benchmark."""

    pair: PairBacktestResult
    comparison: pd.DataFrame
    equity_curves: pd.DataFrame
    uncertainty: pd.DataFrame
    strategy_name: str


def _fixed_oos_returns(
    prices: pd.Series,
    oos_index: pd.Index,
    half_life: float,
    threshold: float,
    backtest: BacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    result = run_backtest(
        prices,
        MomentumConfig(half_life=half_life, threshold=threshold),
        backtest,
    )
    frame = result.frame.loc[oos_index]
    return frame["net_strategy_return"], frame["turnover"]


def _metric_row(
    model: str,
    returns: pd.Series,
    turnover: pd.Series,
    periods_per_year: int,
) -> dict[str, float | str]:
    return {
        "model": model,
        **performance_metrics(returns, turnover, periods_per_year),
    }


def _regime_table(
    prices: pd.Series,
    returns: pd.DataFrame,
    periods_per_year: int,
) -> pd.DataFrame:
    # Shifted so today's return never helps classify today's regime.
    trailing_return = prices.pct_change(126).shift(1).reindex(returns.index)
    regimes = pd.Series(
        "down/flat trailing market", index=returns.index, dtype="object"
    )
    regimes.loc[trailing_return >= 0] = "positive trailing market"
    rows: list[dict[str, float | int | str]] = []
    for model in returns.columns:
        for regime in regimes.dropna().unique():
            mask = regimes == regime
            subset = returns.loc[mask, model]
            metrics = performance_metrics(
                subset, pd.Series(0.0, index=subset.index), periods_per_year
            )
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "observations": len(subset),
                    "annual_return": metrics["annual_return"],
                    "annual_volatility": metrics["annual_volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def run_research(
    prices: pd.Series,
    walk_forward: WalkForwardConfig | None = None,
    backtest: BacktestConfig | None = None,
    fixed_half_life: float = 20.0,
    threshold: float = 0.0,
) -> ResearchResult:
    """Compare an adaptive model with predeclared, investable baselines.

    All reported strategies use exactly the adaptive model's out-of-sample
    dates. The full-sample sensitivity table is diagnostic and is never used to
    choose the predeclared fixed baseline.
    """
    walk_forward = walk_forward or WalkForwardConfig()
    backtest = backtest or BacktestConfig()
    adaptive = run_walk_forward(prices, walk_forward, backtest, threshold)
    oos_index = adaptive.frame.index

    fixed_returns, fixed_turnover = _fixed_oos_returns(
        prices, oos_index, fixed_half_life, threshold, backtest
    )
    adaptive_returns = adaptive.frame["net_strategy_return"]
    adaptive_turnover = adaptive.frame["turnover"]

    buy_hold_turnover = pd.Series(0.0, index=oos_index)
    buy_hold_turnover.iloc[0] = 1.0
    buy_hold_returns = adaptive.frame["asset_return"].copy()
    buy_hold_returns.iloc[0] -= backtest.transaction_cost_bps / 10_000.0

    fixed_name = f"Fixed momentum ({fixed_half_life:g}d half-life)"
    returns = pd.concat(
        [
            adaptive_returns.rename("Adaptive momentum"),
            fixed_returns.rename(fixed_name),
            buy_hold_returns.rename("Buy and hold"),
        ],
        axis=1,
    )
    comparison = pd.DataFrame(
        [
            _metric_row(
                "Adaptive momentum",
                adaptive_returns,
                adaptive_turnover,
                backtest.periods_per_year,
            ),
            _metric_row(
                fixed_name,
                fixed_returns,
                fixed_turnover,
                backtest.periods_per_year,
            ),
            _metric_row(
                "Buy and hold",
                buy_hold_returns,
                buy_hold_turnover,
                backtest.periods_per_year,
            ),
        ]
    ).set_index("model")

    sensitivity_rows: list[dict[str, float]] = []
    for half_life in walk_forward.candidate_half_lives:
        candidate_returns, candidate_turnover = _fixed_oos_returns(
            prices, oos_index, half_life, threshold, backtest
        )
        sensitivity_rows.append(
            {
                "half_life": half_life,
                **performance_metrics(
                    candidate_returns, candidate_turnover, backtest.periods_per_year
                ),
            }
        )

    uncertainty_rows: list[dict[str, float | int | str]] = []
    for benchmark in (fixed_name, "Buy and hold"):
        uncertainty_rows.append(
            {
                "comparison": f"Adaptive momentum minus {benchmark}",
                **block_bootstrap_mean_ci(
                    returns["Adaptive momentum"] - returns[benchmark]
                ),
            }
        )

    return ResearchResult(
        adaptive=adaptive,
        comparison=comparison,
        equity_curves=(1.0 + returns).cumprod(),
        sensitivity=pd.DataFrame(sensitivity_rows).set_index("half_life"),
        regime_metrics=_regime_table(prices, returns, backtest.periods_per_year),
        uncertainty=pd.DataFrame(uncertainty_rows).set_index("comparison"),
    )


def run_pairs_research(
    prices: pd.DataFrame,
    model: PairsConfig | None = None,
    backtest: BacktestConfig | None = None,
) -> PairResearchResult:
    """Compare a pairs strategy with a 50/50 passive holding."""
    model = model or PairsConfig()
    backtest = backtest or BacktestConfig()
    pair = run_pairs_backtest(prices, model, backtest)
    evaluation = pair.frame.iloc[model.lookback :]
    pair_returns = evaluation["net_strategy_return"]
    pair_turnover = evaluation["turnover"]

    passive_turnover = pd.Series(0.0, index=evaluation.index)
    passive_turnover.iloc[0] = 1.0
    passive_returns = 0.5 * evaluation["return_a"] + 0.5 * evaluation["return_b"]
    passive_returns.iloc[0] -= backtest.transaction_cost_bps / 10_000.0
    passive_name = "50/50 buy and hold"
    strategy_name = (
        "Long-only pair rotation"
        if backtest.trading_mode.value == "spot_long_only"
        else "Rolling-beta pairs trade"
    )
    returns = pd.concat(
        [
            pair_returns.rename(strategy_name),
            passive_returns.rename(passive_name),
        ],
        axis=1,
    )
    comparison = pd.DataFrame(
        [
            _metric_row(
                strategy_name,
                pair_returns,
                pair_turnover,
                backtest.periods_per_year,
            ),
            _metric_row(
                passive_name,
                passive_returns,
                passive_turnover,
                backtest.periods_per_year,
            ),
        ]
    ).set_index("model")
    uncertainty = pd.DataFrame(
        [
            {
                "comparison": f"{strategy_name} minus {passive_name}",
                **block_bootstrap_mean_ci(pair_returns - passive_returns),
            }
        ]
    ).set_index("comparison")
    return PairResearchResult(
        pair=pair,
        comparison=comparison,
        equity_curves=(1.0 + returns).cumprod(),
        uncertainty=uncertainty,
        strategy_name=strategy_name,
    )


def save_research_artifacts(result: ResearchResult, output_dir: str | Path) -> None:
    """Save small, diffable CSV outputs for reproducibility."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(destination / "comparison.csv")
    result.equity_curves.to_csv(destination / "equity_curves.csv")
    result.adaptive.selections.to_csv(destination / "selections.csv", index=False)
    result.adaptive.candidate_scores.to_csv(
        destination / "candidate_scores.csv", index=False
    )
    result.sensitivity.to_csv(destination / "sensitivity.csv")
    result.regime_metrics.to_csv(destination / "regime_metrics.csv", index=False)
    result.uncertainty.to_csv(destination / "uncertainty.csv")


def save_pairs_artifacts(result: PairResearchResult, output_dir: str | Path) -> None:
    """Save the complete pairs audit trail and comparison tables."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(destination / "comparison.csv")
    result.equity_curves.to_csv(destination / "equity_curves.csv")
    result.pair.frame.to_csv(destination / "pair_backtest.csv")
    result.uncertainty.to_csv(destination / "uncertainty.csv")
