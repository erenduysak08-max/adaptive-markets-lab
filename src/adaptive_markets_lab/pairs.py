"""Rolling-beta pairs-trading research engine."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig, PairsConfig, TradingMode
from .metrics import performance_metrics


@dataclass(frozen=True)
class PairBacktestResult:
    """Daily pair diagnostics, positions and performance statistics."""

    frame: pd.DataFrame
    metrics: dict[str, float]
    asset_a: str
    asset_b: str


def validate_pair_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return two clean, aligned positive price series."""
    if prices.shape[1] != 2:
        raise ValueError("pair prices must contain exactly two columns")
    clean = prices.dropna().astype(float).sort_index()
    if len(clean) < 2:
        raise ValueError("pair prices must contain at least two observations")
    if not clean.index.is_unique:
        raise ValueError("pair price index must be unique")
    if clean.columns[0] == clean.columns[1]:
        raise ValueError("pair price columns must have different names")
    if (clean <= 0).any().any():
        raise ValueError("pair prices must be positive")
    return clean


def rolling_ols_zscore(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Estimate rolling log-price OLS beta and its current residual z-score.

    Each row uses only the trailing window ending at that row. The computation
    includes an intercept and standardises the newest residual using residuals
    from that same regression window.
    """
    clean = validate_pair_prices(prices)
    if lookback < 10:
        raise ValueError("lookback must be at least 10 observations")
    if len(clean) <= lookback:
        raise ValueError("pair prices must be longer than the lookback")

    logs = np.log(clean)
    beta = pd.Series(np.nan, index=clean.index, name="hedge_ratio")
    intercept = pd.Series(np.nan, index=clean.index, name="intercept")
    z_score = pd.Series(np.nan, index=clean.index, name="z_score")

    for stop in range(lookback, len(logs) + 1):
        window = logs.iloc[stop - lookback : stop]
        y = window.iloc[:, 0].to_numpy()
        x = window.iloc[:, 1].to_numpy()
        x_centered = x - x.mean()
        denominator = float(x_centered @ x_centered)
        if denominator <= np.finfo(float).eps:
            continue
        current_beta = float(x_centered @ (y - y.mean()) / denominator)
        current_intercept = float(y.mean() - current_beta * x.mean())
        residuals = y - current_intercept - current_beta * x
        residual_std = float(residuals.std(ddof=1))
        row = stop - 1
        beta.iloc[row] = current_beta
        intercept.iloc[row] = current_intercept
        if residual_std > np.finfo(float).eps:
            z_score.iloc[row] = residuals[-1] / residual_std

    return pd.concat([beta, intercept, z_score], axis=1)


def pairs_target_state(z_score: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Create +1 long-spread, -1 short-spread and 0 flat target states."""
    if entry_z <= 0 or exit_z < 0 or exit_z >= entry_z:
        raise ValueError("require entry_z > exit_z >= 0")

    target = pd.Series(0.0, index=z_score.index, name="target_state")
    state = 0.0
    for date, value in z_score.items():
        if not np.isfinite(value):
            state = 0.0
        elif state == 0.0:
            if value >= entry_z:
                state = -1.0
            elif value <= -entry_z:
                state = 1.0
        elif state == 1.0 and value >= -exit_z:
            state = 0.0
        elif state == -1.0 and value <= exit_z:
            state = 0.0
        target.loc[date] = state
    return target


def _target_weights(
    state: pd.Series,
    beta: pd.Series,
    backtest: BacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    if backtest.trading_mode is TradingMode.SPOT_LONG_ONLY:
        # A long-only relative-value rotation, not a market-neutral pairs trade.
        weight_a = (state > 0).astype(float)
        weight_b = (state < 0).astype(float)
    else:
        denominator = (1.0 + beta.abs()).replace(0.0, np.nan)
        weight_a = state.div(denominator) * backtest.gross_leverage
        weight_b = -state.mul(beta).div(denominator) * backtest.gross_leverage
    return weight_a.fillna(0.0), weight_b.fillna(0.0)


def run_pairs_backtest(
    prices: pd.DataFrame,
    model: PairsConfig | None = None,
    backtest: BacktestConfig | None = None,
) -> PairBacktestResult:
    """Run a close-to-close, costed pairs backtest without future information."""
    clean = validate_pair_prices(prices)
    model = model or PairsConfig()
    backtest = backtest or BacktestConfig(trading_mode=TradingMode.LONG_SHORT)
    diagnostics = rolling_ols_zscore(clean, model.lookback)
    target_state = pairs_target_state(
        diagnostics["z_score"], model.entry_z, model.exit_z
    )
    target_a, target_b = _target_weights(
        target_state, diagnostics["hedge_ratio"], backtest
    )
    target_a = target_a.rename("target_weight_a")
    target_b = target_b.rename("target_weight_b")

    # Signals made after close t become positions for the t-to-t+1 return.
    weight_a = target_a.shift(1).fillna(0.0).rename("weight_a")
    weight_b = target_b.shift(1).fillna(0.0).rename("weight_b")
    returns = clean.pct_change().fillna(0.0)
    return_a = returns.iloc[:, 0].rename("return_a")
    return_b = returns.iloc[:, 1].rename("return_b")
    turnover = (weight_a.diff().abs() + weight_b.diff().abs()).rename("turnover")
    turnover.iloc[0] = abs(weight_a.iloc[0]) + abs(weight_b.iloc[0])
    costs = (turnover * backtest.transaction_cost_bps / 10_000.0).rename(
        "transaction_cost"
    )
    gross_returns = (weight_a * return_a + weight_b * return_b).rename(
        "gross_strategy_return"
    )
    net_returns = (gross_returns - costs).rename("net_strategy_return")
    held_state = target_state.shift(1).fillna(0.0).rename("state")
    gross_exposure = (weight_a.abs() + weight_b.abs()).rename("gross_exposure")

    frame = pd.concat(
        [
            clean.iloc[:, 0].rename("price_a"),
            clean.iloc[:, 1].rename("price_b"),
            return_a,
            return_b,
            diagnostics,
            target_state,
            held_state,
            target_a,
            target_b,
            weight_a,
            weight_b,
            gross_exposure,
            turnover,
            costs,
            gross_returns,
            net_returns,
            (1.0 + net_returns).cumprod().rename("strategy_growth"),
        ],
        axis=1,
    )
    evaluation = frame.iloc[model.lookback :]
    metrics = performance_metrics(
        evaluation["net_strategy_return"],
        evaluation["turnover"],
        backtest.periods_per_year,
    )
    return PairBacktestResult(
        frame=frame,
        metrics=metrics,
        asset_a=str(clean.columns[0]),
        asset_b=str(clean.columns[1]),
    )
