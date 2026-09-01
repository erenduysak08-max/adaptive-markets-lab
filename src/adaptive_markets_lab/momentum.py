import pandas as pd

from .config import MomentumConfig, TradingMode


def momentum_score(returns: pd.Series, config: MomentumConfig) -> pd.Series:
    """Exponentially weighted historical return, known at each timestamp."""
    clean_returns = returns.astype(float).sort_index()
    return clean_returns.ewm(halflife=config.half_life, adjust=False).mean()


def momentum_signal(
    returns: pd.Series,
    model: MomentumConfig,
    trading_mode: TradingMode,
) -> pd.Series:
    """Convert a momentum score to a target position before leverage."""
    score = momentum_score(returns, model)
    long = score > model.threshold

    if trading_mode is TradingMode.SPOT_LONG_ONLY:
        return long.astype(float).rename("target_position")

    short = score < -model.threshold
    signal = pd.Series(0.0, index=score.index, name="target_position")
    signal.loc[long] = 1.0
    signal.loc[short] = -1.0
    return signal
