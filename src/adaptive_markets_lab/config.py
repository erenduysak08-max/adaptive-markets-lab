from dataclasses import dataclass
from enum import Enum


class TradingMode(str, Enum):
    """Position constraints applied by the backtest."""

    SPOT_LONG_ONLY = "spot_long_only"
    LONG_SHORT = "long_short"


@dataclass(frozen=True)
class MomentumConfig:
    half_life: float = 20.0
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.half_life <= 0:
            raise ValueError("half_life must be positive")
        if self.threshold < 0:
            raise ValueError("threshold cannot be negative")


@dataclass(frozen=True)
class BacktestConfig:
    trading_mode: TradingMode = TradingMode.SPOT_LONG_ONLY
    gross_leverage: float = 1.0
    transaction_cost_bps: float = 0.0
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.gross_leverage <= 0:
            raise ValueError("gross_leverage must be positive")
        if (
            self.trading_mode is TradingMode.SPOT_LONG_ONLY
            and self.gross_leverage != 1.0
        ):
            raise ValueError("spot_long_only requires gross_leverage=1.0")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Rules for repeatedly selecting a model using past data only.

    ``train_periods`` observations are used for each selection. The chosen
    half-life is frozen for ``test_periods`` observations before the rolling
    training window moves forward. Set ``expanding=True`` to keep all history.
    """

    candidate_half_lives: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)
    train_periods: int = 504
    test_periods: int = 63
    selection_metric: str = "sharpe_ratio"
    expanding: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_half_lives:
            raise ValueError("candidate_half_lives cannot be empty")
        if any(value <= 0 for value in self.candidate_half_lives):
            raise ValueError("candidate half-lives must be positive")
        if len(set(self.candidate_half_lives)) != len(self.candidate_half_lives):
            raise ValueError("candidate half-lives must be unique")
        if self.train_periods < 2:
            raise ValueError("train_periods must be at least 2")
        if self.test_periods <= 0:
            raise ValueError("test_periods must be positive")
        if self.selection_metric not in {"sharpe_ratio", "annual_return"}:
            raise ValueError(
                "selection_metric must be 'sharpe_ratio' or 'annual_return'"
            )


@dataclass(frozen=True)
class PairsConfig:
    """Parameters for a rolling-regression pairs strategy."""

    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5

    def __post_init__(self) -> None:
        if self.lookback < 10:
            raise ValueError("lookback must be at least 10 observations")
        if self.entry_z <= 0:
            raise ValueError("entry_z must be positive")
        if self.exit_z < 0:
            raise ValueError("exit_z cannot be negative")
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be smaller than entry_z")
