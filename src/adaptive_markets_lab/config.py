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
        if self.trading_mode is TradingMode.SPOT_LONG_ONLY and self.gross_leverage != 1.0:
            raise ValueError("spot_long_only requires gross_leverage=1.0")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

