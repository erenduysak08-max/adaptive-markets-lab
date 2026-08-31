"""Adaptive Markets Lab research engine."""

from .backtest import BacktestResult, run_backtest
from .config import BacktestConfig, MomentumConfig, TradingMode
from .momentum import momentum_signal

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MomentumConfig",
    "TradingMode",
    "momentum_signal",
    "run_backtest",
]

