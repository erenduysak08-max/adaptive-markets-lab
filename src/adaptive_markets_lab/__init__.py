"""Adaptive Markets Lab research engine."""

from .backtest import BacktestResult, run_backtest
from .config import (
    BacktestConfig,
    MomentumConfig,
    PairsConfig,
    TradingMode,
    WalkForwardConfig,
)
from .momentum import momentum_signal
from .pairs import PairBacktestResult, run_pairs_backtest
from .research import (
    PairResearchResult,
    ResearchResult,
    run_pairs_research,
    run_research,
)
from .walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MomentumConfig",
    "PairBacktestResult",
    "PairResearchResult",
    "PairsConfig",
    "ResearchResult",
    "TradingMode",
    "WalkForwardConfig",
    "WalkForwardResult",
    "momentum_signal",
    "run_backtest",
    "run_pairs_backtest",
    "run_pairs_research",
    "run_research",
    "run_walk_forward",
]
