"""Training module for momentum trader."""

__version__ = "0.1.6"

from .data import DataManager
from .metrics import (
    PerformanceTracker,
    calculate_avg_trade_return,
    calculate_episode_score,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from .trainer import RainbowTrainerModule

__all__ = [
    "RainbowTrainerModule",
    "DataManager",
    "PerformanceTracker",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_avg_trade_return",
    "calculate_episode_score",
]
