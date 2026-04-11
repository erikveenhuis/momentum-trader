"""Configuration management for the trading environment."""

import os
from dataclasses import dataclass


@dataclass
class TradingEnvConfig:
    """Configuration for the trading environment.

    All fields are required (no defaults) to ensure every value
    is explicitly set from training_config.yaml. The only exception
    is render_mode which defaults to None (no rendering).
    """

    data_path: str
    window_size: int
    initial_balance: float
    transaction_fee: float
    reward_scale: float
    invalid_action_penalty: float
    drawdown_penalty_lambda: float
    slippage_bps: float
    opportunity_cost_lambda: float
    min_rebalance_pct: float
    min_trade_value: float

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be > 0, got {self.initial_balance}")
        if not 0 <= self.transaction_fee < 1:
            raise ValueError(f"transaction_fee must be in [0,1), got {self.transaction_fee}")
        if self.reward_scale <= 0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")
