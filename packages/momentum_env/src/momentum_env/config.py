"""Configuration management for the trading environment."""
from dataclasses import dataclass
from typing import Literal, Optional
import os

@dataclass
class TradingEnvConfig:
    """Configuration for the trading environment."""
    
    # Data configuration
    data_path: str
    window_size: int = 60
    
    # Trading parameters
    initial_balance: float = 1000.0
    transaction_fee: float = 0.001  # 0.1% fee
    
    # Reward parameters
    reward_scale: float = 10.0
    invalid_action_penalty: float = -1.0
    
    # Environment parameters
    render_mode: Optional[Literal["human", "terminal"]] = None
    
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
        if self.render_mode is not None and self.render_mode not in ["human", "terminal"]:
            raise ValueError(f"Invalid render_mode: {self.render_mode}. Must be one of: human, terminal, or None")