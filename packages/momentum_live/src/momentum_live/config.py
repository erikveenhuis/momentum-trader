"""Configuration helpers for live trading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class AlpacaCredentials:
    """Connection credentials for the Alpaca crypto data and trading APIs."""

    api_key: str
    secret_key: str
    location: str = "us"
    paper: bool = True  # Use paper trading by default

    @classmethod
    def from_environment(
        cls,
        api_key_var: str = "ALPACA_API_KEY",
        secret_key_var: str = "ALPACA_API_SECRET",
        location_var: str = "ALPACA_CRYPTO_FEED",
        paper_var: str = "ALPACA_PAPER_TRADING",
    ) -> "AlpacaCredentials":
        """Build credentials from environment variables."""

        api_key = os.getenv(api_key_var)
        secret_key = os.getenv(secret_key_var)
        location = os.getenv(location_var, "us")
        paper = os.getenv(paper_var, "true").lower() in ("true", "1", "yes")

        if not api_key:
            raise EnvironmentError(f"Missing Alpaca API key in environment variable '{api_key_var}'.")
        if not secret_key:
            raise EnvironmentError(f"Missing Alpaca API secret in environment variable '{secret_key_var}'.")

        return cls(api_key=api_key, secret_key=secret_key, location=location, paper=paper)


@dataclass(slots=True)
class LiveTradingConfig:
    """Runtime configuration for live inference."""

    symbols: Sequence[str]
    window_size: int
    models_dir: str = "models"
    checkpoint_pattern: str = "checkpoint_trainer_best_*.pt"
    initial_balance: float = 1_000.0
    transaction_fee: float = 0.001
    reward_scale: float = 50.0
    invalid_action_penalty: float = -0.05

    def __post_init__(self) -> None:
        if isinstance(self.symbols, Iterable) and not isinstance(self.symbols, str):
            self.symbols = [symbol.strip() for symbol in self.symbols if symbol]
        else:
            raise ValueError("symbols must be a sequence of non-empty strings")

        if not self.symbols:
            raise ValueError("At least one trading symbol must be provided")

        if self.window_size < 1:
            raise ValueError("window_size must be a positive integer")

        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")

        if not (0.0 <= self.transaction_fee < 1.0):
            raise ValueError("transaction_fee must be in the range [0, 1)")

        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")

        if self.invalid_action_penalty >= 0:
            raise ValueError("invalid_action_penalty should be negative to penalise invalid trades")


def parse_symbols(symbols: str | Sequence[str]) -> List[str]:
    """Parse a comma or space separated list of symbols into a list."""

    if isinstance(symbols, str):
        # Allow both comma and whitespace separated inputs
        parts = [part.strip() for chunk in symbols.split(",") for part in chunk.split()]
    else:
        parts = [str(symbol).strip() for symbol in symbols]

    cleaned = [symbol for symbol in parts if symbol]

    if not cleaned:
        raise ValueError("No valid symbols provided")

    return cleaned


__all__ = ["AlpacaCredentials", "LiveTradingConfig", "parse_symbols"]
