"""Configuration helpers for live trading."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class AlpacaCredentials:
    """Connection credentials for the Alpaca crypto data and trading APIs."""

    api_key: str
    secret_key: str
    location: str = "us"
    paper: bool = True

    @classmethod
    def from_environment(
        cls,
        api_key_var: str = "ALPACA_API_KEY",
        secret_key_var: str = "ALPACA_API_SECRET",
        location_var: str = "ALPACA_CRYPTO_FEED",
        paper_var: str = "ALPACA_PAPER_TRADING",
    ) -> AlpacaCredentials:
        """Build credentials from environment variables."""

        api_key = os.getenv(api_key_var)
        secret_key = os.getenv(secret_key_var)
        location = os.getenv(location_var, "us")
        paper = os.getenv(paper_var, "true").lower() in ("true", "1", "yes")

        if not api_key:
            raise OSError(f"Missing Alpaca API key in environment variable '{api_key_var}'.")
        if not secret_key:
            raise OSError(f"Missing Alpaca API secret in environment variable '{secret_key_var}'.")

        return cls(api_key=api_key, secret_key=secret_key, location=location, paper=paper)


@dataclass(slots=True)
class LiveTradingConfig:
    """Runtime configuration for live inference. No defaults -- all fields required."""

    symbols: Sequence[str]
    window_size: int
    initial_balance: float
    transaction_fee: float
    reward_scale: float
    invalid_action_penalty: float
    drawdown_penalty_lambda: float
    slippage_bps: float
    opportunity_cost_lambda: float
    benchmark_allocation_frac: float
    min_rebalance_pct: float
    min_trade_value: float
    models_dir: str
    checkpoint_pattern: str
    tb_log_dir: str | None = None

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


def parse_symbols(symbols: str | Sequence[str]) -> list[str]:
    """Parse a comma or space separated list of symbols into a list."""

    if isinstance(symbols, str):
        parts = [part.strip() for chunk in symbols.split(",") for part in chunk.split()]
    else:
        parts = [str(symbol).strip() for symbol in symbols]

    cleaned = [symbol for symbol in parts if symbol]

    if not cleaned:
        raise ValueError("No valid symbols provided")

    return cleaned


__all__ = ["AlpacaCredentials", "LiveTradingConfig", "parse_symbols"]
