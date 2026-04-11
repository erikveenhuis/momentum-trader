from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from momentum_live.agent_loader import find_best_checkpoint
from momentum_live.config import LiveTradingConfig, parse_symbols
from momentum_live.trader import BarData, LiveFeatureNormalizer, MomentumLiveTrader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_parse_symbols_handles_comma_and_space():
    assert parse_symbols("BTC/USD, ETH/USD") == ["BTC/USD", "ETH/USD"]
    assert parse_symbols(["SOL/USD", "XRP/USD"]) == ["SOL/USD", "XRP/USD"]


def test_find_best_checkpoint(tmp_path: Path):
    paths = [
        tmp_path / "checkpoint_trainer_best_20250101_ep100_score_0.20.pt",
        tmp_path / "checkpoint_trainer_best_20250102_ep101_score_0.35.pt",
        tmp_path / "checkpoint_trainer_best_20250103_ep102_score_0.10.pt",
    ]

    for idx, path in enumerate(paths):
        path.write_bytes(b"dummy")
        # Stagger modification times to ensure tie breaking by mtime if needed
        atime = mtime = (datetime.now() - timedelta(days=len(paths) - idx)).timestamp()
        Path(path).touch()
        os.utime(path, (atime, mtime))

    best = find_best_checkpoint(tmp_path)
    assert best.name == "checkpoint_trainer_best_20250102_ep101_score_0.35.pt"


def test_live_feature_normalizer_generates_window():
    normalizer = LiveFeatureNormalizer(window_size=3)

    timestamp = datetime.now(UTC)
    for price in (100.0, 101.0, 102.0):
        bar = BarData(
            symbol="BTC/USD",
            open=price,
            high=price,
            low=price,
            close=price,
            volume=10.0,
            timestamp=timestamp,
        )
        normalizer.update(bar)

    window = normalizer.window()
    assert window.shape == (3, 12)
    assert np.isfinite(window).all()


class _StubAgent:
    def __init__(self, action_sequence):
        self._actions = iter(action_sequence)

    def select_action(self, observation):
        return next(self._actions)


def test_momentum_live_trader_process_bar_returns_decision():
    config = LiveTradingConfig(
        symbols=["BTC/USD"],
        window_size=2,
        initial_balance=100.0,
        transaction_fee=0.0,
        reward_scale=1.0,
        invalid_action_penalty=-0.05,
        drawdown_penalty_lambda=0.0,
        slippage_bps=0.0,
        opportunity_cost_lambda=0.0,
        min_rebalance_pct=0.02,
        min_trade_value=1.0,
        models_dir="models",
        checkpoint_pattern="checkpoint_trainer_best_*.pt",
    )

    agent = _StubAgent([1])  # Action 1 = target 20%
    trader = MomentumLiveTrader(agent=agent, config=config)

    timestamp = datetime.now(UTC)
    bar1 = BarData("BTC/USD", 100.0, 100.0, 100.0, 100.0, 5.0, timestamp)
    bar2 = BarData("BTC/USD", 101.0, 101.0, 101.0, 101.0, 5.0, timestamp)

    assert trader.process_bar(bar1) is None

    decision = trader.process_bar(bar2)
    assert decision is not None
    assert decision["target_allocation"] == 0.2
    assert decision["valid"] is True
