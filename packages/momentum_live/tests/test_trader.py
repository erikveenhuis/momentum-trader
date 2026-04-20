from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from momentum_live.agent_loader import find_best_checkpoint
from momentum_live.config import LiveTradingConfig, parse_symbols
from momentum_live.trader import BarData, LiveFeatureNormalizer, MomentumLiveTrader


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
        atime = mtime = (datetime.now() - timedelta(days=len(paths) - idx)).timestamp()
        Path(path).touch()
        os.utime(path, (atime, mtime))

    best = find_best_checkpoint(tmp_path)
    assert best.name == "checkpoint_trainer_best_20250102_ep101_score_0.35.pt"


def test_live_feature_normalizer_generates_window():
    normalizer = LiveFeatureNormalizer(window_size=3)
    timestamp = datetime.now(UTC)
    for price in (100.0, 101.0, 102.0):
        normalizer.update(BarData("BTC/USD", price, price, price, price, 10.0, timestamp))
    window = normalizer.window()
    assert window.shape == (3, 12)
    assert np.isfinite(window).all()


class _StubAgent:
    num_actions = 6

    def __init__(self, action_sequence):
        self._actions = list(action_sequence)
        self._idx = 0

    def select_action(self, observation):
        action = self._actions[self._idx]
        self._idx += 1
        return action


class _CapturingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def tags(self) -> set[str]:
        return {tag for tag, _, _ in self.scalars}


class _StubAgentWithQ(_StubAgent):
    def __init__(self, action_sequence, q_values: np.ndarray):
        super().__init__(action_sequence)
        self._q = np.asarray(q_values, dtype=np.float32)
        self._last_select_q_values = self._q.copy()

    def select_action_with_provenance(self, observation):
        return self.select_action(observation), True


def _make_live_config(tb_log_dir: str | None = None) -> LiveTradingConfig:
    return LiveTradingConfig(
        symbols=["BTC/USD"],
        window_size=2,
        initial_balance=100.0,
        transaction_fee=0.0,
        reward_scale=1.0,
        invalid_action_penalty=-0.05,
        drawdown_penalty_lambda=0.0,
        slippage_bps=0.0,
        opportunity_cost_lambda=0.0,
        benchmark_allocation_frac=0.5,
        min_rebalance_pct=0.02,
        min_trade_value=1.0,
        models_dir="models",
        checkpoint_pattern="checkpoint_trainer_best_*.pt",
        tb_log_dir=tb_log_dir,
    )


class _StubSubaccount:
    """Minimal BrokerSubAccountClient stand-in: just exposes a configurable cash value."""

    def __init__(self, cash: float, account_id: str = "acc-test") -> None:
        self.cash = cash
        self.account_id = account_id

    def get_account_cash(self) -> float:
        return float(self.cash)


def test_trader_uses_subaccount_cash_for_observation():
    config = _make_live_config()
    agent = _StubAgentWithQ(action_sequence=[0], q_values=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    trader = MomentumLiveTrader(
        agent=agent,
        symbol="BTC/USD",
        config=config,
        subaccount_client=_StubSubaccount(cash=42.0),
    )
    assert trader.get_account_cash() == 42.0


def test_trader_emits_action_rate_and_q_scalars_per_bar():
    writer = _CapturingWriter()
    agent = _StubAgentWithQ(
        action_sequence=[1, 2],
        q_values=np.array([0.1, 0.4, 0.7, 0.0, 0.0, 0.0]),
    )
    config = _make_live_config()
    trader = MomentumLiveTrader(agent=agent, symbol="BTC/USD", config=config, writer=writer)

    timestamp = datetime.now(UTC)
    trader.process_bar(BarData("BTC/USD", 100.0, 100.0, 100.0, 100.0, 5.0, timestamp))
    trader.process_bar(BarData("BTC/USD", 101.0, 101.0, 101.0, 101.0, 5.0, timestamp))

    tags = writer.tags()
    assert "Live/Q/Mean" in tags
    assert "Live/Q/MaxAcrossActions" in tags
    assert "Live/Q/ActionMargin" in tags
    assert "Live/Q/Selected" in tags
    assert "Live/Action Rate/2" in tags
    assert "Live/PortfolioValue/BTC/USD" in tags
    assert "Live/Position/BTC/USD" in tags

    # Only the second bar reaches selection (window warms up first), so action 1
    # holds 100% of the cumulative share.
    rate_1_values = [v for tag, v, _ in writer.scalars if tag == "Live/Action Rate/1"]
    assert rate_1_values, "Expected at least one Live/Action Rate/1 emission"
    assert rate_1_values[-1] == 1.0


def test_trader_emits_trade_metrics_when_position_closes(tmp_path):
    writer = _CapturingWriter()
    log_dir = tmp_path / "live_tb"
    config = _make_live_config(tb_log_dir=str(log_dir))
    agent = _StubAgentWithQ(
        action_sequence=[2, 0, 0],
        q_values=np.array([0.0, 0.1, 0.5, 0.2, 0.0, 0.0]),
    )
    trader = MomentumLiveTrader(agent=agent, symbol="BTC/USD", config=config, writer=writer)
    trader._trades_jsonl_path = log_dir / "live_trades.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    bars = [
        BarData("BTC/USD", 100.0, 100.0, 100.0, 100.0, 5.0, timestamp),
        BarData("BTC/USD", 101.0, 101.0, 101.0, 101.0, 5.0, timestamp),
        BarData("BTC/USD", 102.0, 102.0, 102.0, 102.0, 5.0, timestamp),
    ]
    for bar in bars:
        trader.process_bar(bar)

    pre_position = trader.portfolio_state.position
    trader.portfolio_state = type(trader.portfolio_state)(
        balance=trader.portfolio_state.balance,
        position=0.0,
        position_price=trader.portfolio_state.position_price,
        total_transaction_cost=trader.portfolio_state.total_transaction_cost,
    )
    trader._maybe_emit_trade_close(pre_position=max(pre_position, 1e-3))

    trade_tags = {tag for tag, _, _ in writer.scalars if tag.startswith("Live/Trade/")}
    assert trade_tags
    assert any(tag.endswith("Count/BTC/USD") for tag in trade_tags)


def test_trader_writer_is_optional():
    config = _make_live_config(tb_log_dir=None)
    agent = _StubAgentWithQ(action_sequence=[1, 1], q_values=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))
    trader = MomentumLiveTrader(agent=agent, symbol="BTC/USD", config=config)
    assert trader.writer is None
    timestamp = datetime.now(UTC)
    trader.process_bar(BarData("BTC/USD", 100.0, 100.0, 100.0, 100.0, 5.0, timestamp))
    assert trader.process_bar(BarData("BTC/USD", 101.0, 101.0, 101.0, 101.0, 5.0, timestamp)) is not None


def test_process_bar_returns_decision():
    config = _make_live_config()
    agent = _StubAgent([1])
    trader = MomentumLiveTrader(agent=agent, symbol="BTC/USD", config=config)

    timestamp = datetime.now(UTC)
    bar1 = BarData("BTC/USD", 100.0, 100.0, 100.0, 100.0, 5.0, timestamp)
    bar2 = BarData("BTC/USD", 101.0, 101.0, 101.0, 101.0, 5.0, timestamp)

    assert trader.process_bar(bar1) is None

    decision = trader.process_bar(bar2)
    assert decision is not None
    assert decision["target_allocation"] == 0.2
    assert decision["valid"] is True
    assert decision["symbol"] == "BTC/USD"


def test_process_bar_ignores_other_symbols():
    config = _make_live_config()
    agent = _StubAgent([1])
    trader = MomentumLiveTrader(agent=agent, symbol="BTC/USD", config=config)
    timestamp = datetime.now(UTC)
    assert trader.process_bar(BarData("ETH/USD", 1.0, 1.0, 1.0, 1.0, 1.0, timestamp)) is None
