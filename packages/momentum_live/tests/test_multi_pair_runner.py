from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from momentum_live.account_registry import BrokerAccountRegistry
from momentum_live.broker import BrokerAccountManager, BrokerCredentials
from momentum_live.config import AlpacaCredentials, LiveTradingConfig
from momentum_live.multi_pair_runner import MultiPairRunner

_FIRM_ID = "00000000-0000-0000-0000-0000000000ff"


@dataclass
class _FakeAccount:
    id: str


@dataclass
class _FakeTradeAccount:
    cash: float


@dataclass
class _FakeJournal:
    id: str
    status: str = "executed"


class _FakeBrokerClient:
    def __init__(self, *, cash: float = 0.0) -> None:
        self.cash = cash
        self.created: list[str] = []
        self.cancel_orders_calls: list[str] = []
        self.close_positions_calls: list[str] = []
        self.journals: list[dict] = []

    def create_account(self, request):
        new_id = str(uuid.uuid4())
        self.created.append(new_id)
        return _FakeAccount(id=new_id)

    def get_trade_account_by_id(self, account_id):
        return _FakeTradeAccount(cash=self.cash)

    def get_all_positions_for_account(self, account_id):
        return []

    def cancel_orders_for_account(self, account_id):
        self.cancel_orders_calls.append(str(account_id))

    def close_all_positions_for_account(self, account_id, cancel_orders=False):
        self.close_positions_calls.append(str(account_id))

    def create_journal(self, request):
        self.journals.append(
            {
                "from": str(request.from_account),
                "to": str(request.to_account),
                "amount": float(request.amount),
            }
        )
        return _FakeJournal(id=str(uuid.uuid4()))

    def get_journal_by_id(self, journal_id):
        return _FakeJournal(id=journal_id, status="executed")


class _StubAgent:
    num_actions = 6

    def select_action(self, observation):
        return 0


def _make_config(symbols, tmp_path: Path):
    return LiveTradingConfig(
        symbols=list(symbols),
        window_size=2,
        initial_balance=1000.0,
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
        checkpoint_pattern="*.pt",
        tb_log_dir=str(tmp_path / "tb"),
        reset_mode="soft",
        subaccount_registry_path=str(tmp_path / "registry.json"),
    )


class _StubHTTPSession:
    """No-op session: pretends every sub-account is already crypto-enabled."""

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"enabled_assets": ["us_equity", "crypto"], "crypto_status": "APPROVED"}

    def get(self, *args, **kwargs):  # noqa: ARG002
        return self._Resp()

    def patch(self, *args, **kwargs):  # noqa: ARG002
        return self._Resp()


def _make_runner(tmp_path: Path, symbols=("BTC/USD", "ETH/USD")) -> tuple[MultiPairRunner, _FakeBrokerClient]:
    creds = BrokerCredentials(
        api_key="ck",
        secret_key="ss",
        base_url="https://broker-api.sandbox.alpaca.markets",
        funding_account_id=_FIRM_ID,
    )
    fake = _FakeBrokerClient(cash=0.0)
    registry = BrokerAccountRegistry(tmp_path / "registry.json")
    manager = BrokerAccountManager(
        creds,
        registry,
        broker_client=fake,
        http_session=_StubHTTPSession(),
    )
    data_creds = AlpacaCredentials(api_key="dk", secret_key="ds")
    config = _make_config(symbols, tmp_path)
    runner = MultiPairRunner(agent=_StubAgent(), broker=manager, data_credentials=data_creds, config=config)
    return runner, fake


def test_runner_creates_one_subaccount_per_pair(tmp_path: Path) -> None:
    runner, fake = _make_runner(tmp_path)
    assert set(runner.traders.keys()) == {"BTC/USD", "ETH/USD"}
    assert len(fake.created) == 2
    bound_ids = {t.subaccount_client.account_id for t in runner.traders.values()}
    assert bound_ids == set(fake.created)
    for symbol, trader in runner.traders.items():
        assert trader.symbol == symbol


def test_canonical_symbol_handles_alpaca_format(tmp_path: Path) -> None:
    runner, _ = _make_runner(tmp_path, symbols=("BTC/USD",))
    assert runner._canonical_symbol("BTC/USD") == "BTC/USD"
    assert runner._canonical_symbol("BTCUSD") == "BTC/USD"
    assert runner._canonical_symbol("btcusd") == "BTC/USD"
    assert runner._canonical_symbol("DOGE/USD") is None
    assert runner._canonical_symbol(None) is None


def test_reset_all_soft_journals_to_initial_balance(tmp_path: Path) -> None:
    runner, fake = _make_runner(tmp_path)
    summaries = runner.reset_all("soft")

    assert len(summaries) == 2
    # both accounts should have been topped up to 1000 from cash=0
    amounts = sorted(j["amount"] for j in fake.journals)
    assert amounts == [1000.0, 1000.0]
    bound_ids = {t.subaccount_client.account_id for t in runner.traders.values()}
    assert sorted(fake.cancel_orders_calls) == sorted(bound_ids)
    assert sorted(fake.close_positions_calls) == sorted(bound_ids)

    # marker line written to live_trades.jsonl
    marker_path = Path(runner.config.tb_log_dir) / "live_trades.jsonl"
    assert marker_path.exists()
    last = marker_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(last)
    assert payload["marker"] == "reset"
    assert payload["initial_balance"] == 1000.0


def test_reset_all_none_is_noop(tmp_path: Path) -> None:
    runner, fake = _make_runner(tmp_path)
    assert runner.reset_all("none") == []
    assert fake.journals == []


def test_reset_all_hard_not_implemented(tmp_path: Path) -> None:
    runner, _ = _make_runner(tmp_path)
    with pytest.raises(NotImplementedError):
        runner.reset_all("hard")


def test_reset_all_unknown_mode(tmp_path: Path) -> None:
    runner, _ = _make_runner(tmp_path)
    with pytest.raises(ValueError):
        runner.reset_all("nuke")
