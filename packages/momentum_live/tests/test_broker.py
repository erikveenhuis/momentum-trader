from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from momentum_live.account_registry import BrokerAccountRegistry
from momentum_live.broker import BrokerAccountManager, BrokerCredentials

_FIRM_ID = "00000000-0000-0000-0000-000000000001"


@dataclass
class _FakeJournal:
    id: str
    status: str = "executed"


@dataclass
class _FakeAccount:
    id: str
    cash: float = 0.0


@dataclass
class _FakeTradeAccount:
    cash: float


class _FakeHTTPResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


class _FakeHTTPSession:
    """Tracks GET/PATCH calls against ``/v1/accounts/{id}`` for ensure_crypto_enabled."""

    def __init__(self, *, crypto_enabled: bool = True, crypto_status: str = "APPROVED") -> None:
        self.crypto_enabled = crypto_enabled
        self.crypto_status = crypto_status
        self.get_calls: list[str] = []
        self.patch_calls: list[tuple[str, dict]] = []

    def get(self, url: str, auth=None, timeout=None):  # noqa: ARG002 - test double
        self.get_calls.append(url)
        enabled = ["us_equity", "crypto"] if self.crypto_enabled else ["us_equity"]
        return _FakeHTTPResponse(
            200,
            payload={
                "enabled_assets": enabled,
                "crypto_status": self.crypto_status,
            },
        )

    def patch(self, url: str, auth=None, json=None, timeout=None):  # noqa: ARG002 - test double
        self.patch_calls.append((url, dict(json or {})))
        self.crypto_enabled = True
        self.crypto_status = "APPROVED"
        return _FakeHTTPResponse(200, payload={"ok": True})


class _FakeBrokerClient:
    def __init__(self, *, initial_cash: float = 0.0) -> None:
        self.cash = initial_cash
        self.journals: list[dict] = []
        self.created_accounts: list[str] = []
        self.cancel_orders_calls: list[str] = []
        self.close_positions_calls: list[str] = []
        self.positions_by_account: dict[str, list] = {}

    def create_account(self, request):
        new_id = str(uuid.uuid4())
        self.created_accounts.append(new_id)
        return _FakeAccount(id=new_id)

    # ----- account state -----
    def get_trade_account_by_id(self, account_id):
        return _FakeTradeAccount(cash=self.cash)

    def get_all_positions_for_account(self, account_id):
        return self.positions_by_account.get(str(account_id), [])

    def cancel_orders_for_account(self, account_id):
        self.cancel_orders_calls.append(str(account_id))

    def close_all_positions_for_account(self, account_id, cancel_orders=False):
        self.close_positions_calls.append(str(account_id))
        self.positions_by_account[str(account_id)] = []

    def create_journal(self, request):
        amount = float(request.amount)
        self.journals.append(
            {
                "from": str(request.from_account),
                "to": str(request.to_account),
                "amount": amount,
                "description": request.description,
            }
        )
        return _FakeJournal(id=str(uuid.uuid4()))

    def get_journal_by_id(self, journal_id):
        return _FakeJournal(id=journal_id, status="executed")


def _make_manager(
    tmp_path: Path,
    *,
    cash: float = 0.0,
    funding_account_id: str = _FIRM_ID,
    http_session: _FakeHTTPSession | None = None,
) -> tuple[BrokerAccountManager, _FakeBrokerClient, _FakeHTTPSession]:
    creds = BrokerCredentials(
        api_key="ck",
        secret_key="ss",
        base_url="https://broker-api.sandbox.alpaca.markets",
        funding_account_id=funding_account_id,
    )
    registry = BrokerAccountRegistry(tmp_path / "registry.json")
    fake_client = _FakeBrokerClient(initial_cash=cash)
    session = http_session if http_session is not None else _FakeHTTPSession(crypto_enabled=True)
    manager = BrokerAccountManager(creds, registry, broker_client=fake_client, http_session=session)
    return manager, fake_client, session


def test_credentials_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_BROKER_API_KEY", "K")
    monkeypatch.setenv("ALPACA_BROKER_API_SECRET", "S")
    monkeypatch.setenv("ALPACA_BROKER_BASE_URL", "https://broker-api.sandbox.alpaca.markets")
    monkeypatch.setenv("ALPACA_BROKER_ACCOUNT_ID", "FIRM")

    creds = BrokerCredentials.from_environment()
    assert creds.api_key == "K"
    assert creds.is_sandbox is True
    assert creds.funding_account_id == "FIRM"


def test_credentials_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_BROKER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_BROKER_API_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_BROKER_ACCOUNT_ID", raising=False)
    with pytest.raises(OSError, match="Missing Broker API env vars"):
        BrokerCredentials.from_environment()


def test_credentials_firm_account_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Firm/funding account id is optional — only required for JNLC flows."""
    monkeypatch.setenv("ALPACA_BROKER_API_KEY", "K")
    monkeypatch.setenv("ALPACA_BROKER_API_SECRET", "S")
    monkeypatch.delenv("ALPACA_BROKER_ACCOUNT_ID", raising=False)

    creds = BrokerCredentials.from_environment()
    assert creds.api_key == "K"
    assert creds.funding_account_id == ""
    assert creds.has_firm_account is False


def test_ensure_subaccount_creates_then_caches(tmp_path: Path) -> None:
    manager, fake, _ = _make_manager(tmp_path)
    entry1 = manager.ensure_subaccount("BTC/USD")
    assert uuid.UUID(entry1.account_id)
    entry2 = manager.ensure_subaccount("BTC/USD")
    assert entry2 == entry1
    assert len(fake.created_accounts) == 1
    payload = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert "BTC/USD" in payload


def test_ensure_subaccount_enables_crypto_on_new_account(tmp_path: Path) -> None:
    session = _FakeHTTPSession(crypto_enabled=False, crypto_status="INACTIVE")
    manager, _fake, _ = _make_manager(tmp_path, http_session=session)

    entry = manager.ensure_subaccount("BTC/USD")

    assert session.get_calls, "expected a GET to inspect enabled_assets"
    assert session.patch_calls, "expected a PATCH to enable crypto"
    patched_url, patched_body = session.patch_calls[0]
    assert entry.account_id in patched_url
    assert patched_body == {"enabled_assets": ["us_equity", "crypto"]}


def test_ensure_crypto_enabled_is_noop_when_already_enabled(tmp_path: Path) -> None:
    session = _FakeHTTPSession(crypto_enabled=True, crypto_status="APPROVED")
    manager, _fake, _ = _make_manager(tmp_path, http_session=session)

    account_id = str(uuid.uuid4())
    patched = manager.ensure_crypto_enabled(account_id)
    assert patched is False
    assert session.get_calls == [f"https://broker-api.sandbox.alpaca.markets/v1/accounts/{account_id}"]
    assert session.patch_calls == []


def test_adopt_subaccount_upserts_registry(tmp_path: Path) -> None:
    session = _FakeHTTPSession(crypto_enabled=False, crypto_status="INACTIVE")
    manager, fake, _ = _make_manager(tmp_path, http_session=session)
    account_id = "a7880383-9924-446b-8be7-3d8b3bcdf68f"

    entry = manager.adopt_subaccount("BTC/USD", account_id)

    assert entry.account_id == account_id
    assert entry.label == "adopted-BTCUSD"
    assert manager.registry.get("BTC/USD") == entry
    assert fake.created_accounts == []  # no create_account call
    assert session.patch_calls, "adopt should also trigger crypto enablement"


def test_adopt_subaccount_rejects_non_uuid(tmp_path: Path) -> None:
    manager, _fake, _ = _make_manager(tmp_path)
    with pytest.raises(ValueError, match="not a valid UUID"):
        manager.adopt_subaccount("BTC/USD", "not-a-uuid")


def test_adopt_subaccount_overrides_existing_entry(tmp_path: Path) -> None:
    manager, _fake, _ = _make_manager(tmp_path)
    first = manager.ensure_subaccount("BTC/USD")
    new_id = "a7880383-9924-446b-8be7-3d8b3bcdf68f"
    assert first.account_id != new_id

    manager.adopt_subaccount("BTC/USD", new_id)
    assert manager.registry.get("BTC/USD").account_id == new_id


def test_journal_cash_directions(tmp_path: Path) -> None:
    manager, fake, _ = _make_manager(tmp_path, cash=0.0)
    entry = manager.ensure_subaccount("BTC/USD")

    manager.journal_cash(entry.account_id, 1000.0, direction="to_sub", wait=False)
    assert fake.journals[-1]["from"] == _FIRM_ID
    assert fake.journals[-1]["to"] == entry.account_id
    assert fake.journals[-1]["amount"] == 1000.0

    manager.journal_cash(entry.account_id, 250.0, direction="from_sub", wait=False)
    assert fake.journals[-1]["from"] == entry.account_id
    assert fake.journals[-1]["to"] == _FIRM_ID
    assert fake.journals[-1]["amount"] == 250.0


def test_journal_cash_rejects_invalid_args(tmp_path: Path) -> None:
    manager, _, _ = _make_manager(tmp_path)
    entry = manager.ensure_subaccount("BTC/USD")
    with pytest.raises(ValueError):
        manager.journal_cash(entry.account_id, -1.0, direction="to_sub", wait=False)
    with pytest.raises(ValueError):
        manager.journal_cash(entry.account_id, 1.0, direction="sideways", wait=False)


def test_journal_cash_without_firm_account_raises(tmp_path: Path) -> None:
    manager, _, _ = _make_manager(tmp_path, funding_account_id="")
    entry = manager.ensure_subaccount("BTC/USD")
    with pytest.raises(RuntimeError, match="ALPACA_BROKER_ACCOUNT_ID"):
        manager.journal_cash(entry.account_id, 100.0, direction="to_sub", wait=False)


def test_reset_subaccount_topup(tmp_path: Path) -> None:
    manager, fake, _ = _make_manager(tmp_path, cash=200.0)
    entry = manager.ensure_subaccount("BTC/USD")

    summary = manager.reset_subaccount(entry.account_id, target_balance=1000.0, wait_timeout=0.0)

    assert entry.account_id in fake.cancel_orders_calls
    assert entry.account_id in fake.close_positions_calls
    assert summary["cash_before"] == 200.0
    assert any(j["amount"] == 800.0 and j["to"] == entry.account_id for j in fake.journals)


def test_reset_subaccount_skim(tmp_path: Path) -> None:
    manager, fake, _ = _make_manager(tmp_path, cash=1500.0)
    entry = manager.ensure_subaccount("BTC/USD")

    manager.reset_subaccount(entry.account_id, target_balance=1000.0, wait_timeout=0.0)
    assert any(j["amount"] == 500.0 and j["from"] == entry.account_id and j["to"] == _FIRM_ID for j in fake.journals)


def test_reset_subaccount_no_journal_when_within_threshold(tmp_path: Path) -> None:
    manager, fake, _ = _make_manager(tmp_path, cash=1000.005)
    entry = manager.ensure_subaccount("BTC/USD")

    manager.reset_subaccount(entry.account_id, target_balance=1000.0, wait_timeout=0.0)
    assert fake.journals == []
