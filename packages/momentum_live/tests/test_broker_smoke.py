"""End-to-end smoke test for the Broker sub-account crypto flow.

Validates that a pre-existing sandbox sub-account can be adopted, crypto-enabled,
and used to round-trip a small BTC/USD market order. Optionally loads the
trained agent and runs one ``process_bar`` call against real Alpaca minute
bars, which exercises the full adopt + warmup + inference pipeline without
depending on firm-account funding or JNLC.

Prerequisites:
    # .env (loaded by scripts/env-paper.sh)
    ALPACA_API_KEY=...
    ALPACA_API_SECRET=...
    ALPACA_BROKER_API_KEY=...
    ALPACA_BROKER_API_SECRET=...
    # ALPACA_BROKER_ACCOUNT_ID can stay blank — we're in --reset-mode none land.

Usage:
    source scripts/env-paper.sh
    # Required: the sub-account id you adopted via the Brokerdash tutorial /
    # manual bootstrap, already crypto-enabled and funded with some USD.
    export MOMENTUM_LIVE_SMOKE_ACCOUNT_ID=a7880383-9924-446b-8be7-3d8b3bcdf68f

    # Adopt + crypto enablement + account inspection (safe, read-mostly):
    pytest packages/momentum_live/tests/test_broker_smoke.py -m integration -v

    # Round-trip a ~$10 BTC market order on the sub-account (default notional):
    MOMENTUM_LIVE_SMOKE_TRADE=1 pytest packages/momentum_live/tests/test_broker_smoke.py \
        -m integration -v -k roundtrip

    # Full agent + live minute-bar loop:
    MOMENTUM_LIVE_SMOKE_AGENT=1 pytest packages/momentum_live/tests/test_broker_smoke.py \
        -m integration -v -k agent
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_BROKER_CREDS_MISSING = not (os.getenv("ALPACA_BROKER_API_KEY") and os.getenv("ALPACA_BROKER_API_SECRET"))
_DATA_CREDS_MISSING = not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET"))
_SMOKE_ACCOUNT_ID = os.getenv("MOMENTUM_LIVE_SMOKE_ACCOUNT_ID", "").strip()
# Default to $10 because some Alpaca crypto venues reject sub-$10 notionals on
# BTC/USD; the existing paper-account round-trip in test_paper_trading.py uses
# the same $10 floor for the same reason. Override with
# ``MOMENTUM_LIVE_SMOKE_TRADE_NOTIONAL=<usd>`` if you want to spend less / more.
_TRADE_NOTIONAL = float(os.getenv("MOMENTUM_LIVE_SMOKE_TRADE_NOTIONAL", "10.0"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _BROKER_CREDS_MISSING,
        reason="ALPACA_BROKER_API_KEY / ALPACA_BROKER_API_SECRET must be set (source scripts/env-paper.sh).",
    ),
    pytest.mark.skipif(
        not _SMOKE_ACCOUNT_ID,
        reason="MOMENTUM_LIVE_SMOKE_ACCOUNT_ID must be set to the adopted sub-account UUID.",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_account_id() -> str:
    try:
        uuid.UUID(_SMOKE_ACCOUNT_ID)
    except ValueError as exc:
        pytest.skip(f"MOMENTUM_LIVE_SMOKE_ACCOUNT_ID is not a valid UUID: {exc}")
    return _SMOKE_ACCOUNT_ID


@pytest.fixture(scope="module")
def broker_manager(tmp_path_factory: pytest.TempPathFactory):
    """A real ``BrokerAccountManager`` pointed at the sandbox, with a scratch registry."""
    from momentum_live.account_registry import BrokerAccountRegistry
    from momentum_live.broker import BrokerAccountManager, BrokerCredentials

    registry_dir = tmp_path_factory.mktemp("smoke_registry")
    credentials = BrokerCredentials.from_environment()
    registry = BrokerAccountRegistry(registry_dir / "broker_subaccounts.json")
    return BrokerAccountManager(credentials, registry)


@pytest.fixture(scope="module")
def adopted_entry(broker_manager, smoke_account_id: str):
    """Adopt the sub-account into the scratch registry and ensure it's crypto-enabled."""
    entry = broker_manager.adopt_subaccount("BTC/USD", smoke_account_id, label="smoke-BTCUSD")
    assert entry.account_id == smoke_account_id
    return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _account_status(account) -> str:
    status = getattr(account, "status", None)
    if status is None:
        return ""
    return (status.value if hasattr(status, "value") else str(status)).upper()


def _resolve_checkpoint_path() -> Path | None:
    """Find a usable checkpoint, falling back from ``best_*`` to ``latest_*``.

    - ``MOMENTUM_LIVE_CHECKPOINT`` (explicit file path) wins if set.
    - ``MOMENTUM_LIVE_CHECKPOINT_PATTERN`` overrides the glob pattern.
    - Otherwise we try ``checkpoint_trainer_best_*.pt`` first and fall back to
      ``checkpoint_trainer_latest_*.pt`` (matches what training actually
      writes mid-run before a "best" snapshot is selected).
    """
    explicit = os.getenv("MOMENTUM_LIVE_CHECKPOINT")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None

    from momentum_live.agent_loader import find_best_checkpoint

    models_dir = os.getenv("MOMENTUM_LIVE_MODELS_DIR")
    candidate = Path(models_dir).expanduser().resolve() if models_dir else _REPO_ROOT / "models"
    if not candidate.is_dir():
        return None

    patterns: list[str] = []
    override = os.getenv("MOMENTUM_LIVE_CHECKPOINT_PATTERN")
    if override:
        patterns.append(override)
    else:
        patterns.extend(["checkpoint_trainer_best_*.pt", "checkpoint_trainer_latest_*.pt"])

    for pattern in patterns:
        try:
            return find_best_checkpoint(candidate, pattern=pattern)
        except FileNotFoundError:
            continue
    return None


def _fetch_btc_minute_bars(lookback_minutes: int) -> list:
    """Pull recent 1-minute BTC/USD bars from Alpaca (same timeframe as the live stream)."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from momentum_live.trader import BarData

    client = CryptoHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_API_SECRET"],
    )
    end = datetime.now(UTC)
    start = end - timedelta(minutes=lookback_minutes)

    try:
        request = CryptoBarsRequest(
            symbol_or_symbols="BTC/USD",
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=lookback_minutes,
        )
    except TypeError:
        request = CryptoBarsRequest(
            symbol_or_symbols="BTC/USD",
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )

    response = client.get_crypto_bars(request)
    out: list = []

    def _append(raw) -> None:
        out.append(
            BarData(
                symbol="BTC/USD",
                open=float(raw.open),
                high=float(raw.high),
                low=float(raw.low),
                close=float(raw.close),
                volume=float(getattr(raw, "volume", 0.0)),
                timestamp=getattr(raw, "timestamp", datetime.now(UTC)),
            )
        )

    data_attr = getattr(response, "data", None)
    if isinstance(data_attr, dict):
        for _symbol, raw_bars in data_attr.items():
            for raw_bar in raw_bars or []:
                _append(raw_bar)
    elif isinstance(data_attr, (list, tuple)):
        for raw_bar in data_attr:
            _append(raw_bar)

    out.sort(key=lambda b: b.timestamp)
    return out


def _wait_for_order(broker_client, account_id: str, order_id, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    last_status = None
    while time.monotonic() < deadline:
        order = broker_client.get_order_for_account_by_id(account_id, order_id)
        status = getattr(order, "status", None)
        status_str = (status.value if hasattr(status, "value") else str(status) or "").lower()
        if status_str != last_status:
            last_status = status_str
        if status_str in {"filled", "canceled", "rejected", "expired"}:
            return order
        time.sleep(0.5)
    return broker_client.get_order_for_account_by_id(account_id, order_id)


# ---------------------------------------------------------------------------
# 1. Adopt + crypto enablement
# ---------------------------------------------------------------------------


class TestAdoptAndEnableCrypto:
    """Verify the adopt flow lands a crypto-enabled sub-account in the registry."""

    def test_adopt_upserts_registry(self, broker_manager, adopted_entry) -> None:
        stored = broker_manager.registry.get("BTC/USD")
        assert stored is not None
        assert stored.account_id == adopted_entry.account_id
        assert stored.label == "smoke-BTCUSD"

    def test_account_is_active_and_crypto_metadata_present(self, broker_manager, adopted_entry) -> None:
        """Inspect raw ``GET /v1/accounts/{id}`` (alpaca-py drops ``enabled_assets``).

        We intentionally bypass the SDK because ``alpaca-py``'s ``Account`` model
        omits ``enabled_assets`` (returns ``None`` even when the API populates it).
        """
        import requests

        base = broker_manager._credentials.base_url.rstrip("/")
        auth = (
            broker_manager._credentials.api_key,
            broker_manager._credentials.secret_key,
        )
        resp = requests.get(f"{base}/v1/accounts/{adopted_entry.account_id}", auth=auth, timeout=10)
        assert resp.status_code == 200, f"GET account failed: {resp.status_code} {resp.text[:200]}"
        payload = resp.json()

        status = str(payload.get("status", "")).upper()
        assert status in {"ACTIVE", "APPROVED"}, f"Sub-account status={status!r}"

        enabled_assets = [str(x).lower() for x in (payload.get("enabled_assets") or [])]
        crypto_status = str(payload.get("crypto_status") or "").upper()
        assert "crypto" in enabled_assets, (
            f"sub-account {adopted_entry.account_id} enabled_assets={enabled_assets}; ensure_crypto_enabled() should have PATCHed it."
        )
        assert crypto_status in {"APPROVED", "ACTIVE"}, f"crypto_status={crypto_status!r}; expected APPROVED or ACTIVE"

    def test_ensure_crypto_enabled_is_idempotent(self, broker_manager, adopted_entry) -> None:
        patched = broker_manager.ensure_crypto_enabled(adopted_entry.account_id)
        assert patched is False, "crypto should already be enabled after adopt_subaccount"

    def test_trading_engine_actually_accepts_crypto_orders(self, broker_manager, adopted_entry) -> None:
        """Surface the ``PATCH != trading-engine-enabled`` Alpaca quirk early.

        Submits a deliberately oversized BTC/USD market order ($1B notional). A
        well-provisioned account responds with HTTP 403 ``insufficient balance
        for USD`` (the trading subsystem accepted the crypto order but ran out
        of buying power). A misprovisioned account that was crypto-enabled via
        ``PATCH /v1/accounts/{id}`` *after* creation responds with HTTP 422
        ``crypto orders not allowed for account`` even though
        ``enabled_assets`` and ``crypto_status`` look correct. The latter
        means the account must be re-created with ``enabled_assets=[us_equity,
        crypto]`` from scratch (which our ``ensure_subaccount`` flow does).
        """
        import requests

        base = broker_manager._credentials.base_url.rstrip("/")
        auth = (
            broker_manager._credentials.api_key,
            broker_manager._credentials.secret_key,
        )
        body = {
            "symbol": "BTC/USD",
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "notional": "1000000000",
            "client_order_id": f"smoke-probe-{uuid.uuid4().hex[:12]}",
        }
        resp = requests.post(
            f"{base}/v1/trading/accounts/{adopted_entry.account_id}/orders",
            auth=auth,
            json=body,
            timeout=10,
        )
        if resp.status_code == 422 and "crypto orders not allowed" in resp.text.lower():
            pytest.fail(
                "Adopted sub-account "
                f"{adopted_entry.account_id} reports ``enabled_assets=[us_equity, crypto]`` "
                "but the trading engine rejects crypto orders. This account was crypto-enabled "
                "via PATCH /v1/accounts/{id} *after* creation, which only updates metadata. "
                "Alpaca only enables crypto trading for accounts whose ``enabled_assets`` were "
                "set at creation time. Create a fresh sub-account via "
                "``BrokerAccountManager.ensure_subaccount('BTC/USD')`` (which sets "
                "``enabled_assets=[us_equity, crypto]`` on CreateAccountRequest), fund it via "
                "ACH, then point MOMENTUM_LIVE_SMOKE_ACCOUNT_ID at the new account."
            )
        assert resp.status_code in {403, 422}, f"Unexpected probe response: {resp.status_code} {resp.text[:200]}"
        text = resp.text.lower()
        assert ("insufficient balance" in text) or ("buying power" in text), (
            f"Probe order should have been rejected for cash, not for permissions: {resp.status_code} {resp.text[:200]}"
        )


# ---------------------------------------------------------------------------
# 2. Tiny BTC/USD round-trip on the sub-account
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("MOMENTUM_LIVE_SMOKE_TRADE") != "1",
    reason="set MOMENTUM_LIVE_SMOKE_TRADE=1 to place a real ~$10 BTC/USD order on the sub-account.",
)
class TestSubAccountCryptoRoundtrip:
    """Buy ~$1 of BTC on the sub-account, confirm the fill, then close the position."""

    def test_buy_and_close_via_subaccount_client(self, broker_manager, adopted_entry) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        from momentum_live.subaccount_client import BrokerSubAccountClient

        client = BrokerSubAccountClient(broker_manager.client, adopted_entry.account_id)

        starting_cash = client.get_account_cash()
        assert starting_cash >= _TRADE_NOTIONAL + 0.5, (
            f"Sub-account cash ${starting_cash:.2f} is below the ${_TRADE_NOTIONAL:.2f} "
            "smoke-trade notional + buffer. Fund the account or lower "
            "MOMENTUM_LIVE_SMOKE_TRADE_NOTIONAL."
        )

        buy_request = MarketOrderRequest(
            symbol="BTC/USD",
            notional=_TRADE_NOTIONAL,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            client_order_id=f"smoke-buy-{uuid.uuid4().hex[:12]}",
        )
        buy_order = client.submit_order(buy_request)
        assert getattr(buy_order, "id", None) is not None

        filled_buy = _wait_for_order(broker_manager.client, adopted_entry.account_id, buy_order.id)
        filled_status = getattr(filled_buy, "status", None)
        filled_status_str = (filled_status.value if hasattr(filled_status, "value") else str(filled_status) or "").lower()
        assert filled_status_str == "filled", f"Buy did not fill (status={filled_status_str!r})"
        assert float(getattr(filled_buy, "filled_qty", 0.0) or 0.0) > 0.0

        positions = client.get_all_positions()
        btc_position = None
        for pos in positions:
            pos_symbol = getattr(pos, "symbol", "")
            if pos_symbol in {"BTC/USD", "BTCUSD"}:
                btc_position = pos
                break
        assert btc_position is not None, "BTC position not reflected after buy"

        sell_qty = float(btc_position.qty)
        sell_request = MarketOrderRequest(
            symbol="BTC/USD",
            qty=sell_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            client_order_id=f"smoke-sell-{uuid.uuid4().hex[:12]}",
        )
        sell_order = client.submit_order(sell_request)
        filled_sell = _wait_for_order(broker_manager.client, adopted_entry.account_id, sell_order.id)
        sell_status = getattr(filled_sell, "status", None)
        sell_status_str = (sell_status.value if hasattr(sell_status, "value") else str(sell_status) or "").lower()
        assert sell_status_str == "filled", f"Sell did not fill (status={sell_status_str!r})"


# ---------------------------------------------------------------------------
# 3. Full agent loop against live 1-minute bars (adopt + warmup + inference)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("MOMENTUM_LIVE_SMOKE_AGENT") != "1",
    reason="set MOMENTUM_LIVE_SMOKE_AGENT=1 to load the trained agent and run one process_bar.",
)
@pytest.mark.skipif(
    _DATA_CREDS_MISSING,
    reason="ALPACA_API_KEY / ALPACA_API_SECRET (data API) must be set for historical bar warmup.",
)
class TestAgentSmokeLoop:
    """Load the trained agent and feed it one real minute bar through the full trader."""

    def test_agent_decides_on_adopted_subaccount(self, broker_manager, adopted_entry) -> None:
        from momentum_live.agent_loader import load_agent_from_checkpoint
        from momentum_live.config import LiveTradingConfig
        from momentum_live.subaccount_client import BrokerSubAccountClient
        from momentum_live.trader import MomentumLiveTrader

        checkpoint_path = _resolve_checkpoint_path()
        if checkpoint_path is None:
            pytest.skip("No checkpoint found (set MOMENTUM_LIVE_CHECKPOINT or place one under models/)")

        config = LiveTradingConfig(
            symbols=["BTC/USD"],
            window_size=60,
            initial_balance=float(os.getenv("MOMENTUM_LIVE_SMOKE_INITIAL_BALANCE", "10000.0")),
            transaction_fee=0.001,
            reward_scale=1.0,
            invalid_action_penalty=-0.1,
            drawdown_penalty_lambda=0.5,
            slippage_bps=5.0,
            opportunity_cost_lambda=0.1,
            benchmark_allocation_frac=0.5,
            min_rebalance_pct=0.02,
            min_trade_value=1.0,
            models_dir="models",
            checkpoint_pattern="checkpoint_trainer_best_*.pt",
            reset_mode="none",
        )

        agent = load_agent_from_checkpoint(checkpoint_path)
        client = BrokerSubAccountClient(broker_manager.client, adopted_entry.account_id)
        trader = MomentumLiveTrader(
            agent=agent,
            symbol="BTC/USD",
            config=config,
            subaccount_client=client,
        )

        bars = _fetch_btc_minute_bars(lookback_minutes=max(config.window_size * 3, 180))
        assert len(bars) >= config.window_size, f"Need at least {config.window_size} 1-minute bars from Alpaca, got {len(bars)}"

        warmup = bars[: config.window_size - 1]
        latest = bars[config.window_size - 1]
        trader.preload_history(warmup)

        decision = trader.process_bar(latest)
        assert decision is not None
        assert 0 <= decision["action_index"] < 6
        assert decision["target_allocation"] in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
