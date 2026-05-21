"""End-to-end smoke test for the Broker sub-account crypto flow.

By default this suite **creates an ephemeral sandbox sub-account** via
:class:`~momentum_live.broker.BrokerAccountManager.ensure_subaccount`, runs the
tests, then **closes it** (liquidate + journal USD back when a firm account is
configured + ``BrokerClient.close_account``). No manual sub-account UUID is
required.

Legacy adopt mode: set ``MOMENTUM_LIVE_SMOKE_ACCOUNT_ID`` to reuse an existing
sub-account; that account is **not** closed after the run.

Optional integration knobs:

- ``MOMENTUM_LIVE_SMOKE_TRADE=1`` — real ~$10 BTC/USD round-trip on the
  sub-account. Ephemeral (and low-cash adopt) runs need ``ALPACA_BROKER_ACCOUNT_ID``
  so the suite can JNLC buying power onto the sub-account and skim it back during
  teardown.

Prerequisites:

    source scripts/env-paper.sh   # ALPACA_BROKER_* ; ALPACA_BROKER_ACCOUNT_ID for trade/teardown

Usage:

    # Ephemeral create → inspect → destroy (default)
    pytest packages/momentum_live/tests/test_broker_smoke.py -m integration -v

    # Optional round-trip (~$10 BTC); requires firm id for JNLC top-up + teardown skim
    MOMENTUM_LIVE_SMOKE_TRADE=1 pytest packages/momentum_live/tests/test_broker_smoke.py \\
        -m integration -v -k roundtrip

    # Optional agent + live minute-bar warmup (needs data API keys + checkpoint)
    MOMENTUM_LIVE_SMOKE_AGENT=1 pytest packages/momentum_live/tests/test_broker_smoke.py \\
        -m integration -v -k agent

    # Legacy: fixed sub-account, left open after the run
    export MOMENTUM_LIVE_SMOKE_ACCOUNT_ID=<uuid>
    pytest packages/momentum_live/tests/test_broker_smoke.py -m integration -v
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
_EXISTING_SMOKE_ACCOUNT = bool(_SMOKE_ACCOUNT_ID)

_TRADE_NOTIONAL = float(os.getenv("MOMENTUM_LIVE_SMOKE_TRADE_NOTIONAL", "10.0"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _BROKER_CREDS_MISSING,
        reason="ALPACA_BROKER_API_KEY / ALPACA_BROKER_API_SECRET must be set (source scripts/env-paper.sh).",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


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
def smoke_subaccount_entry(broker_manager):
    """Ephemeral sub-account (default) or adopted UUID when ``MOMENTUM_LIVE_SMOKE_ACCOUNT_ID`` is set."""
    from momentum_core.logging import get_logger

    log = get_logger(__name__)

    if _EXISTING_SMOKE_ACCOUNT:
        try:
            uuid.UUID(_SMOKE_ACCOUNT_ID)
        except ValueError as exc:
            pytest.fail(f"MOMENTUM_LIVE_SMOKE_ACCOUNT_ID is not a valid UUID: {exc}")
        entry = broker_manager.adopt_subaccount("BTC/USD", _SMOKE_ACCOUNT_ID, label="smoke-BTCUSD")
        yield entry
        return

    entry = broker_manager.ensure_subaccount("BTC/USD", label_prefix="smoke-ephemeral")
    try:
        yield entry
    finally:
        try:
            broker_manager.close_subaccount(entry.account_id)
        except Exception as exc:
            log.warning(
                "Ephemeral smoke sub-account teardown failed for %s: %s",
                entry.account_id,
                exc,
                exc_info=True,
            )


@pytest.fixture
def journal_smoke_trade_float(broker_manager, smoke_subaccount_entry):
    """JNLC top-up when ``MOMENTUM_LIVE_SMOKE_TRADE=1`` and cash is below trade needs."""
    if os.getenv("MOMENTUM_LIVE_SMOKE_TRADE") != "1":
        return

    need = _TRADE_NOTIONAL + 2.0
    aid = smoke_subaccount_entry.account_id
    cash = broker_manager.get_account_cash(aid)
    if cash >= need:
        return

    shortfall = round(need - cash, 2)
    if shortfall <= 0:
        return

    if not broker_manager.funding_account_id:
        pytest.skip(
            "Smoke BTC round-trip needs ALPACA_BROKER_ACCOUNT_ID to JNLC funds onto the sub-account "
            f"(need ~${need:.2f} USD; currently ${cash:.2f})."
        )

    broker_manager.journal_cash(
        aid,
        shortfall,
        direction="to_sub",
        description="momentum-live smoke trade float",
    )


def _resolve_checkpoint_path() -> Path | None:
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


class TestSmokeSubaccountCryptoMetadata:
    """Registry + raw Broker API checks for the smoke sub-account."""

    def test_registry_contains_smoke_pair(self, broker_manager, smoke_subaccount_entry) -> None:
        stored = broker_manager.registry.get("BTC/USD")
        assert stored is not None
        assert stored.account_id == smoke_subaccount_entry.account_id
        if _EXISTING_SMOKE_ACCOUNT:
            assert stored.label == "smoke-BTCUSD"
        else:
            assert stored.label == "smoke-ephemeral-BTCUSD"

    def test_account_is_active_and_crypto_metadata_present(self, broker_manager, smoke_subaccount_entry) -> None:
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
        resp = requests.get(f"{base}/v1/accounts/{smoke_subaccount_entry.account_id}", auth=auth, timeout=10)
        assert resp.status_code == 200, f"GET account failed: {resp.status_code} {resp.text[:200]}"
        payload = resp.json()

        status = str(payload.get("status", "")).upper()
        assert status in {"ACTIVE", "APPROVED"}, f"Sub-account status={status!r}"

        enabled_assets = [str(x).lower() for x in (payload.get("enabled_assets") or [])]
        crypto_status = str(payload.get("crypto_status") or "").upper()
        assert "crypto" in enabled_assets, (
            f"sub-account {smoke_subaccount_entry.account_id} enabled_assets={enabled_assets}; "
            "ensure_crypto_enabled() should have PATCHed it."
        )
        assert crypto_status in {"APPROVED", "ACTIVE"}, f"crypto_status={crypto_status!r}; expected APPROVED or ACTIVE"

    def test_ensure_crypto_enabled_is_idempotent(self, broker_manager, smoke_subaccount_entry) -> None:
        patched = broker_manager.ensure_crypto_enabled(smoke_subaccount_entry.account_id)
        assert patched is False, "crypto should already be enabled after adopt_subaccount / ensure_subaccount"

    def test_trading_engine_actually_accepts_crypto_orders(self, broker_manager, smoke_subaccount_entry) -> None:
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
            f"{base}/v1/trading/accounts/{smoke_subaccount_entry.account_id}/orders",
            auth=auth,
            json=body,
            timeout=10,
        )
        if resp.status_code == 422 and "crypto orders not allowed" in resp.text.lower():
            pytest.fail(
                "Smoke sub-account "
                f"{smoke_subaccount_entry.account_id} reports ``enabled_assets=[us_equity, crypto]`` "
                "but the trading engine rejects crypto orders. This account was crypto-enabled "
                "via PATCH /v1/accounts/{id} *after* creation, which only updates metadata. "
                "Alpaca only enables crypto trading for accounts whose ``enabled_assets`` were "
                "set at creation time. Use the default ephemeral smoke run "
                "(unset ``MOMENTUM_LIVE_SMOKE_ACCOUNT_ID``) so ``ensure_subaccount`` creates "
                "a fresh account, or ``BrokerAccountManager.ensure_subaccount('BTC/USD')``."
            )
        assert resp.status_code in {403, 422}, f"Unexpected probe response: {resp.status_code} {resp.text[:200]}"
        text = resp.text.lower()
        assert ("insufficient balance" in text) or ("buying power" in text), (
            f"Probe order should have been rejected for cash, not for permissions: {resp.status_code} {resp.text[:200]}"
        )


@pytest.mark.skipif(
    os.getenv("MOMENTUM_LIVE_SMOKE_TRADE") != "1",
    reason="set MOMENTUM_LIVE_SMOKE_TRADE=1 to place a real ~$10 BTC/USD order on the sub-account.",
)
class TestSubAccountCryptoRoundtrip:
    """Buy ~$10 of BTC on the sub-account, confirm the fill, then close the position."""

    @pytest.mark.usefixtures("journal_smoke_trade_float")
    def test_buy_and_close_via_subaccount_client(self, broker_manager, smoke_subaccount_entry) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        from momentum_live.subaccount_client import BrokerSubAccountClient

        client = BrokerSubAccountClient(broker_manager.client, smoke_subaccount_entry.account_id)

        starting_cash = client.get_account_cash()
        assert starting_cash >= _TRADE_NOTIONAL + 0.5, (
            f"Sub-account cash ${starting_cash:.2f} is below the ${_TRADE_NOTIONAL:.2f} "
            "smoke-trade notional + buffer. Fund the account, set ALPACA_BROKER_ACCOUNT_ID "
            "for automatic JNLC top-up, or lower MOMENTUM_LIVE_SMOKE_TRADE_NOTIONAL."
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

        filled_buy = _wait_for_order(broker_manager.client, smoke_subaccount_entry.account_id, buy_order.id)
        filled_status = getattr(filled_buy, "status", None)
        filled_status_str = (
            filled_status.value if hasattr(filled_status, "value") else str(filled_status) or ""
        ).lower()
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
        filled_sell = _wait_for_order(broker_manager.client, smoke_subaccount_entry.account_id, sell_order.id)
        sell_status = getattr(filled_sell, "status", None)
        sell_status_str = (sell_status.value if hasattr(sell_status, "value") else str(sell_status) or "").lower()
        assert sell_status_str == "filled", f"Sell did not fill (status={sell_status_str!r})"


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

    def test_agent_decides_on_smoke_subaccount(self, broker_manager, smoke_subaccount_entry) -> None:
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
        client = BrokerSubAccountClient(broker_manager.client, smoke_subaccount_entry.account_id)
        trader = MomentumLiveTrader(
            agent=agent,
            symbol="BTC/USD",
            config=config,
            subaccount_client=client,
        )

        bars = _fetch_btc_minute_bars(lookback_minutes=max(config.window_size * 3, 180))
        assert len(bars) >= config.window_size, (
            f"Need at least {config.window_size} 1-minute bars from Alpaca, got {len(bars)}"
        )

        warmup = bars[: config.window_size - 1]
        latest = bars[config.window_size - 1]
        trader.preload_history(warmup)

        decision = trader.process_bar(latest)
        assert decision is not None
        assert 0 <= decision["action_index"] < 6
        assert decision["target_allocation"] in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
