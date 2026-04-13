"""Integration tests against Alpaca paper trading.

Run with:
    source scripts/env-paper.sh
    pytest packages/momentum_live/tests/test_paper_trading.py -m integration -v

Agent + 1-minute bar test additionally needs a GPU and a checkpoint:
    export MOMENTUM_LIVE_CHECKPOINT=/path/to/checkpoint.pt
    # or place checkpoints under models/ or set MOMENTUM_LIVE_MODELS_DIR
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_MISSING_CREDS = not (os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET"))
_SKIP_REASON = "ALPACA_API_KEY and ALPACA_API_SECRET must be set (source scripts/env-paper.sh)"

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_checkpoint_path() -> Path | None:
    """Return a checkpoint file path, or None if none is available."""

    explicit = os.getenv("MOMENTUM_LIVE_CHECKPOINT")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None

    from momentum_live.agent_loader import find_best_checkpoint

    models_dir = os.getenv("MOMENTUM_LIVE_MODELS_DIR")
    if models_dir:
        candidate = Path(models_dir).expanduser().resolve()
    else:
        candidate = _REPO_ROOT / "models"
    if not candidate.is_dir():
        return None
    try:
        return find_best_checkpoint(candidate)
    except FileNotFoundError:
        return None


def _live_trading_config_for_agent_test():
    from momentum_live.config import LiveTradingConfig

    return LiveTradingConfig(
        symbols=["BTC/USD"],
        window_size=60,
        initial_balance=100_000.0,
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
    )


def _fetch_one_minute_bars_btc(
    api_key: str,
    secret_key: str,
    *,
    location: str,
    lookback_minutes: int,
) -> list:
    """Fetch recent 1-minute crypto bars from Alpaca (same timeframe as the live stream)."""

    from alpaca.data.enums import CryptoFeed
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from momentum_live.trader import BarData

    client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    end = datetime.now(UTC)
    start = end - timedelta(minutes=lookback_minutes)

    try:
        feed = CryptoFeed(location)
    except ValueError:
        feed = CryptoFeed.US

    request_kwargs: dict = {
        "symbol_or_symbols": "BTC/USD",
        "timeframe": TimeFrame.Minute,
        "start": start,
        "end": end,
        "limit": lookback_minutes,
    }
    request_kwargs["feed"] = feed
    try:
        request = CryptoBarsRequest(**request_kwargs)
    except TypeError:
        request_kwargs.pop("limit", None)
        request = CryptoBarsRequest(**request_kwargs)

    bars_response = client.get_crypto_bars(request)
    out: list = []

    def append_raw(raw_bar, _fallback_symbol: str | None = None) -> None:
        out.append(
            BarData(
                symbol="BTC/USD",
                open=float(getattr(raw_bar, "open")),
                high=float(getattr(raw_bar, "high")),
                low=float(getattr(raw_bar, "low")),
                close=float(getattr(raw_bar, "close")),
                volume=float(getattr(raw_bar, "volume", 0.0)),
                timestamp=getattr(raw_bar, "timestamp", datetime.now(UTC)),
            )
        )

    data_attr = getattr(bars_response, "data", None)
    if isinstance(data_attr, dict):
        for symbol_key, raw_bars in data_attr.items():
            for raw_bar in raw_bars or []:
                append_raw(raw_bar, symbol_key)
    elif isinstance(data_attr, (list, tuple)):
        for raw_bar in data_attr:
            append_raw(raw_bar)
    else:
        bars_attr = getattr(bars_response, "bars", None)
        if isinstance(bars_attr, dict):
            for symbol_key, raw_bars in bars_attr.items():
                for raw_bar in raw_bars or []:
                    append_raw(raw_bar, symbol_key)
        elif isinstance(bars_attr, (list, tuple)):
            for raw_bar in bars_attr:
                append_raw(raw_bar)
        else:
            try:
                for raw_bar in bars_response:
                    append_raw(raw_bar)
            except TypeError:
                pass

    out.sort(key=lambda b: b.timestamp)
    return out


def _trading_client():
    from alpaca.trading.client import TradingClient

    return TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_API_SECRET"],
        paper=True,
    )


def _wait_for_fill(client, order_id: str, timeout: float = 30.0, interval: float = 0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        order = client.get_order_by_id(order_id)
        status = str(getattr(order, "status", "")).lower()
        if hasattr(order.status, "value"):
            status = order.status.value.lower()
        if status in ("filled", "partially_filled", "canceled", "expired", "rejected"):
            return order
        time.sleep(interval)
    return client.get_order_by_id(order_id)


@pytest.mark.skipif(_MISSING_CREDS, reason=_SKIP_REASON)
class TestAlpacaPaperConnectivity:
    """Verify that the paper trading account is reachable and functional."""

    def test_account_is_active(self):
        client = _trading_client()
        account = client.get_account()

        status = account.status.value if hasattr(account.status, "value") else str(account.status)
        assert status.upper() == "ACTIVE", f"Account status: {status}"
        assert not account.trading_blocked
        assert not account.account_blocked
        assert float(account.cash) > 0

    def test_can_list_positions(self):
        client = _trading_client()
        positions = client.get_all_positions()
        assert isinstance(positions, list)

    def test_can_list_orders(self):
        from alpaca.trading.requests import GetOrdersRequest

        client = _trading_client()
        request = GetOrdersRequest(status="all", limit=5)
        orders = client.get_orders(filter=request)
        assert isinstance(orders, list)


@pytest.mark.skipif(_MISSING_CREDS, reason=_SKIP_REASON)
class TestAlpacaPaperTradeRoundTrip:
    """Buy $10 of BTC, confirm the fill, then close the position."""

    SYMBOL = "BTC/USD"
    ALPACA_SYMBOL = "BTCUSD"

    def test_buy_and_sell_roundtrip(self):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        client = _trading_client()

        idempotency_key = f"test-roundtrip-{uuid.uuid4().hex[:12]}"
        buy_request = MarketOrderRequest(
            symbol=self.ALPACA_SYMBOL,
            notional=10.0,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            client_order_id=idempotency_key,
        )

        buy_order = client.submit_order(buy_request)
        assert buy_order.id is not None

        filled_buy = _wait_for_fill(client, buy_order.id)
        buy_status = str(getattr(filled_buy, "status", ""))
        if hasattr(filled_buy.status, "value"):
            buy_status = filled_buy.status.value
        assert buy_status.lower() == "filled", f"Buy order did not fill: {buy_status}"
        assert float(filled_buy.filled_qty) > 0
        assert float(filled_buy.filled_avg_price) > 0

        positions = client.get_all_positions()
        btc_position = None
        for pos in positions:
            if getattr(pos, "symbol", "") in (self.SYMBOL, self.ALPACA_SYMBOL):
                btc_position = pos
                break
        assert btc_position is not None, "BTC position not found after buy"
        assert float(btc_position.qty) > 0

        sell_qty = float(btc_position.qty)
        sell_request = MarketOrderRequest(
            symbol=self.ALPACA_SYMBOL,
            qty=sell_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        sell_order = client.submit_order(sell_request)
        filled_sell = _wait_for_fill(client, sell_order.id)
        sell_status = str(getattr(filled_sell, "status", ""))
        if hasattr(filled_sell.status, "value"):
            sell_status = filled_sell.status.value
        assert sell_status.lower() == "filled", f"Sell order did not fill: {sell_status}"


@pytest.mark.skipif(_MISSING_CREDS, reason=_SKIP_REASON)
class TestAgentOneMinuteBarPipeline:
    """Load a trained agent, ingest Alpaca 1-minute BTC bars, run ``process_bar``, assert a decision.

    Uses the same minute aggregates as the live WebSocket feed (REST). Optional WebSocket test below.
    """

    def test_agent_decides_on_sequential_one_minute_bars(self):
        checkpoint_path = _resolve_checkpoint_path()
        if checkpoint_path is None:
            pytest.skip(
                "No checkpoint found: set MOMENTUM_LIVE_CHECKPOINT or place checkpoints under models/ or set MOMENTUM_LIVE_MODELS_DIR"
            )

        from momentum_live.agent_loader import load_agent_from_checkpoint
        from momentum_live.trader import MomentumLiveTrader

        agent = load_agent_from_checkpoint(checkpoint_path)
        config = _live_trading_config_for_agent_test()
        trader = MomentumLiveTrader(agent=agent, config=config)

        location = os.getenv("ALPACA_CRYPTO_FEED", "us")
        bars = _fetch_one_minute_bars_btc(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_API_SECRET"],
            location=location,
            lookback_minutes=180,
        )
        need = config.window_size
        assert len(bars) >= need, f"Need at least {need} 1-minute bars from Alpaca, got {len(bars)}"

        warmup = bars[: need - 1]
        last_bar = bars[need - 1]
        trader.preload_history(warmup)

        decision = trader.process_bar(last_bar)
        assert decision is not None
        assert 0 <= decision["action_index"] < 6
        assert decision["target_allocation"] in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    @pytest.mark.skipif(
        os.getenv("MOMENTUM_LIVE_STREAM_TEST") != "1",
        reason="set MOMENTUM_LIVE_STREAM_TEST=1 to run live WebSocket bar test (can take up to ~2 min)",
    )
    def test_listen_live_one_minute_bar_and_decide(self):
        """Warm up from REST, then subscribe to the live crypto bar WebSocket and process one bar."""

        import threading

        from alpaca.data.enums import CryptoFeed
        from alpaca.data.live import CryptoDataStream
        from momentum_live.agent_loader import load_agent_from_checkpoint
        from momentum_live.trader import BarData, MomentumLiveTrader

        checkpoint_path = _resolve_checkpoint_path()
        if checkpoint_path is None:
            pytest.skip(
                "No checkpoint found: set MOMENTUM_LIVE_CHECKPOINT or place checkpoints under models/ or set MOMENTUM_LIVE_MODELS_DIR"
            )

        agent = load_agent_from_checkpoint(checkpoint_path)
        config = _live_trading_config_for_agent_test()
        trader = MomentumLiveTrader(agent=agent, config=config)

        location = os.getenv("ALPACA_CRYPTO_FEED", "us")
        try:
            feed = CryptoFeed(location)
        except ValueError:
            feed = CryptoFeed.US

        bars = _fetch_one_minute_bars_btc(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_API_SECRET"],
            location=location,
            lookback_minutes=180,
        )
        need = config.window_size
        assert len(bars) >= need
        trader.preload_history(bars[:need])

        decision_box: dict[str, object | None] = {"decision": None}
        error_box: dict[str, BaseException | None] = {"err": None}

        stream = CryptoDataStream(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_API_SECRET"],
            feed=feed,
        )

        async def handle_bar(bar) -> None:
            try:
                bar_data = BarData.from_alpaca(bar)
                decision_box["decision"] = trader.process_bar(bar_data)
            except BaseException as exc:
                error_box["err"] = exc
            finally:
                stream.stop()

        stream.subscribe_bars(handle_bar, "BTC/USD")

        def _run() -> None:
            stream.run()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=130.0)
        assert not thread.is_alive(), "WebSocket did not receive a bar within 130s (check feed / connection limits)"

        if error_box["err"] is not None:
            raise error_box["err"]

        decision = decision_box["decision"]
        assert decision is not None
        assert 0 <= decision["action_index"] < 6
