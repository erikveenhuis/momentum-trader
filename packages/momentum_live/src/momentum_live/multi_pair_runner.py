"""Multi-pair live trading runner.

Owns:

- One :class:`alpaca.data.live.CryptoDataStream` shared across all pairs (Alpaca
  caps concurrent stream connections per credential set).
- One :class:`momentum_live.trader.MomentumLiveTrader` per pair, each wired to its
  own :class:`BrokerSubAccountClient` (one Broker sub-account).
- The warmup loop that primes each pair's normalizer from REST 1-minute history
  before subscribing to the live websocket.
- The reset workflow that pre-trades all sub-accounts back to ``initial_balance``.

The agent network is shared across all per-pair traders (cheap inference, identical
policy per pair — matches training).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from momentum_agent import RainbowDQNAgent
from momentum_core.logging import get_logger

from .broker import BrokerAccountManager
from .config import AlpacaCredentials, LiveTradingConfig
from .subaccount_client import BrokerSubAccountClient
from .trader import BarData, MomentumLiveTrader

LOGGER = get_logger(__name__)


class MultiPairRunner:
    """Drives N per-pair :class:`MomentumLiveTrader`s off one shared bar stream."""

    def __init__(
        self,
        agent: RainbowDQNAgent,
        broker: BrokerAccountManager,
        data_credentials: AlpacaCredentials,
        config: LiveTradingConfig,
    ):
        if not config.symbols:
            raise ValueError("LiveTradingConfig.symbols is empty")

        self.agent = agent
        self.broker = broker
        self.data_credentials = data_credentials
        self.config = config

        self._stream = None
        self._historical_client = None
        self._history_primed = False

        self.traders: dict[str, MomentumLiveTrader] = {}
        self._symbol_alias_map: dict[str, str] = {}

        self._init_traders()

    # ------------------------------------------------------------------
    # Construction / wiring
    # ------------------------------------------------------------------

    def _init_traders(self) -> None:
        trades_jsonl_path: Path | None = None
        if self.config.tb_log_dir:
            trades_jsonl_path = Path(self.config.tb_log_dir) / "live_trades.jsonl"
            trades_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        for symbol in self.config.symbols:
            entry = self.broker.ensure_subaccount(symbol)
            client = BrokerSubAccountClient(self.broker.client, entry.account_id)

            trader = MomentumLiveTrader(
                agent=self.agent,
                symbol=symbol,
                config=self.config,
                subaccount_client=client,
                trades_jsonl_path=trades_jsonl_path,
            )
            self.traders[symbol] = trader

            cleaned = symbol.replace("/", "").upper()
            self._symbol_alias_map[cleaned] = symbol
            self._symbol_alias_map[symbol.upper()] = symbol
            LOGGER.info("Bound %s to sub-account %s", symbol, entry.account_id)

    # ------------------------------------------------------------------
    # Reset workflow
    # ------------------------------------------------------------------

    def reset_all(self, mode: str = "soft") -> list[dict[str, object]]:
        """Apply the configured reset to every registered sub-account.

        Modes:
          - ``"none"``: no-op (returns []).
          - ``"soft"``: cancel orders + close positions + JNLC back to initial_balance.
          - ``"hard"``: not implemented in this revision; raises NotImplementedError.

        A marker line is appended to ``live_trades.jsonl`` (when enabled) so
        downstream tooling can segment the trace by checkpoint.
        """
        if mode == "none":
            LOGGER.info("Reset mode 'none' — leaving sub-accounts untouched")
            return []
        if mode == "hard":
            raise NotImplementedError("Hard reset (recreate sub-accounts) is not implemented yet")
        if mode != "soft":
            raise ValueError(f"Unknown reset mode {mode!r}; expected one of: none, soft, hard")

        results: list[dict[str, object]] = []
        for symbol, trader in self.traders.items():
            client = trader.subaccount_client
            if client is None:  # pragma: no cover - defensive
                LOGGER.warning("[%s] no sub-account client; skipping reset", symbol)
                continue
            summary = self.broker.reset_subaccount(client.account_id, self.config.initial_balance)
            summary["symbol"] = symbol
            results.append(summary)
            self._reset_local_trader_state(trader)

        self._write_reset_marker(results)
        return results

    def _reset_local_trader_state(self, trader: MomentumLiveTrader) -> None:
        """Wipe a trader's in-memory position bookkeeping after a soft reset."""
        from momentum_env.trading import PortfolioState

        trader.portfolio_state = PortfolioState(
            balance=0.0,
            position=0.0,
            position_price=0.0,
            total_transaction_cost=0.0,
        )
        trader.bars_in_position = 0
        trader.prev_portfolio_value = self.config.initial_balance
        trader.step_records.clear()
        trader.closed_trade_count = 0

    def _write_reset_marker(self, results: list[dict[str, object]]) -> None:
        if not self.config.tb_log_dir:
            return
        path = Path(self.config.tb_log_dir) / "live_trades.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = {
            "marker": "reset",
            "ts": datetime.now(UTC).isoformat(),
            "initial_balance": self.config.initial_balance,
            "results": results,
        }
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(marker, default=float) + "\n")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Subscribe to the bar stream and dispatch bars per symbol.

        Blocks until the stream stops (Ctrl-C, connection error, etc).
        """
        self._ensure_clients()
        stream = self._stream
        assert stream is not None

        self._prime_history()

        LOGGER.info("Subscribing to bars for symbols: %s", ", ".join(self.traders.keys()))
        stream.subscribe_bars(self._handle_bar, *self.traders.keys())

        try:
            stream.run()
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            LOGGER.info("Live stream interrupted by user")
        except ValueError as exc:
            if "connection limit exceeded" in str(exc).lower():
                LOGGER.error("Alpaca connection limit exceeded. Close other instances or wait.")
                raise RuntimeError("Alpaca connection limit exceeded") from exc
            raise

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                LOGGER.info("Requested Alpaca stream shutdown")
            except (RuntimeError, OSError, AttributeError) as exc:  # pragma: no cover - defensive
                LOGGER.warning("Error during stream shutdown: %s", exc)

    async def _handle_bar(self, bar) -> None:
        try:
            bar_data = BarData.from_alpaca(bar)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            LOGGER.exception("Could not normalize incoming bar: %s", exc)
            return

        canonical = self._canonical_symbol(bar_data.symbol)
        if canonical is None:
            LOGGER.debug("No trader for symbol %s", bar_data.symbol)
            return
        if canonical != bar_data.symbol:
            bar_data = BarData(
                symbol=canonical,
                open=bar_data.open,
                high=bar_data.high,
                low=bar_data.low,
                close=bar_data.close,
                volume=bar_data.volume,
                timestamp=bar_data.timestamp,
            )
        try:
            self.traders[canonical].process_bar(bar_data)
        except Exception as exc:  # noqa: BLE001 - live-trading safety boundary; must keep stream alive across other pairs
            LOGGER.exception("[%s] error while processing bar: %s", canonical, exc)

    # ------------------------------------------------------------------
    # Warmup (REST history -> per-pair normalizer)
    # ------------------------------------------------------------------

    def _ensure_clients(self) -> None:
        if self._stream is not None:
            return
        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.live import CryptoDataStream
        except ImportError as exc:  # pragma: no cover - handled in tests
            raise RuntimeError("alpaca-py is required for live streaming. Install 'alpaca-py'.") from exc

        self._stream = CryptoDataStream(
            api_key=self.data_credentials.api_key,
            secret_key=self.data_credentials.secret_key,
            feed=self.data_credentials.location,
        )
        try:
            self._historical_client = CryptoHistoricalDataClient(
                api_key=self.data_credentials.api_key,
                secret_key=self.data_credentials.secret_key,
            )
            LOGGER.info("Initialized Alpaca historical data client for warmup")
        except (ImportError, RuntimeError, ValueError, OSError, AttributeError) as exc:  # pragma: no cover - defensive
            self._historical_client = None
            LOGGER.warning("Could not initialize historical client: %s", exc)

    def _prime_history(self) -> None:
        if self._history_primed:
            return
        window_size = self.config.window_size
        symbols = list(self.traders.keys())
        if window_size <= 0 or not symbols:
            self._history_primed = True
            return
        if self._historical_client is None:
            LOGGER.info("Historical client unavailable; skipping warmup")
            self._history_primed = True
            return

        feed = self._resolve_feed()
        lookback = max(window_size * 2, window_size + 10)
        end = datetime.now(UTC)
        bars_by_symbol = self._collect_historical_bars(symbols, lookback, end, feed)

        missing = [s for s in symbols if len(bars_by_symbol.get(s, [])) < window_size]
        if missing:
            LOGGER.info(
                "Historical warmup missing data for %s; requesting extended lookback",
                ", ".join(missing),
            )
            extended_lookback = max(lookback * 3, window_size * 10)
            extended = self._collect_historical_bars(missing, extended_lookback, end, feed)
            for s, bars in extended.items():
                bars_by_symbol.setdefault(s, []).extend(bars)

        if not bars_by_symbol:
            LOGGER.warning("Historical warmup returned no bars")
            self._history_primed = True
            return

        for symbol, bars in bars_by_symbol.items():
            if not bars:
                continue
            bars.sort(key=lambda b: b.timestamp)
            deduped: dict[datetime, BarData] = {}
            for b in bars:
                deduped[b.timestamp] = b
            trimmed = list(deduped.values())[-window_size:]
            if len(trimmed) < window_size:
                LOGGER.warning(
                    "Historical warmup collected %d/%d bars for %s",
                    len(trimmed),
                    window_size,
                    symbol,
                )
            self.traders[symbol].preload_history(trimmed)

        LOGGER.info("Historical warmup complete | symbols=%s", ", ".join(sorted(bars_by_symbol.keys())))
        self._history_primed = True

    def _resolve_feed(self):
        try:
            from alpaca.common.enums import CryptoFeed

            return CryptoFeed(self.data_credentials.location)
        except (ImportError, ValueError):
            return None

    def _canonical_symbol(self, symbol: str | None) -> str | None:
        if symbol is None:
            return None
        if symbol in self.traders:
            return symbol
        cleaned = symbol.replace("/", "").upper()
        return self._symbol_alias_map.get(cleaned)

    def _collect_historical_bars(
        self,
        symbols: list[str],
        lookback_minutes: int,
        end: datetime,
        feed,
    ) -> dict[str, list[BarData]]:
        if not symbols:
            return {}
        try:
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError as exc:
            LOGGER.warning("alpaca-py historical helpers unavailable (%s); skipping warmup", exc)
            return {}

        start = end - timedelta(minutes=lookback_minutes)
        request_kwargs: dict = {
            "symbol_or_symbols": symbols[0] if len(symbols) == 1 else list(symbols),
            "timeframe": TimeFrame.Minute,
            "start": start,
            "end": end,
            "limit": lookback_minutes,
        }
        if feed is not None:
            request_kwargs["feed"] = feed
        try:
            request = CryptoBarsRequest(**request_kwargs)
        except TypeError:
            request_kwargs.pop("limit", None)
            request = CryptoBarsRequest(**request_kwargs)

        try:
            bars_response = self._historical_client.get_crypto_bars(request)
        except Exception as exc:
            LOGGER.warning(
                "Failed to fetch historical bars (lookback=%d) for %s: %s",
                lookback_minutes,
                ", ".join(symbols),
                exc,
            )
            return {}

        bars_by_symbol: dict[str, list[BarData]] = {}

        def append_bar(raw_bar, fallback_symbol: str | None = None) -> None:
            raw_symbol = getattr(raw_bar, "symbol", None) or fallback_symbol
            canonical = self._canonical_symbol(raw_symbol)
            if canonical is None:
                return
            try:
                bar = BarData(
                    symbol=canonical,
                    open=float(raw_bar.open),
                    high=float(raw_bar.high),
                    low=float(raw_bar.low),
                    close=float(raw_bar.close),
                    volume=float(getattr(raw_bar, "volume", 0.0)),
                    timestamp=getattr(raw_bar, "timestamp", datetime.now(UTC)),
                )
            except Exception as exc:
                LOGGER.debug("Skipping malformed historical bar: %s", exc)
                return
            bars_by_symbol.setdefault(canonical, []).append(bar)

        data_attr = getattr(bars_response, "data", None)
        if isinstance(data_attr, dict):
            for symbol_key, raw_bars in data_attr.items():
                for raw_bar in raw_bars or []:
                    append_bar(raw_bar, fallback_symbol=symbol_key)
        elif isinstance(data_attr, (list, tuple)):
            for raw_bar in data_attr:
                append_bar(raw_bar)
        else:
            bars_attr = getattr(bars_response, "bars", None)
            if isinstance(bars_attr, dict):
                for symbol_key, raw_bars in bars_attr.items():
                    for raw_bar in raw_bars or []:
                        append_bar(raw_bar, fallback_symbol=symbol_key)
            elif isinstance(bars_attr, (list, tuple)):
                for raw_bar in bars_attr:
                    append_bar(raw_bar)
            else:
                try:
                    for raw_bar in bars_response:
                        append_bar(raw_bar)
                except TypeError:
                    pass

        return bars_by_symbol


def preload_history_iterable(trader: MomentumLiveTrader, bars: Iterable[BarData]) -> None:
    """Convenience helper for tests: prime a single trader from an arbitrary iterable."""
    trader.preload_history(list(bars))


__all__ = ["MultiPairRunner", "preload_history_iterable"]
