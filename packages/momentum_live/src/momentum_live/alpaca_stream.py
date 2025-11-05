"""Integration with Alpaca's live crypto stream."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from momentum_core.logging import get_logger

from .config import AlpacaCredentials
from .trader import BarData, MomentumLiveTrader

LOGGER = get_logger("momentum_live.alpaca")


class AlpacaStreamRunner:
    """Wrapper around ``alpaca-py``'s ``CryptoDataStream`` and trading client."""

    def __init__(self, credentials: AlpacaCredentials, trader: MomentumLiveTrader):
        self.credentials = credentials
        self.trader = trader
        self._stream = None
        self._trading_client = None
        self._historical_client = None
        self._history_primed = False
        self._symbol_alias_map = {}
        for symbol in trader.symbols:
            cleaned = symbol.replace("/", "").upper()
            self._symbol_alias_map[cleaned] = symbol
            self._symbol_alias_map[symbol.upper()] = symbol

    def _ensure_clients(self):
        if self._stream is None or self._trading_client is None:
            try:
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.live import CryptoDataStream
                from alpaca.trading.client import TradingClient
            except ImportError as exc:  # pragma: no cover - handled via tests
                raise RuntimeError("alpaca-py is required for live streaming and trading. Install the 'alpaca-py' package.") from exc

            # Data streaming client
            self._stream = CryptoDataStream(
                api_key=self.credentials.api_key,
                secret_key=self.credentials.secret_key,
                feed=self.credentials.location,
            )

            # Trading client
            self._trading_client = TradingClient(
                api_key=self.credentials.api_key,
                secret_key=self.credentials.secret_key,
                paper=self.credentials.paper,
            )

            try:
                self._historical_client = CryptoHistoricalDataClient(
                    api_key=self.credentials.api_key,
                    secret_key=self.credentials.secret_key,
                )
                LOGGER.info("Initialized Alpaca historical data client for warmup")
            except Exception as exc:  # pragma: no cover - defensive logging
                self._historical_client = None
                LOGGER.warning("Unable to initialize Alpaca historical client: %s", exc)

            # Pass trading client to trader
            self.trader.set_trading_client(self._trading_client)

            LOGGER.info(f"Initialized Alpaca clients (paper trading: {self.credentials.paper})")

    async def _handle_bar(self, bar) -> None:
        try:
            bar_data = BarData.from_alpaca(bar)
            self.trader.process_bar(bar_data)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Error while processing bar from Alpaca: %s", exc)

    def _prime_history(self) -> None:
        if self._history_primed:
            return

        window_size = getattr(self.trader.config, "window_size", 0)
        symbols = list(self.trader.symbols)

        if window_size <= 0 or not symbols:
            self._history_primed = True
            return

        if self._historical_client is None:
            LOGGER.info("Historical client unavailable; skipping warmup")
            self._history_primed = True
            return

        feed = self._resolve_feed()

        lookback = max(window_size * 2, window_size + 10)
        end = datetime.now(timezone.utc)

        bars_by_symbol = self._collect_historical_bars(symbols, lookback, end, feed)

        missing_symbols = [symbol for symbol in symbols if len(bars_by_symbol.get(symbol, [])) < window_size]
        if missing_symbols:
            LOGGER.info(
                "Historical warmup missing data for symbols %s; requesting extended lookback",
                ", ".join(missing_symbols),
            )
            extended_lookback = max(lookback * 3, window_size * 10)
            extended_bars = self._collect_historical_bars(missing_symbols, extended_lookback, end, feed)
            for symbol, bars in extended_bars.items():
                bars_by_symbol.setdefault(symbol, []).extend(bars)

        if not bars_by_symbol:
            LOGGER.warning("Historical warmup returned no bars")
            self._history_primed = True
            return

        window_bars: list[BarData] = []
        for symbol, bars in bars_by_symbol.items():
            if not bars:
                LOGGER.warning("Historical warmup still missing data for %s", symbol)
                continue

            bars.sort(key=lambda item: item.timestamp)
            deduped: dict[datetime, BarData] = {}
            for bar in bars:
                deduped[bar.timestamp] = bar

            trimmed = list(deduped.values())[-window_size:]
            if len(trimmed) < window_size:
                LOGGER.warning(
                    "Historical warmup collected %d/%d bars for %s",
                    len(trimmed),
                    window_size,
                    symbol,
                )

            window_bars.extend(trimmed)

        if not window_bars:
            LOGGER.warning("No historical bars matched configured symbols for warmup")
            self._history_primed = True
            return

        window_bars.sort(key=lambda item: (item.timestamp, item.symbol))
        self.trader.preload_history(window_bars)
        LOGGER.info(
            "Historical warmup complete | symbols=%s",
            ", ".join(sorted(bars_by_symbol.keys())),
        )

        self._history_primed = True

    def _canonical_symbol(self, symbol: str | None) -> str | None:
        if symbol is None:
            return None

        cleaned = symbol.replace("/", "").upper()
        return self._symbol_alias_map.get(cleaned, symbol)

    def _resolve_feed(self):
        try:
            from alpaca.common.enums import CryptoFeed

            return CryptoFeed(self.credentials.location)
        except (ImportError, ValueError):
            return None

    def _collect_historical_bars(self, symbols: list[str], lookback_minutes: int, end: datetime, feed) -> dict[str, list[BarData]]:
        if not symbols:
            return {}

        try:
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError as exc:
            LOGGER.warning("alpaca-py historical request helpers unavailable (%s); skipping warmup", exc)
            return {}

        start = end - timedelta(minutes=lookback_minutes)
        request_kwargs = {
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
                "Failed to fetch historical bars (lookback=%d minutes) for %s: %s",
                lookback_minutes,
                ", ".join(symbols),
                exc,
            )
            return {}

        bars_by_symbol: dict[str, list[BarData]] = {}

        def append_bar(raw_bar, fallback_symbol=None):
            raw_symbol = getattr(raw_bar, "symbol", None) or fallback_symbol
            canonical_symbol = self._canonical_symbol(raw_symbol)
            if canonical_symbol not in self.trader.symbols:
                LOGGER.debug("Skipping warmup bar for untracked symbol %s", raw_symbol)
                return

            try:
                bar_data = BarData(
                    symbol=canonical_symbol,
                    open=float(getattr(raw_bar, "open")),
                    high=float(getattr(raw_bar, "high")),
                    low=float(getattr(raw_bar, "low")),
                    close=float(getattr(raw_bar, "close")),
                    volume=float(getattr(raw_bar, "volume", 0.0)),
                    timestamp=getattr(raw_bar, "timestamp", datetime.utcnow()),
                )
            except Exception as exc:
                LOGGER.debug("Skipping malformed historical bar: %s", exc)
                return

            bars_by_symbol.setdefault(canonical_symbol, []).append(bar_data)

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

    def run_forever(self) -> None:
        self._ensure_clients()
        stream = self._stream
        assert stream is not None

        if not self.trader.symbols:
            raise RuntimeError("No symbols configured for live trading")

        self._prime_history()

        LOGGER.info("Subscribing to bars for symbols: %s", ", ".join(self.trader.symbols))
        stream.subscribe_bars(self._handle_bar, *self.trader.symbols)

        try:
            stream.run()
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            LOGGER.info("Live stream interrupted by user")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            LOGGER.info("Requested Alpaca stream shutdown")


__all__ = ["AlpacaStreamRunner"]
