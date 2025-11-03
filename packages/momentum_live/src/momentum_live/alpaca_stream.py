"""Integration with Alpaca's live crypto stream."""

from __future__ import annotations

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

    def _ensure_clients(self):
        if self._stream is None or self._trading_client is None:
            try:
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

            # Pass trading client to trader
            self.trader.set_trading_client(self._trading_client)

            LOGGER.info(f"Initialized Alpaca clients (paper trading: {self.credentials.paper})")

    async def _handle_bar(self, bar) -> None:
        try:
            bar_data = BarData.from_alpaca(bar)
            self.trader.process_bar(bar_data)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Error while processing bar from Alpaca: %s", exc)

    def run_forever(self) -> None:
        self._ensure_clients()
        stream = self._stream
        assert stream is not None

        if not self.trader.symbols:
            raise RuntimeError("No symbols configured for live trading")

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
