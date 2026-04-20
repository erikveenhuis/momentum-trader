"""Per-sub-account trading shim built on the Broker ``*_for_account`` endpoints.

``MomentumLiveTrader`` was originally written against the regular ``TradingClient``
surface (``get_account()``, ``get_all_positions()``, ``submit_order(req)``,
``get_order_by_id(id)``). The Broker API exposes the same operations under
``BrokerClient.*_for_account(account_id, ...)``. This thin shim mirrors the
expected surface so the trader doesn't need to know it's talking to a Broker
sub-account.
"""

from __future__ import annotations

from momentum_core.logging import get_logger

LOGGER = get_logger("momentum_live.subaccount_client")


class BrokerSubAccountClient:
    """``TradingClient``-shaped facade over a single Broker sub-account."""

    def __init__(self, broker_client, account_id: str):
        self._client = broker_client
        self._account_id = account_id

    @property
    def account_id(self) -> str:
        return self._account_id

    def get_account(self):
        return self._client.get_trade_account_by_id(self._account_id)

    def get_account_cash(self) -> float:
        return float(self.get_account().cash)

    def get_all_positions(self):
        return list(self._client.get_all_positions_for_account(self._account_id) or [])

    def submit_order(self, order_request):
        return self._client.submit_order_for_account(self._account_id, order_request)

    def get_order_by_id(self, order_id):
        return self._client.get_order_for_account_by_id(self._account_id, order_id)


__all__ = ["BrokerSubAccountClient"]
