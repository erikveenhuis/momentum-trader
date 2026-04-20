from __future__ import annotations

from dataclasses import dataclass

from momentum_live.subaccount_client import BrokerSubAccountClient


@dataclass
class _FakeAccount:
    cash: str | float


class _FakeBroker:
    def __init__(self, cash: float = 1000.0) -> None:
        self.cash = cash
        self.calls: list[tuple] = []
        self.positions: list[object] = []

    def get_trade_account_by_id(self, account_id):
        self.calls.append(("get_trade_account_by_id", account_id))
        return _FakeAccount(cash=self.cash)

    def get_all_positions_for_account(self, account_id):
        self.calls.append(("get_all_positions_for_account", account_id))
        return self.positions

    def submit_order_for_account(self, account_id, request):
        self.calls.append(("submit_order_for_account", account_id, request))
        return {"id": "order-1"}

    def get_order_for_account_by_id(self, account_id, order_id):
        self.calls.append(("get_order_for_account_by_id", account_id, order_id))
        return {"id": order_id, "status": "filled"}


def test_routes_calls_with_account_id() -> None:
    broker = _FakeBroker(cash=1234.5)
    client = BrokerSubAccountClient(broker, "acc-X")

    assert client.account_id == "acc-X"
    assert client.get_account_cash() == 1234.5
    assert client.get_all_positions() == []
    assert client.submit_order("req")["id"] == "order-1"
    assert client.get_order_by_id("order-1")["status"] == "filled"

    assert ("get_trade_account_by_id", "acc-X") in broker.calls
    assert ("get_all_positions_for_account", "acc-X") in broker.calls
    assert ("submit_order_for_account", "acc-X", "req") in broker.calls
    assert ("get_order_for_account_by_id", "acc-X", "order-1") in broker.calls
