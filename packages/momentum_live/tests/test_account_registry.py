from __future__ import annotations

import json
from pathlib import Path

import pytest
from momentum_live.account_registry import BrokerAccountRegistry, SubAccountEntry


def test_load_empty_when_file_missing(tmp_path: Path) -> None:
    registry = BrokerAccountRegistry(tmp_path / "missing.json")
    registry.load()
    assert registry.all() == {}


def test_set_persists_atomically(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = BrokerAccountRegistry(path)

    entry = SubAccountEntry.new("BTC/USD", "acc-1", "bot-BTCUSD")
    registry.set(entry)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "BTC/USD": {
            "account_id": "acc-1",
            "label": "bot-BTCUSD",
            "created_at": entry.created_at,
        }
    }

    reloaded = BrokerAccountRegistry(path)
    reloaded.load()
    assert reloaded.get("BTC/USD") == entry
    assert (path.parent / (path.name + ".tmp")).exists() is False


def test_set_overwrites_existing_entry(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = BrokerAccountRegistry(path)
    registry.set(SubAccountEntry.new("BTC/USD", "acc-1", "lbl-1"))
    registry.set(SubAccountEntry.new("BTC/USD", "acc-2", "lbl-2"))

    reloaded = BrokerAccountRegistry(path)
    assert reloaded.get("BTC/USD").account_id == "acc-2"


def test_remove_entry(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = BrokerAccountRegistry(path)
    registry.set(SubAccountEntry.new("BTC/USD", "acc-1", "lbl"))
    removed = registry.remove("BTC/USD")
    assert removed is not None
    assert removed.account_id == "acc-1"
    assert registry.get("BTC/USD") is None
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text("[]", encoding="utf-8")
    registry = BrokerAccountRegistry(path)
    with pytest.raises(RuntimeError, match="must be a JSON object"):
        registry.load()
