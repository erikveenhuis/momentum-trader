"""Persisted mapping of crypto pair to Broker sub-account id.

The registry is a JSON file (default ``models/broker_subaccounts.json``) that survives
restarts so we never re-create a sub-account that already exists for a pair. Writes
are atomic (``.tmp`` + ``rename``) so a crashed process never leaves a partially
written file behind.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from momentum_core.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class SubAccountEntry:
    """Single registry row for one trading pair."""

    pair: str
    account_id: str
    label: str
    created_at: str

    @classmethod
    def new(cls, pair: str, account_id: str, label: str) -> SubAccountEntry:
        return cls(
            pair=pair,
            account_id=account_id,
            label=label,
            created_at=datetime.now(UTC).isoformat(),
        )


class BrokerAccountRegistry:
    """File-backed pair to ``SubAccountEntry`` mapping with atomic writes."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, SubAccountEntry] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        """Read the registry file (or initialize empty if it doesn't exist)."""
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._loaded:
            return

        if not self._path.exists():
            LOGGER.info("Registry file %s does not exist yet; starting empty", self._path)
            self._entries = {}
            self._loaded = True
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Registry file {self._path} is not valid JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise RuntimeError(f"Registry file {self._path} must be a JSON object, got {type(raw).__name__}")

        entries: dict[str, SubAccountEntry] = {}
        for pair, payload in raw.items():
            if not isinstance(payload, dict):
                raise RuntimeError(f"Registry entry for {pair!r} must be an object, got {type(payload).__name__}")
            entries[pair] = SubAccountEntry(
                pair=pair,
                account_id=str(payload["account_id"]),
                label=str(payload.get("label", pair)),
                created_at=str(payload.get("created_at", datetime.now(UTC).isoformat())),
            )

        self._entries = entries
        self._loaded = True
        LOGGER.info("Loaded registry from %s (%d entries)", self._path, len(entries))

    def get(self, pair: str) -> SubAccountEntry | None:
        with self._lock:
            self._load_locked()
            return self._entries.get(pair)

    def all(self) -> dict[str, SubAccountEntry]:
        with self._lock:
            self._load_locked()
            return dict(self._entries)

    def set(self, entry: SubAccountEntry) -> None:
        with self._lock:
            self._load_locked()
            self._entries[entry.pair] = entry
            self._flush_locked()

    def remove(self, pair: str) -> SubAccountEntry | None:
        with self._lock:
            self._load_locked()
            removed = self._entries.pop(pair, None)
            if removed is not None:
                self._flush_locked()
            return removed

    def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            pair: {k: v for k, v in asdict(entry).items() if k != "pair"} for pair, entry in self._entries.items()
        }
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self._path)


__all__ = ["BrokerAccountRegistry", "SubAccountEntry"]
