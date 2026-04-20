"""Ad-hoc soft-reset for the registered Broker sub-accounts.

This is the manual counterpart to ``momentum-live --reset-mode soft``: cancel
orders, close positions, JNLC each sub-account back to ``--initial-balance``.
Useful between checkpoints when you want to evaluate a freshly trained model
from identical starting capital across pairs.

An advisory file lock (``models/broker_subaccounts.lock``) prevents concurrent
resets and concurrent execution while the live runner is active. The script is
idempotent: re-running it is safe (delta < skip_threshold yields no journal).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from momentum_core.logging import get_logger, setup_package_logging

from .account_registry import BrokerAccountRegistry
from .broker import BrokerAccountManager, BrokerCredentials
from .config import parse_symbols

LOGGER = get_logger("momentum_live.reset_cli")


def _configure_logging(level: str) -> None:
    setup_package_logging(
        "momentum_live",
        root_level=level.upper(),
        console_level=level.upper(),
        log_filename="live_trading.log",
    )


@contextmanager
def _file_lock(lock_path: Path):
    """Advisory exclusive lock so concurrent resets / live runs collide loudly."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(lock_path, "w")
    try:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise RuntimeError(
                    f"Reset lock {lock_path} is held by another process. Stop the live runner / other reset before retrying."
                ) from exc
            raise
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        finally:
            fp.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Soft-reset Broker sub-accounts to a target balance")
    parser.add_argument("--initial-balance", type=float, required=True, help="Per-account target USD balance")
    parser.add_argument(
        "--registry",
        default="models/broker_subaccounts.json",
        help="Path to the sub-account registry JSON (default: models/broker_subaccounts.json)",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated subset of pairs to reset (default: all registered)",
    )
    parser.add_argument(
        "--lock",
        default="models/broker_subaccounts.lock",
        help="Path to the advisory lockfile (default: models/broker_subaccounts.lock)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=BrokerAccountManager.DEFAULT_RESET_WAIT_TIMEOUT,
        help="Seconds to wait for positions to close after close_all_positions",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    credentials = BrokerCredentials.from_environment()
    if not credentials.has_firm_account:
        LOGGER.error("momentum-live-reset requires ALPACA_BROKER_ACCOUNT_ID (JNLC funding account). Set it in .env and re-run.")
        return 1

    registry = BrokerAccountRegistry(args.registry)
    registry.load()

    entries = registry.all()
    if not entries:
        LOGGER.warning("Registry %s is empty; nothing to reset", args.registry)
        return 0

    if args.symbols:
        wanted = set(parse_symbols(args.symbols))
        unknown = sorted(wanted - set(entries))
        if unknown:
            LOGGER.error("Unknown pairs: %s. Registered pairs: %s", unknown, sorted(entries))
            return 1
        entries = {pair: entry for pair, entry in entries.items() if pair in wanted}

    manager = BrokerAccountManager(credentials, registry)

    summaries: list[dict[str, object]] = []
    try:
        with _file_lock(Path(args.lock)):
            for pair, entry in entries.items():
                LOGGER.info("Resetting %s (account_id=%s)", pair, entry.account_id)
                summary = manager.reset_subaccount(
                    entry.account_id,
                    args.initial_balance,
                    wait_timeout=args.wait_timeout,
                )
                summary["symbol"] = pair
                summaries.append(summary)
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 2

    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
