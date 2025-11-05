"""Utilities for resetting the Alpaca paper trading account."""

from __future__ import annotations

import argparse
import sys
import time

from momentum_core.logging import get_logger, setup_package_logging

from .config import AlpacaCredentials

LOGGER = get_logger("momentum_live.reset_account")


def _configure_logging(level: str) -> None:
    setup_package_logging(
        "momentum_live",
        root_level=level.upper(),
        console_level=level.upper(),
        log_filename="live_trading.log",
    )


def reset_paper_account(wait_interval: float = 2.0, timeout: float = 60.0) -> None:
    """Cancel open orders and close positions in the paper trading account.

    Parameters
    ----------
    wait_interval:
        Seconds to wait between position status checks.
    timeout:
        Maximum number of seconds to wait for positions to close. A timeout does not
        raise but a warning is logged if positions remain open.
    """

    try:
        from alpaca.common.exceptions import APIError  # type: ignore[attr-defined]
        from alpaca.trading.client import TradingClient
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("alpaca-py is required to reset the paper account") from exc

    credentials = AlpacaCredentials.from_environment()

    if not credentials.paper:
        raise RuntimeError("Refusing to reset account because ALPACA_PAPER_TRADING is disabled.")

    LOGGER.info("Connecting to Alpaca trading client (paper=%s)...", credentials.paper)
    trading_client = TradingClient(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
        paper=True,
    )

    try:
        LOGGER.info("Submitting request to cancel all open orders...")
        trading_client.cancel_orders()
        LOGGER.info("Cancel request submitted for all open orders")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to cancel open orders: %s", exc)

    try:
        LOGGER.info("Submitting request to close all open positions...")
        trading_client.close_all_positions(cancel_orders=False)
        LOGGER.info("Close-all request submitted")
    except APIError as exc:
        if getattr(exc, "status_code", None) == 404:
            LOGGER.info("No open positions to close")
        else:  # pragma: no cover - defensive
            raise

    if wait_interval <= 0 or timeout <= 0:
        return

    LOGGER.info(
        "Waiting up to %.1f seconds for all positions to close (polling every %.1f seconds)...",
        timeout,
        wait_interval,
    )

    deadline = time.time() + timeout
    last_remaining = None

    while time.time() < deadline:
        try:
            positions = list(trading_client.get_all_positions())
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Unable to fetch open positions: %s", exc)
            break

        remaining = len(positions)

        if remaining == 0:
            LOGGER.info("All positions successfully closed")
            return

        if remaining != last_remaining:
            LOGGER.info("Waiting on %d remaining position(s) to close", remaining)
            last_remaining = remaining

        time.sleep(wait_interval)

    try:
        remaining_positions = list(trading_client.get_all_positions())
    except Exception:  # pragma: no cover - defensive
        remaining_positions = []

    if remaining_positions:
        LOGGER.warning(
            "Timeout reached while waiting for positions to close (%d still open)",
            len(remaining_positions),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset the Alpaca paper trading account")
    parser.add_argument(
        "--wait-interval",
        type=float,
        default=2.0,
        help="Seconds between polling for open positions",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum number of seconds to wait for positions to close",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level to use during reset",
    )

    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    try:
        reset_paper_account(wait_interval=args.wait_interval, timeout=args.timeout)
    except Exception as exc:  # pragma: no cover - CLI guard
        LOGGER.exception("Failed to reset Alpaca paper account: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
