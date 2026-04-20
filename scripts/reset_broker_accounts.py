"""Shim entry point for the broker sub-account soft-reset CLI.

Real implementation lives at ``momentum_live.reset_cli`` so it can also be
exposed as the ``momentum-live-reset`` console script.

Usage::

    source scripts/env-paper.sh
    python scripts/reset_broker_accounts.py --initial-balance 1000
    # or restrict to a subset:
    python scripts/reset_broker_accounts.py --initial-balance 1000 --symbols BTC/USD,ETH/USD
"""

from __future__ import annotations

import sys

from momentum_live.reset_cli import main

if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
