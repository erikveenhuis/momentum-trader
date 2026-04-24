"""Backward-compat shim.

The canonical home for ``ACCOUNT_STATE_DIM`` (and the other shape contracts
shared across packages) is now :mod:`momentum_core.constants`. This module
re-exports the symbol so existing imports like
``from momentum_agent.constants import ACCOUNT_STATE_DIM`` keep working.
"""

from momentum_core.constants import ACCOUNT_STATE_DIM

__all__ = ["ACCOUNT_STATE_DIM"]
