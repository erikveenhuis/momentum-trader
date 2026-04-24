"""Tests for ``momentum_train.utils.memory_utils``.

The module is intentionally a thin wrapper over glibc ``mallopt`` and
``malloc_trim`` plus a ``/proc/self/status`` RSS reader, with no-op
fallbacks for non-glibc hosts. Tests cover:

* Happy path on Linux with glibc: ``configure_glibc_arenas`` returns True,
  ``release_memory_to_os`` doesn't raise, ``current_rss_gb`` returns a
  plausible positive float.
* Graceful no-op when the libc probe fails (monkeypatched).
* ``configure_glibc_arenas`` clamps ``n < 1`` to 1 rather than passing a
  negative value to ``mallopt`` (which would be a silent no-op at best
  and undefined at worst).
"""

from __future__ import annotations

import sys

import pytest
from momentum_train.utils import memory_utils
from momentum_train.utils.memory_utils import (
    configure_glibc_arenas,
    current_rss_gb,
    release_memory_to_os,
)


@pytest.fixture(autouse=True)
def _reset_libc_cache(monkeypatch):
    """Force each test to re-probe libc so monkeypatches don't leak across tests."""
    monkeypatch.setattr(memory_utils, "_libc", None, raising=False)
    monkeypatch.setattr(memory_utils, "_libc_probed", False, raising=False)


# --- Linux + glibc happy path --------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc-specific; skipped on non-Linux")
def test_configure_glibc_arenas_returns_true_on_glibc():
    """On a real glibc host, mallopt(M_ARENA_MAX, 2) must succeed."""
    assert configure_glibc_arenas(2) is True


@pytest.mark.unit
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc-specific; skipped on non-Linux")
def test_release_memory_to_os_does_not_raise():
    """malloc_trim(0) is safe to call even when nothing is trimmable."""
    # Return value may be True (something released) or False (nothing to
    # release); both are legal. The contract is "does not raise".
    result = release_memory_to_os()
    assert isinstance(result, bool)


@pytest.mark.unit
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Requires /proc/self/status")
def test_current_rss_gb_returns_plausible_positive_value():
    rss = current_rss_gb()
    assert rss is not None, "Should get an RSS reading on Linux"
    assert rss > 0.0, f"RSS should be positive, got {rss}"
    # Pytest worker processes should use well under 100 GiB; upper bound is a
    # sanity check that we haven't misread the KiB-to-GiB conversion.
    assert rss < 100.0, f"RSS reading seems implausibly large: {rss} GiB"


# --- No-op path (libc unavailable) ---------------------------------------


@pytest.mark.unit
def test_configure_glibc_arenas_returns_false_when_libc_unavailable(monkeypatch):
    """If the libc probe returns None, we must not crash; just return False."""
    monkeypatch.setattr(memory_utils, "_get_libc", lambda: None)
    assert configure_glibc_arenas(2) is False


@pytest.mark.unit
def test_release_memory_to_os_returns_false_when_libc_unavailable(monkeypatch):
    """gc.collect should still run; malloc_trim path becomes a no-op."""
    called = {"gc": False}

    def _fake_collect():
        called["gc"] = True
        return 0

    monkeypatch.setattr(memory_utils, "_get_libc", lambda: None)
    monkeypatch.setattr(memory_utils.gc, "collect", _fake_collect)
    assert release_memory_to_os() is False
    assert called["gc"], "gc.collect must run regardless of platform so Python-side cleanup is consistent"


# --- Argument clamping ---------------------------------------------------


@pytest.mark.unit
def test_configure_glibc_arenas_clamps_negative_n_to_one(monkeypatch):
    """n < 1 would be undefined behavior in mallopt; caller must be protected."""
    received = []

    class FakeLibc:
        def mallopt(self, key, value):
            received.append((key, value))
            return 1  # success

        def malloc_trim(self, _pad):
            return 0

    monkeypatch.setattr(memory_utils, "_get_libc", lambda: FakeLibc())
    configure_glibc_arenas(-5)
    assert received == [(-8, 1)], f"Expected M_ARENA_MAX(-8) set to clamped 1, got {received}"


@pytest.mark.unit
def test_configure_glibc_arenas_passes_through_valid_n(monkeypatch):
    received = []

    class FakeLibc:
        def mallopt(self, key, value):
            received.append((key, value))
            return 1

        def malloc_trim(self, _pad):
            return 0

    monkeypatch.setattr(memory_utils, "_get_libc", lambda: FakeLibc())
    configure_glibc_arenas(4)
    assert received == [(-8, 4)]


# --- _get_libc probe behavior -------------------------------------------


@pytest.mark.unit
def test_get_libc_returns_none_on_non_linux(monkeypatch):
    monkeypatch.setattr(memory_utils.sys, "platform", "darwin")
    assert memory_utils._get_libc() is None


@pytest.mark.unit
def test_get_libc_returns_none_when_cdll_raises(monkeypatch):
    """Treat ``OSError`` from CDLL load (e.g. musl libc without malloc_trim) as non-glibc."""
    monkeypatch.setattr(memory_utils.sys, "platform", "linux")
    monkeypatch.setattr(memory_utils.ctypes.util, "find_library", lambda name: "libc.so.6")

    def _raise(*_a, **_k):
        raise OSError("cannot load library")

    monkeypatch.setattr(memory_utils.ctypes, "CDLL", _raise)
    assert memory_utils._get_libc() is None
