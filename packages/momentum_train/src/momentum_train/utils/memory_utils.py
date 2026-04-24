"""Glibc-aware memory hygiene helpers.

Why this module exists
----------------------

Training episodes on this project trigger two kinds of memory pressure:

1. **Per-save pickle spikes.** ``_save_checkpoint`` hands a dict that embeds
   the ~6 GB PER buffer to ``torch.save``, which serializes it to an
   in-memory pickle byte stream (~8 GB) before flushing to disk. The spike
   is transient, but glibc's default allocator (ptmalloc2) almost never
   ``munmap``s freed arena pages back to the kernel, so RSS ratchets up
   every save cycle and eventually crosses the 59 GiB system ceiling.

2. **Per-thread arena fragmentation.** ptmalloc2 creates one arena per
   thread up to 8 × cores by default. Torch and numpy's worker threads
   each get their own arena, and each arena keeps its own free-list. The
   total resident set is much larger than the live working set.

The OOM kills we've observed (Apr 22 and Apr 23 2026, both at ~54 GiB
anon-rss after 8-13 h) fit the model above. The two mitigations below
address those drivers directly without touching on-disk format:

* ``configure_glibc_arenas(n=2)`` — cap the per-thread arena count via
  ``mallopt(M_ARENA_MAX, n)``. Reduces steady-state RSS 20-40% on
  multi-threaded numpy/torch workloads with no code-shape change.

* ``release_memory_to_os()`` — after the transient spike caller is
  finished (i.e. ``torch.save`` returned), force Python to drop any stale
  references via ``gc.collect()``, then ask glibc to return freed arena
  pages to the kernel with ``malloc_trim(0)``. Without this, RSS stays at
  the peak high-water mark indefinitely.

Both helpers are **no-ops on non-glibc platforms** (macOS, Windows,
musl-libc). Callers do not need to guard the call site themselves; they
get a ``bool`` back indicating whether anything was actually done, and
failures are logged at DEBUG (not ERROR) because the fallback is simply
"allocator behaves as before".
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import sys
import threading

from momentum_core.logging import get_logger

logger = get_logger(__name__)

# glibc malloc.h: ``M_ARENA_MAX`` is ``-8``. Defined as a negative so
# ``mallopt`` can distinguish it from positive numeric params.
_M_ARENA_MAX = -8

_libc_lock = threading.Lock()
_libc: ctypes.CDLL | None = None
_libc_probed = False


def _get_libc() -> ctypes.CDLL | None:
    """Return a handle to glibc's libc, or ``None`` on non-glibc platforms.

    Cached after the first call. The probe is thread-safe so repeated calls
    from background writer threads don't race on ``ctypes.CDLL`` loading.
    """
    global _libc, _libc_probed
    if _libc_probed:
        return _libc
    with _libc_lock:
        if _libc_probed:
            return _libc
        _libc_probed = True
        if not sys.platform.startswith("linux"):
            logger.debug("Not on Linux (%s); glibc memory helpers are no-ops.", sys.platform)
            return None
        # Use ctypes.util.find_library to be tolerant of different libc paths
        # (libc.so.6 on most glibc distros; musl or alpine would fail here,
        # which is the desired behavior -- musl has no malloc_trim anyway).
        lib_name = ctypes.util.find_library("c") or "libc.so.6"
        try:
            libc = ctypes.CDLL(lib_name, use_errno=True)
        except OSError as exc:
            logger.debug("Could not load %s (%s); glibc memory helpers disabled.", lib_name, exc)
            return None
        # Confirm this libc actually exposes the glibc-specific entry points
        # we rely on. musl libc also lives at libc.so.* on some setups but
        # doesn't implement malloc_trim / mallopt with the same signatures.
        if not hasattr(libc, "malloc_trim") or not hasattr(libc, "mallopt"):
            logger.debug("Loaded %s lacks malloc_trim/mallopt; treating as non-glibc.", lib_name)
            return None
        libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        libc.mallopt.restype = ctypes.c_int
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        _libc = libc
        return _libc


def configure_glibc_arenas(n: int = 2) -> bool:
    """Cap glibc's per-thread arena count to ``n``.

    Equivalent to setting ``MALLOC_ARENA_MAX=n`` in the environment BEFORE
    the process starts, except it also works after Python has already
    initialized the allocator (which we have no choice about -- glibc
    reads the env var during its own ctor, so setting it from Python after
    startup has no effect). The in-process call uses the same underlying
    ``mallopt`` that the env var maps to.

    Args:
        n: Max number of arenas. ``1`` serializes all allocations through
            one mutex (highest RSS savings, most contention). ``2`` is the
            usual "tight but not pathological" setting for ML workloads.
            Values < 1 are coerced to 1.

    Returns:
        ``True`` if the cap was successfully applied, ``False`` if we're
        on a non-glibc platform or the syscall failed.
    """
    if n < 1:
        logger.debug("configure_glibc_arenas called with n=%d; clamping to 1.", n)
        n = 1
    libc = _get_libc()
    if libc is None:
        return False
    rc = libc.mallopt(_M_ARENA_MAX, n)
    if rc == 1:
        logger.info(
            "glibc arenas capped at %d via mallopt(M_ARENA_MAX, %d). This reduces per-thread "
            "allocator fragmentation for numpy/torch workloads.",
            n,
            n,
        )
        return True
    logger.debug("mallopt(M_ARENA_MAX, %d) returned %d (failure); leaving defaults.", n, rc)
    return False


def release_memory_to_os() -> bool:
    """Force a full Python GC sweep, then ask glibc to return freed pages.

    Call this right after a known transient-peak allocation (e.g. a
    ``torch.save`` that pickles a multi-GB dict). Without the ``malloc_trim``
    step, glibc keeps freed arena pages on its own free list forever --
    RSS stays pinned at the peak even though the Python objects are gone.

    Returns ``True`` if ``malloc_trim`` reported that memory was released.
    ``gc.collect`` always runs regardless of platform so the Python side
    of the cleanup is consistent.
    """
    # Drop any cycle-referenced / deferred objects first, then trim.
    gc.collect()
    libc = _get_libc()
    if libc is None:
        return False
    rc = libc.malloc_trim(0)
    # malloc_trim returns 1 if it actually freed something, 0 otherwise.
    # Either outcome is non-error; we just surface which happened at DEBUG.
    logger.debug("malloc_trim(0) returned %d (1 = pages released).", rc)
    return rc == 1


def current_rss_gb() -> float | None:
    """Return the process's current resident set size in GiB.

    Reads ``/proc/self/status`` on Linux (zero-dependency, no psutil
    required). Returns ``None`` on non-Linux platforms or if the file is
    unreadable for any reason. The caller should tolerate ``None`` because
    this helper is only used for diagnostic logging, not decisions.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    # Format: "VmRSS:    12345678 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        kib = int(parts[1])
                        return kib / (1024 * 1024)  # -> GiB
    except OSError as exc:
        logger.debug("Could not read /proc/self/status for RSS: %s", exc)
    return None
