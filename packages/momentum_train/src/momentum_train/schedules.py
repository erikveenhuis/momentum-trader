"""Hyperparameter schedules used by the trainer.

Each schedule is a small pure function that takes the current absolute
episode index and returns the value to use. Anchoring on the absolute
episode index (rather than a counter that resets at resume) means
``--resume`` continues an in-flight schedule instead of restarting it.
"""

from __future__ import annotations


def compute_benchmark_frac(
    episode: int,
    start: float,
    end: float,
    anneal_episodes: int,
) -> float:
    """Linearly anneal ``benchmark_allocation_frac`` from ``start`` to ``end``.

    The reward formula subtracts ``benchmark_allocation_frac * price_return``
    from the agent's per-step return. A high benchmark (e.g. 0.5) is useful
    early as anti-collapse pressure (sitting flat in any uptrending bar
    earns negative reward), but once the agent has demonstrated non-flat
    action diversity the same pressure becomes a structural long bias that
    fights the sniper objective. This schedule lets us start high and relax
    the pressure as training progresses.

    Args:
        episode: Absolute episode index (0-based). Negative values are
            clamped to 0; values beyond ``anneal_episodes`` clamp to ``end``.
        start: Value at episode 0.
        end: Value at and after ``anneal_episodes``. May be smaller or
            larger than ``start`` (the schedule simply interpolates).
        anneal_episodes: Number of episodes over which to interpolate. If
            ``<= 0`` the schedule is treated as a constant ``end``.

    Returns:
        The interpolated value, clamped to ``[min(start, end), max(start, end)]``.
    """
    start_f = float(start)
    end_f = float(end)
    anneal_int = int(anneal_episodes)
    ep = max(0, int(episode))

    if anneal_int <= 0:
        return end_f
    if ep >= anneal_int:
        return end_f
    progress = ep / anneal_int
    interpolated = start_f + (end_f - start_f) * progress
    lo = min(start_f, end_f)
    hi = max(start_f, end_f)
    if interpolated < lo:
        return lo
    if interpolated > hi:
        return hi
    return interpolated


def compute_curriculum_sampling(
    episode: int,
    num_episodes: int,
    *,
    mode: str,
    start_frac: float,
    end_frac: float,
    recent_frac: float,
) -> tuple[float, bool]:
    """Return ``(pool_frac, sample_from_recent)`` for training file selection.

    Modes (``trainer.curriculum_mode`` in ``training_config.yaml``):

    * ``flat`` — uniform random over **all** training files every episode.
    * ``linear_chronological`` — legacy ramp: earliest ``start_frac`` of files
      at episode 0, expanding linearly to ``end_frac`` by the final episode.
    * ``recent`` — uniform random over the newest ``recent_frac`` of files
      (chronological sort); useful when val/test are recent months.
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm == "flat":
        return 1.0, False
    if mode_norm == "recent":
        frac = min(max(float(recent_frac), 0.0), 1.0)
        return frac, True
    if mode_norm == "linear_chronological":
        start_f = float(start_frac)
        end_f = float(end_frac)
        ep_denom = max(int(num_episodes), 1)
        ep = max(0, int(episode))
        interpolated = start_f + (end_f - start_f) * (ep / ep_denom)
        lo = min(start_f, end_f)
        hi = max(start_f, end_f)
        pool_frac = min(max(interpolated, lo), hi)
        return min(max(pool_frac, 0.0), 1.0), False
    raise ValueError(f"Unknown curriculum_mode {mode!r}; expected flat, linear_chronological, or recent.")
