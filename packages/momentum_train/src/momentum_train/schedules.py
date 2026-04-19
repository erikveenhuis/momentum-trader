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
