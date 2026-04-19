#!/usr/bin/env python3
"""One-shot recovery tool for collapsed training runs.

Picks a healthy pre-collapse checkpoint, optionally bakes in resets
(strip optimizer / scheduler state, refill NoisyLinear sigma), saves it
as the canonical resume target ``models/checkpoint_trainer_latest.pt``,
and prints the exact ``run_training`` command to use next.

Pure-stdlib + torch + the existing :mod:`momentum_train.utils.checkpoint_utils` —
no training loop and no GPU required.

Usage
-----

::

    python scripts/recover_from_collapse.py [--from-episode N]
                                            [--strip-optimizer]
                                            [--reset-noisy]
                                            [--noisy-sigma-init 0.5]
                                            [--reset-best-validation]
                                            [--benchmark-frac-override 0.15]
                                            [--model-dir models]
                                            [--config-path config/training_config.yaml]
                                            [--dry-run]

Behaviour
---------

1. **Pick checkpoint** in ``--model-dir``:

   * If any ``checkpoint_trainer_best_*.pt`` exists, pick the highest-score one.
   * Else if ``--from-episode N`` is provided, pick
     ``checkpoint_trainer_latest_*_ep{N}_*.pt`` (exact match).
   * Else **auto-pick**: scan ``validation_results_*.json`` files in the
     model dir, pair each with the nearest preceding
     ``checkpoint_trainer_latest_*_epK_*.pt``, and choose the most recent
     pairing whose ``validation_score`` is finite and not in the bottom
     25% of available scores.

2. **Mutate** the loaded checkpoint dict according to the flags:

   * ``--strip-optimizer`` removes ``optimizer_state_dict`` /
     ``scheduler_state_dict`` / ``scaler_state_dict`` (same effect as
     ``--reset-lr-on-resume`` baked in).
   * ``--reset-noisy`` builds the agent's ``RainbowNetwork`` from the saved
     ``agent_config``, loads the saved network state, calls
     :meth:`RainbowDQNAgent.reset_noisy_sigma` (online + target), then
     writes the modified state back into both ``network_state_dict`` and
     ``target_network_state_dict``.
   * ``--reset-best-validation`` zeros ``best_validation_metric`` (to
     ``-inf``) and ``early_stopping_counter`` (to ``0``). Use this when the
     validation distribution is about to change (e.g. you widened the
     validation window in ``split_config.yaml``) so old scores no longer
     compare to new ones, and the early-stop counter shouldn't carry over.

3. **Save** as a new episode-numbered file with
   ``episode = (max existing latest episode) + 1`` and a sentinel reward
   marker ``recover``. ``find_latest_checkpoint`` orders
   ``checkpoint_trainer_latest_*_ep*_reward*.pt`` by episode number, so this
   guarantees the recovery checkpoint wins the resume picker over any
   newer (collapsed) episode-numbered file. The ``rewardrecover`` marker
   makes recovery-origin checkpoints obvious in directory listings.

4. **Print** the exact resume command tailored to the flags chosen.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from momentum_train.utils.checkpoint_utils import load_checkpoint  # noqa: E402

LOGGER_NAME = "momentum_train.RecoverFromCollapse"
logger = logging.getLogger(LOGGER_NAME)


# ----------------------------------------------------------------------
# Checkpoint discovery
# ----------------------------------------------------------------------


_LATEST_FILE_RE = re.compile(r"^checkpoint_trainer_latest_(\d{8})_ep(\d+)_reward(.+)\.pt$")
_BEST_FILE_RE = re.compile(r"^checkpoint_trainer_best_(\d{8})_ep(\d+)(?:_score_(.+))?\.pt$")
_VALIDATION_RESULTS_RE = re.compile(r"^validation_results_(\d{8})_(\d{6})\.json$")


def _list_latest_checkpoints(model_dir: Path) -> list[tuple[int, Path]]:
    """Return ``(episode, path)`` tuples for every ``latest_*_ep*`` file."""
    out: list[tuple[int, Path]] = []
    for p in model_dir.glob("checkpoint_trainer_latest_*_ep*_reward*.pt"):
        m = _LATEST_FILE_RE.match(p.name)
        if not m:
            continue
        try:
            episode = int(m.group(2))
        except ValueError:
            continue
        out.append((episode, p))
    out.sort(key=lambda x: x[0])
    return out


def _list_best_checkpoints(model_dir: Path) -> list[tuple[float, int, Path]]:
    """Return ``(score, episode, path)`` for every ``best_*`` file with a score."""
    out: list[tuple[float, int, Path]] = []
    for p in model_dir.glob("checkpoint_trainer_best_*.pt"):
        m = _BEST_FILE_RE.match(p.name)
        if not m:
            continue
        score_str = m.group(3)
        if score_str is None:
            continue
        try:
            score = float(score_str)
            episode = int(m.group(2))
        except ValueError:
            continue
        if not math.isfinite(score):
            continue
        out.append((score, episode, p))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _load_validation_score(path: Path) -> float | None:
    """Best-effort extraction of an aggregate validation score from a results JSON.

    Different versions of the trainer have written slightly different shapes;
    we look at the most common keys and fall back to ``None`` rather than
    crashing on an unfamiliar layout.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to parse %s: %s", path, exc)
        return None

    for key in ("aggregate_score", "validation_score", "score"):
        if isinstance(data, dict) and key in data:
            try:
                value = float(data[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value

    if isinstance(data, dict):
        per_file = data.get("per_file_results") or data.get("per_file") or data.get("files")
        if isinstance(per_file, list) and per_file:
            scores: list[float] = []
            for entry in per_file:
                if not isinstance(entry, dict):
                    continue
                for key in ("score", "validation_score", "total_return_pct", "sharpe_ratio"):
                    if key in entry:
                        try:
                            value = float(entry[key])
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(value):
                            scores.append(value)
                            break
            if scores:
                return float(sum(scores) / len(scores))
    return None


def _validation_results(model_dir: Path) -> list[tuple[datetime, float, Path]]:
    """Return ``(timestamp, score, path)`` for every parseable validation results file."""
    out: list[tuple[datetime, float, Path]] = []
    for p in model_dir.glob("validation_results_*.json"):
        m = _VALIDATION_RESULTS_RE.match(p.name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            continue
        score = _load_validation_score(p)
        if score is None:
            continue
        out.append((ts, score, p))
    out.sort(key=lambda x: x[0])
    return out


def auto_pick_checkpoint(model_dir: Path) -> tuple[Path, str]:
    """Pick a healthy pre-collapse latest checkpoint via validation results.

    Algorithm:

    1. Build the score distribution from every parseable
       ``validation_results_*.json`` and drop the bottom 25%.
    2. Walk validation results from newest to oldest; for each one whose
       score survived the bottom-25% cut, find the latest ``epK`` checkpoint
       whose modification time is at or before that validation file's
       modification time. Return the first match.
    3. If nothing survives, fall back to the most recent latest checkpoint
       and warn the operator.

    Returns ``(checkpoint_path, justification_string)``.
    """
    val_results = _validation_results(model_dir)
    latest_files = _list_latest_checkpoints(model_dir)
    if not latest_files:
        raise FileNotFoundError(f"No checkpoint_trainer_latest_*_ep*_reward*.pt files found in {model_dir}; nothing to recover from.")

    if not val_results:
        episode, path = latest_files[-1]
        return path, f"no validation_results_*.json found; falling back to most recent latest checkpoint (ep {episode})."

    scores = [s for _, s, _ in val_results]
    scores_sorted = sorted(scores)
    # Drop (at least) the bottom 25% of validation scores. We round the drop
    # count UP so a small history with a clearly-bad recent run still excludes
    # that run (with N=5, ceil(0.25*5)=2 dropped, 3 survive). For N<4 we keep
    # everything — one or two data points isn't enough to call any of them bad.
    if len(scores_sorted) >= 4:
        cutoff_drop = max(1, -(-len(scores_sorted) // 4))  # ceil(N/4)
        cutoff = scores_sorted[cutoff_drop - 1]
    else:
        cutoff = float("-inf")

    latest_by_mtime = sorted(latest_files, key=lambda x: x[1].stat().st_mtime)

    for ts, score, vr_path in reversed(val_results):
        if score <= cutoff:
            continue
        vr_mtime = vr_path.stat().st_mtime
        candidate: tuple[int, Path] | None = None
        for episode, ckpt_path in latest_by_mtime:
            if ckpt_path.stat().st_mtime <= vr_mtime + 1.0:
                candidate = (episode, ckpt_path)
            else:
                break
        if candidate is None:
            continue
        episode, ckpt_path = candidate
        return ckpt_path, (
            f"auto-picked ep {episode} based on validation_results from "
            f"{ts.strftime('%Y-%m-%d %H:%M:%S UTC')} (score={score:.4f}, "
            f"bottom-25% cutoff={cutoff:.4f})."
        )

    episode, path = latest_files[-1]
    return path, (
        f"no validation_results survived the bottom-25% cutoff ({cutoff:.4f}); "
        f"falling back to most recent latest checkpoint (ep {episode})."
    )


def pick_checkpoint(model_dir: Path, from_episode: int | None) -> tuple[Path, str]:
    """Top-level checkpoint picker honouring the precedence in the docstring."""
    best_files = _list_best_checkpoints(model_dir)
    if best_files:
        score, episode, path = best_files[0]
        return path, f"highest-score best checkpoint (score={score:.4f}, ep {episode})."

    if from_episode is not None:
        for episode, path in _list_latest_checkpoints(model_dir):
            if episode == int(from_episode):
                return path, f"explicit --from-episode {episode}."
        raise FileNotFoundError(f"No checkpoint_trainer_latest_*_ep{from_episode}_*.pt found in {model_dir}.")

    return auto_pick_checkpoint(model_dir)


# ----------------------------------------------------------------------
# Mutations
# ----------------------------------------------------------------------


def strip_optimizer_state(checkpoint: dict) -> list[str]:
    """Drop optimizer / scheduler / scaler state from the checkpoint.

    Returns the names of keys that were present and removed (for logging).
    """
    removed: list[str] = []
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"):
        if checkpoint.pop(key, None) is not None:
            removed.append(key)
    return removed


def reset_best_validation_state(checkpoint: dict) -> dict[str, tuple[object, object]]:
    """Reset ``best_validation_metric`` and ``early_stopping_counter`` in-place.

    Use this when the validation distribution changes (e.g. validation window
    widened in ``split_config.yaml``). Old scores aren't comparable to new
    ones, so carrying ``best_validation_metric`` forward would suppress every
    new "best" update until the new run accidentally beats the old scale,
    and ``early_stopping_counter`` would burn its budget against unrelated
    cycles.

    Returns ``{key: (old_value, new_value)}`` for the keys that were touched
    (used for human-readable log output). Keys absent from the checkpoint are
    skipped silently — they'll be populated with sane defaults by the trainer
    on the first validation cycle.
    """
    changed: dict[str, tuple[object, object]] = {}
    if "best_validation_metric" in checkpoint:
        old = checkpoint["best_validation_metric"]
        new = float("-inf")
        checkpoint["best_validation_metric"] = new
        changed["best_validation_metric"] = (old, new)
    if "early_stopping_counter" in checkpoint:
        old = checkpoint["early_stopping_counter"]
        new = 0
        checkpoint["early_stopping_counter"] = new
        changed["early_stopping_counter"] = (old, new)
    return changed


def reset_noisy_in_checkpoint(checkpoint: dict, std_init: float | None) -> int:
    """Refill NoisyLinear sigma in the checkpoint's network state.

    We instantiate a CPU :class:`RainbowDQNAgent` in ``inference_only`` mode,
    load the saved weights, run :meth:`reset_noisy_sigma`, then write the
    modified state back into both ``network_state_dict`` and
    ``target_network_state_dict``.

    Returns the number of NoisyLinear layers that were touched.
    """
    from momentum_agent import RainbowDQNAgent

    agent_config = checkpoint.get("agent_config")
    if not isinstance(agent_config, dict):
        raise ValueError("Checkpoint missing valid agent_config; cannot rebuild agent for NoisyNet reset.")

    agent = RainbowDQNAgent(config=agent_config, device="cpu", inference_only=True)
    if not agent.load_state(checkpoint):
        raise RuntimeError("agent.load_state(checkpoint) returned False; cannot proceed with NoisyNet reset.")

    count = agent.reset_noisy_sigma(std_init=std_init)

    # Preserve the source checkpoint's key-prefix convention (`_orig_mod.*`
    # for torch.compile wrapped, plain otherwise) so the resume-side load
    # doesn't have to guess. The inference-only agent built above is always
    # eager (plain keys), so we re-add `_orig_mod.` when the source had it.
    def _format_like(src: dict | None, fresh_state: dict) -> dict:
        fresh_cpu = {k: v.detach().cpu() for k, v in fresh_state.items()}
        if isinstance(src, dict) and src and all(isinstance(k, str) and k.startswith("_orig_mod.") for k in src.keys()):
            return {f"_orig_mod.{k}": v for k, v in fresh_cpu.items()}
        return fresh_cpu

    inner_online = getattr(agent.network, "_orig_mod", agent.network)
    inner_target = getattr(agent.target_network, "_orig_mod", agent.target_network)
    checkpoint["network_state_dict"] = _format_like(checkpoint.get("network_state_dict"), inner_online.state_dict())
    checkpoint["target_network_state_dict"] = _format_like(checkpoint.get("target_network_state_dict"), inner_target.state_dict())
    return count


# ----------------------------------------------------------------------
# Save + command rendering
# ----------------------------------------------------------------------


def write_recovered_checkpoint(checkpoint: dict, model_dir: Path, src_episode: int) -> Path:
    """Persist the mutated checkpoint so resume picks it up.

    ``find_latest_checkpoint`` orders files by episode number, so we write
    the recovered checkpoint with episode = (max existing latest episode)
    + 1 to guarantee it wins. The reward marker is the literal string
    ``recover`` so it's obvious in directory listings that this file came
    from this script and not from training.
    """
    latest_files = _list_latest_checkpoints(model_dir)
    next_episode = (latest_files[-1][0] + 1) if latest_files else int(src_episode) + 1
    today = datetime.now().strftime("%Y%m%d")
    out_path = model_dir / f"checkpoint_trainer_latest_{today}_ep{next_episode}_rewardrecover.pt"

    checkpoint["episode"] = int(next_episode)

    torch.save(checkpoint, out_path)
    return out_path


def render_resume_command(
    *,
    benchmark_frac_override: float | None,
    config_path: str,
) -> str:
    parts = ["python -m momentum_train.run_training"]
    if config_path != "config/training_config.yaml":
        parts.append(f"--config_path {config_path}")
    parts.append("--resume")
    if benchmark_frac_override is not None:
        parts.append(f"--benchmark-frac-override {benchmark_frac_override}")
    return " ".join(parts)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing checkpoint_trainer_* and validation_results_*.json files.",
    )
    parser.add_argument(
        "--from-episode",
        type=int,
        default=None,
        help="Pick checkpoint_trainer_latest_*_ep{N}_*.pt explicitly (overrides auto-pick when no best exists).",
    )
    parser.add_argument(
        "--strip-optimizer",
        action="store_true",
        help="Drop optimizer/scheduler/scaler state from the checkpoint (equivalent to baking --reset-lr-on-resume in).",
    )
    parser.add_argument(
        "--reset-noisy",
        action="store_true",
        help="Refill NoisyLinear sigma parameters; mu (the deterministic part) is left untouched.",
    )
    parser.add_argument(
        "--noisy-sigma-init",
        type=float,
        default=None,
        help="Override the std_init scalar used when --reset-noisy is set; defaults to each layer's constructor value.",
    )
    parser.add_argument(
        "--reset-best-validation",
        action="store_true",
        help=(
            "Zero best_validation_metric (-> -inf) and early_stopping_counter "
            "(-> 0). Use when the validation distribution is about to change "
            "(e.g. you widened the validation window) so old scores no longer "
            "compare to new ones."
        ),
    )
    parser.add_argument(
        "--benchmark-frac-override",
        type=float,
        default=None,
        help="Forwarded to the printed resume command (does not modify the checkpoint).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/training_config.yaml",
        help="Forwarded to the printed resume command if it differs from the default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the chosen checkpoint and intended mutations without writing anything.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        logger.error("Model dir %s does not exist.", model_dir)
        return 2

    src_path, justification = pick_checkpoint(model_dir, args.from_episode)
    logger.info("Recovery source: %s", src_path)
    logger.info("Reason: %s", justification)

    checkpoint = load_checkpoint(str(src_path))
    if checkpoint is None:
        logger.error("Failed to load source checkpoint at %s.", src_path)
        return 3

    src_episode = int(checkpoint.get("episode", 0))

    mutations: list[str] = []
    if args.strip_optimizer:
        removed = strip_optimizer_state(checkpoint)
        if removed:
            mutations.append(f"strip-optimizer: removed {removed}")
        else:
            mutations.append("strip-optimizer: nothing to remove (already absent)")

    if args.reset_noisy:
        if args.dry_run:
            mutations.append("reset-noisy: SKIPPED in dry-run (would rebuild agent and refill sigma)")
        else:
            try:
                count = reset_noisy_in_checkpoint(checkpoint, std_init=args.noisy_sigma_init)
            except Exception as exc:
                logger.error("Failed to reset NoisyLinear sigma: %s", exc, exc_info=True)
                return 4
            label = "per-layer" if args.noisy_sigma_init is None else f"{float(args.noisy_sigma_init):.4f}"
            mutations.append(f"reset-noisy: refilled {count} online + {count} target NoisyLinear layer(s) (std_init={label})")

    if args.reset_best_validation:
        changed = reset_best_validation_state(checkpoint)
        if changed:
            details = ", ".join(f"{key}: {old!r} -> {new!r}" for key, (old, new) in changed.items())
            mutations.append(f"reset-best-validation: {details}")
        else:
            mutations.append("reset-best-validation: nothing to reset (best_validation_metric / early_stopping_counter both absent)")

    if args.dry_run:
        logger.info("Dry run: no checkpoint written.")
        if mutations:
            logger.info("Would apply: %s", "; ".join(mutations))
        cmd = render_resume_command(
            benchmark_frac_override=args.benchmark_frac_override,
            config_path=args.config_path,
        )
        logger.info("Would print resume command: %s", cmd)
        return 0

    out_path = write_recovered_checkpoint(checkpoint, model_dir, src_episode)
    logger.info("Wrote recovered checkpoint to %s", out_path)
    if mutations:
        for line in mutations:
            logger.info("  - %s", line)
    else:
        logger.info("  - no mutations applied (pure rollback)")

    cmd = render_resume_command(
        benchmark_frac_override=args.benchmark_frac_override,
        config_path=args.config_path,
    )
    print()
    print("Recovery checkpoint ready. Next step:")
    print(f"  {cmd}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
