#!/usr/bin/env python3
"""Tier 5b: full PER buffer audit.

Loads the prioritized replay buffer from a trainer checkpoint and emits a
comprehensive JSON report:

* Reward histogram (binned + raw quantiles).
* Per-action priority statistics (count, mean, p50, p90, p99, top-share).
* FIFO half-life — proxy for "how recent is the data the learner sees?"
  estimated from the SumTree write pointer and the buffer fill ratio.
* Top-100 highest-priority transitions, with their action, reward, and
  approximate age in steps.

Unlike the in-loop Tier 4a audit (4096-sample, throttled), this script walks
*every* stored transition and is meant to be run offline when something looks
off in the live PER scalars.

Output: ``output/audit_per_buffer_<timestamp>.json`` (or ``--output-path``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_train.run_training import configure_logging
from momentum_train.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint

LOGGER_NAME = "momentum_train.AuditPerBuffer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=str, default="config/training_config.yaml")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit trainer checkpoint. Defaults to latest in run.model_dir.",
    )
    parser.add_argument("--checkpoint-prefix", type=str, default="checkpoint_trainer")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to write the JSON report (default: output/audit_per_buffer_<timestamp>.json).",
    )
    parser.add_argument(
        "--reward-bins",
        type=int,
        default=50,
        help="Number of histogram bins for the reward distribution.",
    )
    parser.add_argument("--top-n", type=int, default=100, help="Top-N highest-priority transitions to dump.")
    parser.add_argument("--log-level", type=str, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level config in {path} must be a mapping.")
    return cfg


def _resolve_checkpoint(args: argparse.Namespace, run_cfg: dict) -> Path | None:
    if args.checkpoint_path:
        p = Path(args.checkpoint_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    model_dir = Path(run_cfg.get("model_dir", "models")).expanduser()
    latest = find_latest_checkpoint(str(model_dir), args.checkpoint_prefix)
    return Path(latest).resolve() if latest else None


def _extract_buffer_state(checkpoint: dict) -> dict:
    """Pull the PER buffer state dict out of a trainer checkpoint."""
    buf_state = checkpoint.get("buffer_state")
    if not isinstance(buf_state, dict):
        raise ValueError("Checkpoint does not contain a buffer_state. Was it saved before PER persistence was enabled?")
    for key in ("buffer", "tree_state", "buffer_write_idx", "max_priority", "capacity"):
        if key not in buf_state:
            raise ValueError(f"Buffer state is missing required key {key!r}.")
    return buf_state


def _compute_buffer_arrays(buf_state: dict) -> dict:
    """Build numpy arrays of (rewards, actions, priorities, age_in_writes)."""
    experiences = list(buf_state["buffer"])
    capacity = int(buf_state["capacity"])
    tree_arr = np.asarray(buf_state["tree_state"]["tree"])
    write_ptr = int(buf_state["tree_state"]["write"])
    size = int(buf_state["tree_state"]["size"])

    # Leaf priorities live in the last ``capacity`` entries of the SumTree.
    leaves = tree_arr[capacity - 1 : 2 * capacity - 1]
    rewards = np.empty(len(experiences), dtype=np.float64)
    actions = np.empty(len(experiences), dtype=np.int64)
    priorities = np.empty(len(experiences), dtype=np.float64)
    age_in_writes = np.empty(len(experiences), dtype=np.int64)

    # Lockstep writes (PrioritizedReplayBuffer.store) mean buffer index k <->
    # SumTree leaf k for every k < size. Older entries (relative to ``write``)
    # have larger ``age``; we cap at ``size`` for clarity.
    for i, exp in enumerate(experiences):
        rewards[i] = float(getattr(exp, "reward", 0.0))
        actions[i] = int(getattr(exp, "action", 0))
        priorities[i] = float(leaves[i]) if i < len(leaves) else 0.0
        # Age = number of writes ago this slot was last written. Newest entry
        # (just before ``write``) is age 1; oldest is age ``size``.
        age_in_writes[i] = ((write_ptr - 1 - i) % capacity) + 1

    return {
        "rewards": rewards,
        "actions": actions,
        "priorities": priorities,
        "ages": age_in_writes,
        "size": size,
        "capacity": capacity,
        "write_ptr": write_ptr,
    }


def _percentile_or_nan(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _build_report(arrays: dict, *, reward_bins: int, top_n: int) -> dict:
    rewards = arrays["rewards"]
    actions = arrays["actions"]
    priorities = arrays["priorities"]
    ages = arrays["ages"]
    size = arrays["size"]
    capacity = arrays["capacity"]

    # Reward histogram + robust quantiles.
    if rewards.size:
        hist_counts, hist_edges = np.histogram(rewards, bins=reward_bins)
    else:
        hist_counts = np.zeros(reward_bins, dtype=np.int64)
        hist_edges = np.zeros(reward_bins + 1, dtype=np.float64)
    reward_summary = {
        "count": int(rewards.size),
        "mean": float(rewards.mean()) if rewards.size else float("nan"),
        "std": float(rewards.std(ddof=0)) if rewards.size else float("nan"),
        "min": float(rewards.min()) if rewards.size else float("nan"),
        "max": float(rewards.max()) if rewards.size else float("nan"),
        "p1": _percentile_or_nan(rewards, 1),
        "p50": _percentile_or_nan(rewards, 50),
        "p99": _percentile_or_nan(rewards, 99),
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        },
    }

    # Per-action priority breakdown.
    counter = Counter(int(a) for a in actions.tolist())
    by_action: dict[int, dict[str, float]] = {}
    total_priority = float(priorities.sum()) if priorities.size else 0.0
    for action, cnt in sorted(counter.items()):
        mask = actions == action
        sample = priorities[mask]
        by_action[int(action)] = {
            "count": int(cnt),
            "share": float(cnt / actions.size) if actions.size else 0.0,
            "priority_mean": float(sample.mean()) if sample.size else 0.0,
            "priority_p50": _percentile_or_nan(sample, 50),
            "priority_p90": _percentile_or_nan(sample, 90),
            "priority_p99": _percentile_or_nan(sample, 99),
            "priority_share": float(sample.sum() / total_priority) if total_priority > 0 else 0.0,
        }

    # FIFO half-life. Under uniform sampling and FIFO eviction the median
    # transition age (in environment steps) is half the buffer fill.
    fifo_half_life = float(size) / 2.0 if size > 0 else 0.0

    # Top-N highest priority transitions (most-shaping examples).
    top_records: list[dict] = []
    if priorities.size:
        n = min(int(top_n), priorities.size)
        top_idx = np.argpartition(-priorities, kth=max(0, n - 1))[:n]
        top_idx = top_idx[np.argsort(-priorities[top_idx])]
        for rank, idx in enumerate(top_idx, start=1):
            top_records.append(
                {
                    "rank": rank,
                    "buffer_index": int(idx),
                    "priority": float(priorities[idx]),
                    "reward": float(rewards[idx]),
                    "action": int(actions[idx]),
                    "age_in_writes": int(ages[idx]),
                }
            )

    return {
        "buffer_meta": {
            "size": int(size),
            "capacity": int(capacity),
            "fill_ratio": float(size / capacity) if capacity > 0 else 0.0,
            "write_ptr": int(arrays["write_ptr"]),
            "fifo_half_life_steps": fifo_half_life,
            "total_priority": total_priority,
            "max_priority": float(priorities.max()) if priorities.size else 0.0,
        },
        "reward": reward_summary,
        "by_action": by_action,
        "top_n": top_records,
    }


def main() -> int:
    args = parse_args()
    cfg = _load_config(Path(args.config_path).expanduser().resolve())
    configure_logging(args.log_level, cfg.get("logging"))
    logger = logging.getLogger(LOGGER_NAME)

    run_cfg = cfg.setdefault("run", {})
    ckpt_path = _resolve_checkpoint(args, run_cfg)
    if ckpt_path is None:
        logger.error("No checkpoint found and none provided via --checkpoint-path; aborting.")
        return 2
    logger.info("Loading checkpoint %s", ckpt_path)
    checkpoint = load_checkpoint(str(ckpt_path))
    if not checkpoint:
        logger.error("load_checkpoint returned an empty payload for %s", ckpt_path)
        return 3

    buf_state = _extract_buffer_state(checkpoint)
    arrays = _compute_buffer_arrays(buf_state)
    report = _build_report(arrays, reward_bins=args.reward_bins, top_n=args.top_n)
    report["_meta"] = {
        "checkpoint_path": str(ckpt_path),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "reward_bins": args.reward_bins,
        "top_n": args.top_n,
    }

    out_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else Path("output").resolve() / f"audit_per_buffer_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    logger.info("PER buffer audit written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
