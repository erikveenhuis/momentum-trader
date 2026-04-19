#!/usr/bin/env python3
"""Tier 5a: deterministic greedy rollout over train/val/test splits.

Loads the latest checkpoint (or one specified via ``--checkpoint-path``), runs
the agent in :func:`agent.greedy` mode (epsilon=0, NoisyNet noise frozen) over a
configurable subset of files from one or more splits, and persists the artifacts
the offline analyzer (Tier 5c) consumes:

* ``<file>.steps.parquet`` — one row per environment step with action,
  position, price, portfolio value, reward, transaction cost, and
  ``was_greedy`` (always True under greedy mode, kept for downstream parity).
* ``<file>.trades.jsonl`` — closed trades produced by
  :func:`segment_trades`, one JSON object per line.
* ``<file>.q_snapshots.npz`` — captured ``[K, num_actions, num_atoms]``
  Q-distributions sampled every ``--q-snapshot-every`` steps (configurable),
  plus the support array. Lets us inspect distribution skew / collapse offline.
* ``<file>.non_a0_triggers.jsonl`` — every step where the greedy action was
  non-zero, including the full Q-vector. This is the sniper's trigger log
  and is the most useful artifact for debugging "why did we just open a
  position?".
* ``summary.json`` — per-split aggregates including mean per-trade Sharpe,
  hit rate, expectancy, and basic Q-value summaries.

Outputs are written under ``--output-dir/eval_greedy_<timestamp>/<split>/``.
The script is read-only with respect to the model — it never touches the
optimizer, scheduler, or PER buffer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_agent import RainbowDQNAgent
from momentum_env import TradingEnv
from momentum_env.config import TradingEnvConfig
from momentum_train.data import DataManager
from momentum_train.run_training import configure_logging
from momentum_train.trade_metrics import (
    StepRecord,
    aggregate_trade_metrics,
    segment_trades,
)
from momentum_train.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint
from momentum_train.utils.utils import set_seeds

LOGGER_NAME = "momentum_train.EvalGreedy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=str, default="config/training_config.yaml")
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated subset of {train,val,test}.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Random files to sample per split (capped at availability).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Parent directory for the per-run output bundle.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Explicit trainer checkpoint path. Defaults to latest in run.model_dir.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="checkpoint_trainer",
        help="Prefix used for autodiscovery in run.model_dir.",
    )
    parser.add_argument(
        "--q-snapshot-every",
        type=int,
        default=50,
        help="Capture the full Q-distribution every Nth step (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for file sampling.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU when no GPU/MPS is available.",
    )
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


def _select_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("No CUDA/MPS device available; pass --allow-cpu to use CPU.")


def _resolve_split_files(data_manager: DataManager, splits: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if "train" in splits:
        out["train"] = list(data_manager.get_training_files())
    if "val" in splits:
        out["val"] = list(data_manager.get_validation_files())
    if "test" in splits:
        out["test"] = list(data_manager.get_test_files())
    return out


def _sample_subset(files: list[Path], count: int, rng: np.random.Generator) -> list[Path]:
    if not files:
        return []
    if count <= 0 or count >= len(files):
        return sorted(files)
    idx = rng.choice(len(files), size=count, replace=False)
    return sorted([files[int(i)] for i in idx])


def _obs_to_batch(obs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a single observation into the (market, account) tensor pair."""
    market = torch.from_numpy(np.asarray(obs["market_data"])).to(device, dtype=torch.float32)
    account = torch.from_numpy(np.asarray(obs["account_state"])).to(device, dtype=torch.float32)
    if market.dim() == 2:
        market = market.unsqueeze(0)
    if account.dim() == 1:
        account = account.unsqueeze(0)
    return market, account


def _q_snapshot(agent: RainbowDQNAgent, market: torch.Tensor, account: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_values [num_actions], q_distribution [num_actions, num_atoms]).

    Computed under no_grad in eval mode. Used both for capturing snapshots
    every Nth step and for the non-a0 trigger log on every non-flat action.
    """
    net = getattr(agent.network, "_orig_mod", agent.network)
    with torch.no_grad():
        log_probs = net(market, account)  # [B, A, atoms]
        probs = torch.exp(log_probs)
        support = net.support  # [atoms]
        q_values = (probs * support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    return (
        q_values[0].detach().cpu().numpy().astype(np.float32),
        probs[0].detach().cpu().numpy().astype(np.float32),
    )


def _build_env(config: dict, data_path: str) -> TradingEnv:
    env_cfg = dict(config.get("env", {}))
    env_cfg["data_path"] = data_path
    return TradingEnv(config=TradingEnvConfig(**env_cfg))


def _rollout(
    agent: RainbowDQNAgent,
    env: TradingEnv,
    *,
    device: torch.device,
    q_snapshot_every: int,
) -> dict:
    """Run a single greedy episode; return per-step records + Q snapshots."""
    obs, _ = env.reset()
    steps: list[StepRecord] = []
    rewards: list[float] = []
    q_means: list[float] = []
    q_max_actions: list[int] = []
    snapshot_steps: list[int] = []
    snapshot_dists: list[np.ndarray] = []
    non_a0_triggers: list[dict] = []

    done = False
    step_index = 0
    while not done:
        market, account = _obs_to_batch(obs, device)
        # Compute Q-values directly so we can record provenance and snapshots
        # without paying for two forwards. The agent's select_action would do
        # the same forward internally.
        q_values_np, q_dist_np = _q_snapshot(agent, market, account)
        action = int(np.argmax(q_values_np))
        q_means.append(float(q_values_np.mean()))
        q_max_actions.append(action)

        if q_snapshot_every > 0 and step_index % q_snapshot_every == 0:
            snapshot_steps.append(step_index)
            snapshot_dists.append(q_dist_np)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        rewards.append(float(reward))
        step_index += 1

        steps.append(
            StepRecord(
                step_index=step_index,
                portfolio_value=float(info.get("portfolio_value", 0.0) or 0.0),
                position=float(info.get("position", 0.0) or 0.0),
                price=float(info.get("price", 0.0) or 0.0),
                action=int(action),
                transaction_cost=float(info.get("step_transaction_cost", 0.0) or 0.0),
                was_greedy=True,
            )
        )

        if action != 0:
            non_a0_triggers.append(
                {
                    "step_index": step_index,
                    "action": action,
                    "price": float(info.get("price", 0.0) or 0.0),
                    "portfolio_value": float(info.get("portfolio_value", 0.0) or 0.0),
                    "position": float(info.get("position", 0.0) or 0.0),
                    "reward": float(reward),
                    "q_values": q_values_np.tolist(),
                    "q_argmax_margin": float(np.partition(q_values_np, -2)[-1] - np.partition(q_values_np, -2)[-2]),
                }
            )

        obs = next_obs

    return {
        "steps": steps,
        "rewards": rewards,
        "q_means": q_means,
        "q_max_actions": q_max_actions,
        "snapshot_steps": snapshot_steps,
        "snapshot_dists": snapshot_dists,
        "non_a0_triggers": non_a0_triggers,
        "support": getattr(getattr(agent.network, "_orig_mod", agent.network), "support").detach().cpu().numpy(),
    }


def _write_steps_table(steps: list[StepRecord], rewards: list[float], path: Path) -> None:
    """Persist per-step trace as parquet (preferred) or CSV fallback."""
    rows = []
    for step, r in zip(steps, rewards):
        d = asdict(step)
        d["reward"] = float(r)
        rows.append(d)
    try:
        import pandas as pd  # local import keeps script importable without pandas

        df = pd.DataFrame(rows)
        try:
            df.to_parquet(path.with_suffix(".steps.parquet"), index=False)
            return
        except Exception:  # pragma: no cover - parquet engine missing
            df.to_csv(path.with_suffix(".steps.csv"), index=False)
            return
    except ImportError:
        # Fall through to JSONL so we never lose the data.
        with path.with_suffix(".steps.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


def _write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + "\n")


def _write_q_snapshots(
    snapshot_steps: list[int],
    snapshot_dists: list[np.ndarray],
    support: np.ndarray,
    path: Path,
) -> None:
    if not snapshot_dists:
        return
    np.savez_compressed(
        path.with_suffix(".q_snapshots.npz"),
        steps=np.asarray(snapshot_steps, dtype=np.int64),
        distributions=np.stack(snapshot_dists, axis=0).astype(np.float32),
        support=np.asarray(support, dtype=np.float32),
    )


def _summarize_split(
    per_file: list[dict],
) -> dict:
    if not per_file:
        return {"files": 0}
    trade_counts = [r["trade_metrics"].get("num_trades", 0) for r in per_file]
    sharpes = [r["trade_metrics"].get("per_trade_sharpe", float("nan")) for r in per_file]
    hit_rates = [r["trade_metrics"].get("hit_rate", float("nan")) for r in per_file]
    expectancies = [r["trade_metrics"].get("expectancy_pct", float("nan")) for r in per_file]
    pvs = [r["final_portfolio_value"] for r in per_file]
    q_means = [r["q_mean"] for r in per_file if r["q_mean"] is not None]
    return {
        "files": len(per_file),
        "trades_total": int(sum(trade_counts)),
        "trades_mean": float(np.mean(trade_counts)) if trade_counts else 0.0,
        "per_trade_sharpe_mean": float(np.nanmean(sharpes)) if sharpes else float("nan"),
        "hit_rate_mean": float(np.nanmean(hit_rates)) if hit_rates else float("nan"),
        "expectancy_pct_mean": float(np.nanmean(expectancies)) if expectancies else float("nan"),
        "final_portfolio_value_mean": float(np.mean(pvs)) if pvs else float("nan"),
        "q_mean_overall": float(np.mean(q_means)) if q_means else float("nan"),
    }


def main() -> int:
    args = parse_args()
    cfg = _load_config(Path(args.config_path).expanduser().resolve())
    configure_logging(args.log_level, cfg.get("logging"))
    logger = logging.getLogger(LOGGER_NAME)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    valid_splits = {"train", "val", "test"}
    bad = sorted(set(splits) - valid_splits)
    if bad:
        raise ValueError(f"Unknown splits requested: {bad}; valid options: {sorted(valid_splits)}")

    seed = args.seed if args.seed is not None else cfg.get("trainer", {}).get("seed")
    if seed is not None:
        set_seeds(int(seed))
    rng = np.random.default_rng(seed)

    run_cfg = cfg.setdefault("run", {})
    agent_cfg = cfg.setdefault("agent", {})

    data_manager = DataManager(base_dir=run_cfg.get("data_base_dir", "data"))
    data_manager.organize_data()
    split_files = _resolve_split_files(data_manager, splits)

    device = _select_device(args.allow_cpu)
    logger.info("Greedy eval on device %s; splits=%s", device, splits)

    agent = RainbowDQNAgent(config=agent_cfg, device=device, scaler=None)

    model_dir = Path(run_cfg.get("model_dir", "models")).expanduser()
    ckpt_path = (
        Path(args.checkpoint_path).expanduser().resolve()
        if args.checkpoint_path
        else (Path(p).resolve() if (p := find_latest_checkpoint(str(model_dir), args.checkpoint_prefix)) else None)
    )
    if ckpt_path is None:
        logger.warning("No checkpoint found; running with randomly initialized weights (sanity check only).")
    else:
        logger.info("Loading checkpoint %s", ckpt_path)
        ckpt = load_checkpoint(str(ckpt_path))
        if ckpt:
            agent.load_state(ckpt)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_dir).expanduser().resolve() / f"eval_greedy_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Writing outputs under %s", output_root)

    overall_summary: dict[str, dict] = {}
    greedy_ctx = agent.greedy() if hasattr(agent, "greedy") else nullcontext()

    with greedy_ctx:
        for split in splits:
            files = split_files.get(split, [])
            if not files:
                logger.warning("Split %s has no files; skipping.", split)
                continue
            sampled = _sample_subset(files, args.num_files, rng)
            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            per_file_records: list[dict] = []
            for f in sampled:
                t0 = time.perf_counter()
                env = _build_env(cfg, str(f))
                try:
                    rollout = _rollout(agent, env, device=device, q_snapshot_every=args.q_snapshot_every)
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

                trades = segment_trades(rollout["steps"])
                trade_metrics = aggregate_trade_metrics(trades)
                base = split_dir / f.stem
                _write_steps_table(rollout["steps"], rollout["rewards"], base)
                _write_jsonl((t.to_dict() for t in trades), base.with_suffix(".trades.jsonl"))
                _write_jsonl(rollout["non_a0_triggers"], base.with_suffix(".non_a0_triggers.jsonl"))
                _write_q_snapshots(rollout["snapshot_steps"], rollout["snapshot_dists"], rollout["support"], base)

                final_pv = float(rollout["steps"][-1].portfolio_value) if rollout["steps"] else float("nan")
                per_file_records.append(
                    {
                        "file": f.name,
                        "steps": len(rollout["steps"]),
                        "elapsed_sec": time.perf_counter() - t0,
                        "final_portfolio_value": final_pv,
                        "trade_metrics": trade_metrics,
                        "q_mean": float(np.mean(rollout["q_means"])) if rollout["q_means"] else None,
                        "non_a0_count": len(rollout["non_a0_triggers"]),
                    }
                )
                logger.info(
                    "[%s] %s: steps=%d, trades=%d, final_pv=%.2f, non_a0=%d, sharpe=%.3f, hit_rate=%.3f",
                    split,
                    f.name,
                    len(rollout["steps"]),
                    trade_metrics.get("num_trades", 0),
                    final_pv,
                    len(rollout["non_a0_triggers"]),
                    trade_metrics.get("per_trade_sharpe", float("nan")),
                    trade_metrics.get("hit_rate", float("nan")),
                )

            split_summary = {
                "files_evaluated": [r["file"] for r in per_file_records],
                "per_file": per_file_records,
                "aggregate": _summarize_split(per_file_records),
            }
            with (split_dir / "summary.json").open("w", encoding="utf-8") as fh:
                json.dump(split_summary, fh, indent=2, default=float)
            overall_summary[split] = split_summary["aggregate"]

    overall_summary["_meta"] = {
        "timestamp_utc": timestamp,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "config_path": args.config_path,
        "splits": splits,
        "num_files_per_split": args.num_files,
        "q_snapshot_every": args.q_snapshot_every,
    }
    with (output_root / "overall_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(overall_summary, fh, indent=2, default=float)
    logger.info("Greedy eval complete: %s", output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
