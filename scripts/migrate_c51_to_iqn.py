#!/usr/bin/env python3
"""One-shot migration: warm-start a fresh IQN agent from a pre-IQN (C51) checkpoint.

The Beyond-the-Rainbow upgrade replaces the C51 distributional head with an
IQN head, layers Munchausen-DQN onto the target, and (optionally) wraps the
dueling heads with spectral normalization. Pre-IQN checkpoints are
deliberately *not* loadable by the new agent (the architecture rule
`Old checkpoints incompatible after architecture changes` is enforced via
`agent_checkpoint._assert_iqn_state_dict`). To preserve the encoder's many
hours of training, this script transfers only the *encoder + auxiliary*
tensors from a chosen source ``best_*.pt`` into a freshly-initialised
IQN-shaped checkpoint, copies the closest replay-buffer side-car next to
it with all PER priorities reset to ``priority_max`` (i.e. uniform
sampling for the first few thousand learner steps until IQN TD-errors
re-prioritise organically), and discards the optimizer / scheduler /
scaler / validation-metadata that no longer apply.

Usage
-----

::

    python scripts/migrate_c51_to_iqn.py \\
        --source-checkpoint models/checkpoint_trainer_best_20260426_ep6803_score_0.4479.pt \\
        --buffer-source     models/checkpoint_trainer_latest_20260427_ep8403_reward0.4479.buffer \\
        --config            config/training_config.yaml \\
        --output-stem       models/checkpoint_trainer_latest_20260427_ep6803_iqn_warmstart \\
        [--dry-run]

The script never modifies the source files. It writes only
``<output-stem>.pt`` (and, when a buffer source is supplied,
``<output-stem>.buffer/``). With ``--dry-run`` it prints the summary and
writes nothing.

Tensors transferred (after stripping any ``_orig_mod.`` prefix that
``torch.compile`` adds):

* ``feature_embedding.{weight,bias}``
* ``cls_token``
* ``pos_encoder.pe`` (buffer)
* ``transformer_encoder.layers.*``
* ``account_processor.*.{weight,bias}``
* ``head_norm.{weight,bias}``
* ``aux_return_head.*.{weight,bias}``

Tensors *not* transferred (left at the IQN agent's fresh init):

* ``value_stream.*`` and ``advantage_stream.*`` (C51 → IQN shape change)
* ``tau_embedding.*`` (does not exist in the C51 graph)
* any ``parametrizations.weight_mu.*`` (spectral-norm artefacts)
* anything matching ``support`` / ``delta_z`` (C51-only buffers)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Make the script runnable both directly and via ``python -m``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "packages" / "momentum_agent" / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "packages" / "momentum_agent" / "src"))

from momentum_agent.agent import RainbowDQNAgent  # noqa: E402

logger = logging.getLogger("migrate_c51_to_iqn")


# ---------------------------------------------------------------------------
# Tensor-transfer policy. The list below is intentionally exhaustive; if a
# new encoder-side parameter is added later, extend ``_ALLOWED_PREFIXES``
# and re-run the migration tests. ``_FORBIDDEN_PATTERNS`` is the no-fly
# zone (C51 leftovers we explicitly drop).
# ---------------------------------------------------------------------------

_ALLOWED_PREFIXES: tuple[str, ...] = (
    "feature_embedding.",
    "cls_token",
    "pos_encoder.",
    "transformer_encoder.",
    "account_processor.",
    "head_norm.",
    "aux_return_head.",
)

_FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^value_stream\."),
    re.compile(r"^advantage_stream\."),
    re.compile(r"^tau_embedding\."),
    re.compile(r"^support$"),
    re.compile(r"^delta_z$"),
    re.compile(r"\.parametrizations\."),  # spectral-norm artefacts
)

_ORIG_MOD_PREFIX = "_orig_mod."


@dataclass(frozen=True)
class TransferReport:
    transferred: tuple[str, ...]
    fresh: tuple[str, ...]
    skipped_forbidden: tuple[str, ...]
    missing_in_source: tuple[str, ...]


def _strip_orig_mod(name: str) -> str:
    """``torch.compile`` wraps the module under ``_orig_mod``; strip that
    prefix once so the transfer logic operates on canonical names."""
    return name[len(_ORIG_MOD_PREFIX) :] if name.startswith(_ORIG_MOD_PREFIX) else name


def _is_allowed(name: str) -> bool:
    return any(name.startswith(p) for p in _ALLOWED_PREFIXES)


def _is_forbidden(name: str) -> bool:
    return any(p.search(name) for p in _FORBIDDEN_PATTERNS)


def _normalize_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collapse the ``_orig_mod.`` prefix and drop forbidden keys."""
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        canonical = _strip_orig_mod(key)
        if _is_forbidden(canonical):
            continue
        out[canonical] = value
    return out


def _is_pre_iqn_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """Heuristic: pre-IQN state dicts have ``support``/``delta_z`` buffers
    OR have no ``tau_embedding.linear.weight`` / ``tau_embedding.indices``
    parameter (the IQN signature)."""
    has_iqn_marker = any(_strip_orig_mod(k).startswith("tau_embedding.") for k in state_dict)
    has_c51_marker = any(_strip_orig_mod(k) in ("support", "delta_z") for k in state_dict)
    if has_c51_marker:
        return True
    return not has_iqn_marker


def _transfer_encoder_tensors(
    source: dict[str, torch.Tensor],
    fresh: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], TransferReport]:
    """Copy whitelisted tensors from ``source`` into ``fresh``; return the
    merged dict + a report of what happened."""
    transferred: list[str] = []
    fresh_kept: list[str] = []
    skipped_forbidden: list[str] = []
    missing_in_source: list[str] = []

    src_norm = _normalize_state_dict(source)
    out: dict[str, torch.Tensor] = dict(fresh)

    for name in fresh:
        canonical = _strip_orig_mod(name)
        if not _is_allowed(canonical):
            fresh_kept.append(name)
            continue
        if canonical in src_norm:
            src_tensor = src_norm[canonical]
            if src_tensor.shape != fresh[name].shape:
                # Shape mismatch on an encoder-side tensor is a hard error:
                # it would silently corrupt the warm-start.
                raise ValueError(
                    f"Shape mismatch on '{canonical}': source={tuple(src_tensor.shape)} "
                    f"fresh={tuple(fresh[name].shape)}. Refusing to migrate."
                )
            out[name] = src_tensor.clone().to(dtype=fresh[name].dtype)
            transferred.append(canonical)
        else:
            missing_in_source.append(canonical)

    for src_key in src_norm:
        if _is_forbidden(src_key):
            skipped_forbidden.append(src_key)

    return out, TransferReport(
        transferred=tuple(sorted(set(transferred))),
        fresh=tuple(sorted(set(fresh_kept))),
        skipped_forbidden=tuple(sorted(set(skipped_forbidden))),
        missing_in_source=tuple(sorted(set(missing_in_source))),
    )


# ---------------------------------------------------------------------------
# Buffer side-car: copy + reset PER priorities to ``priority_max``.
# ---------------------------------------------------------------------------


def _reset_buffer_priorities_in_place(buffer_dir: Path) -> tuple[int, float]:
    """Overwrite all SumTree priorities at ``buffer_dir/sumtree.npz`` with
    the buffer's stored ``max_priority`` (alpha-space). Internal sum nodes
    are recomputed bottom-up. Returns ``(num_leaves_reset, priority)``.
    """
    sumtree_path = buffer_dir / "sumtree.npz"
    meta_path = buffer_dir / "meta.json"
    if not sumtree_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Buffer side-car at {buffer_dir} is missing sumtree.npz or meta.json.")

    meta = json.loads(meta_path.read_text())
    capacity = int(meta["capacity"])
    size = int(meta["size"])
    priority = float(meta.get("max_priority", 1.0)) or 1.0

    with np.load(sumtree_path) as npz:
        tree = np.array(npz["tree"])
        data_indices = np.array(npz["data_indices"])
        write = np.int64(npz["write"])
        size_npz = np.int64(npz["size"])

    if tree.shape != (2 * capacity - 1,):
        raise ValueError(f"SumTree shape {tree.shape} inconsistent with capacity {capacity}.")

    # Reset all leaves: leaves live at indices [capacity-1 : 2*capacity-1].
    # The first ``size`` of those are populated; the rest stay zero.
    tree[:] = 0.0
    leaf_start = capacity - 1
    if size > 0:
        tree[leaf_start : leaf_start + size] = priority
    # Rebuild internal nodes bottom-up: tree[i] = tree[2i+1] + tree[2i+2].
    for i in range(leaf_start - 1, -1, -1):
        tree[i] = tree[2 * i + 1] + tree[2 * i + 2]

    np.savez(
        sumtree_path,
        tree=tree,
        data_indices=data_indices,
        write=write,
        size=size_npz,
    )
    return size, priority


def _copy_buffer_sidecar(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Main migration entrypoint.
# ---------------------------------------------------------------------------


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _build_fresh_iqn_state_dicts(yaml_config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Construct a fresh, randomly-initialised IQN agent purely so we can
    extract its ``network.state_dict()`` (target identical) and the
    ``agent_config`` slice that `RainbowDQNAgent.__init__` would consume.
    Built on CPU with ``inference_only=True`` so no CUDA / no compile is
    required."""
    agent_yaml = yaml_config.get("agent")
    if agent_yaml is None:
        raise KeyError("YAML config is missing the top-level 'agent:' section.")
    # The agent expects keys merged from the env stage (e.g. window_size,
    # n_features, num_actions). Reuse the merged shape that the trainer
    # uses at runtime.
    merged: dict[str, Any] = dict(agent_yaml)
    env_yaml = yaml_config.get("environment") or {}
    for key in ("window_size", "n_features", "num_actions"):
        if key in env_yaml and key not in merged:
            merged[key] = env_yaml[key]
    # Mirror run_training.py: surface the trainer-level seed into the agent
    # config so AgentConfig.from_dict() validates cleanly.
    trainer_yaml = yaml_config.get("trainer") or {}
    if "seed" not in merged and "seed" in trainer_yaml:
        merged["seed"] = trainer_yaml["seed"]
    agent = RainbowDQNAgent(config=merged, device="cpu", inference_only=True)
    fresh_state = {k: v.clone() for k, v in agent.network.state_dict().items()}
    return fresh_state, merged


def migrate_checkpoint(
    *,
    source_path: Path,
    output_stem: Path,
    yaml_config_path: Path,
    buffer_source: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Run the migration. Returns a structured summary suitable for tests."""
    logger.info("Loading source checkpoint from %s", source_path)
    source_blob = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source_blob, dict):
        raise TypeError(f"Expected source checkpoint to deserialize as dict, got {type(source_blob).__name__}")

    src_network_state = source_blob.get("network_state_dict")
    if src_network_state is None:
        raise KeyError(
            "Source checkpoint has no 'network_state_dict'; this script "
            "expects a trainer-saved .pt produced by CheckpointMixin._save_checkpoint."
        )

    if not _is_pre_iqn_state_dict(src_network_state):
        raise RuntimeError(
            f"Source checkpoint at {source_path} already looks like an IQN "
            "checkpoint (it has 'tau_embedding.*' or no C51 markers). "
            "Refusing to migrate; use --resume directly instead."
        )

    yaml_config = _load_yaml_config(yaml_config_path)
    fresh_network_state, merged_agent_cfg = _build_fresh_iqn_state_dicts(yaml_config)

    # Same target architecture for both online + target nets, so we can
    # reuse the encoder transfer for both.
    new_network_state, report = _transfer_encoder_tensors(
        source=src_network_state,
        fresh=fresh_network_state,
    )
    new_target_state, _ = _transfer_encoder_tensors(
        source=source_blob.get("target_network_state_dict") or src_network_state,
        fresh=fresh_network_state,
    )

    new_checkpoint: dict[str, Any] = {
        "episode": int(source_blob.get("episode", 0)),
        "total_train_steps": int(source_blob.get("total_train_steps", 0)),
        "best_validation_metric": float("-inf"),
        "early_stopping_counter": 0,
        "buffer_state": None,
        "buffer_sidecar_relpath": None,
        "agent_config": merged_agent_cfg,
        "agent_total_steps": int(source_blob.get("agent_total_steps", 0)),
        "agent_env_steps": int(source_blob.get("agent_env_steps", 0)),
        "total_steps": int(source_blob.get("agent_total_steps", 0)),
        "network_state_dict": new_network_state,
        "target_network_state_dict": new_target_state,
        # Optimizer / scheduler / scaler are intentionally OMITTED (not written
        # as ``None``): the IQN head + cosine tau embedding are new params that
        # the old optimizer's momentum state has nothing useful to say about.
        # Omitting the keys lets the agent's ``"key in dict"`` guards short-
        # circuit cleanly without raising on ``load_state_dict(None)``.
        # Traceability metadata so a future operator can identify the warm-start.
        "_iqn_warmstart_from": str(source_path.name),
    }

    summary: dict[str, Any] = {
        "source_checkpoint": str(source_path),
        "source_episode": int(source_blob.get("episode", 0)),
        "source_validation_score": source_blob.get("validation_score"),
        "transferred_tensor_count": len(report.transferred),
        "fresh_tensor_count": len(report.fresh),
        "skipped_forbidden_keys": list(report.skipped_forbidden),
        "missing_in_source_keys": list(report.missing_in_source),
        "output_pt": str(output_stem.with_suffix(".pt")),
        "output_buffer": None,
        "buffer_priority_reset_count": 0,
        "buffer_priority_value": None,
        "dry_run": bool(dry_run),
    }

    if buffer_source is not None:
        if not buffer_source.exists() or not buffer_source.is_dir():
            raise FileNotFoundError(f"Buffer source {buffer_source} does not exist or is not a directory.")
        output_buffer = output_stem.with_suffix(".buffer")
        summary["output_buffer"] = str(output_buffer)
        if not dry_run:
            _copy_buffer_sidecar(buffer_source, output_buffer)
            count, priority = _reset_buffer_priorities_in_place(output_buffer)
            summary["buffer_priority_reset_count"] = int(count)
            summary["buffer_priority_value"] = float(priority)
            new_checkpoint["buffer_sidecar_relpath"] = output_buffer.name

    if not dry_run:
        output_pt = output_stem.with_suffix(".pt")
        output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_checkpoint, output_pt)
        logger.info("Wrote migrated checkpoint to %s", output_pt)

    summary["transferred_tensors"] = list(report.transferred)
    summary["fresh_tensors"] = list(report.fresh)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("c51 -> iqn migration summary")
    print("=" * 72)
    print(f"  source                : {summary['source_checkpoint']}")
    print(f"  source episode        : {summary['source_episode']}")
    print(f"  source val score      : {summary['source_validation_score']}")
    print(f"  output (.pt)          : {summary['output_pt']}")
    print(f"  output (.buffer/)     : {summary['output_buffer']}")
    print(f"  tensors transferred   : {summary['transferred_tensor_count']}")
    print(f"  tensors left fresh    : {summary['fresh_tensor_count']}")
    print(f"  skipped (forbidden)   : {len(summary['skipped_forbidden_keys'])}")
    print(f"  missing in source     : {len(summary['missing_in_source_keys'])}")
    if summary["output_buffer"]:
        print(f"  buffer leaves reset   : {summary['buffer_priority_reset_count']}")
        print(f"  buffer priority value : {summary['buffer_priority_value']}")
    print(f"  dry-run               : {summary['dry_run']}")
    print("=" * 72)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--output-stem",
        type=Path,
        required=True,
        help="Path stem (no suffix); .pt and .buffer/ are appended.",
    )
    p.add_argument(
        "--buffer-source",
        type=Path,
        default=None,
        help="Optional .buffer/ side-car to copy + reset priorities.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    summary = migrate_checkpoint(
        source_path=args.source_checkpoint,
        output_stem=args.output_stem,
        yaml_config_path=args.config,
        buffer_source=args.buffer_source,
        dry_run=args.dry_run,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via tests
    raise SystemExit(main())
