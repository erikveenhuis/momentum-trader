#!/usr/bin/env python3
"""Profile validation on a limited random subset of validation data."""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_train.data import DataManager
from momentum_train.run_training import configure_logging
from momentum_train.trainer import RainbowTrainerModule
from momentum_train.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint
from momentum_train.utils.utils import set_seeds

from momentum_agent import RainbowDQNAgent

LOGGER_NAME = "momentum_train.ProfileValidation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation on a random subset of validation files for profiling.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/training_config.yaml",
        help="Path to the main training configuration YAML file.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of random validation files to include in the profiling run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible file sampling.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override logging level (e.g. DEBUG, INFO).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a specific trainer checkpoint to load before validation.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="checkpoint_trainer",
        help="Checkpoint filename prefix when searching the model directory.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow validation to fall back to CPU when no GPU or MPS device is detected.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run under cProfile and emit aggregated statistics.",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default=None,
        help="Optional path to save raw cProfile stats (e.g. validate_subset.prof).",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=40,
        help="Number of functions to display in the cProfile cumulative summary (default: 40).",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Enable PyTorch profiler around the validation loop (records CPU/GPU activity).",
    )
    parser.add_argument(
        "--torch-profiler-trace",
        type=str,
        default=None,
        help="Optional path to export the PyTorch profiler trace (Chrome tracing format).",
    )
    parser.add_argument(
        "--torch-profiler-rowlimit",
        type=int,
        default=40,
        help="Row limit for the PyTorch profiler summarized table (default: 40).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must be a mapping.")
    return config


def sample_subset(files: list[Path], count: int) -> list[Path]:
    if not files:
        raise ValueError("No files available to sample from.")
    if count >= len(files):
        subset = list(files)
    else:
        subset = random.sample(files, count)
    return sorted(subset)


def select_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if allow_cpu:
        logging.getLogger(LOGGER_NAME).warning("Falling back to CPU for validation profiling. Expect slower execution.")
        return torch.device("cpu")
    raise RuntimeError("No CUDA or MPS device available. Re-run with --allow-cpu to profile on CPU.")


def resolve_checkpoint_path(explicit_path: str | None, model_dir: Path, prefix: str) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Specified checkpoint does not exist: {candidate}")
        return candidate
    latest = find_latest_checkpoint(str(model_dir), prefix)
    if latest:
        return Path(latest).resolve()
    return None


def _log_profile_stats(profiler: cProfile.Profile, top_n: int, logger: logging.Logger) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(top_n)
    logger.info("cProfile cumulative top %d entries:\n%s", top_n, stream.getvalue())


def run(args: argparse.Namespace) -> dict[str, dict]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    config_path = Path(args.config_path).resolve()

    t0 = time.perf_counter()
    config = load_config(config_path)
    timings["config_load"] = time.perf_counter() - t0

    configure_logging(args.log_level, config.get("logging"))
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Configuration loaded from %s", config_path)

    run_config = config.setdefault("run", {})
    trainer_config = config.setdefault("trainer", {})
    agent_config = config.setdefault("agent", {})

    seed_value = args.seed if args.seed is not None else trainer_config.get("seed")
    if seed_value is not None:
        set_seeds(int(seed_value))

    data_base_dir = run_config.get("data_base_dir", "data")

    data_start = time.perf_counter()
    data_manager = DataManager(base_dir=data_base_dir)
    data_manager.organize_data()
    timings["data_organization"] = time.perf_counter() - data_start

    val_sample_start = time.perf_counter()
    validation_files = data_manager.get_validation_files()
    subset_size = max(1, args.num_files)
    if subset_size > len(validation_files):
        logger.warning(
            "Requested %d validation files but only %d are available. Using the full validation set.",
            subset_size,
            len(validation_files),
        )
    validation_subset = sample_subset(validation_files, subset_size)
    data_manager.val_files = list(validation_subset)
    timings["sample_validation_subset"] = time.perf_counter() - val_sample_start

    logger.info("Profiling validation on %d validation file(s).", len(validation_subset))
    logger.info("Selected validation files: %s", ", ".join(f.name for f in validation_subset))

    if args.seed is not None:
        trainer_config["seed"] = args.seed
        agent_config["seed"] = args.seed

    device_start = time.perf_counter()
    device = select_device(args.allow_cpu)
    timings["device_selection"] = time.perf_counter() - device_start
    logger.info("Using device %s for validation profiling.", device)

    run_config["mode"] = "train"
    run_config["skip_evaluation"] = True

    init_start = time.perf_counter()
    agent = RainbowDQNAgent(config=agent_config, device=device, scaler=None)
    trainer = RainbowTrainerModule(
        agent=agent,
        device=device,
        data_manager=data_manager,
        config=config,
        scaler=None,
        writer=None,
    )
    timings["agent_trainer_init"] = time.perf_counter() - init_start

    model_dir = Path(run_config.get("model_dir", "models")).expanduser()

    checkpoint_resolve_start = time.perf_counter()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path, model_dir, args.checkpoint_prefix)
    timings["checkpoint_resolve"] = time.perf_counter() - checkpoint_resolve_start

    checkpoint_data = None
    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint_load_start = time.perf_counter()
        checkpoint_data = load_checkpoint(str(checkpoint_path))
        timings["checkpoint_load"] = time.perf_counter() - checkpoint_load_start
    else:
        timings["checkpoint_load"] = 0.0

    checkpoint_apply_start = time.perf_counter()
    if checkpoint_path and checkpoint_data:
        loaded = agent.load_state(checkpoint_data)
        if loaded:
            trainer.best_validation_metric = checkpoint_data.get(
                "best_validation_metric",
                trainer.best_validation_metric,
            )
            trainer.early_stopping_counter = checkpoint_data.get(
                "early_stopping_counter",
                trainer.early_stopping_counter,
            )
            trainer.total_train_steps = checkpoint_data.get("total_train_steps", 0)
            agent.total_steps = checkpoint_data.get("agent_total_steps", agent.total_steps)
            logger.info("Checkpoint loaded successfully.")
        else:
            logger.warning("Failed to load agent state from checkpoint. Continuing with randomly initialised weights.")
    elif checkpoint_path and not checkpoint_data:
        logger.warning(
            "Checkpoint %s could not be loaded. Continuing with randomly initialised weights.",
            checkpoint_path,
        )
    else:
        logger.warning("No checkpoint found; validation will run with randomly initialised weights.")
    timings["checkpoint_apply"] = time.perf_counter() - checkpoint_apply_start

    torch_profiler_summary: str | None = None
    validation_start = time.perf_counter()
    if args.torch_profiler:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        profiler_kwargs = {
            "activities": activities,
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": False,
        }
        with profile(**profiler_kwargs) as torch_prof:
            try:
                _, validation_score, avg_metrics = trainer.validate(validation_subset)
            finally:
                trainer.close_cached_environments()
        timings["validation"] = time.perf_counter() - validation_start

        if args.torch_profiler_trace:
            trace_path = Path(args.torch_profiler_trace).expanduser().resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            torch_prof.export_chrome_trace(str(trace_path))
            logger.info("Torch profiler chrome trace saved to %s", trace_path)

        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        row_limit = max(1, args.torch_profiler_rowlimit)
        torch_profiler_summary = torch_prof.key_averages().table(sort_by=sort_key, row_limit=row_limit)
    else:
        try:
            _, validation_score, avg_metrics = trainer.validate(validation_subset)
        finally:
            timings["validation"] = time.perf_counter() - validation_start
            trainer.close_cached_environments()

    timings["total"] = time.perf_counter() - total_start

    logger.info("Validation score: %.4f", validation_score)
    if avg_metrics:
        logger.info(
            "Average metrics | Reward: %.2f | Portfolio: %.2f | Return: %.2f%% | Sharpe: %.4f | Max Drawdown: %.2f%% | Transaction Costs: %.2f",
            avg_metrics.get("avg_reward", float("nan")),
            avg_metrics.get("portfolio_value", float("nan")),
            avg_metrics.get("total_return", float("nan")),
            avg_metrics.get("sharpe_ratio", float("nan")),
            avg_metrics.get("max_drawdown", float("nan")) * 100,
            avg_metrics.get("transaction_costs", float("nan")),
        )

    logger.info("Validation profiling complete.")
    logger.info("Timing summary (seconds):")
    for name, value in timings.items():
        logger.info("  %-28s %.3f", name + ":", value)

    validation_time = timings.get("validation", 0.0)
    if validation_time > 0 and validation_subset:
        logger.info(
            "Avg %.3f s/file (%d file(s))",
            validation_time / len(validation_subset),
            len(validation_subset),
        )

    if torch_profiler_summary:
        logger.info("PyTorch profiler summary (sorted by %s):\n%s", sort_key, torch_profiler_summary)

    return {
        "timings": timings,
        "metrics": {
            "validation_files": len(validation_subset),
            "validation_score": validation_score,
        },
    }


def main() -> None:
    args = parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        result = profiler.runcall(run, args)
        if args.profile_output:
            profiler.dump_stats(args.profile_output)
        logger = logging.getLogger(LOGGER_NAME)
        top_n = max(1, args.profile_top)
        _log_profile_stats(profiler, top_n, logger)
        return result

    run(args)


if __name__ == "__main__":
    main()
