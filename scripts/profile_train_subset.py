#!/usr/bin/env python3
"""Profile training on a limited random subset of training data."""

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
from momentum_train.run_training import configure_logging, run_training
from momentum_train.utils.utils import set_seeds

LOGGER_NAME = "momentum_train.ProfileTrain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused training session on a small random subset of training files for profiling.")
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
        help="Number of random training files to include in the profiling run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes to run. Defaults to the number of sampled files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible file sampling.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Override trainer warmup_steps before learning starts (defaults to 0 for profiling).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint when available.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override logging level (e.g. DEBUG, INFO).",
    )
    parser.add_argument(
        "--evaluate-after-training",
        action="store_true",
        help="Run the post-training evaluation step when profiling completes.",
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
        help="Optional path to save raw cProfile stats (e.g. train_subset.prof).",
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
        help="Enable PyTorch profiler around the training loop (records CPU/GPU activity).",
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

    train_sample_start = time.perf_counter()
    train_files = data_manager.get_training_files()
    subset_size = max(1, args.num_files)
    if subset_size > len(train_files):
        logger.warning(
            "Requested %d training files but only %d are available. Using the full training set.",
            subset_size,
            len(train_files),
        )
    train_subset = sample_subset(train_files, subset_size)
    data_manager.train_files = list(train_subset)
    timings["sample_train_subset"] = time.perf_counter() - train_sample_start

    validation_sample_start = time.perf_counter()
    validation_files = data_manager.get_validation_files()
    val_subset_size = min(len(validation_files), subset_size)
    if val_subset_size == 0:
        validation_subset: list[Path] = []
        data_manager.val_files = []
        logger.warning("No validation files available; training will proceed without validation.")
    else:
        if subset_size > len(validation_files):
            logger.warning(
                "Requested %d validation files but only %d are available. Using the full validation set.",
                subset_size,
                len(validation_files),
            )
        validation_subset = sample_subset(validation_files, val_subset_size)
        data_manager.val_files = list(validation_subset)
    timings["sample_validation_subset"] = time.perf_counter() - validation_sample_start

    logger.info("Profiling training on %d training file(s).", len(train_subset))
    logger.info("Selected training files: %s", ", ".join(f.name for f in train_subset))
    if data_manager.val_files:
        logger.info("Using %d validation file(s) during the run.", len(data_manager.val_files))
        logger.debug("Selected validation files: %s", ", ".join(f.name for f in data_manager.val_files))

    if args.seed is not None:
        trainer_config["seed"] = args.seed
        agent_config["seed"] = args.seed

    if args.warmup_steps is not None:
        trainer_config["warmup_steps"] = max(0, int(args.warmup_steps))

    episodes = args.episodes if args.episodes and args.episodes > 0 else len(train_subset)
    run_config["episodes"] = max(1, episodes)
    run_config["skip_evaluation"] = not args.evaluate_after_training
    run_config["resume"] = bool(args.resume)
    run_config["specific_file"] = None
    run_config["mode"] = "train"

    logger.info(
        "Starting profiling run with %d episode(s), resume=%s, evaluation=%s",
        run_config["episodes"],
        run_config["resume"],
        not run_config["skip_evaluation"],
    )

    torch_profiler_summary: str | None = None
    train_start = time.perf_counter()
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
            trained_agent, trained_trainer = run_training(
                config,
                data_manager,
                resume_training_flag=run_config["resume"],
                reset_lr_on_resume=False,
            )
        timings["training"] = time.perf_counter() - train_start

        if args.torch_profiler_trace:
            trace_path = Path(args.torch_profiler_trace).expanduser().resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            torch_prof.export_chrome_trace(str(trace_path))
            logger.info("Torch profiler chrome trace saved to %s", trace_path)

        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        row_limit = max(1, args.torch_profiler_rowlimit)
        torch_profiler_summary = torch_prof.key_averages().table(sort_by=sort_key, row_limit=row_limit)
    else:
        trained_agent, trained_trainer = run_training(
            config,
            data_manager,
            resume_training_flag=run_config["resume"],
            reset_lr_on_resume=False,
        )
        timings["training"] = time.perf_counter() - train_start

    timings["total"] = time.perf_counter() - total_start

    total_steps = getattr(trained_trainer, "total_train_steps", None)
    learner_updates = getattr(trained_agent, "total_steps", None)

    metrics = {
        "episodes": run_config["episodes"],
        "total_steps": total_steps,
        "learner_updates": learner_updates,
        "train_files": len(train_subset),
        "validation_files": len(data_manager.val_files),
    }

    logger.info("Profiling training run completed.")
    logger.info("Timing summary (seconds):")
    for name, value in timings.items():
        logger.info("  %-28s %.3f", name + ":", value)

    training_time = timings.get("training", 0.0)
    if training_time > 0 and total_steps:
        logger.info(
            "Env steps: %s | Avg %.6f s/step | %.2f steps/s",
            f"{total_steps:,}",
            training_time / total_steps,
            total_steps / training_time,
        )
    if training_time > 0 and metrics["episodes"]:
        logger.info(
            "Episodes: %s | Avg %.3f s/episode",
            f"{metrics['episodes']:,}",
            training_time / metrics["episodes"],
        )
    if learner_updates is not None and training_time > 0:
        logger.info(
            "Learner updates: %s | Avg %.6f s/update",
            f"{learner_updates:,}",
            training_time / max(learner_updates, 1),
        )

    if torch_profiler_summary:
        logger.info("PyTorch profiler summary (sorted by %s):\n%s", sort_key, torch_profiler_summary)

    return {"timings": timings, "metrics": metrics}


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
