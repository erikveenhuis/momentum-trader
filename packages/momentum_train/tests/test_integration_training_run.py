from __future__ import annotations

import itertools
import json
import random
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from momentum_agent import RainbowDQNAgent
from momentum_train.data import DataManager
from momentum_train.run_training import evaluate_on_test_data
from momentum_train.trainer import RainbowTrainerModule
from torch.amp import GradScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _copy_random_files(
    src_dir: Path,
    dst_dir: Path,
    count: int,
    rng: random.Random,
    *,
    max_rows: int | None = None,
) -> list[Path]:
    if not src_dir.is_dir():
        pytest.skip(f"Source directory not found: {src_dir}")

    source_files = sorted(src_dir.glob("*.csv"))
    if len(source_files) < count:
        pytest.skip(f"Not enough CSV files in {src_dir} (needed {count}, found {len(source_files)})")

    dst_dir.mkdir(parents=True, exist_ok=True)
    selected = rng.sample(source_files, count)

    copied_paths = []
    for src_file in selected:
        dst_file = dst_dir / src_file.name
        if max_rows is None:
            shutil.copy2(src_file, dst_file)
        else:
            with src_file.open("r", encoding="utf-8") as src_fp:
                header = src_fp.readline()
                limited_rows = list(itertools.islice(src_fp, max_rows))

            with dst_file.open("w", encoding="utf-8") as dst_fp:
                dst_fp.write(header)
                dst_fp.writelines(limited_rows)

        copied_paths.append(dst_file)

    return copied_paths


@pytest.mark.integration
def test_end_to_end_training_validation_and_testing(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for this integration test")

    TRAIN_EPISODES = 10
    VALIDATION_FILES = 5
    TEST_FILES = 3
    MAX_ROWS_PER_FILE = 600

    processed_dir = tmp_path / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"

    rng = random.Random(1234)
    production_processed_dir = PROJECT_ROOT / "data" / "processed"
    _copy_random_files(
        production_processed_dir / "train",
        train_dir,
        count=TRAIN_EPISODES,
        rng=rng,
        max_rows=MAX_ROWS_PER_FILE,
    )
    _copy_random_files(
        production_processed_dir / "validation",
        val_dir,
        count=VALIDATION_FILES,
        rng=rng,
        max_rows=MAX_ROWS_PER_FILE,
    )
    _copy_random_files(
        production_processed_dir / "test",
        test_dir,
        count=TEST_FILES,
        rng=rng,
        max_rows=MAX_ROWS_PER_FILE,
    )

    data_manager = DataManager(base_dir=str(tmp_path))
    data_manager.organize_data()

    train_file_count = len(data_manager.get_training_files())
    val_file_count = len(data_manager.get_validation_files())
    test_file_count = len(data_manager.get_test_files())

    assert train_file_count == TRAIN_EPISODES
    assert val_file_count == VALIDATION_FILES
    assert test_file_count == TEST_FILES

    model_dir = tmp_path / "models"

    config_path = PROJECT_ROOT / "config" / "training_config.yaml"
    config = yaml.safe_load(config_path.read_text())

    # Use the production configuration but scope outputs and episode counts for the sampled files.
    config["run"]["model_dir"] = str(model_dir)
    config["run"]["episodes"] = TRAIN_EPISODES
    config["run"]["skip_evaluation"] = False
    config["run"]["specific_file"] = None

    config["trainer"]["validation_freq"] = TRAIN_EPISODES
    config["trainer"]["checkpoint_save_freq"] = TRAIN_EPISODES

    # Mirror run_training: propagate trainer seed into agent config for reproducibility.
    config["agent"]["seed"] = config["trainer"]["seed"]

    assert len(data_manager.get_training_files()) == train_file_count
    assert len(data_manager.get_validation_files()) == val_file_count
    assert len(data_manager.get_test_files()) == test_file_count

    scaler = GradScaler("cuda")

    agent = RainbowDQNAgent(config=config["agent"], device="cuda", scaler=scaler)

    trainer = RainbowTrainerModule(
        agent=agent,
        device=torch.device("cuda"),
        data_manager=data_manager,
        config=config,
        scaler=scaler,
        writer=None,
    )

    trainer.train(
        num_episodes=config["run"]["episodes"],
        start_episode=0,
        start_total_steps=0,
        initial_best_score=float("-inf"),
        initial_early_stopping_counter=0,
        specific_file=config["run"].get("specific_file"),
    )

    assert getattr(trainer, "total_train_steps", 0) > 0

    validation_results = list(model_dir.glob("validation_results_*.json"))
    assert len(validation_results) == 1, "Validation should run once after processing train files"

    with validation_results[-1].open("r", encoding="utf-8") as f:
        validation_payload = json.load(f)

    assert validation_payload.get("average_metrics"), "Validation metrics missing"

    assert validation_results, "Validation summary JSON not generated"

    evaluate_on_test_data(agent, trainer, config)

    test_results = list(model_dir.glob("test_results_*.json"))
    assert test_results, "Test evaluation results JSON not generated"

    with test_results[-1].open("r", encoding="utf-8") as f:
        test_payload = json.load(f)

    assert "average_metrics" in test_payload
    assert "detailed_results" in test_payload
    assert isinstance(test_payload["detailed_results"], list)
