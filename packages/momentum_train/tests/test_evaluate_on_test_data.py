import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from momentum_train.run_training import evaluate_on_test_data


class StubTrainer:
    def __init__(self, files, results, avg_metrics):
        self.data_manager = SimpleNamespace(get_test_files=lambda: files)
        self._results_iter = iter(results)
        self._expected_metrics_call = avg_metrics
        self.collected_metrics = None

    def _validate_single_file(self, file_path):
        return next(self._results_iter)

    def _calculate_average_validation_metrics(self, metrics):
        self.collected_metrics = metrics
        return self._expected_metrics_call


@pytest.mark.unittest
def test_evaluate_on_test_data_writes_results(tmp_path):
    test_files = [tmp_path / "test1.csv", tmp_path / "test2.csv"]

    file_metrics_list = [
        {
            "avg_reward": 1.0,
            "portfolio_value": 1005.0,
            "total_return": 5.0,
            "sharpe_ratio": 0.10,
            "max_drawdown": 0.02,
            "transaction_costs": 1.5,
        },
        {
            "avg_reward": 2.0,
            "portfolio_value": 1010.0,
            "total_return": 6.0,
            "sharpe_ratio": 0.20,
            "max_drawdown": 0.01,
            "transaction_costs": 2.0,
        },
    ]

    detailed_results = [
        {"file": "test1.csv", "reward": 1.23},
        {"file": "test2.csv", "reward": 2.34},
    ]

    results = [
        {
            "file_metrics": file_metrics_list[0],
            "detailed_result": detailed_results[0],
            "episode_score": 0.4,
        },
        {
            "file_metrics": file_metrics_list[1],
            "detailed_result": detailed_results[1],
            "episode_score": 0.6,
        },
    ]

    avg_metrics = {
        "avg_reward": float(np.mean([1.0, 2.0])),
        "portfolio_value": float(np.mean([1005.0, 1010.0])),
        "total_return": float(np.mean([5.0, 6.0])),
        "sharpe_ratio": float(np.mean([0.10, 0.20])),
        "max_drawdown": float(np.mean([0.02, 0.01])),
        "transaction_costs": float(np.mean([1.5, 2.0])),
    }

    trainer = StubTrainer(test_files, results, avg_metrics)

    config = {"run": {"model_dir": str(tmp_path)}}

    evaluate_on_test_data(agent=None, trainer=trainer, config=config)

    expected_metrics = [res["file_metrics"] for res in results]
    assert trainer.collected_metrics == expected_metrics

    output_files = list(Path(tmp_path).glob("test_results_*.json"))
    assert len(output_files) == 1

    with output_files[0].open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["average_metrics"] == avg_metrics
    assert payload["average_episode_score"] == pytest.approx(np.mean([0.4, 0.6]))
    assert payload["detailed_results"] == detailed_results


@pytest.mark.unittest
def test_evaluate_on_test_data_handles_empty_test_set(tmp_path):
    trainer = StubTrainer([], [], {})
    config = {"run": {"model_dir": str(tmp_path)}}

    # Should return without writing files when no test data is available.
    evaluate_on_test_data(agent=None, trainer=trainer, config=config)

    assert not list(Path(tmp_path).glob("test_results_*.json"))
