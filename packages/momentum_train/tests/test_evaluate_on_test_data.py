import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from momentum_train.run_training import evaluate_on_test_data


class StubTrainer:
    def __init__(self, files, results, avg_metrics, *, writer=None, agent=None):
        self.data_manager = SimpleNamespace(get_test_files=lambda: files)
        self._results_iter = iter(results)
        self._expected_metrics_call = avg_metrics
        self.collected_metrics = None
        self.writer = writer
        self.agent = agent

    def _validate_single_file(self, file_path, **kwargs):
        return next(self._results_iter)

    def _calculate_average_validation_metrics(self, metrics):
        self.collected_metrics = metrics
        return self._expected_metrics_call

    def close_cached_environments(self):
        pass


class _CapturingWriter:
    """Test double for ``SummaryWriter.add_scalar`` capture (Tier 1a / 1b / 2b assertions)."""

    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value, step):  # pragma: no cover - trivial
        self.scalars.append((tag, float(value), int(step)))

    def tags(self) -> set[str]:
        return {tag for tag, *_ in self.scalars}

    def value_for(self, tag: str) -> float:
        for t, v, _ in self.scalars:
            if t == tag:
                return v
        raise KeyError(tag)


@pytest.mark.unit
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


@pytest.mark.unit
def test_evaluate_on_test_data_handles_empty_test_set(tmp_path):
    trainer = StubTrainer([], [], {})
    config = {"run": {"model_dir": str(tmp_path)}}

    # Should return without writing files when no test data is available.
    evaluate_on_test_data(agent=None, trainer=trainer, config=config)

    assert not list(Path(tmp_path).glob("test_results_*.json"))


@pytest.mark.unit
@pytest.mark.tb_logging
def test_evaluate_on_test_data_mirrors_results_to_tensorboard(tmp_path):
    """Tier 1a: aggregate Test/* scalars and Test/Action Rate/{0..5} land on the writer."""
    test_files = [tmp_path / "tA.csv", tmp_path / "tB.csv"]

    detailed_results = [
        {
            "file": "tA.csv",
            "reward": 1.0,
            "steps": 100,
            "action_counts": {0: 70, 1: 5, 2: 10, 3: 5, 4: 5, 5: 5},
            "trade_metrics": {
                "PerTradeSharpe": 0.5,
                "HitRate": 0.6,
                "Expectancy": 0.0010,
                "MAE_Mean": 0.0020,
                "NTrades": 20.0,
            },
        },
        {
            "file": "tB.csv",
            "reward": 2.0,
            "steps": 200,
            "action_counts": {0: 150, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10},
            "trade_metrics": {
                "PerTradeSharpe": 0.7,
                "HitRate": 0.55,
                "Expectancy": 0.0020,
                "MAE_Mean": 0.0030,
                "NTrades": 30.0,
            },
        },
    ]

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

    results = [
        {"file_metrics": file_metrics_list[0], "detailed_result": detailed_results[0], "episode_score": 0.4},
        {"file_metrics": file_metrics_list[1], "detailed_result": detailed_results[1], "episode_score": 0.6},
    ]

    avg_metrics = {
        "avg_reward": 1.5,
        "portfolio_value": 1007.5,
        "total_return": 5.5,
        "sharpe_ratio": 0.15,
        "max_drawdown": 0.015,
        "transaction_costs": 1.75,
        "avg_position": 0.4,
        "avg_abs_position": 0.5,
        "avg_exposure_pct": 30.0,
        "max_exposure_pct": 80.0,
    }

    writer = _CapturingWriter()
    agent = SimpleNamespace(total_steps=12345)
    trainer = StubTrainer(test_files, results, avg_metrics, writer=writer, agent=agent)
    config = {"run": {"model_dir": str(tmp_path)}}

    evaluate_on_test_data(agent=agent, trainer=trainer, config=config)

    tags = writer.tags()
    # Headline aggregate scalars must be present.
    for tag in (
        "Test/Score",
        "Test/Avg Reward",
        "Test/Avg Portfolio",
        "Test/Transaction Costs",
        "Test/Portfolio/Total Return Pct",
        "Test/Portfolio/Sharpe Ratio",
        "Test/Portfolio/Max Drawdown Pct",
        "Test/Avg Exposure Pct",
        "Test/Max Exposure Pct",
    ):
        assert tag in tags, f"missing TB tag {tag}"

    # All 6 actions should be reported as rates.
    for k in range(6):
        assert f"Test/Action Rate/{k}" in tags

    # Trade metrics should mirror through.
    assert "Test/Trade/PerTradeSharpe" in tags
    assert "Test/Trade/HitRate" in tags
    assert "Test/Trade/MAE_Mean" in tags

    # X-axis must use the agent's total_steps so it lines up with training curves.
    for _, _, step in writer.scalars:
        assert step == 12345

    # Sanity check a couple of computed values.
    assert writer.value_for("Test/Score") == pytest.approx(np.mean([0.4, 0.6]))
    assert writer.value_for("Test/Portfolio/Max Drawdown Pct") == pytest.approx(0.015 * 100.0)
    assert writer.value_for("Test/Trade/PerTradeSharpe") == pytest.approx(np.mean([0.5, 0.7]))


@pytest.mark.unit
def test_evaluate_on_test_data_emits_no_scalars_when_writer_is_none(tmp_path):
    """Tier 1a regression: with writer=None nothing is mirrored (no AttributeError)."""
    test_files = [tmp_path / "tA.csv"]
    detailed_results = [{"file": "tA.csv", "reward": 1.0, "steps": 50, "action_counts": {0: 50}}]
    file_metrics_list = [
        {
            "avg_reward": 1.0,
            "portfolio_value": 1005.0,
            "total_return": 5.0,
            "sharpe_ratio": 0.10,
            "max_drawdown": 0.02,
            "transaction_costs": 1.5,
        }
    ]
    results = [
        {
            "file_metrics": file_metrics_list[0],
            "detailed_result": detailed_results[0],
            "episode_score": 0.4,
        }
    ]
    avg_metrics = file_metrics_list[0]
    trainer = StubTrainer(test_files, results, avg_metrics)  # writer=None default
    config = {"run": {"model_dir": str(tmp_path)}}

    evaluate_on_test_data(agent=None, trainer=trainer, config=config)
    # Should not raise; JSON file should still be produced.
    assert list(Path(tmp_path).glob("test_results_*.json"))
