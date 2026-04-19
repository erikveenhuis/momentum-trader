"""Smoke tests for ``scripts/eval_greedy.py`` (Tier 5a).

These exercise the *pure* helper functions: split sampling, summary
aggregation, and JSONL persistence. The end-to-end rollout requires a real
checkpoint + dataset and is exercised manually from the CLI.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "eval_greedy.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def eval_greedy_mod():
    spec = importlib.util.spec_from_file_location("eval_greedy", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_greedy"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.unit
def test_sample_subset_returns_full_when_count_exceeds_pool(eval_greedy_mod, tmp_path):
    """Tier 5a: requesting more files than available returns the full set sorted."""
    files = [tmp_path / f"f{i}.csv" for i in range(3)]
    rng = np.random.default_rng(0)
    out = eval_greedy_mod._sample_subset(files, count=10, rng=rng)
    assert sorted(files) == out


@pytest.mark.unit
def test_sample_subset_random_subsample_is_deterministic_per_seed(eval_greedy_mod, tmp_path):
    files = [tmp_path / f"f{i}.csv" for i in range(20)]
    out1 = eval_greedy_mod._sample_subset(files, count=5, rng=np.random.default_rng(42))
    out2 = eval_greedy_mod._sample_subset(files, count=5, rng=np.random.default_rng(42))
    assert out1 == out2
    assert len(out1) == 5


@pytest.mark.unit
def test_sample_subset_empty_pool_returns_empty(eval_greedy_mod):
    assert eval_greedy_mod._sample_subset([], count=5, rng=np.random.default_rng(0)) == []


@pytest.mark.unit
def test_summarize_split_aggregates_basic_stats(eval_greedy_mod):
    """Tier 5a: per-split summary collapses the per-file records into one dict."""
    per_file = [
        {
            "file": "a.csv",
            "trade_metrics": {"num_trades": 4, "per_trade_sharpe": 0.5, "hit_rate": 0.5, "expectancy_pct": 1.0},
            "final_portfolio_value": 1100.0,
            "q_mean": 0.2,
        },
        {
            "file": "b.csv",
            "trade_metrics": {"num_trades": 2, "per_trade_sharpe": 1.0, "hit_rate": 1.0, "expectancy_pct": 2.0},
            "final_portfolio_value": 1200.0,
            "q_mean": 0.4,
        },
    ]
    summary = eval_greedy_mod._summarize_split(per_file)
    assert summary["files"] == 2
    assert summary["trades_total"] == 6
    assert summary["trades_mean"] == pytest.approx(3.0)
    assert summary["per_trade_sharpe_mean"] == pytest.approx(0.75)
    assert summary["hit_rate_mean"] == pytest.approx(0.75)
    assert summary["final_portfolio_value_mean"] == pytest.approx(1150.0)
    assert summary["q_mean_overall"] == pytest.approx(0.3)


@pytest.mark.unit
def test_summarize_split_empty_returns_minimal_payload(eval_greedy_mod):
    assert eval_greedy_mod._summarize_split([]) == {"files": 0}


@pytest.mark.unit
def test_write_jsonl_roundtrip(eval_greedy_mod, tmp_path):
    """Tier 5a: jsonl writer survives a roundtrip and accepts NaN via default=float."""
    records = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": float("nan")}]
    path = tmp_path / "out.jsonl"
    eval_greedy_mod._write_jsonl(records, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed_first = json.loads(lines[0])
    assert parsed_first["a"] == 1.0
