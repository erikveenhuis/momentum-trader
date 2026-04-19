"""Tier 5c: tests for ``scripts/analyze_trades.py`` pure helpers + end-to-end report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_trades.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def analyze_mod():
    spec = importlib.util.spec_from_file_location("analyze_trades", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_trades"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _make_trade(pnl_pct: float, *, pct_greedy: float | None = 1.0) -> dict:
    return {
        "entry_index": 0,
        "exit_index": 5,
        "duration_steps": 5,
        "entry_price": 100.0,
        "exit_price": 100.0 * (1 + pnl_pct / 100.0),
        "entry_portfolio_value": 1000.0,
        "exit_portfolio_value": 1000.0 * (1 + pnl_pct / 100.0),
        "pnl_pct": float(pnl_pct),
        "pnl_absolute": 1000.0 * pnl_pct / 100.0,
        "max_position": 1.0,
        "mae_pct": min(pnl_pct, 0.0),
        "mfe_pct": max(pnl_pct, 0.0),
        "transaction_cost_total": 0.0,
        "actions_taken": [5, 0, 0, 0, 0],
        "pct_greedy_actions": pct_greedy,
    }


@pytest.mark.unit
def test_split_by_greedy_buckets_correctly(analyze_mod):
    """Tier 5c: trades split at pct_greedy >= 0.5; NaN provenance lands in unknown."""
    trades = [
        _make_trade(1.0, pct_greedy=0.9),
        _make_trade(-0.5, pct_greedy=0.1),
        _make_trade(0.0, pct_greedy=0.5),
        _make_trade(2.0, pct_greedy=float("nan")),
        _make_trade(0.5, pct_greedy=None),
    ]
    out = analyze_mod._split_by_greedy(trades)
    assert len(out["greedy"]) == 2  # 0.9 and 0.5
    assert len(out["eps"]) == 1  # 0.1
    assert len(out["unknown"]) == 2  # NaN + None


@pytest.mark.unit
def test_percentile_table_has_expected_keys(analyze_mod):
    table = analyze_mod._percentile_table([1.0, 2.0, 3.0, 4.0, 5.0])
    assert "min" in table and "max" in table
    assert table["min"] == pytest.approx(1.0)
    assert table["max"] == pytest.approx(5.0)
    assert table["p50"] == pytest.approx(3.0)
    # 5%-step grid p5..p95 inclusive (every 5%).
    expected = {f"p{q}" for q in range(5, 100, 5)}
    assert expected.issubset(table.keys())


@pytest.mark.unit
def test_percentile_table_empty_returns_empty(analyze_mod):
    assert analyze_mod._percentile_table([]) == {}


@pytest.mark.unit
def test_bootstrap_ci_returns_finite_numbers(analyze_mod):
    """Tier 5c: bootstrap CIs are finite when there is variance + iters > 0."""
    rng = np.random.default_rng(0)
    ci = analyze_mod._bootstrap_ci([1.0, 2.0, 3.0, -1.0, 0.5], iters=200, alpha=0.05, rng=rng)
    assert np.isfinite(ci["sharpe"]["low"])
    assert np.isfinite(ci["sharpe"]["high"])
    assert ci["sharpe"]["low"] <= ci["sharpe"]["median"] <= ci["sharpe"]["high"]
    assert ci["expectancy_pct"]["low"] <= ci["expectancy_pct"]["median"] <= ci["expectancy_pct"]["high"]


@pytest.mark.unit
def test_bootstrap_ci_too_few_returns_nan(analyze_mod):
    rng = np.random.default_rng(0)
    ci = analyze_mod._bootstrap_ci([1.0], iters=200, alpha=0.05, rng=rng)
    assert np.isnan(ci["sharpe"]["median"])
    assert np.isnan(ci["expectancy_pct"]["median"])


@pytest.mark.unit
def test_summarize_trades_includes_pnl_percentiles(analyze_mod):
    trades = [_make_trade(1.0), _make_trade(-0.5), _make_trade(2.0)]
    summary = analyze_mod._summarize_trades(trades)
    assert summary["num_trades"] == 3
    assert "pnl_percentiles" in summary


@pytest.mark.unit
def test_load_trades_skips_blank_and_malformed_lines(analyze_mod, tmp_path):
    """Tier 5c: blank lines + JSON parse errors are tolerated rather than aborting."""
    p = tmp_path / "x.trades.jsonl"
    valid = json.dumps(_make_trade(1.0))
    p.write_text(f"{valid}\n\n{{not-json}}\n{valid}\n", encoding="utf-8")
    out = analyze_mod._load_trades(p)
    assert len(out) == 2


@pytest.mark.unit
def test_build_report_full_path_with_rolling_and_provenance(analyze_mod):
    """Tier 5c: end-to-end report has overall + provenance + bootstrap + rolling."""
    trades = [
        _make_trade(1.0, pct_greedy=1.0),
        _make_trade(-0.2, pct_greedy=0.0),
        _make_trade(0.3, pct_greedy=0.7),
        _make_trade(0.5, pct_greedy=0.6),
        _make_trade(-0.1, pct_greedy=0.2),
    ]
    report = analyze_mod._build_report(
        trades,
        rolling_window=3,
        bootstrap_iters=100,
        bootstrap_alpha=0.05,
        seed=42,
        equity_curve={"files": 1, "steps_total": 100, "mean": 1000.0, "min": 990.0, "max": 1020.0, "per_file": []},
    )
    assert report["trade_count_total"] == 5
    assert "overall" in report and report["overall"]["num_trades"] == 5
    assert "by_provenance" in report
    assert report["by_provenance"]["greedy"]["num_trades"] >= 1
    assert "bootstrap_ci" in report
    assert "rolling_window" in report and report["rolling_window"]["window"] == 3
    assert report["rolling_window"]["metrics"]["num_trades"] == 3
    assert report["equity_curve"]["mean"] == pytest.approx(1000.0)


@pytest.mark.unit
def test_gather_trade_files_from_directory_picks_up_jsonl(analyze_mod, tmp_path):
    (tmp_path / "a.trades.jsonl").write_text(json.dumps(_make_trade(1.0)) + "\n", encoding="utf-8")
    sub = tmp_path / "split"
    sub.mkdir()
    (sub / "b.trades.jsonl").write_text(json.dumps(_make_trade(0.5)) + "\n", encoding="utf-8")
    files = analyze_mod._gather_trade_files(tmp_path)
    assert len(files) == 2
    names = {f.name for f in files}
    assert names == {"a.trades.jsonl", "b.trades.jsonl"}


@pytest.mark.unit
def test_gather_trade_files_single_file_returned_as_list(analyze_mod, tmp_path):
    p = tmp_path / "x.trades.jsonl"
    p.write_text(json.dumps(_make_trade(1.0)) + "\n", encoding="utf-8")
    files = analyze_mod._gather_trade_files(p)
    assert files == [p]
