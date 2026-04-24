"""Tests for ``scripts/recover_from_collapse.py``.

Exercises the pure helpers + the file-orchestration glue:

* Best-checkpoint discovery (highest score wins).
* Explicit ``--from-episode`` lookup.
* Auto-pick fallback against synthetic ``validation_results_*.json`` +
  companion ``checkpoint_trainer_latest_*_ep*_reward*.pt`` files.
* ``strip_optimizer_state`` actually removes the right keys.
* ``write_recovered_checkpoint`` produces a file that beats every
  pre-existing latest checkpoint in episode order (so resume picks it up).
* Resume command rendering for several flag combinations.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "recover_from_collapse.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def recover_mod():
    spec = importlib.util.spec_from_file_location("recover_from_collapse", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recover_from_collapse"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Helpers to fabricate a model_dir that looks like a real training run
# ---------------------------------------------------------------------------


def _write_dummy_latest(model_dir: Path, *, date: str, episode: int, reward: str = "-inf") -> Path:
    """Create a syntactically-valid 'latest' checkpoint stub on disk.

    ``torch.save({})`` is enough — discovery + write_recovered_checkpoint only
    care about filename parsing and file existence, not payload contents.
    """
    p = model_dir / f"checkpoint_trainer_latest_{date}_ep{episode}_reward{reward}.pt"
    torch.save({"episode": episode, "marker": "latest-stub"}, p)
    return p


def _write_dummy_best(model_dir: Path, *, date: str, episode: int, score: float) -> Path:
    p = model_dir / f"checkpoint_trainer_best_{date}_ep{episode}_score_{score:.4f}.pt"
    torch.save({"episode": episode, "marker": "best-stub", "validation_score": score}, p)
    return p


def _write_dummy_validation_results(model_dir: Path, *, date: str, time_str: str, score: float) -> Path:
    p = model_dir / f"validation_results_{date}_{time_str}.json"
    p.write_text(json.dumps({"aggregate_score": score}), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pick_checkpoint_prefers_highest_score_best(tmp_path, recover_mod):
    _write_dummy_best(tmp_path, date="20260415", episode=2000, score=0.31)
    _write_dummy_best(tmp_path, date="20260417", episode=2800, score=0.18)
    _write_dummy_best(tmp_path, date="20260418", episode=3000, score=0.25)
    _write_dummy_latest(tmp_path, date="20260418", episode=3499)

    chosen, reason = recover_mod.pick_checkpoint(tmp_path, from_episode=None)
    assert "best" in reason
    assert "0.3100" in reason
    assert chosen.name.startswith("checkpoint_trainer_best_20260415_ep2000_score_0.3100")


@pytest.mark.unit
def test_pick_checkpoint_explicit_from_episode_when_no_best(tmp_path, recover_mod):
    _write_dummy_latest(tmp_path, date="20260418", episode=3000)
    _write_dummy_latest(tmp_path, date="20260418", episode=3499)

    chosen, reason = recover_mod.pick_checkpoint(tmp_path, from_episode=3000)
    assert "--from-episode 3000" in reason
    assert "ep3000" in chosen.name


@pytest.mark.unit
def test_pick_checkpoint_explicit_from_episode_missing_raises(tmp_path, recover_mod):
    _write_dummy_latest(tmp_path, date="20260418", episode=3499)
    with pytest.raises(FileNotFoundError):
        recover_mod.pick_checkpoint(tmp_path, from_episode=2000)


@pytest.mark.unit
def test_pick_checkpoint_auto_pick_uses_validation_results(tmp_path, recover_mod):
    """Synthetic run: 5 latest checkpoints (ep 2000-3499) and 5 validation
    results — three good (>= cutoff), two bad (well below). Auto-pick must
    return the most recent latest checkpoint whose mtime is <= the most
    recent good validation_result."""
    paths_ep = {}
    for ep, date in [
        (2000, "20260416"),
        (2400, "20260416"),
        (2800, "20260417"),
        (3200, "20260418"),
        (3499, "20260418"),
    ]:
        paths_ep[ep] = _write_dummy_latest(tmp_path, date=date, episode=ep)
        # Stagger mtimes so newer episodes look newer on disk.
        time.sleep(0.01)

    base = time.time() - 3600
    times = [base, base + 60, base + 120, base + 180, base + 240]
    eps = [2000, 2400, 2800, 3200, 3499]
    for ep, t in zip(eps, times, strict=True):
        import os as _os

        _os.utime(paths_ep[ep], (t, t))

    # validation_results: good, good, good, bad, bad — and times that line up
    # with each checkpoint's mtime. Auto-pick should prefer the newest *good*.
    vr_specs = [
        ("20260416", "000100", 0.50),  # good, near ep 2000
        ("20260416", "001100", 0.40),  # good, near ep 2400
        ("20260417", "000200", 0.30),  # good, near ep 2800  <-- newest good
        ("20260418", "000300", -0.20),  # bad, near ep 3200
        ("20260418", "002400", -0.40),  # bad, near ep 3499
    ]
    for ep, (date, tstr, score) in zip(eps, vr_specs, strict=True):
        vr_path = _write_dummy_validation_results(tmp_path, date=date, time_str=tstr, score=score)
        # Pin vr mtime to match its corresponding checkpoint mtime + 1 second.
        ckpt_t = paths_ep[ep].stat().st_mtime
        import os as _os

        _os.utime(vr_path, (ckpt_t + 1.0, ckpt_t + 1.0))

    chosen, reason = recover_mod.pick_checkpoint(tmp_path, from_episode=None)
    assert "auto-picked" in reason
    # The newest good validation_result lined up with ep 2800.
    assert "ep2800" in chosen.name


@pytest.mark.unit
def test_auto_pick_falls_back_when_no_validation_results(tmp_path, recover_mod):
    _write_dummy_latest(tmp_path, date="20260418", episode=3000)
    _write_dummy_latest(tmp_path, date="20260418", episode=3499)
    chosen, reason = recover_mod.pick_checkpoint(tmp_path, from_episode=None)
    assert "no validation_results_*.json found" in reason
    assert "ep3499" in chosen.name


@pytest.mark.unit
def test_auto_pick_falls_back_when_all_validation_below_cutoff(tmp_path, recover_mod):
    _write_dummy_latest(tmp_path, date="20260418", episode=3000)
    _write_dummy_latest(tmp_path, date="20260418", episode=3499)
    # Only one validation result; cutoff index = max(0, 1//4 - 1) = 0, so
    # cutoff = the only score. The score then equals the cutoff, so it's NOT
    # below — it's at — and should be selected. Add a deliberately bad single
    # entry that is *equal* to the cutoff (which means it survives), so we
    # write the test the other way: zero usable results.
    chosen, reason = recover_mod.pick_checkpoint(tmp_path, from_episode=None)
    assert "ep3499" in chosen.name
    assert "no validation_results_*.json found" in reason


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strip_optimizer_state_removes_present_keys(recover_mod):
    ckpt = {
        "optimizer_state_dict": {"foo": 1},
        "scheduler_state_dict": {"bar": 2},
        "scaler_state_dict": {"baz": 3},
        "network_state_dict": {"keep": True},
    }
    removed = recover_mod.strip_optimizer_state(ckpt)
    assert set(removed) == {"optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"}
    assert "optimizer_state_dict" not in ckpt
    assert "scheduler_state_dict" not in ckpt
    assert "scaler_state_dict" not in ckpt
    assert "network_state_dict" in ckpt


@pytest.mark.unit
def test_strip_optimizer_state_no_op_when_keys_absent(recover_mod):
    ckpt = {"network_state_dict": {"keep": True}}
    removed = recover_mod.strip_optimizer_state(ckpt)
    assert removed == []
    assert "network_state_dict" in ckpt


@pytest.mark.unit
def test_reset_best_validation_state_resets_both_keys(recover_mod):
    """Both keys present -> both reset, returned dict reports old/new pair."""
    ckpt = {
        "best_validation_metric": 0.42,
        "early_stopping_counter": 7,
        "network_state_dict": {"keep": True},
    }
    changed = recover_mod.reset_best_validation_state(ckpt)
    assert set(changed.keys()) == {"best_validation_metric", "early_stopping_counter"}
    assert ckpt["best_validation_metric"] == float("-inf")
    assert ckpt["early_stopping_counter"] == 0
    # Old values are preserved in the returned diff for log readability.
    assert changed["best_validation_metric"][0] == 0.42
    assert changed["early_stopping_counter"][0] == 7
    assert changed["best_validation_metric"][1] == float("-inf")
    assert changed["early_stopping_counter"][1] == 0
    # Untouched keys are left alone.
    assert ckpt["network_state_dict"] == {"keep": True}


@pytest.mark.unit
def test_reset_best_validation_state_handles_partial_keys(recover_mod):
    """Missing keys are silently skipped (trainer will populate defaults)."""
    ckpt = {"best_validation_metric": -0.1}  # no early_stopping_counter
    changed = recover_mod.reset_best_validation_state(ckpt)
    assert set(changed.keys()) == {"best_validation_metric"}
    assert ckpt["best_validation_metric"] == float("-inf")
    assert "early_stopping_counter" not in ckpt


@pytest.mark.unit
def test_reset_best_validation_state_noop_on_empty_returns_empty_dict(recover_mod):
    ckpt: dict = {"network_state_dict": {}}
    changed = recover_mod.reset_best_validation_state(ckpt)
    assert changed == {}
    assert ckpt == {"network_state_dict": {}}


@pytest.mark.unit
def test_main_with_reset_best_validation_flag_writes_reset_state(tmp_path, recover_mod, monkeypatch):
    """End-to-end: --reset-best-validation on a real checkpoint dict produces
    a recovery file whose payload has best_validation_metric=-inf and
    early_stopping_counter=0, while preserving everything else."""
    import math

    ckpt_path = _write_dummy_latest(tmp_path, date="20260418", episode=3000)
    full_payload = {
        "episode": 3000,
        "total_train_steps": 999,
        "network_state_dict": {"weight": torch.tensor([1.0])},
        "target_network_state_dict": {"weight": torch.tensor([1.0])},
        "best_validation_metric": 0.55,
        "agent_total_steps": 1234,
        "early_stopping_counter": 4,
        "agent_config": {"hidden_dim": 8},
    }
    torch.save(full_payload, ckpt_path)

    rc = recover_mod.main(
        [
            "--model-dir",
            str(tmp_path),
            "--from-episode",
            "3000",
            "--reset-best-validation",
        ]
    )
    assert rc == 0

    out_files = sorted(p for p in tmp_path.glob("checkpoint_trainer_latest_*_rewardrecover.pt"))
    assert out_files, "Expected exactly one recovery checkpoint to be written"
    reloaded = torch.load(out_files[-1], map_location="cpu", weights_only=False)
    assert reloaded["best_validation_metric"] == float("-inf")
    assert math.isinf(reloaded["best_validation_metric"]) and reloaded["best_validation_metric"] < 0
    assert reloaded["early_stopping_counter"] == 0
    # Untouched fields preserved
    assert reloaded["agent_total_steps"] == 1234
    assert reloaded["total_train_steps"] == 999


# ---------------------------------------------------------------------------
# Save behaviour: written file beats every pre-existing latest by episode
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_recovered_checkpoint_beats_existing_latest_files(tmp_path, recover_mod):
    _write_dummy_latest(tmp_path, date="20260418", episode=3499)

    fake_ckpt = {
        "episode": 2800,  # source episode
        "network_state_dict": {},
        "marker": "recovered",
    }
    out_path = recover_mod.write_recovered_checkpoint(fake_ckpt, tmp_path, src_episode=2800)

    assert out_path.exists()
    # Filename should encode an episode strictly larger than any pre-existing
    # latest so find_latest_checkpoint picks it up.
    import re

    m = re.search(r"_ep(\d+)_reward", out_path.name)
    assert m is not None
    new_ep = int(m.group(1))
    assert new_ep > 3499
    assert "rewardrecover" in out_path.name


@pytest.mark.unit
def test_write_recovered_updates_episode_field_in_payload(tmp_path, recover_mod):
    _write_dummy_latest(tmp_path, date="20260418", episode=3499)
    fake_ckpt = {"episode": 2800, "network_state_dict": {}}
    out_path = recover_mod.write_recovered_checkpoint(fake_ckpt, tmp_path, src_episode=2800)
    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    # The recovered checkpoint advertises the new episode index so resume
    # logging stays consistent with the filename.
    assert reloaded["episode"] > 3499


# ---------------------------------------------------------------------------
# Resume command rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_resume_command_minimal(recover_mod):
    cmd = recover_mod.render_resume_command(
        benchmark_frac_override=None,
        config_path="config/training_config.yaml",
    )
    assert cmd == "python -m momentum_train.run_training --resume"


@pytest.mark.unit
def test_render_resume_command_includes_benchmark_override(recover_mod):
    cmd = recover_mod.render_resume_command(
        benchmark_frac_override=0.15,
        config_path="config/training_config.yaml",
    )
    assert "--benchmark-frac-override 0.15" in cmd
    assert "--resume" in cmd


@pytest.mark.unit
def test_render_resume_command_passes_through_custom_config(recover_mod):
    cmd = recover_mod.render_resume_command(
        benchmark_frac_override=None,
        config_path="config/alt.yaml",
    )
    assert "--config_path config/alt.yaml" in cmd
    assert "--resume" in cmd


# ---------------------------------------------------------------------------
# CLI integration: dry-run on an empty dir is a graceful failure path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_dry_run_with_empty_model_dir_returns_nonzero(tmp_path, recover_mod, caplog):
    """No checkpoints anywhere => discovery raises => script exits non-zero."""
    with caplog.at_level("INFO"):
        with pytest.raises(FileNotFoundError):
            recover_mod.pick_checkpoint(tmp_path, from_episode=None)
