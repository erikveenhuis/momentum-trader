"""Tests for checkpoint_utils: find_latest_checkpoint and load_checkpoint."""

from pathlib import Path

import pytest
import torch
from momentum_train.utils.checkpoint_utils import (
    _probe_checkpoint_usable,
    find_latest_checkpoint,
    load_checkpoint,
)


def _write_valid_checkpoint(path: Path, *, payload: dict | None = None) -> Path:
    """Write a real (tiny) torch-saved ZIP so ``find_latest_checkpoint``'s
    usability probe accepts it. Stub bytes like ``b"x"`` no longer pass the
    probe, so any test that needs a "valid enough to be picked" file should
    use this helper.
    """
    torch.save(payload if payload is not None else {"stub": True}, path)
    return path


REQUIRED_KEYS = [
    "episode",
    "total_train_steps",
    "network_state_dict",
    "best_validation_metric",
    "target_network_state_dict",
    "agent_total_steps",
    "early_stopping_counter",
    "agent_config",
]

OPTIONAL_KEYS = [
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
]


def _make_valid_checkpoint(*, with_optimizer: bool = True) -> dict:
    ckpt = {key: 0 for key in REQUIRED_KEYS}
    if with_optimizer:
        for key in OPTIONAL_KEYS:
            ckpt[key] = 0
    return ckpt


# --- find_latest_checkpoint ---


@pytest.mark.unit
def test_find_latest_checkpoint_by_episode_number(tmp_path):
    for ep in (10, 50, 25):
        _write_valid_checkpoint(tmp_path / f"checkpoint_trainer_latest_20260401_ep{ep}_reward0.50.pt")

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is not None
    assert "ep50" in result


@pytest.mark.unit
def test_find_latest_checkpoint_falls_back_to_latest_pt(tmp_path):
    _write_valid_checkpoint(tmp_path / "checkpoint_trainer_latest.pt")

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is not None
    assert result.endswith("checkpoint_trainer_latest.pt")


@pytest.mark.unit
def test_find_latest_checkpoint_falls_back_to_best_pt(tmp_path):
    _write_valid_checkpoint(tmp_path / "checkpoint_trainer_best.pt")

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is not None
    assert result.endswith("checkpoint_trainer_best.pt")


@pytest.mark.unit
def test_find_latest_checkpoint_returns_none_empty_dir(tmp_path):
    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is None


@pytest.mark.unit
def test_find_latest_checkpoint_skips_zero_byte_and_falls_back(tmp_path, caplog):
    """Zero-byte stubs from an interrupted torch.save must be skipped.

    Reproduces the April 22 OOM-during-save scenario: the new "latest"
    filename was created before any bytes were written, so the loader picked
    the empty stub and raised ``EOFError: Ran out of input``, which the resume
    path then (pre-fix) swallowed into a silent "start from scratch".
    """
    _write_valid_checkpoint(tmp_path / "checkpoint_trainer_latest_20260401_ep100_reward0.50.pt")
    (tmp_path / "checkpoint_trainer_latest_20260401_ep200_reward-inf.pt").write_bytes(b"")

    with caplog.at_level("WARNING", logger="CheckpointUtils"):
        result = find_latest_checkpoint(model_dir=str(tmp_path))

    assert result is not None, "Expected fallback to the non-empty candidate, got None."
    assert "ep100" in result, f"Expected to pick ep100 after skipping empty ep200, got {result}"
    assert any("zero-byte" in rec.message.lower() for rec in caplog.records), (
        "Expected a WARNING log about skipping the zero-byte checkpoint."
    )


@pytest.mark.unit
def test_find_latest_checkpoint_skips_truncated_zip_and_falls_back(tmp_path, caplog):
    """Truncated ZIP archives (torch.save OOM'd mid-write) must also be skipped.

    Reproduces the April 23 2026 failure: the rotation+save produced an
    ep3700 file of ~479 MB (vs ~8.3 GB expected). It was non-empty, so the
    old size-only guard accepted it, and then torch.load blew up with
    ``PytorchStreamReader failed reading zip archive: failed finding
    central directory``. The probe now catches this via
    ``zipfile.is_zipfile`` returning False.
    """
    good = _write_valid_checkpoint(tmp_path / "checkpoint_trainer_latest_20260423_ep3652_reward-inf.pt")
    # Build a real torch checkpoint then truncate it in place -- this mirrors
    # an OOM during torch.save far more faithfully than raw garbage bytes
    # (which would also be rejected but via a different code path).
    truncated = tmp_path / "checkpoint_trainer_latest_20260423_ep3700_reward-inf.pt"
    torch.save({"stub": True}, truncated)
    full_bytes = truncated.read_bytes()
    truncated.write_bytes(full_bytes[: len(full_bytes) // 2])
    assert truncated.stat().st_size > 0, "Test precondition: truncated file must be non-zero"

    with caplog.at_level("WARNING", logger="CheckpointUtils"):
        result = find_latest_checkpoint(model_dir=str(tmp_path))

    assert result is not None, "Expected fallback to ep3652, got None."
    assert "ep3652" in result, f"Expected to pick ep3652 after skipping the truncated ep3700, got {result}"
    assert str(good) == result
    assert any("central directory" in rec.message.lower() or "not a valid zip" in rec.message.lower() for rec in caplog.records), (
        "Expected a WARNING log explaining the truncated-ZIP skip."
    )


@pytest.mark.unit
def test_find_latest_checkpoint_returns_none_when_all_candidates_empty(tmp_path):
    """If every episode-tagged checkpoint is zero-byte, we must not return one."""
    for ep in (100, 200, 300):
        (tmp_path / f"checkpoint_trainer_latest_20260401_ep{ep}_reward-inf.pt").write_bytes(b"")

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is None


@pytest.mark.unit
def test_find_latest_checkpoint_returns_none_when_all_candidates_truncated(tmp_path):
    """Back-to-back save crashes could truncate every candidate; still return None."""
    for ep in (100, 200, 300):
        path = tmp_path / f"checkpoint_trainer_latest_20260401_ep{ep}_reward-inf.pt"
        torch.save({"stub": True}, path)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 3])

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is None


@pytest.mark.unit
def test_find_latest_checkpoint_legacy_latest_pt_zero_byte_is_skipped(tmp_path):
    """The legacy ``{prefix}_latest.pt`` fallback must also honor the probe."""
    (tmp_path / "checkpoint_trainer_latest.pt").write_bytes(b"")
    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is None


@pytest.mark.unit
def test_find_latest_checkpoint_legacy_latest_pt_truncated_is_skipped(tmp_path):
    """Legacy fallback also rejects truncated ZIPs, not just empty files."""
    legacy = tmp_path / "checkpoint_trainer_latest.pt"
    torch.save({"stub": True}, legacy)
    data = legacy.read_bytes()
    legacy.write_bytes(data[: len(data) // 2])
    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is None


# --- _probe_checkpoint_usable: direct unit tests for the probe itself ---


@pytest.mark.unit
def test_probe_accepts_valid_torch_save(tmp_path):
    path = tmp_path / "valid.pt"
    torch.save({"foo": 42}, path)
    usable, reason = _probe_checkpoint_usable(str(path))
    assert usable, f"Valid torch.save rejected by probe: {reason!r}"
    assert reason == ""


@pytest.mark.unit
def test_probe_rejects_zero_byte_file(tmp_path):
    path = tmp_path / "empty.pt"
    path.write_bytes(b"")
    usable, reason = _probe_checkpoint_usable(str(path))
    assert not usable
    assert "zero-byte" in reason.lower()


@pytest.mark.unit
def test_probe_rejects_truncated_zip(tmp_path):
    path = tmp_path / "truncated.pt"
    torch.save({"foo": 42}, path)
    full = path.read_bytes()
    path.write_bytes(full[: len(full) // 2])
    usable, reason = _probe_checkpoint_usable(str(path))
    assert not usable
    assert "central directory" in reason.lower() or "zip" in reason.lower()


@pytest.mark.unit
def test_probe_rejects_random_garbage(tmp_path):
    """Random non-ZIP bytes (not a torch.save output at all) must also be rejected."""
    path = tmp_path / "garbage.pt"
    path.write_bytes(b"\x00" * 4096 + b"not a zip anywhere in here")
    usable, reason = _probe_checkpoint_usable(str(path))
    assert not usable
    assert "zip" in reason.lower()


@pytest.mark.unit
def test_probe_rejects_nonexistent_file(tmp_path):
    usable, reason = _probe_checkpoint_usable(str(tmp_path / "does_not_exist.pt"))
    assert not usable
    assert "stat failed" in reason.lower()


# --- load_checkpoint ---


@pytest.mark.unit
def test_load_checkpoint_success(tmp_path):
    ckpt = _make_valid_checkpoint()
    ckpt["best_validation_metric"] = 0.42
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    loaded = load_checkpoint(str(path))
    assert loaded is not None
    assert loaded["best_validation_metric"] == pytest.approx(0.42)
    for key in REQUIRED_KEYS:
        assert key in loaded


@pytest.mark.unit
def test_load_checkpoint_missing_keys(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"episode": 1}, path)

    result = load_checkpoint(str(path))
    assert result is None


@pytest.mark.unit
def test_load_checkpoint_succeeds_without_optimizer_state(tmp_path):
    """A checkpoint stripped by --reset-lr-on-resume / --strip-optimizer must still load.

    The optimizer / scheduler / scaler keys are intentionally optional
    because both `--reset-lr-on-resume` and
    `scripts/recover_from_collapse.py --strip-optimizer` produce
    checkpoints without them.
    """
    ckpt = _make_valid_checkpoint(with_optimizer=False)
    path = tmp_path / "stripped.pt"
    torch.save(ckpt, path)

    loaded = load_checkpoint(str(path))
    assert loaded is not None
    for key in OPTIONAL_KEYS:
        assert key not in loaded


@pytest.mark.unit
def test_load_checkpoint_file_not_found(tmp_path):
    result = load_checkpoint(str(tmp_path / "nonexistent.pt"))
    assert result is None


@pytest.mark.unit
def test_load_checkpoint_none_path():
    result = load_checkpoint(None)
    assert result is None


@pytest.mark.unit
def test_load_checkpoint_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a valid torch checkpoint")

    result = load_checkpoint(str(path))
    assert result is None
