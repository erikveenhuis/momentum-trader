"""Unit tests for the top-K best-validation ring (CheckpointMixin).

The ring pins the K highest-scoring validation checkpoints on disk in a
separate ``checkpoint_trainer_topk_<DATE>_ep<N>_score_<X>.pt`` stream that
runs alongside the threshold-gated ``best_*`` save. It exists specifically
to catch true peaks whose improvement is smaller than
``min_validation_threshold`` (the case observed at episode 6400 in the
April 2026 run, where score 0.4424 over best 0.4416 was rejected).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from momentum_train.trainer import RainbowTrainerModule


class _Net:
    def state_dict(self) -> dict:
        return {"w": 0}


class _Optim:
    def state_dict(self) -> dict:
        return {"o": 0}


def _make_trainer(tmp_path: Path, *, top_k: int, min_eps: int = 0) -> RainbowTrainerModule:
    """Build a minimal trainer wired for ``_save_checkpoint`` + topk ring."""
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.best_validation_metric = 0.5
    trainer.early_stopping_counter = 0
    trainer.min_episodes_before_early_stopping = min_eps
    trainer.top_k_best_checkpoints = top_k
    trainer._checkpoints_saved_this_run = 0
    trainer.writer = None
    trainer.latest_trainer_checkpoint_path = str(tmp_path / "checkpoint_trainer_latest.pt")
    trainer.best_trainer_checkpoint_base_path = str(tmp_path / "checkpoint_trainer_best")
    trainer.best_model_base_prefix = str(tmp_path / "rainbow_best")
    trainer.run_config = {"model_dir": str(tmp_path)}
    trainer.latest_checkpoint_keep_last_n = 0
    trainer.agent = SimpleNamespace(
        config={"lr": 1e-4},
        total_steps=100,
        env_steps=100,
        network=_Net(),
        target_network=_Net(),
        optimizer=_Optim(),
        scheduler=None,
        lr_scheduler_enabled=False,
        # No buffer attribute → ``_save_buffer_sidecar`` returns None (test
        # path doesn't exercise side-car write).
    )
    return trainer


def _topk_files(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob("checkpoint_trainer_topk_*_ep*_score_*.pt"))


@pytest.mark.unit
def test_topk_save_no_op_when_disabled(tmp_path):
    """Default ``top_k_best_checkpoints=0`` must keep the new code path dormant."""
    trainer = _make_trainer(tmp_path, top_k=0)
    saved = trainer._maybe_save_topk_checkpoint(episode=100, total_steps=10_000, validation_score=0.9)
    assert saved is None
    assert _topk_files(tmp_path) == []


@pytest.mark.unit
def test_topk_save_skips_non_finite_score(tmp_path):
    """``-inf`` / NaN never enters the ring — the rest of the trainer treats
    these as "validation failed", not as a real peak."""
    trainer = _make_trainer(tmp_path, top_k=3)
    assert trainer._maybe_save_topk_checkpoint(episode=10, total_steps=1, validation_score=float("-inf")) is None
    assert trainer._maybe_save_topk_checkpoint(episode=10, total_steps=1, validation_score=float("nan")) is None
    assert _topk_files(tmp_path) == []


@pytest.mark.unit
def test_topk_save_respects_min_episodes_gate(tmp_path):
    """Until the ``min_episodes_before_early_stopping`` gate has been crossed
    the ring must defer (mirrors the threshold-gated ``best_*`` eligibility)."""
    trainer = _make_trainer(tmp_path, top_k=3, min_eps=5000)
    saved = trainer._maybe_save_topk_checkpoint(episode=4999, total_steps=1, validation_score=0.99)
    assert saved is None
    assert _topk_files(tmp_path) == []
    # At/after the gate, saves resume.
    saved = trainer._maybe_save_topk_checkpoint(episode=5000, total_steps=1, validation_score=0.99)
    assert saved is not None
    assert len(_topk_files(tmp_path)) == 1


@pytest.mark.unit
def test_topk_save_fills_ring_unconditionally_until_full(tmp_path):
    """Below capacity the ring saves every finite score, even one that wouldn't
    beat the current minimum (there is no current minimum yet)."""
    trainer = _make_trainer(tmp_path, top_k=3)
    for ep, score in [(100, 0.10), (200, 0.05), (300, 0.20)]:
        saved = trainer._maybe_save_topk_checkpoint(episode=ep, total_steps=ep, validation_score=score)
        assert saved is not None, f"expected save at ep={ep}, score={score}"
    files = _topk_files(tmp_path)
    assert len(files) == 3
    # All three scores are present in filenames (ordering is filesystem-glob
    # dependent, so check membership not order).
    score_strs = {f.name.split("_score_")[1].rsplit(".pt", 1)[0] for f in files}
    assert score_strs == {"0.1000", "0.0500", "0.2000"}


@pytest.mark.unit
def test_topk_save_evicts_lowest_when_full_and_score_wins(tmp_path):
    """Full ring + new score beats the minimum → evict min, save new."""
    trainer = _make_trainer(tmp_path, top_k=3)
    for ep, score in [(100, 0.10), (200, 0.05), (300, 0.20)]:
        trainer._maybe_save_topk_checkpoint(episode=ep, total_steps=ep, validation_score=score)
    assert len(_topk_files(tmp_path)) == 3

    # 0.15 beats the ring min (0.05) → 0.05 evicts, ring becomes {0.10, 0.15, 0.20}.
    saved = trainer._maybe_save_topk_checkpoint(episode=400, total_steps=400, validation_score=0.15)
    assert saved is not None
    score_strs = {f.name.split("_score_")[1].rsplit(".pt", 1)[0] for f in _topk_files(tmp_path)}
    assert score_strs == {"0.1000", "0.1500", "0.2000"}


@pytest.mark.unit
def test_topk_save_skips_when_full_and_score_does_not_win(tmp_path):
    """Full ring + new score below current minimum → no save, no eviction."""
    trainer = _make_trainer(tmp_path, top_k=2)
    trainer._maybe_save_topk_checkpoint(episode=100, total_steps=100, validation_score=0.40)
    trainer._maybe_save_topk_checkpoint(episode=200, total_steps=200, validation_score=0.50)
    assert len(_topk_files(tmp_path)) == 2

    saved = trainer._maybe_save_topk_checkpoint(episode=300, total_steps=300, validation_score=0.30)
    assert saved is None
    score_strs = {f.name.split("_score_")[1].rsplit(".pt", 1)[0] for f in _topk_files(tmp_path)}
    assert score_strs == {"0.4000", "0.5000"}


@pytest.mark.unit
def test_topk_save_skips_on_tie_with_ring_minimum(tmp_path):
    """Strict ``>`` to avoid pointless filename churn on a tie at the minimum."""
    trainer = _make_trainer(tmp_path, top_k=2)
    trainer._maybe_save_topk_checkpoint(episode=100, total_steps=100, validation_score=0.40)
    trainer._maybe_save_topk_checkpoint(episode=200, total_steps=200, validation_score=0.40)
    assert len(_topk_files(tmp_path)) == 2  # Both saved while ring was filling up.

    saved = trainer._maybe_save_topk_checkpoint(episode=300, total_steps=300, validation_score=0.40)
    assert saved is None
    assert len(_topk_files(tmp_path)) == 2


@pytest.mark.unit
def test_topk_ring_survives_simulated_resume(tmp_path):
    """The ring is rescanned from filenames each call, so a fresh trainer
    instance (the ``--resume`` case) immediately sees the prior ring."""
    trainer_a = _make_trainer(tmp_path, top_k=3)
    for ep, score in [(100, 0.10), (200, 0.20), (300, 0.30)]:
        trainer_a._maybe_save_topk_checkpoint(episode=ep, total_steps=ep, validation_score=score)
    assert len(_topk_files(tmp_path)) == 3

    # Brand-new trainer instance pointing at the same ``model_dir`` — no
    # in-memory ring state passed across, but the on-disk files still gate
    # the next save.
    trainer_b = _make_trainer(tmp_path, top_k=3)
    saved = trainer_b._maybe_save_topk_checkpoint(episode=400, total_steps=400, validation_score=0.05)
    assert saved is None  # 0.05 doesn't beat ring min (0.10)
    saved = trainer_b._maybe_save_topk_checkpoint(episode=500, total_steps=500, validation_score=0.25)
    assert saved is not None  # 0.25 > 0.10, beats ring min
    score_strs = {f.name.split("_score_")[1].rsplit(".pt", 1)[0] for f in _topk_files(tmp_path)}
    assert score_strs == {"0.2000", "0.2500", "0.3000"}


@pytest.mark.unit
def test_topk_save_writes_loadable_checkpoint(tmp_path):
    """Saved file must contain the same keys as the ``best_*`` checkpoint so
    it's drop-in usable with the existing loaders (no ``buffer_sidecar_relpath``,
    no buffer blob — the latest stream provides those for resume)."""
    trainer = _make_trainer(tmp_path, top_k=3)
    saved = trainer._maybe_save_topk_checkpoint(episode=42, total_steps=4200, validation_score=0.4424)
    assert saved is not None
    assert saved.exists()

    cp = torch.load(saved, map_location="cpu", weights_only=False)
    for key in (
        "episode",
        "total_train_steps",
        "best_validation_metric",
        "early_stopping_counter",
        "buffer_state",
        "buffer_sidecar_relpath",
        "agent_config",
        "agent_total_steps",
        "total_steps",
        "network_state_dict",
        "target_network_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "validation_score",
    ):
        assert key in cp, f"missing key in topk checkpoint: {key}"
    assert cp["buffer_state"] is None
    assert cp["buffer_sidecar_relpath"] is None
    assert cp["validation_score"] == pytest.approx(0.4424)
    assert cp["episode"] == 42


@pytest.mark.unit
def test_save_checkpoint_invokes_topk_save_when_validation_score_present(tmp_path):
    """Integration: ``_save_checkpoint(..., validation_score=...)`` must trigger
    the ring save even when ``is_best=False`` (the missed-peak scenario)."""
    trainer = _make_trainer(tmp_path, top_k=3)
    trainer._save_checkpoint(episode=10, total_steps=1000, is_best=False, validation_score=0.6)
    assert len(_topk_files(tmp_path)) == 1


@pytest.mark.unit
def test_save_checkpoint_skips_topk_when_no_validation_score(tmp_path):
    """Periodic saves (no validation_score) must NOT touch the topk ring."""
    trainer = _make_trainer(tmp_path, top_k=3)
    trainer._save_checkpoint(episode=10, total_steps=1000, is_best=False, validation_score=None)
    assert _topk_files(tmp_path) == []


@pytest.mark.unit
def test_topk_disabled_via_runtime_attribute_still_safe(tmp_path):
    """Defensive: missing ``top_k_best_checkpoints`` attribute (e.g. a partially
    constructed trainer in tests) must be treated as disabled, not crash."""
    trainer = _make_trainer(tmp_path, top_k=0)
    # Strip the attribute entirely, mimicking an older checkpoint of trainer
    # state that predates this feature.
    del trainer.top_k_best_checkpoints
    saved = trainer._maybe_save_topk_checkpoint(episode=10, total_steps=10, validation_score=0.9)
    assert saved is None
    assert _topk_files(tmp_path) == []


@pytest.mark.unit
def test_topk_ring_handles_lower_k_on_resume_lazily(tmp_path):
    """Resuming with a smaller K than the on-disk ring leaves stale files in
    place until the next eligible save, at which point eviction trims down to K.

    This is intentional: the ring never deletes files outside of a save flow,
    so the user can shrink K mid-run without surprise data loss until the
    next validation lands on the ring.
    """
    trainer_a = _make_trainer(tmp_path, top_k=5)
    for ep, score in [(100, 0.10), (200, 0.20), (300, 0.30), (400, 0.40), (500, 0.50)]:
        trainer_a._maybe_save_topk_checkpoint(episode=ep, total_steps=ep, validation_score=score)
    assert len(_topk_files(tmp_path)) == 5

    # Resume with K=2; before any save fires the old 5 files are untouched.
    trainer_b = _make_trainer(tmp_path, top_k=2)
    assert len(_topk_files(tmp_path)) == 5

    # The next save: ring is currently 5/2, score 0.45 beats min 0.10. We save
    # → ring becomes 6 on disk → eviction trims overflow=4 lowest entries
    # (0.10, 0.20, 0.30, 0.40), leaving {0.45, 0.50}.
    saved = trainer_b._maybe_save_topk_checkpoint(episode=600, total_steps=600, validation_score=0.45)
    assert saved is not None
    score_strs = {f.name.split("_score_")[1].rsplit(".pt", 1)[0] for f in _topk_files(tmp_path)}
    assert score_strs == {"0.4500", "0.5000"}


@pytest.mark.unit
def test_topk_ring_ignores_non_canonical_filenames(tmp_path):
    """Files that don't match the canonical filename schema (manual copies,
    third-party tooling outputs) must be left alone — never parsed, never
    counted against the ring, never deleted."""
    decoy = tmp_path / "checkpoint_trainer_topk_NOTADATE_score.pt"
    decoy.write_text("nope")
    decoy2 = tmp_path / "checkpoint_trainer_topk_20260101_ep5_score_invalid.pt"
    decoy2.write_text("nope")

    trainer = _make_trainer(tmp_path, top_k=2)
    entries = trainer._scan_topk_ring()
    assert entries == []  # decoys ignored

    trainer._maybe_save_topk_checkpoint(episode=10, total_steps=10, validation_score=0.1)
    trainer._maybe_save_topk_checkpoint(episode=20, total_steps=20, validation_score=0.2)
    # Saves succeeded; decoys still on disk.
    assert decoy.exists()
    assert decoy2.exists()


@pytest.mark.unit
def test_topk_save_skipped_when_agent_networks_missing(tmp_path):
    """Defensive: if the agent's networks/optimizer haven't been initialised
    yet (early lifecycle, mocked tests), the ring save quietly skips rather
    than writing a malformed .pt."""
    trainer = _make_trainer(tmp_path, top_k=3)
    trainer.agent.network = None
    saved = trainer._maybe_save_topk_checkpoint(episode=10, total_steps=10, validation_score=0.5)
    assert saved is None
    assert _topk_files(tmp_path) == []


@pytest.mark.unit
def test_topk_save_evicts_oldest_episode_on_score_tie_within_ring(tmp_path):
    """When two existing entries share the lowest score, the eviction tie-breaks
    on episode (oldest evicts first via the ``(score, episode)`` sort key).
    Stable so a long run can't silently keep an old, low-quality checkpoint
    forever just because some intermediate score happens to match."""
    trainer = _make_trainer(tmp_path, top_k=2)
    trainer._maybe_save_topk_checkpoint(episode=100, total_steps=100, validation_score=0.30)
    trainer._maybe_save_topk_checkpoint(episode=200, total_steps=200, validation_score=0.30)
    assert len(_topk_files(tmp_path)) == 2

    # 0.40 beats min (0.30) → evict the older 0.30 (episode 100), keep 200.
    saved = trainer._maybe_save_topk_checkpoint(episode=300, total_steps=300, validation_score=0.40)
    assert saved is not None
    names = sorted(p.name for p in _topk_files(tmp_path))
    # ep100 file gone, ep200 + ep300 remain.
    assert not any("_ep100_" in n for n in names)
    assert any("_ep200_" in n for n in names)
    assert any("_ep300_" in n for n in names)


@pytest.mark.unit
def test_topk_save_does_not_mutate_best_validation_metric(tmp_path):
    """The ring must be a pure save sink — it must NEVER mutate
    ``best_validation_metric`` (which only the threshold-gated path is
    allowed to update). Otherwise we'd silently lower the bar that the
    ``best_*`` save uses on the next call."""
    trainer = _make_trainer(tmp_path, top_k=3)
    trainer.best_validation_metric = 0.50
    trainer._maybe_save_topk_checkpoint(episode=10, total_steps=10, validation_score=0.45)
    assert trainer.best_validation_metric == pytest.approx(0.50)
    trainer._maybe_save_topk_checkpoint(episode=20, total_steps=20, validation_score=0.55)
    assert trainer.best_validation_metric == pytest.approx(0.50)


@pytest.mark.unit
def test_topk_ring_uses_separate_filename_stream_from_best(tmp_path):
    """Ring filenames must not collide with ``checkpoint_trainer_best_*`` so
    the existing best stream is left alone (rotation, recovery scripts, etc.)."""
    trainer = _make_trainer(tmp_path, top_k=2)
    saved = trainer._maybe_save_topk_checkpoint(episode=42, total_steps=42, validation_score=0.42)
    assert saved is not None
    assert saved.name.startswith("checkpoint_trainer_topk_")
    # The threshold-gated ``best_*`` stream is untouched.
    assert list(tmp_path.glob("checkpoint_trainer_best_*.pt")) == []


@pytest.mark.unit
def test_topk_save_deferred_score_smaller_than_threshold_still_lands(tmp_path):
    """Regression for the April 2026 missed-peak scenario: a +0.0008 improvement
    over the prior best (smaller than ``min_validation_threshold=0.001``) is
    rejected by the threshold-gated path but MUST land in the top-K ring."""
    trainer = _make_trainer(tmp_path, top_k=5)
    # Simulate the threshold-gated path having declined this score (by NOT
    # updating ``best_validation_metric``); the ring decision is independent.
    trainer.best_validation_metric = 0.4416
    saved = trainer._maybe_save_topk_checkpoint(episode=6400, total_steps=6_400_000, validation_score=0.4424)
    assert saved is not None
    files = _topk_files(tmp_path)
    assert len(files) == 1
    assert "_score_0.4424.pt" in files[0].name
    # And ``best_validation_metric`` is left at 0.4416 — only the threshold
    # path may bump it.
    assert trainer.best_validation_metric == pytest.approx(0.4416)
    # Sanity: actually loadable.
    cp = torch.load(files[0], map_location="cpu", weights_only=False)
    assert cp["validation_score"] == pytest.approx(0.4424)
    # The persisted ``best_validation_metric`` reflects the live trainer state
    # (still 0.4416 because the ring save doesn't bump it).
    assert cp["best_validation_metric"] == pytest.approx(0.4416)
    # numpy import only used for parity with other tests; keep here so this
    # module's imports stay used regardless of which subset of tests runs.
    assert np.isfinite(cp["validation_score"])
