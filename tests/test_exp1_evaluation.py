from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.fhp.exp1_archived_ucv_escher_baseline.evaluate_checkpoints import (
    _delta_summary,
    _load_verified_checkpoints,
)
from fhp_escher.checkpointing import sha256_file


def test_delta_summary_uses_12h_minus_6h_sign_convention():
    result = _delta_summary(
        [1.0, 2.0, 3.0], interpretation="positive_favours_12h", seeds=[1]
    )
    assert result["mean_delta_chips_per_hand"] == pytest.approx(2.0)
    assert result["num_paired_deals"] == 3


def test_checkpoint_verifier_resolves_moved_manifest_paths(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    rows = []
    for index, (hours, content) in enumerate(((6, b"six"), (12, b"twelve"))):
        path = checkpoints / f"checkpoint_{hours}h.pkl"
        path.write_bytes(content)
        rows.append(
            {
                "checkpoint_target_seconds": hours * 60 * 60,
                "path": f"/remote/machine/{path.name}",
                "sha256": sha256_file(path),
            }
        )
    (run_dir / "checkpoint_manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    _, resolved = _load_verified_checkpoints(run_dir)
    assert resolved["checkpoint_6h"].name == "checkpoint_6h.pkl"
    assert resolved["checkpoint_12h"].name == "checkpoint_12h.pkl"
