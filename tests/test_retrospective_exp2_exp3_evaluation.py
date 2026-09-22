from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from experiments.fhp.retrospective_exp2_exp3_evaluation.run import (
    EXPERIMENTS,
    _build_tasks,
    _collapse_lbr_shards,
    discover_checkpoints,
)
from fhp_escher.checkpointing import sha256_file
from gcp.retrospective_exp2_exp3_evaluation_batch import build_job


def _fake_records():
    return [
        {
            "experiment": experiment,
            "experiment_label": EXPERIMENTS[experiment]["label"],
            "seed": seed,
            "training_hours": hour,
            "checkpoint_path": f"/{experiment}/{seed}/{hour}.pkl",
            "nodes_touched": hour * (10 if experiment == "exp2" else 8),
        }
        for experiment in EXPERIMENTS
        for seed in (0, 1, 2)
        for hour in (6, 12, 18, 24)
    ]


def test_production_task_schedule_covers_all_evaluation_tiers():
    args = Namespace(
        smoke=False,
        rule_deals=10_000,
        lbr_deals=1_000,
        lbr_shard_deals=10,
        lbr_rollouts=4_096,
        crossplay_deals=50_000,
        base_seed=20_260_922,
    )
    tasks = _build_tasks(_fake_records(), args)
    counts = {}
    for task in tasks:
        counts[task["kind"]] = counts.get(task["kind"], 0) + 1
    assert counts == {
        "rule": 120,
        "lbr": 2_400,
        "temporal_crossplay": 36,
        "direct_crossplay": 12,
        "node_matched_crossplay": 3,
    }
    assert len({task["task_id"] for task in tasks}) == len(tasks)


def test_lbr_shards_are_recombined_before_training_seed_inference():
    common = {
        "experiment": "exp2",
        "training_seed": 0,
        "training_hours": 24,
        "policy_a_name": "target",
        "evaluation_seed": 10,
        "lbr_seed": 20,
        "lbr_rollouts": 32,
    }
    rows = [
        {
            **common,
            "shard_index": 0,
            "_paired_values": [1.0, 3.0],
            "_player_zero_values": [2.0, 4.0],
            "_player_one_values": [0.0, 2.0],
        },
        {
            **common,
            "shard_index": 1,
            "evaluation_seed": 11,
            "_paired_values": [5.0, 7.0],
            "_player_zero_values": [6.0, 8.0],
            "_player_one_values": [4.0, 6.0],
        },
    ]
    collapsed = _collapse_lbr_shards(rows)
    assert len(collapsed) == 1
    assert collapsed[0]["num_deal_pairs"] == 4
    assert collapsed[0]["mean_chips_per_hand"] == pytest.approx(4.0)
    assert collapsed[0]["num_shards"] == 2


def test_checkpoint_discovery_verifies_three_seeds_and_four_times(tmp_path: Path):
    root = tmp_path / "run"
    contract = EXPERIMENTS["exp2"]
    for seed in (0, 1, 2):
        worker = root / "workers" / f"task_{seed:03d}"
        checkpoints = worker / "checkpoints"
        checkpoints.mkdir(parents=True)
        (worker / "SUCCESS.json").write_text('{"status":"complete"}', encoding="utf-8")
        (worker / "run_manifest.json").write_text(
            json.dumps(
                {
                    "experiment_name": contract["experiment_name"],
                    "algorithm_id": contract["algorithm_id"],
                    "seed": seed,
                }
            ),
            encoding="utf-8",
        )
        rows = []
        for hour in (6, 12, 18, 24):
            path = checkpoints / f"seed_{seed}_{hour}h.pkl"
            path.write_bytes(f"{seed}-{hour}".encode())
            rows.append(
                {
                    "checkpoint_id": f"time_{hour:02d}h",
                    "checkpoint_target_seconds": hour * 3600,
                    "actual_training_elapsed_seconds": hour * 3600 + 1,
                    "outer_iteration": hour,
                    "nodes_touched": hour * 100,
                    "path": f"checkpoints/{path.name}",
                    "sha256": sha256_file(path),
                }
            )
        (worker / "checkpoint_manifest.json").write_text(
            json.dumps(rows), encoding="utf-8"
        )
    records = discover_checkpoints(root, "exp2")
    assert len(records) == 12
    assert records[-1]["training_hours"] == 24
    assert records[-1]["seed"] == 2


def test_batch_job_is_one_standard_n2_vm_with_cloud_smoke():
    args = Namespace(
        repo_url="https://example.invalid/repo.git",
        repo_ref="deadbeef",
        bucket_root="gs://bucket",
        run_id="evaluation-run",
        exp2_run_id="exp2-run",
        exp3_run_id="exp3-run",
        service_account="runner@example.invalid",
        rule_deals=10_000,
        lbr_deals=1_000,
        lbr_rollouts=4_096,
        lbr_shard_deals=10,
        crossplay_deals=50_000,
        base_seed=20_260_922,
    )
    job = build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == 1
    assert group["parallelism"] == 1
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "--smoke" in script
    assert "--workers 8" in script
    assert "training_states" in script
