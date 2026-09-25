from __future__ import annotations

from argparse import Namespace

from experiments.fhp.retrospective_exp1_exp2_exp3_evaluation.run import (
    EXPERIMENTS,
    _build_new_tasks,
)
from gcp.retrospective_exp1_exp2_exp3_evaluation_batch import build_job


def _fake_records():
    return [
        {
            "experiment": experiment,
            "experiment_label": EXPERIMENTS[experiment]["label"],
            "seed": seed,
            "training_hours": hour,
            "checkpoint_path": f"/{experiment}/{seed}/{hour}.pkl",
            "nodes_touched": hour * {"exp1": 13, "exp2": 10, "exp3": 8}[experiment],
        }
        for experiment in EXPERIMENTS
        for seed in (0, 1, 2)
        for hour in (6, 12, 18, 24)
    ]


def test_new_task_schedule_only_computes_missing_experiment_one_work():
    args = Namespace(
        smoke=False,
        rule_deals=10_000,
        lbr_deals=1_000,
        lbr_shard_deals=10,
        lbr_rollouts=4_096,
        crossplay_deals=50_000,
        base_seed=20_260_922,
    )
    tasks = _build_new_tasks(_fake_records(), args)
    counts = {}
    for task in tasks:
        counts[task["kind"]] = counts.get(task["kind"], 0) + 1
    assert counts == {
        "rule": 60,
        "lbr": 1_200,
        "temporal_crossplay": 18,
        "direct_crossplay": 24,
        "node_matched_crossplay": 6,
    }
    assert len({task["task_id"] for task in tasks}) == len(tasks)


def test_direct_pairs_share_the_frozen_common_deal_schedule():
    args = Namespace(
        smoke=False,
        rule_deals=10_000,
        lbr_deals=1_000,
        lbr_shard_deals=10,
        lbr_rollouts=4_096,
        crossplay_deals=50_000,
        base_seed=20_260_922,
    )
    tasks = _build_new_tasks(_fake_records(), args)
    direct = [
        task for task in tasks
        if task["kind"] == "direct_crossplay"
        and task["training_seed"] == 0 and task["training_hours"] == 24
    ]
    assert len(direct) == 2
    assert {task["evaluation_seed"] for task in direct} == {23_260_946}


def test_batch_job_uses_one_standard_vm_and_imports_frozen_reference():
    args = Namespace(
        repo_url="https://example.invalid/repo.git",
        repo_ref="deadbeef",
        bucket_root="gs://bucket",
        run_id="evaluation-run",
        exp1_run_id="exp1-run",
        exp2_run_id="exp2-run",
        exp3_run_id="exp3-run",
        reference_run_id="reference-run",
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
    assert group["taskSpec"]["maxRunDuration"] == "64800s"
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "--smoke" in script
    assert "--workers 8" in script
    assert script.count("--exclude='.*training_states.*'") == 3
    assert "$REFERENCE_RUN_ID/analysis" in script
