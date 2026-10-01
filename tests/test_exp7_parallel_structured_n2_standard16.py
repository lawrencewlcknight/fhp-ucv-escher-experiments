"""Experiment 7's immutable budget, cloud wiring, and real-Ray recovery."""
from copy import deepcopy
import csv
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.fhp.exp2_fhp_lossless_structured_ucv import config as exp2
from experiments.fhp.exp6_fhp_structured_n2_standard16 import config as exp6
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16 import config as exp7
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.aggregate import aggregate_workers, task_name
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.run import main
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.preflight import capacity_preflight
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.training_state import STATE_TYPE, read_training_state
from gcp import exp7_parallel_structured_n2_standard16_batch as builder
from gcp import exp2_lossless_structured_batch as old_builder
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.compare import compare_runs
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16.training_state import restore_training_state


def test_only_parallel_execution_changes():
    assert exp7.EXPERIMENT_CONFIG == exp6.EXPERIMENT_CONFIG == exp2.EXPERIMENT_CONFIG
    assert exp7.EXPERIMENT_CONFIG is not exp6.EXPERIMENT_CONFIG
    assert exp7.PRODUCTION_SEEDS == exp6.PRODUCTION_SEEDS == (0, 1, 2)
    assert exp7.REFERENCE_VM == exp6.REFERENCE_VM
    assert exp7.checkpoint_schedule() == exp6.checkpoint_schedule()
    assert exp7.CHECKPOINT_HOURS == (6, 12, 18, 24)
    assert exp7.EXPERIMENT_CONFIG["num_traversals"] == 10_000
    settings = exp7.parallel_settings()
    assert settings["parallel_num_workers"] == 8
    assert settings["parallel_collection_chunk_size"] == 1200
    assert settings["parallel_ray_object_store_memory"] == 4 * 1024**3
    assert not settings["parallelize_independent_learners"]
    assert exp7.LEARNER_THREADS == exp6.LEARNER_THREADS == 8
    assert not exp7.contract_manifest()["frozen_critic_target_cache"]
    exp7.validate_contract(seeds=(0, 1, 2), schedule=exp7.checkpoint_schedule(),
                          config=exp7.EXPERIMENT_CONFIG, smoke=False)
    changed = deepcopy(exp7.EXPERIMENT_CONFIG)
    changed["num_traversals"] *= 8
    with pytest.raises(ValueError, match="num_traversals"):
        exp7.validate_contract(seeds=(0, 1, 2), schedule=exp7.checkpoint_schedule(),
                              config=changed, smoke=False)


def test_incompatible_parallel_resume_rejected():
    solver = SimpleNamespace(_parallel_num_workers=8, _parallel_run_seed=0,
                             _parallel_collection_chunk_size=1200,
                             _parallel_cache_actor_snapshots=True,
                             _parallelize_independent_learners=False)
    with pytest.raises(ValueError, match="schema"):
        restore_training_state(solver, {"type": "serial-state"})
    with pytest.raises(ValueError, match="contract differs"):
        restore_training_state(solver, {"type": STATE_TYPE, "parallel": {"contract": {}}})


def test_analysis_only_comparison(tmp_path):
    def write_csv(path, rows):
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for config, factor in ((exp6, 1), (exp7, 2)):
        directory = tmp_path / str(config.EXPERIMENT_ID) / "analysis"
        directory.mkdir(parents=True)
        (directory / "SUCCESS.json").write_text('{}')
        (directory / "experiment_manifest.json").write_text(json.dumps({
            "experiment_id": config.EXPERIMENT_ID, "contract": config.contract_manifest()}))
        write_csv(directory / "seed_summaries.csv", [
            {"seed": seed, "smoke": False, "final_nodes_touched": 1000 * factor,
             "final_iteration": 10 * factor, "final_training_elapsed_seconds": 86400}
            for seed in (0, 1, 2)])
        write_csv(directory / "checkpoint_index.csv", [
            {"seed": seed, "checkpoint_target_hours": hour,
             "actual_training_elapsed_seconds": hour * 3600,
             "nodes_touched": 1000 * hour * factor}
            for seed in (0, 1, 2) for hour in (6, 12, 18, 24)])
    result = compare_runs(tmp_path / "6", tmp_path / "7", tmp_path / "comparison")
    assert result["mean_node_throughput_ratio"] == 2
    assert result["mean_iteration_throughput_ratio"] == 2
    assert (tmp_path / "comparison/sequential_vs_parallel_nodes.png").is_file()
    with pytest.raises(ValueError, match="Expected Experiment 6"):
        compare_runs(tmp_path / "7", tmp_path / "6", tmp_path / "wrong")


@pytest.mark.parametrize("kind", ["controller", "smoke", "train", "aggregate"])
def test_cloud_contract(kind, tmp_path):
    args = SimpleNamespace(kind=kind, repo_url=builder.REPO_URL, repo_ref="abc123",
                           bucket_root="gs://test-bucket", run_id="exp7-par8-test",
                           service_account="runner@example.com", parallelism=3,
                           project_id="test-project", region="europe-west1",
                           controller_action="orchestrate")
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    machine = job["allocationPolicy"]["instances"][0]["policy"]["machineType"]
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert group["taskCountPerNode"] == 1
    if kind in ("train", "smoke"):
        assert machine == "n2-standard-16"
        assert spec["computeResource"] == {"cpuMilli": 16000, "memoryMib": 62000}
    else:
        assert machine == ("e2-small" if kind == "controller" else "n2-standard-8")
    if kind == "train":
        assert spec["maxRunDuration"] == "129600s"
        assert spec["maxRetryCount"] == 0
    script = spec["runnables"][0]["script"]["text"]
    assert "exp2_fhp_lossless_structured_ucv" not in script
    assert "EXP2_REMOTE" not in script and "EXP6_REMOTE" not in script
    assert "checkpoint-smoke" not in script
    assert ("capacity-preflight" in script) == (kind == "smoke")
    if kind in ("train", "smoke"):
        assert "--requested-memory-mib 62000" in script
        assert "export OMP_NUM_THREADS=8" in script
    if kind == "train":
        assert "parallel_structured_ucv_escher_n2_16" in script
        assert "EXP7_REMOTE_TASK_URI" in script
    path = tmp_path / f"{kind}.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)
    args.kind = "train"
    old = old_builder.build_job(args)
    assert old["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"


@pytest.mark.ray
def test_eight_actor_resume_smoke_and_capacity_preflight(tmp_path):
    main(["smoke", "--output-root", str(tmp_path), "--no-resume"])
    worker = tmp_path / "workers" / task_name(0, 0)
    summary = json.loads((worker / "summary.json").read_text())
    assert not summary["resumed_from_training_state"]
    assert summary["training_state_retention"] == "none"
    assert summary["final_episode"] == 32  # total, NOT eight times 32
    assert summary["final_iteration"] == 1
    diag = summary["execution_diagnostics"]
    assert diag["parallel_num_workers"] == 8
    assert diag["parallel_dispatch_count"] == 4
    assert diag["parallel_worker_snapshot_reload_count"] == 16
    assert diag["effective_parallel_learner_threads"] == 1
    runtime = json.loads((worker / "runtime_manifest.json").read_text())
    assert runtime["torch_intraop_threads"] == 1
    assert runtime["traversal_execution"] == "ray_parallel"
    rows = json.loads((worker / "checkpoint_manifest.json").read_text())
    assert all("training_state_path" not in row for row in rows)
    assert not list(tmp_path.rglob("*.pt"))
    for row in rows:
        assert row["outer_iteration"] == 1
        assert row["execution_diagnostics"]["parallel_dispatch_count"] == 4
    assert json.loads((tmp_path / "restart_validation/restart_validation.json").read_text())["learning_state_bitwise_identical"]
    assert (tmp_path / "analysis/training_time_breakdown.png").is_file()
    assert (tmp_path / "analysis/SUCCESS.json").is_file()
    torch.set_num_threads(1)
    # Same allocation/serialization path, tiny capacity for the local test.
    result = capacity_preflight(tmp_path / "preflight", config=exp7.smoke_config(), smoke=True)
    assert result["status"] == "passed" and result["serialized_bytes"] > 0
    assert not list((tmp_path / "preflight").glob("exp7-capacity-*"))
    manifest_path = worker / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["execution_backend"] = "sequential_seed_worker"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Not an Experiment 7"):
        aggregate_workers(tmp_path, seeds=(0,))
