"""Pinned continuation transport, validation, workflow wiring and final restart."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.fhp.exp36_fhp_strength_critic2500 import config as learner
from experiments.fhp.exp38_fhp_strength_critic2500_48h import contract as c, report
from gcp.exp38_strength_critic2500_48h_batch import build_job, REPO_URL

ROOT = Path(__file__).resolve().parents[1]


def update_json(path, values):
    data = json.loads(path.read_text())
    data.update(values)
    path.write_text(json.dumps(data))
    return data


def source_fixture(tmp_path, monkeypatch):
    from tests import test_exp20_evaluation as prior
    monkeypatch.setattr(prior.run, "CONFIGS", {"exp36": learner})
    monkeypatch.setattr(prior, "task_name", lambda index, seed: c.worker_name(seed))
    root = prior.source_fixture(tmp_path, monkeypatch)["exp36"]
    for seed in c.SEEDS:
        worker = root/"workers"/c.worker_name(seed)
        update_json(worker/"run_manifest.json", dict(repository_commit=c.TRAINING_REF,
            feature_encoder=learner.encoder().metadata()))
        update_json(worker/"summary.json", dict(final_nodes_touched=2400, final_iteration=24))
        update_json(worker/"SUCCESS.json", dict(summary_sha256=hashlib.sha256((worker/"summary.json").read_bytes()).hexdigest()))
        rows = json.loads((worker/"checkpoint_manifest.json").read_text())
        state = worker/rows[-1]["training_state_path"]
        state.parent.mkdir()
        state.write_bytes(b"transport-fixture-not-a-real-torch-state")
        rows[-1].update(training_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(), training_state_size_bytes=state.stat().st_size)
        (worker/"checkpoint_manifest.json").write_text(json.dumps(rows))
    return root


def cloud_reader(roots, calls):
    def cloud(*args):
        calls.append(args)
        uri = args[2] if args[0] == "objects" else args[1]
        run, relative = uri.removeprefix("gs://test-bucket/").split("/", 1)
        path = roots[run]/relative
        if args[0] == "objects":
            return json.dumps(dict(size=path.stat().st_size, generation="123")).encode()
        assert args[0] == "cat" and path.suffix == ".json"
        return path.read_bytes()
    return cloud


def test_frozen_contract_matches_actual_source():
    assert c.CONFIG_SHA256 == c.digest(learner.EXPERIMENT_CONFIG)
    assert c.PARALLEL_SETTINGS == learner.parallel_settings()
    assert learner.EXPERIMENT_CONFIG["baseline_network_train_steps"] == 2500
    assert c.SOURCE_RUN_ID == "exp36-rich2500-20261018-205125"
    assert c.TOTAL_HOURS == 48 and c.SOURCE_HOURS == 24 and c.SEEDS == (0, 1, 2)
    assert c.NEW_HOURS == (30, 36, 42, 48)
    assert c.HOURS == tuple(range(6, 49, 6))
    raw = json.loads((ROOT/"cloud_outputs"/c.SOURCE_RUN_ID/"analysis/throughput_summary.json").read_text()) if (ROOT/"cloud_outputs"/c.SOURCE_RUN_ID/"analysis/throughput_summary.json").exists() else None
    if raw:
        assert raw["repository_commit"] == c.TRAINING_REF


def test_cloud_preflight_is_metadata_only(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, monkeypatch)
    calls = []
    receipt = c.cloud_preflight("gs://test-bucket", cloud=cloud_reader({c.SOURCE_RUN_ID: source}, calls))
    assert len(receipt["source_states"]) == 3
    assert len(calls) == 18 and len([a for a in calls if a[0] == "objects"]) == 3
    assert all(a[0] == "objects" or a[1].endswith(".json") for a in calls)
    assert receipt["automatic_evaluation"] is False
    assert all(r["remaining_active_seconds"] == 86399 for r in receipt["source_states"])


@pytest.mark.parametrize("change", ("config", "seed", "runtime", "budget", "encoder", "hash", "schedule", "policy_path", "size"))
def test_bad_source_rejected_before_paid_training(tmp_path, monkeypatch, change):
    root = source_fixture(tmp_path, monkeypatch)
    worker = root/"workers"/c.worker_name(0)
    raw = {name: (worker/name).read_bytes() for name in c.METADATA_FILES}
    filename = {"config": "run_manifest.json", "seed": "run_manifest.json", "runtime": "runtime_manifest.json",
        "budget": "summary.json", "encoder": "run_manifest.json", "hash": "SUCCESS.json",
        "schedule": "checkpoint_manifest.json", "policy_path": "checkpoint_manifest.json", "size": "checkpoint_manifest.json"}[change]
    data = json.loads(raw[filename])
    if change == "config": data["training_config"]["baseline_network_train_steps"] = 5000
    elif change == "seed": data["seed"] = 2
    elif change == "runtime": data["python_version"] = "3.11.17"
    elif change == "budget": data["execution_diagnostics"]["critic_train_steps_per_fit"] = [5000, 2500]
    elif change == "encoder": data["feature_encoder"]["id"] = "other"
    elif change == "hash": data["summary_sha256"] = "0"*64
    elif change == "schedule": data.pop(0)
    elif change == "policy_path": data[-1]["path"] = "../dangerous.pkl"
    elif change == "size": data[-1]["training_state_size_bytes"] += 1
    (worker/filename).write_text(json.dumps(data))
    with pytest.raises(ValueError):
        c.cloud_preflight("gs://test-bucket", cloud=cloud_reader({c.SOURCE_RUN_ID: root}, []))


def test_native_transport_preserves_source_and_imported_policies(tmp_path, monkeypatch):
    from experiments.fhp.strategic_feature_suite.continuation import stage_source
    root = source_fixture(tmp_path, monkeypatch)
    source = root/"workers"/c.worker_name(0)
    original = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    destination = tmp_path/"new48"
    result = stage_source(str(source), destination, total_hours=48, seed=0,
        source_commit=c.TRAINING_REF, experiment=learner.EXPERIMENT)
    assert result["source_total_hours"] == 24 and result["total_hours"] == 48
    assert (destination/"continuation_inputs/source_state.pt").exists()
    assert not (destination/"training_states").exists() and not (destination/"SUCCESS.json").exists()
    rows = json.loads((destination/"checkpoint_manifest.json").read_text())
    assert len(rows) == 4 and not any("training_state_path" in r for r in rows)
    assert original == {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    assert stage_source(str(source), destination, total_hours=48, seed=0,
        source_commit=c.TRAINING_REF, experiment=learner.EXPERIMENT) == result
    source_rows = json.loads((source/"checkpoint_manifest.json").read_text())
    (source/source_rows[-1]["training_state_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="full-training-state checksum"):
        stage_source(str(source), tmp_path/"bad48", total_hours=48, seed=0,
            source_commit=c.TRAINING_REF, experiment=learner.EXPERIMENT)
    assert not (tmp_path/"bad48").exists()


def output_fixture(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, monkeypatch)
    cloud_roots = {c.SOURCE_RUN_ID: source}
    receipt = c.cloud_preflight("gs://test-bucket", cloud=cloud_reader(cloud_roots, []))
    root = tmp_path/"exp38"
    root.mkdir()
    (root/"source_contract.json").write_text(json.dumps(receipt))
    for old in receipt["source_states"]:
        seed = old["seed"]
        worker = root/"workers"/c.worker_name(seed)
        shutil.copytree(source/"workers"/c.worker_name(seed), worker, ignore=shutil.ignore_patterns("training_states"))
        update_json(worker/"run_manifest.json", dict(training_duration_seconds=48*3600,
            checkpoint_schedule=list(learner.checkpoint_schedule(total_hours=48))))
        update_json(worker/"summary.json", dict(final_nodes_touched=4800, final_iteration=48,
            final_training_elapsed_seconds=48*3600+1, checkpoint_count=8, resumed_from_training_state=True))
        update_json(worker/"SUCCESS.json", dict(summary_sha256=hashlib.sha256((worker/"summary.json").read_bytes()).hexdigest()))
        rows = json.loads((worker/"checkpoint_manifest.json").read_text())
        for row in rows:
            for k in list(row):
                if k.startswith("training_state_"): row.pop(k)
        for hour in c.NEW_HOURS:
            path = worker/"checkpoints"/f"time_{hour:02d}h.pkl"
            path.write_bytes(f"future-policy-{hour}".encode())
            rows.append(dict(checkpoint_id=f"time_{hour:02d}h", path=f"checkpoints/{path.name}",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checkpoint_target_seconds=hour*3600))
        state = worker/"training_states/time_48h.pt"
        state.parent.mkdir()
        state.write_bytes(b"new-training-state-transport-fixture")
        rows[-1].update(training_state_path="training_states/time_48h.pt",
            training_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(), training_state_size_bytes=state.stat().st_size)
        (worker/"checkpoint_manifest.json").write_text(json.dumps(rows))
        (worker/"continuation_source.json").write_text(json.dumps(dict(seed=seed,
            source_worker=old["source_worker"], source_state_sha256=old["state_sha256"],
            source_commit=c.TRAINING_REF, source_summary_sha256=old["metadata_sha256"]["summary.json"],
            source_total_hours=24, total_hours=48)))
    (root/"analysis").mkdir()
    for name in ("experiment_manifest.json", "throughput_summary.json"):
        (root/"analysis"/name).write_text(json.dumps(dict(experiment_id=36)))
    (root/"analysis/SUCCESS.json").write_text('{"status":"complete"}')
    cloud_roots["exp38-test"] = root
    return root, cloud_reader(cloud_roots, [])


def test_reporting_full_state_inventory_and_lineage(tmp_path, monkeypatch):
    root, cloud = output_fixture(tmp_path, monkeypatch)
    for seed in c.SEEDS:
        worker = root/"workers"/c.worker_name(seed)
        c.validate_metadata({k: (worker/k).read_bytes() for k in c.METADATA_FILES}, seed=seed, hours=48)
    states = report.finish(root, "a"*40, "gs://test-bucket/exp38-test", cloud=cloud)
    assert len(states) == 3 and all(r["reload_for_training"] for r in states)
    assert (root/"analysis/learner_experiment_manifest.json").exists()
    assert json.loads((root/"analysis/experiment_manifest.json").read_text())["experiment_id"] == 38
    assert json.loads((root/"analysis/SUCCESS.json").read_text())["full_restart_states"] == 3
    assert "No evaluation" in (root/"analysis/analysis_summary.md").read_text()
    assert report.finish(root, "a"*40, "gs://test-bucket/exp38-test", cloud=cloud) == states
    bad = root/"workers"/c.worker_name(1)/"training_states/time_48h.pt"
    bad.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="incomplete uploaded"):
        report.finish(root, "a"*40, "gs://test-bucket/exp38-test", cloud=cloud)
    assert not (root/"analysis/SUCCESS.json").exists()


def job_args(kind):
    return SimpleNamespace(kind=kind, run_id="exp38-test", repo_ref="a"*40, bucket_root="gs://test-bucket",
        service_account="runner@test-project.iam.gserviceaccount.com", project_id="test-project",
        region="europe-west1", parallelism=3, repo_url=REPO_URL, max_hours=48, resume=False)


@pytest.mark.parametrize("kind", ("controller", "smoke", "train", "aggregate"))
def test_batch_uses_original_learner_and_new_workflow(kind):
    job = build_job(job_args(kind))
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    text = spec["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert spec["maxRetryCount"] == 0
    assert job["labels"]["experiment"].startswith("exp38-")
    assert "exp38-test" in text and "exp36-test" not in text
    if kind == "controller":
        assert "export REPO_REF="+"a"*40 in text
        assert "EXP38_REMOTE_CONTROLLER=1" in text and "EXP38_PARALLELISM=3" in text
        assert "gcp/run_exp38_strength_critic2500_48h.sh" in text
    else:
        assert "REPO_REF="+c.TRAINING_REF in text and "uv python install 3.11.16" in text
        assert "exp36_fhp_strength_critic2500.run" in text
    if kind == "train":
        assert "stage-source" in text and "--total-hours" in text
        assert spec["environment"]["variables"]["EXP36_SOURCE_RUN_ID"] == c.SOURCE_RUN_ID
        assert spec["environment"]["variables"]["EXP36_TOTAL_HOURS"] == "48"
        assert spec["maxRunDuration"] == "259200s"
        assert "continuation_inputs" in text
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    if kind == "smoke":
        assert "final_restart_validation.json" in text and "capacity-preflight" in text
    if kind == "aggregate":
        assert "restart_inventory.json" in text and "experiment38_contract.json" in text
        assert "objects" in text and "full_restart_states=3" in text


def test_stdlib_cli_launcher_and_parallel_waves(tmp_path):
    args = job_args("controller")
    args.parallelism = 1
    assert build_job(args)["taskGroups"][0]["taskSpec"]["maxRunDuration"] == str((3*72+12)*3600)+"s"
    command = [sys.executable, "-S", str(ROOT/"gcp/exp38_strength_critic2500_48h_batch.py"), "--output", str(tmp_path/"job.json")]
    for key, value in vars(job_args("train")).items():
        if key not in ("max_hours", "resume"):
            command += ["--"+key.replace("_", "-"), str(value)]
    subprocess.run(command, check=True)
    launcher = ROOT/"gcp/run_exp38_strength_critic2500_48h.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert "for stage in smoke train aggregate; do ensure_stage" in text
    assert "for stage in smoke train aggregate evaluate" not in text
    assert 'archive "$REPO_REF"' in text and "no_active_jobs" in text and "new_namespace" in text
    args.run_id = c.SOURCE_RUN_ID
    with pytest.raises(ValueError, match="exp38-"):
        build_job(args)


def test_archived_workflow_needs_no_learning_dependencies(tmp_path):
    # The launcher archives gcp/ and this package only. Exercise that layout
    # with site-packages disabled, not the complete local checkout.
    isolated = tmp_path/"pinned"
    shutil.copytree(ROOT/"gcp", isolated/"gcp", ignore=shutil.ignore_patterns("__pycache__"))
    package = "experiments/fhp/exp38_fhp_strength_critic2500_48h"
    shutil.copytree(ROOT/package, isolated/package, ignore=shutil.ignore_patterns("__pycache__"))
    command = [sys.executable, "-S", str(isolated/"gcp/exp38_strength_critic2500_48h_batch.py"),
               "--output", str(tmp_path/"isolated-job.json")]
    for key, value in vars(job_args("aggregate")).items():
        if key not in ("max_hours", "resume"):
            command += ["--"+key.replace("_", "-"), str(value)]
    subprocess.run(command, cwd=isolated, check=True)
    script = json.loads((tmp_path/"isolated-job.json").read_text())["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "restart_inventory.json" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


@pytest.mark.ray
def test_real_native_24_to_48_to_72_restart(tmp_path):
    if os.environ.get("RUN_EXP38_RAY_TESTS") != "1":
        pytest.skip("Set RUN_EXP38_RAY_TESTS=1 for actual eight-actor continuation smoke")
    from experiments.fhp.exp36_fhp_strength_critic2500.run import main
    from experiments.fhp.exp38_fhp_strength_critic2500_48h.smoke import verify_final_restart
    main(["smoke", "--output-root", str(tmp_path), "--no-resume"])
    assert json.loads((tmp_path/"restart_validation/restart_validation.json").read_text())["learning_state_bitwise_identical"]
    result = verify_final_restart(tmp_path)
    assert result["status"] == "passed" and result["new_full_states"] == 1
