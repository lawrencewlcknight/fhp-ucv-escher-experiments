"""48-hour contract, cloud wiring, and completed-endpoint extension checks."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16 import config as exp7
from experiments.fhp.exp8_fhp_parallel_48h import config as exp8
from experiments.fhp.exp8_fhp_parallel_48h.aggregate import task_name
from experiments.fhp.exp8_fhp_parallel_48h.run import main
from experiments.fhp.exp8_fhp_parallel_48h.continuation import stage_source
from experiments.fhp.exp8_fhp_parallel_48h.training_state import STATE_TYPE, read_training_state
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _config_sha256
from fhp_escher.checkpointing import sha256_file
from gcp import exp8_parallel_48h_batch as builder
from gcp import exp7_parallel_structured_n2_standard16_batch as previous_builder


def test_same_learning_configuration_longer_horizon():
    assert exp8.EXPERIMENT_CONFIG == exp7.EXPERIMENT_CONFIG
    assert exp8.EXPERIMENT_CONFIG is not exp7.EXPERIMENT_CONFIG
    assert exp8.parallel_settings() == exp7.parallel_settings()
    assert exp8.REFERENCE_VM == exp7.REFERENCE_VM
    assert exp8.PRODUCTION_SEEDS == (0, 1, 2)
    assert exp8.CHECKPOINT_HOURS == (6, 12, 18, 24, 30, 36, 42, 48)
    assert exp8.checkpoint_schedule()[-1]["target_training_seconds"] == 48 * 3600
    assert len(exp8.checkpoint_schedule(smoke=True)) == 8
    assert len(exp8.checkpoint_schedule(total_hours=72)) == 12
    assert exp8.contract_manifest()["training_hours_per_seed"] == 48
    exp8.validate_contract(seeds=(0, 1, 2), config=exp8.EXPERIMENT_CONFIG,
                          schedule=exp8.checkpoint_schedule(), smoke=False)
    with pytest.raises(ValueError, match="learning configuration"):
        exp8.validate_contract(seeds=(0, 1, 2), config=dict(exp8.EXPERIMENT_CONFIG, num_traversals=80_000),
                              schedule=exp8.checkpoint_schedule(), smoke=False)
    for bad in (24, 49, 48.5, True):
        with pytest.raises(ValueError, match="multiple"):
            exp8.checkpoint_schedule(total_hours=bad)


def job_args(kind, **kwargs):
    return SimpleNamespace(kind=kind, repo_url=builder.REPO_URL, repo_ref="abc123",
                           bucket_root="gs://test-bucket", run_id="exp8-par48-test",
                           service_account="runner@example.com", parallelism=3,
                           project_id="test-project", region="europe-west1",
                           controller_action="orchestrate", **kwargs)


@pytest.mark.parametrize("kind", ["controller", "smoke", "train", "aggregate"])
@pytest.mark.parametrize("extension", [False, True])
def test_cloud_jobs(kind, extension, tmp_path):
    args = job_args(kind, total_hours=72 if extension else 48,
                    source_run_id="exp8-original" if extension else "")
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert group["taskCountPerNode"] == 1
    env = spec["environment"]["variables"]
    assert env["EXP8_TOTAL_HOURS"] == ("72" if extension else "48")
    assert env["EXP8_SOURCE_RUN_ID"] == ("exp8-original" if extension else "")
    if kind in ("smoke", "train"):
        assert spec["computeResource"] == {"cpuMilli": 16000, "memoryMib": 62000}
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    if kind == "train":
        assert spec["maxRunDuration"] == "259200s"
        assert spec["maxRetryCount"] == 0
    script = spec["runnables"][0]["script"]["text"]
    assert "exp7_fhp_parallel" not in script and "EXP7_" not in script
    if kind == "train":
        assert " stage-source " in script
        assert '--total-hours "$EXP8_TOTAL_HOURS"' in script
    if kind == "smoke":
        assert "capacity-preflight" in script and "checkpoint-smoke" not in script
    path = tmp_path / (kind + ".sh")
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)
    old = previous_builder.build_job(job_args("train"))
    assert old["taskGroups"][0]["taskSpec"]["maxRunDuration"] == "129600s"


def test_cloud_extension_requires_distinct_source():
    for source, hours in (("", 72), ("exp8-par48-test", 72), ("exp8-source", 48)):
        with pytest.raises(ValueError):
            builder.build_job(job_args("train", total_hours=hours, source_run_id=source))


def fake_source(folder):
    folder.mkdir()
    schedule = exp8.checkpoint_schedule()
    rows = []
    for point in schedule:
        path = folder / "checkpoints" / (point["checkpoint_id"] + ".pkl")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(point["checkpoint_id"].encode())
        rows.append({"checkpoint_id": point["checkpoint_id"],
                     "path": str(path.relative_to(folder)), "sha256": sha256_file(path)})
    state = folder / "training_states" / "final.pt"
    state.parent.mkdir()
    state.write_bytes(b"synthetic-state")
    rows[-1].update(training_state_path="training_states/final.pt",
                    training_state_sha256=sha256_file(state))
    manifest = dict(experiment_id=8, experiment_name=exp8.EXPERIMENT_NAME,
                    algorithm_id=exp8.ALGORITHM_ID, seed=0, smoke=False,
                    execution_backend="ray_parallel", reference_vm=exp8.REFERENCE_VM,
                    training_config_sha256=_config_sha256(exp8.EXPERIMENT_CONFIG),
                    repository_commit="test", checkpoint_schedule=list(schedule))
    summary = dict(seed=0, status="complete", checkpoint_count=8,
                   final_training_elapsed_seconds=48 * 3600 + 12)
    for name, data in (("run_manifest.json", manifest), ("summary.json", summary),
                       ("checkpoint_manifest.json", rows), ("runtime_manifest.json", {})):
        (folder / name).write_text(json.dumps(data))
    (folder / "SUCCESS.json").write_text(json.dumps({
        "status": "complete", "summary_sha256": sha256_file(folder / "summary.json")}))
    return state


def test_import_preserves_source_and_checks_integrity(tmp_path):
    source = tmp_path / "source"
    state = fake_source(source)
    original_hash = sha256_file(state)
    destination = tmp_path / "extended"
    result = stage_source(str(source), destination, seed=0, total_hours=72, source_commit="test")
    assert result["source_total_hours"] == 48
    assert result["total_hours"] == 72
    assert sha256_file(state) == original_hash
    assert sha256_file(destination / "continuation_inputs/source_state.pt") == original_hash
    assert not (destination / "training_states").exists()
    assert not (destination / "SUCCESS.json").exists()
    assert not (destination / "run_manifest.json").exists()
    assert len(list((destination / "checkpoints").iterdir())) == 8
    assert stage_source(str(source), destination, seed=0, total_hours=72, source_commit="test") == result
    with pytest.raises(ValueError, match="another continuation"):
        stage_source(str(source), destination, seed=0, total_hours=78, source_commit="test")
    with pytest.raises(ValueError, match="compatible"):
        stage_source(str(source), tmp_path / "bad-seed", seed=1, total_hours=72, source_commit="test")
    with pytest.raises(ValueError, match="add 6 to 48"):
        stage_source(str(source), tmp_path / "too-long", seed=0, total_hours=102, source_commit="test")
    state.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        stage_source(str(source), tmp_path / "bad-state", seed=0, total_hours=72, source_commit="test")
    assert not (tmp_path / "bad-state").exists()


@pytest.mark.ray
def test_completed_endpoint_can_extend_without_retraining(tmp_path):
    main(["smoke", "--output-root", str(tmp_path), "--no-resume"])
    worker = tmp_path / "workers" / task_name(0, 0)
    extended = tmp_path / "extension_smoke" / "workers" / task_name(0, 0)
    initial = json.loads((worker / "summary.json").read_text())
    final = json.loads((extended / "summary.json").read_text())
    assert initial["checkpoint_count"] == 8 and initial["final_iteration"] == 1
    assert final["checkpoint_count"] == 12 and final["final_iteration"] == 2
    assert final["final_episode"] == 64
    assert final["resumed_from_training_state"]
    initial_rows = json.loads((worker / "checkpoint_manifest.json").read_text())
    final_rows = json.loads((extended / "checkpoint_manifest.json").read_text())
    assert all("training_state_path" not in r for r in initial_rows[:-1])
    assert all("training_state_path" not in r for r in final_rows[:-1])
    assert len(list(worker.rglob("*.pt"))) == 1
    assert len(list(extended.rglob("*.pt"))) == 1
    assert not (extended / "continuation_inputs").exists()
    assert not list((tmp_path / "restart_validation").rglob("*.pt"))
    assert [row["sha256"] for row in initial_rows] == [row["sha256"] for row in final_rows[:8]]
    payload = read_training_state(worker / initial_rows[-1]["training_state_path"])
    assert payload["type"] == STATE_TYPE and len(payload["parallel"]["actors"]) == 8
    assert (tmp_path / "analysis/SUCCESS.json").is_file()
    assert (tmp_path / "extension_smoke/analysis/SUCCESS.json").is_file()
    assert len(json.loads((tmp_path / "analysis/throughput_summary.json").read_text())["checkpoint_means"]) == 8
