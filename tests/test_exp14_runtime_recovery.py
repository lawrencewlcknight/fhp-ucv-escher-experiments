"""Offline regression tests: runtime drift must fail before replay downloads."""

from argparse import Namespace
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp14_runtime_gate", REPO / "gcp/exp14_runtime_preflight.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
AUDIT = "54a3269f62189b8ac7190e59c9a3ea70efb4969c"
LAUNCHER = "a" * 40
RUN = "exp14-recovery-test"
SOURCE = "exp2-source"
PYTHON = "3.11.16 (main, Sep 29 2026, 14:59:26) [Clang 22.1.3 ]"


@pytest.fixture
def archive():
    config = {"experiment_id": 14, "experiment_name": gate.EXPERIMENT, "smoke": False,
              "gamma": 2.0, "seeds": [0, 1, 2], "nullable": None}
    identity = {"config": config, "audit_commit": AUDIT, "audit_source_sha256": "f" * 64,
                "python": PYTHON, "torch": "2.7.0+cpu", "numpy": "1.26.4"}
    objects, manifests = {}, {}
    prefix = f"gs://bucket/{RUN}"
    for seed in range(3):
        row = {"checkpoint_id": "time_24h", "checkpoint_target_seconds": 86400,
               "sha256": f"policy{seed}", "training_state_sha256": f"state{seed}",
               "outer_iteration": 123, "nodes_touched": 456}
        provenance = {"seed": seed, "checkpoint_id": "time_24h", "source_commit": "b" * 40,
                      "source_policy_sha256": row["sha256"], "source_state_sha256": row["training_state_sha256"],
                      "outer_iteration": 123, "nodes_touched": 456}
        manifest = {**deepcopy(identity), "provenance": provenance}
        manifests[seed] = manifest
        worker = f"{prefix}/workers/seed_{seed}"
        objects[worker + "/manifest.json"] = manifest
        objects[worker + "/SCREEN_SUCCESS.json"] = {"seed": seed, "smoke": False, "status": "screen_complete"}
        source = f"gs://bucket/{SOURCE}/workers/task_{seed:03d}_lossless_structured_ucv_escher_seed_{seed}"
        objects[source + "/run_manifest.json"] = {"seed": seed, "repository_commit": "b" * 40,
                                                  "experiment_name": "exp2_fhp_lossless_structured_ucv"}
        objects[source + "/checkpoint_manifest.json"] = [row]
    selection = {"config": deepcopy(config), "source_manifest_hashes": {
        str(seed): gate.object_hash(manifest) for seed, manifest in manifests.items()},
        "selection_uses_test_or_gameplay": False, "selected": {"dense": {"updates": 5000}}}
    selection["selection_sha256"] = gate.object_hash(selection)
    objects[prefix + "/selection.json"] = selection
    args = Namespace(bucket_root="gs://bucket", run_id=RUN, source_run_id=SOURCE, audit_ref=AUDIT,
                     python_version="3.11.16", stage="recover", seed=None, repository=None)
    return args, identity, objects


def fetcher(objects, calls=None):
    def fetch(uri, *, optional=False):
        if calls is not None:
            calls.append(uri)
        if uri not in objects and optional:
            return None
        return deepcopy(objects[uri])
    return fetch


def test_read_only_recovery_preserves_manifests_and_selection(archive):
    args, identity, objects = archive
    before = json.dumps(objects, sort_keys=True)
    result = gate.check(args, fetch=fetcher(objects))
    assert result["checked_seeds"] == [0, 1, 2]
    assert gate.check(args, fetch=fetcher(objects), observed=identity)["status"] == "compatible"
    assert json.dumps(objects, sort_keys=True) == before


@pytest.mark.parametrize("field,value,match", [
    ("python", PYTHON.replace("3.11.16", "3.11.17"), "Python patch"),
    ("python", PYTHON.replace("14:59:26", "15:00:00"), '"python"'),
    ("torch", "2.8.0+cpu", '"torch"'),
    ("numpy", "2.0.0", '"numpy"'),
    ("audit_commit", "c" * 40, "audit checkout"),
    ("audit_source_sha256", "c" * 64, '"audit_source_sha256"'),
    ("config", {"changed": True}, '"config'),
])
def test_identity_mismatch_fails_before_source_download(archive, field, value, match):
    args, identity, objects = archive
    identity[field] = value
    calls = []
    with pytest.raises(ValueError, match=match):
        gate.check(args, fetch=fetcher(objects, calls), observed=identity)
    assert not any(f"/{SOURCE}/" in uri for uri in calls)
    assert not any(uri.endswith((".pt", ".pkl")) for uri in calls)


def test_missing_nullable_config_field_is_not_silently_accepted(archive):
    args, identity, objects = archive
    identity["config"].pop("nullable")
    with pytest.raises(ValueError, match="missing_from"):
        gate.check(args, fetch=fetcher(objects), observed=identity)


@pytest.mark.parametrize("change", ["selection", "manifest", "source", "screen", "missing", "mixed_runtime"])
def test_recovery_rejects_incomplete_or_changed_inputs(archive, change):
    args, _, objects = archive
    manifest = objects[f"gs://bucket/{RUN}/workers/seed_1/manifest.json"]
    if change == "selection":
        objects[f"gs://bucket/{RUN}/selection.json"]["selected"]["dense"]["updates"] = 999
    elif change == "manifest":
        manifest["extra_field"] = "unlocked change"
    elif change == "source":
        source = f"gs://bucket/{SOURCE}/workers/task_001_lossless_structured_ucv_escher_seed_1"
        objects[source + "/checkpoint_manifest.json"][0]["training_state_sha256"] = "changed"
    elif change == "screen":
        objects[f"gs://bucket/{RUN}/workers/seed_1/SCREEN_SUCCESS.json"]["status"] = "incomplete"
    elif change == "missing":
        del objects[f"gs://bucket/{RUN}/workers/seed_1/manifest.json"]
    else:
        manifest["python"] = PYTHON.replace("14:59:26", "15:00:00")
    with pytest.raises((ValueError, KeyError)):
        gate.check(args, fetch=fetcher(objects))


def test_optional_missing_object_does_not_hide_permission_or_network_errors(monkeypatch):
    def response(error):
        monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: Namespace(returncode=1, stderr=error))
    response("ERROR: One or more URLs matched no objects.")
    assert gate.fetch_json("gs://bucket/missing", optional=True) is None
    for error in ("PERMISSION_DENIED", "DNS resolution failed", "invalid credentials"):
        response(error)
        with pytest.raises(RuntimeError, match=error):
            gate.fetch_json("gs://bucket/missing", optional=True)


def test_cli_writes_actionable_failure_json(tmp_path):
    failure = tmp_path / "diagnostics/failure.json"
    result = subprocess.run([sys.executable, str(REPO / "gcp/exp14_runtime_preflight.py"),
        "--bucket-root", "gs://unused", "--run-id", RUN, "--source-run-id", SOURCE,
        "--audit-ref", AUDIT, "--python-version", "3.11", "--stage", "train",
        "--failure-file", str(failure)], text=True, capture_output=True)
    assert result.returncode == 2
    detail = json.loads(failure.read_text())
    assert detail["stage"] == "runtime_preflight" and "patch version" in detail["error"]


def build(tmp_path, kind, recovery):
    path = tmp_path / f"{kind}.json"
    command = [sys.executable, str(REPO / "gcp/exp14_card_architecture_batch.py"),
        "--kind", kind, "--output", str(path), "--run-id", RUN, "--bucket-root", "gs://bucket",
        "--service-account", "runner@test", "--repo-ref", LAUNCHER,
        "--source-run-id", SOURCE, "--project-id", "project", "--region", "europe-west1"]
    if recovery:
        command += ["--audit-ref", AUDIT, "--controller-action", "orchestrate-recover", "--resume-tag", "abc123"]
    subprocess.run(command, check=True)
    job = json.loads(path.read_text())
    return job, job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]


@pytest.mark.parametrize("kind", ["controller", "smoke", "screen", "select", "train", "aggregate"])
@pytest.mark.parametrize("recovery", [False, True])
def test_generated_jobs_pin_runtime_and_preserve_audit_checkout(tmp_path, kind, recovery):
    job, script = build(tmp_path, kind, recovery)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    if kind == "controller":
        assert f"REPO_REF={LAUNCHER}" in script
        assert f"export EXP14_AUDIT_REF={AUDIT if recovery else LAUNCHER}" in script
        assert "export EXP14_PYTHON_VERSION=3.11.16" in script
        if recovery:
            assert "CONTROLLER_ACTION=orchestrate-recover" in script
            assert "export RESUME_TAG=abc123" in script
    else:
        assert f"REPO_REF={AUDIT if recovery else LAUNCHER}" in script
        assert "uv python install 3.11.16\n" in script
        assert "uv venv --python 3.11.16 --seed" in script
        assert "uv python install 3.11\n" not in script
        assert '--failure-file "$DIAGNOSTIC_DIR/failure.json"' in script
        assert script.index("trap cleanup EXIT") < script.index("python - --bucket-root")
        if kind in ("smoke", "screen", "train"):
            assert script.index("EXP14_RUNTIME_GATE\n") < script.index(" fetch-source --bucket")
        assert job["taskGroups"][0]["taskSpec"]["maxRetryCount"] == 0


def test_recovery_controller_runs_only_deployment_and_aggregation(archive, tmp_path):
    _, _, objects = archive
    fixture = tmp_path / "objects.json"
    fixture.write_text(json.dumps(objects))
    calls = tmp_path / "calls.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gcloud"
    fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
a = sys.argv[1:]
root = pathlib.Path(os.environ["FHP_TEST_OBJECTS"]).parent
with open(os.environ["FHP_TEST_CALLS"], "a") as out:
    out.write(json.dumps(a) + "\\n")
if a[:2] == ["storage", "cat"]:
    print(json.dumps(json.loads(pathlib.Path(os.environ["FHP_TEST_OBJECTS"]).read_text())[a[2]]))
elif a[:3] == ["batch", "jobs", "list"]:
    active = os.environ.get("FHP_TEST_ACTIVE_JOB")
    if active:
        print(active + ",RUNNING")
elif a[:3] == ["batch", "jobs", "describe"]:
    if not (root / a[3]).exists():
        sys.exit(1)
    print("SUCCEEDED")
elif a[:3] == ["batch", "jobs", "submit"]:
    (root / a[3]).touch()
else:
    raise RuntimeError(a)
''')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "PROJECT_ID": "project", "REGION": "europe-west1", "BUCKET": "gs://bucket",
           "SA_EMAIL": "runner@test", "REPO_REF": LAUNCHER, "EXP14_AUDIT_REF": AUDIT,
           "EXP14_PYTHON_VERSION": "3.11.16", "EXP2_RUN_ID": SOURCE, "RUN_ID": RUN,
           "EXP14_REMOTE_CONTROLLER": "1", "RESUME_TAG": "abc123",
           "FHP_TEST_OBJECTS": str(fixture), "FHP_TEST_CALLS": str(calls),
           "FHP_TEST_ACTIVE_JOB": RUN + "-controller-recover-abc123"}
    command = ["bash", str(REPO / "gcp/run_exp14_card_architecture.sh"), "orchestrate-recover"]
    subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    submitted = [a[3] for a in map(json.loads, calls.read_text().splitlines())
                 if a[:3] == ["batch", "jobs", "submit"]]
    assert submitted == [RUN + "-runtime-retry-abc123", RUN + "-reaggregate-abc123"]
    assert json.loads(fixture.read_text()) == objects
    calls.write_text("")
    env["FHP_TEST_ACTIVE_JOB"] = RUN + "-train"
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 2 and "another job is active" in result.stderr
    assert not any(a[:3] == ["batch", "jobs", "submit"]
                   for a in map(json.loads, calls.read_text().splitlines()))
