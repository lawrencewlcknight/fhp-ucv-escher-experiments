"""UCV-root delegation to the immutable, shared three-seed response experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "gcp/run_exp9_best_response_production.sh"


@pytest.fixture
def environment(tmp_path):
    evaluation = tmp_path / "evaluation checkout with spaces"
    for name in ("gcp/run_exp5_ucv_exp9_br_production.sh", "gcp/finalize_exp5_ucv_exp9_br_production.sh",
                 "gcp/aggregate_exp5_ucv_exp9_br_production.sh", "gcp/requirements-br-pilot.txt",
                 "fhp_evaluation/best_response/production.py", "pyproject.toml"):
        path = evaluation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    (evaluation / "gcp/exp5_ucv_exp9_br_production_batch.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['LAUNCH_TEST_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ.get('LAUNCH_TEST_EXIT', '0')))\n")
    output = tmp_path / "prepared output"
    output.mkdir()
    (output / "request.json").write_text("{}")
    return dict(os.environ, PROJECT_ID="test-project", REGION="europe-west1",
                BUCKET="gs://test-bucket", SA_EMAIL="batch@test-project.iam.gserviceaccount.com",
                RUN_ID="fhp-br-exp9-prod-root-test", FHP_EVAL_REPO=str(evaluation),
                FHP_BATCH_OUTPUT_DIR=str(output), PYTHON=sys.executable,
                RECOVERY_TAG="r1", LAUNCH_TEST_LOG=str(tmp_path / "arguments.json"))


@pytest.mark.parametrize("action,mode,smoke", [
    ("prepare", "prepare", False), ("prepare-smoke", "prepare", True),
    ("run", "submit-workers", False), ("smoke", "submit-workers", True),
    ("aggregate", "submit-aggregate", False), ("aggregate-smoke", "submit-aggregate", True),
    ("recover", "submit-recovery", False), ("recover-smoke", "submit-recovery", True),
])
def test_explicit_actions(environment, tmp_path, action, mode, smoke):
    result = subprocess.run(["bash", str(LAUNCHER), action], env=environment, cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(environment["LAUNCH_TEST_LOG"]).read_text())
    assert args[0] == mode and ("--smoke" in args) == smoke
    assert args[args.index("--native-repo") + 1] == str(ROOT)
    assert args[args.index("--run-dir") + 1] == environment["FHP_BATCH_OUTPUT_DIR"]
    if action.startswith("recover"):
        assert args[args.index("--recovery-tag") + 1] == "r1"


def test_default_preparation_is_under_ucv_root(environment, tmp_path):
    environment.pop("FHP_BATCH_OUTPUT_DIR")
    subprocess.run(["bash", str(LAUNCHER), "prepare"], env=environment, cwd=tmp_path, check=True)
    args = json.loads(Path(environment["LAUNCH_TEST_LOG"]).read_text())
    assert args[args.index("--run-dir") + 1] == str(ROOT / "outputs/batch" / environment["RUN_ID"])


@pytest.mark.parametrize("problem", ["missing_checkout", "missing_request", "bad_run_id", "missing_recovery_tag"])
def test_fail_before_delegation(environment, tmp_path, problem):
    action = "run"
    if problem == "missing_checkout": environment["FHP_EVAL_REPO"] = str(tmp_path / "absent")
    elif problem == "missing_request": environment["FHP_BATCH_OUTPUT_DIR"] = str(tmp_path / "absent")
    elif problem == "bad_run_id": environment["RUN_ID"] = "exp9-training"
    else:
        environment.pop("RECOVERY_TAG")
        action = "recover"
    result = subprocess.run(["bash", str(LAUNCHER), action], env=environment, capture_output=True)
    assert result.returncode != 0
    assert not Path(environment["LAUNCH_TEST_LOG"]).exists()


def test_exit_code_is_preserved(environment):
    environment["LAUNCH_TEST_EXIT"] = "7"
    result = subprocess.run(["bash", str(LAUNCHER), "run"], env=environment, capture_output=True)
    assert result.returncode == 7


def test_status_is_read_only_and_needs_no_local_request(environment, tmp_path):
    gcloud = tmp_path / "gcloud"
    gcloud.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
                      "Path(os.environ['LAUNCH_TEST_LOG']).write_text(json.dumps(sys.argv[1:]))\n")
    gcloud.chmod(0o755)
    environment.update(PATH=str(tmp_path) + os.pathsep + environment["PATH"], FHP_EVAL_REPO="/absent")
    for key in ("BUCKET", "SA_EMAIL", "FHP_BATCH_OUTPUT_DIR"):
        environment.pop(key)
    subprocess.run(["bash", str(LAUNCHER), "status"], env=environment, check=True)
    args = json.loads(Path(environment["LAUNCH_TEST_LOG"]).read_text())
    assert args[:3] == ["batch", "jobs", "list"]
    assert "--filter=name:fhp-br-exp9-prod-root-test-" in args


def test_help_and_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    result = subprocess.run(["bash", str(LAUNCHER)], env={"PATH": os.environ["PATH"]},
                            capture_output=True, text=True, check=True)
    assert "Usage:" in result.stdout
