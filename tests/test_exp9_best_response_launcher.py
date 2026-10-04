"""The root launcher delegates unchanged settings and never auto-submits a pilot."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "gcp/run_exp9_best_response_pilot.sh"


@pytest.fixture
def launch_env(tmp_path):
    evaluation = tmp_path / "evaluation checkout with spaces"
    for name in ("gcp/run_exp4_ucv_exp9_br_pilot.sh", "gcp/finalize_exp4_ucv_exp9_br_pilot.sh",
                 "gcp/requirements-br-pilot.txt", "fhp_evaluation/best_response/pilot.py", "pyproject.toml"):
        path = evaluation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    builder = evaluation / "gcp/exp4_ucv_exp9_br_pilot_batch.py"
    builder.write_text("import json, os, sys\nfrom pathlib import Path\n"
                       "Path(os.environ['FHP_LAUNCH_TEST_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
                       "sys.exit(int(os.environ.get('FHP_LAUNCH_TEST_EXIT', '0')))\n")
    environment = dict(os.environ, PROJECT_ID="test-project", REGION="europe-west1",
                       BUCKET="gs://test-bucket", SA_EMAIL="batch@test-project.iam.gserviceaccount.com",
                       RUN_ID="fhp-br-exp9-root-test", FHP_EVAL_REPO=str(evaluation),
                       PYTHON=sys.executable, FHP_LAUNCH_TEST_LOG=str(tmp_path / "arguments.json"))
    environment.pop("FHP_BATCH_OUTPUT_DIR", None)
    environment.pop("FHP_LAUNCH_TEST_EXIT", None)
    return environment


@pytest.mark.parametrize("action,mode,smoke,suffix", [
    ("run", "submit", False, ""), ("smoke", "submit", True, ""),
    ("dry-run", "dry-run", False, "-preview"),
    ("dry-run-smoke", "dry-run", True, "-smoke-preview"),
])
def test_dispatches_to_shared_builder_with_ucv_paths(launch_env, tmp_path, action, mode, smoke, suffix):
    result = subprocess.run(["bash", str(LAUNCHER), action], cwd=tmp_path,
                            env=launch_env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    arguments = json.loads(Path(launch_env["FHP_LAUNCH_TEST_LOG"]).read_text())
    assert arguments[0] == mode
    assert ("--smoke" in arguments) == smoke
    assert arguments[arguments.index("--native-repo") + 1] == str(ROOT)
    assert arguments[arguments.index("--output-dir") + 1] == str(
        ROOT / "outputs/batch" / (launch_env["RUN_ID"] + suffix))
    assert arguments.count("--native-repo") == 1


def test_custom_output_and_exit_code_propagation(launch_env, tmp_path):
    launch_env.update(FHP_BATCH_OUTPUT_DIR=str(tmp_path / "custom output"), FHP_LAUNCH_TEST_EXIT="7")
    result = subprocess.run(["bash", str(LAUNCHER), "dry-run"], env=launch_env, capture_output=True)
    assert result.returncode == 7
    arguments = json.loads(Path(launch_env["FHP_LAUNCH_TEST_LOG"]).read_text())
    assert arguments[arguments.index("--output-dir") + 1] == launch_env["FHP_BATCH_OUTPUT_DIR"]


def test_missing_shared_checkout_fails_before_delegation(launch_env, tmp_path):
    launch_env["FHP_EVAL_REPO"] = str(tmp_path / "missing")
    result = subprocess.run(["bash", str(LAUNCHER), "run"], env=launch_env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Set FHP_EVAL_REPO" in result.stderr
    assert not Path(launch_env["FHP_LAUNCH_TEST_LOG"]).exists()


@pytest.mark.parametrize("run_id", ["exp9-cache24-20261001-132550", "fhp-br-exp9-bad;command", "fhp-br-exp9-bad-"])
def test_invalid_run_ids_fail_before_delegation(launch_env, run_id):
    launch_env["RUN_ID"] = run_id
    result = subprocess.run(["bash", str(LAUNCHER), "smoke"], env=launch_env, capture_output=True)
    assert result.returncode == 2
    assert not Path(launch_env["FHP_LAUNCH_TEST_LOG"]).exists()


def test_status_does_not_require_shared_checkout_or_storage_settings(launch_env, tmp_path):
    fake_gcloud = tmp_path / "gcloud"
    fake_gcloud.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
                          "Path(os.environ['FHP_LAUNCH_TEST_LOG']).write_text(json.dumps(sys.argv[1:]))\n")
    fake_gcloud.chmod(0o755)
    launch_env.update(PATH=str(tmp_path) + os.pathsep + launch_env["PATH"], FHP_EVAL_REPO="/missing")
    del launch_env["BUCKET"], launch_env["SA_EMAIL"]
    subprocess.run(["bash", str(LAUNCHER), "status"], env=launch_env, check=True)
    arguments = json.loads(Path(launch_env["FHP_LAUNCH_TEST_LOG"]).read_text())
    assert arguments[:4] == ["batch", "jobs", "describe", launch_env["RUN_ID"]]


def test_help_is_safe_without_environment():
    result = subprocess.run(["bash", str(LAUNCHER)], env={"PATH": os.environ["PATH"]},
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
