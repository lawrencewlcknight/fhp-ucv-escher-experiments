"""The UCV entry point is explicit, offline by default and repository-root independent."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "gcp/run_exp21_adversarial_preservation.sh"


@pytest.mark.parametrize("action,stage,smoke", [
    ("prepare", "prepare", False), ("prepare-smoke", "prepare", True),
    ("qualify", "qualify", False), ("train-smoke", "train", True),
    ("evaluate", "evaluate", False), ("aggregate-smoke", "aggregate", True),
])
def test_dispatch(tmp_path, action, stage, smoke):
    suite = tmp_path / "evaluation suite"
    (suite / "gcp").mkdir(parents=True)
    log = tmp_path / "arguments.json"
    (suite / "gcp/exp21_adversarial_preservation_batch.py").write_text(
        "import sys,json,os\nfrom pathlib import Path\n"
        "Path(os.environ['TEST_ARGS']).write_text(json.dumps(sys.argv[1:]))\n")
    env = dict(os.environ, PROJECT_ID="test-project", REGION="europe-west1", BUCKET="gs://test-bucket",
               SA_EMAIL="batch@test-project.iam.gserviceaccount.com", RUN_ID="exp21-adv-test",
               FHP_EVAL_REPO=str(suite), PYTHON=sys.executable, TEST_ARGS=str(log))
    subprocess.run(["bash", str(SCRIPT), action], cwd=tmp_path, env=env, check=True)
    args = json.loads(log.read_text())
    assert args[0] == stage and ("--smoke" in args) == smoke
    assert args[args.index("--native-repo") + 1] == str(ROOT)


def test_help_and_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(["bash", str(SCRIPT)], env={"PATH": os.environ["PATH"]},
                            capture_output=True, text=True, check=True)
    assert "offline" in result.stdout and "paid" in result.stdout


def test_refuses_invalid_run_id():
    env = dict(os.environ, PROJECT_ID="test-project", REGION="europe-west1", RUN_ID="exp10-original")
    result = subprocess.run(["bash", str(SCRIPT), "qualify"], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "Invalid RUN_ID" in result.stderr
