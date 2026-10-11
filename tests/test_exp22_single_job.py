"""Execute the generated one-job shell workflow with no cloud/ML side effects."""
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gcp.exp22_critic_objectives_batch import actions


FAKE_TOOL = r"""
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
tool = Path(sys.argv[0]).name
if tool == "timeout":
    os.execvp(args[1], args[1:])
if tool == "gcloud":
    assert args[:3] == ["storage", "rsync", "--recursive"]
    event = "upload:" + args[-1].rsplit("/", 1)[-1]
else:
    assert args[:2] == ["-m", "experiments.fhp.exp22_fhp_critic_objectives.run"]
    action = args[2]
    event = action + ":" + (args[args.index("--seed")+1] if "--seed" in args
                            else "pilot" if "--pilot" in args else "all")
with open(os.environ["TEST_CALL_LOG"], "a") as handle:
    handle.write(json.dumps(dict(event=event, args=args,
                                remote=os.environ.get("EXP22_REMOTE_WORKER"))) + "\n")
if event == os.environ.get("TEST_FAIL"):
    sys.exit(7)
if tool != "gcloud":
    if action == "fetch-source":
        Path(args[args.index("--directory")+1]).mkdir(parents=True)
    elif action == "worker":
        Path(args[args.index("--output")+1]).mkdir(parents=True)
    else:
        destination = Path(args[args.index("--root")+1]) / "analysis"
        destination.mkdir()
        (destination / "report.md").write_text(event)
"""

EXPECTED = [
    "fetch-source:0", "worker:0", "aggregate:pilot", "upload:pilot_analysis",
    "fetch-source:1", "worker:1", "fetch-source:2", "worker:2",
    "aggregate:all", "upload:analysis",
]


@pytest.mark.parametrize("failure", [None, "fetch-source:0", "worker:0", "aggregate:pilot",
                                   "upload:pilot_analysis", "worker:1", "aggregate:all",
                                   "upload:analysis"])
def test_single_job_order_and_fail_closed(tmp_path, failure):
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("python", "timeout", "gcloud"):
        path = tools / name
        path.write_text(f"#!{sys.executable}\n" + FAKE_TOOL)
        path.chmod(0o755)
    work = tmp_path / "workspace with spaces"
    output = work / "output"
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ, PATH=str(tools)+os.pathsep+os.environ["PATH"],
               WORK_ROOT=str(work), OUTPUT_ROOT=str(output),
               BUCKET_ROOT="gs://test-bucket", RUN_ID="exp22-critic-test",
               TEST_CALL_LOG=str(log), TEST_FAIL=failure or "")
    script = "set -Eeuo pipefail\n" + actions(Namespace(kind="run"))
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    events = [r["event"] for r in rows]
    if failure:
        assert result.returncode == 7, result.stderr
        assert events == EXPECTED[:EXPECTED.index(failure)+1]
    else:
        assert result.returncode == 0, result.stderr
        assert events == EXPECTED
        assert (output / "pilot_analysis/report.md").read_text() == "aggregate:pilot"
        assert (output / "analysis/report.md").read_text() == "aggregate:all"
        for seed in range(3):
            assert (work / f"source/seed_{seed}").is_dir()
            assert (output / f"diagnostics/task_0/seed_{seed}/source_fetch.json").is_file()
    for row in rows:
        if row["event"].startswith("worker:"):
            assert row["remote"].endswith("/workers/seed_" + row["event"][-1])
        if row["event"].startswith("aggregate:"):
            assert row["remote"] is None


def test_single_job_preserves_worker_limits_and_does_not_submit_child_jobs():
    commands = actions(Namespace(kind="run"))
    assert "timeout 21600 python" in commands
    assert commands.count("timeout 1800 python") == 3  # fetch, pilot aggregation, final aggregation
    assert "for SEED in 0 1 2" in commands
    assert "smoke" not in commands
    assert "batch jobs" not in commands
