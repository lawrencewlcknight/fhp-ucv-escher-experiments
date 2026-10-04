#!/usr/bin/env python3
"""Experiment 16: source-pinned Experiment 10 continuation and frozen evaluation."""
from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.fhp.exp16_fhp_hand_board_48h import contract as c
from gcp import exp10_hand_board_features_batch as training
from gcp import retrospective_exp7_exp8_evaluation_batch as evaluation

REPO_URL = training.REPO_URL
MODULE = "experiments.fhp.exp16_fhp_hand_board_48h.evaluate"


def validate(args):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,33}[a-z0-9]", args.run_id):
        raise ValueError("RUN_ID must be 2–35 lowercase letters, digits or hyphens")
    if args.run_id in (c.SOURCE_RUN_ID, c.EXP8_RUN_ID, c.EXP9_RUN_ID):
        raise ValueError("Use a new run ID: source runs must not be overwritten")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("Pin the workflow REPO_REF to a full commit SHA")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", args.bucket_root):
        raise ValueError("Expected a gs:// bucket without a path")
    if not 1 <= args.parallelism <= 3:
        raise ValueError("Training parallelism must be 1–3")
    if not 1 <= args.max_hours <= 72:
        raise ValueError("Evaluation wall limit must be 1–72 hours")


def _replace(script, old, new):
    if old not in script:
        raise RuntimeError(f"Inherited cloud template changed; missing: {old}")
    return script.replace(old, new)


def evaluation_job(args):
    # Reuse the established smoke-gated evaluator's upload/diagnostic lifecycle.
    inherited = copy(args)
    inherited.exp7_run_id, inherited.exp8_run_id = args.run_id, c.EXP8_RUN_ID
    inherited.smoke_only = False
    # Its original validation forbids an evaluation run sharing its input root;
    # ours uses a separate /evaluation output subtree, never the /workers input.
    script = evaluation._script(inherited)
    script = _replace(script, "fhp-eval78", "fhp-exp16-eval")
    script = _replace(script, "experiments.fhp.retrospective_exp7_exp8_evaluation.run", MODULE)
    script = _replace(script, '"$BUCKET_ROOT/$RUN_ID"', '"$BUCKET_ROOT/$RUN_ID/evaluation"')
    script = _replace(script, "--exp7-run", "--exp16-run")
    extra_sources = f'''
mkdir -p "$INPUT_ROOT/source10" "$INPUT_ROOT/exp9"
gcloud storage rsync --recursive \\
  --exclude='.*training_states.*|.*continuation_inputs.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \\
  "$BUCKET_ROOT/{c.SOURCE_RUN_ID}/workers" "$INPUT_ROOT/source10/workers"
gcloud storage rsync --recursive \\
  --exclude='.*training_states.*|.*continuation_inputs.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \\
  "$BUCKET_ROOT/{c.EXP9_RUN_ID}/workers" "$INPUT_ROOT/exp9/workers"
'''
    script = _replace(script, "# Atomic task-result JSON files", extra_sources + "\n# Atomic task-result JSON files")
    script = _replace(script, '--exp8-run "$INPUT_ROOT/exp8"',
                      '--exp8-run "$INPUT_ROOT/exp8" --exp9-run "$INPUT_ROOT/exp9" --source10-run "$INPUT_ROOT/source10"')
    script = script.replace(".*training_states.*|.*reservoir", ".*training_states.*|.*continuation_inputs.*|.*reservoir")
    # Do not rsync a 16 GB final state into an evaluation-only VM.
    return {
        "taskGroups": [{"taskCount": 1, "parallelism": 1, "taskCountPerNode": 1,
                        "taskSpec": {"runnables": [{"script": {"text": script}}],
                                     "computeResource": {"cpuMilli": 16000, "memoryMib": 62000},
                                     "maxRetryCount": 0, "maxRunDuration": f"{args.max_hours * 3600}s"}}],
        "allocationPolicy": {"serviceAccount": {"email": args.service_account},
                             "instances": [{"policy": {"machineType": "n2-standard-16",
                                                        "provisioningModel": "STANDARD",
                                                        "bootDisk": {"sizeGb": 100, "type": "pd-balanced"}}}]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }


def build_job(args):
    validate(args)
    if args.kind == "evaluate":
        job = evaluation_job(args)
    else:
        inherited = copy(args)
        inherited.total_hours, inherited.source_run_id = c.TOTAL_HOURS, c.SOURCE_RUN_ID
        inherited.repo_ref = args.repo_ref if args.kind == "controller" else c.TRAINING_REF
        inherited.controller_action = "orchestrate"
        job = training.build_job(inherited)
        script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]
        if args.kind == "controller":
            script["text"] = _replace(script["text"], "gcp/run_exp10_hand_board_features.sh",
                                       "gcp/run_exp16_hand_board_48h.sh")
            script["text"] = _replace(script["text"], "export EXP10_REMOTE_CONTROLLER=1",
                                       f"export EXP16_REMOTE_CONTROLLER=1\nexport EVAL_MAX_HOURS={args.max_hours}")
        else:
            script["text"] = _replace(script["text"], "uv python install 3.11\n", "uv python install 3.11.16\n")
            script["text"] = _replace(script["text"], "uv venv --python 3.11 ", "uv venv --python 3.11.16 ")
            check = f'''python - <<'PINNED_RUNTIME'
import platform, subprocess, numpy, torch, ray
assert platform.python_version() == '3.11.16', platform.python_version()
assert (numpy.__version__, torch.__version__, ray.__version__) == ('1.26.4', '2.7.0+cpu', '2.51.2')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == '{c.TRAINING_REF}'
PINNED_RUNTIME
'''
            script["text"] = _replace(script["text"], 'cd "$REPOSITORY"\n', 'cd "$REPOSITORY"\n' + check)
    # No surprise paid retries, especially with final-state-only retention.
    job["taskGroups"][0]["taskSpec"]["maxRetryCount"] = 0
    job["labels"] = {"experiment": "exp16-fhp-hand-board-48h", "stage": args.kind}
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("controller", "smoke", "train", "aggregate", "evaluate"))
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "project-id", "region"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--max-hours", type=int, default=48)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2) + "\n")


if __name__ == "__main__":
    main()
