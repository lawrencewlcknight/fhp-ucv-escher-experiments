#!/usr/bin/env python3
"""New workflow, original Experiment 36 learner revision; training only."""
import argparse
from copy import copy
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.fhp.exp38_fhp_strength_critic2500_48h import contract as c
from gcp import exp36_strength_critic2500_batch as training

REPO_URL = training.lifecycle.REPO_URL
MODULE = "experiments.fhp.exp36_fhp_strength_critic2500.run"
PACKAGE = Path(__file__).resolve().parents[1]/"experiments/fhp/exp38_fhp_strength_critic2500_48h"


def replace(script, old, new):
    if old not in script:
        raise RuntimeError(f"Inherited training template changed: {old}")
    return script.replace(old, new)


def build_job(args):
    if not re.fullmatch(r"exp38-[a-z0-9][a-z0-9-]{0,27}[a-z0-9]", args.run_id):
        raise ValueError("Use a new exp38- RUN_ID, at most 35 characters")
    inherited = copy(args)
    # The established builder requires an exp36 prefix; replace only this
    # temporary output identity afterward, NEVER the archived source run ID.
    inherited.run_id = "exp36-" + args.run_id.removeprefix("exp38-")
    inherited.source_run_id, inherited.total_hours = c.SOURCE_RUN_ID, c.TOTAL_HOURS
    inherited.repo_ref = args.repo_ref if args.kind == "controller" else c.TRAINING_REF
    inherited.controller_action = "orchestrate"
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("Pin workflow REPO_REF to a full pushed commit SHA")
    job = training.build_job(inherited)
    spec = job["taskGroups"][0]["taskSpec"]
    text = replace(spec["runnables"][0]["script"]["text"], inherited.run_id, args.run_id)
    if args.kind == "controller":
        text = replace(text, "gcp/run_exp36_strength_critic2500.sh", "gcp/run_exp38_strength_critic2500_48h.sh")
        text = replace(text, "export EXP36_REMOTE_CONTROLLER=1", "export EXP38_REMOTE_CONTROLLER=1")
        text = replace(text, "export PARALLELISM=", "export EXP38_PARALLELISM=")
        # Covers sequential waves without silently killing later seed workers.
        spec["maxRunDuration"] = f"{(72*((3+args.parallelism-1)//args.parallelism)+12)*3600}s"
    elif args.kind == "smoke":
        command = f'python -m {MODULE} smoke --output-root "$OUTPUT_ROOT" --no-resume\n'
        verify = (PACKAGE/"smoke.py").read_text()
        text = replace(text, command, command+'python - "$OUTPUT_ROOT" <<\'EXP38_RESTART\'\n'+verify+'\nEXP38_RESTART\n')
    elif args.kind == "aggregate":
        command = f'python -m {MODULE} aggregate --output-root "$OUTPUT_ROOT" --total-hours "$EXP36_TOTAL_HOURS"\n'
        finish = (PACKAGE/"report.py").read_text()
        text = replace(text, command, command+
            'gcloud storage cp "$BUCKET_ROOT/$RUN_ID/experiment38_contract.json" "$OUTPUT_ROOT/source_contract.json"\n'+
            f'python - "$OUTPUT_ROOT" "{args.repo_ref}" "$BUCKET_ROOT/$RUN_ID" <<\'EXP38_REPORT\'\n'+finish+'\nEXP38_REPORT\n')
    elif args.kind != "train":
        raise ValueError("Only controller, smoke, train and aggregate are supported; evaluation is separate")
    spec["runnables"][0]["script"]["text"] = text
    spec["maxRetryCount"] = 0
    job["labels"] = dict(experiment="exp38-fhp-strength-critic2500-48h", stage=args.kind)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("controller", "smoke", "train", "aggregate"), required=True)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "project-id", "region"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=3)
    args = parser.parse_args()
    # Compatibility with the inherited lifecycle; no evaluator is scheduled.
    args.max_hours, args.resume = 48, False
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2)+"\n")


if __name__ == "__main__":
    main()
