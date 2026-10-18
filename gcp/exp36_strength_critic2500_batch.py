#!/usr/bin/env python3
"""Experiment 36: proven training lifecycle, explicitly separate evaluation."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcp import exp20_half_critic_updates_batch as lifecycle

CONTROL_RUNS = {20: "exp20-critic24-20261006-164357", 33: "exp33-features-20261012-143026"}


def build_job(args):
    if not args.run_id.startswith("exp36-"):
        raise ValueError("RUN_ID must start with exp36-")
    if args.source_run_id and not args.source_run_id.startswith("exp36-"):
        raise ValueError("Only Experiment 36 full states may be continued")
    job = lifecycle.build_job(args)
    spec = job["taskGroups"][0]["taskSpec"]
    script = spec["runnables"][0]["script"]["text"]
    for old, new in (
        ("exp20_fhp_half_critic_updates", "exp36_fhp_strength_critic2500"),
        ("gcp/run_exp20_half_critic_updates.sh", "gcp/run_exp36_strength_critic2500.sh"),
        ("half_critic_hand_board_ucv_escher", "strategic_ae_critic2500_ucv_escher"),
        ("--exp20-run", "--candidate-run"), ("EXP20", "EXP36"),
        ("Experiment 20", "Experiment 36"), ("exp20-", "exp36-"),
    ):
        script = script.replace(old, new)
    if args.kind == "evaluate":
        # Add the two archived comparators; training never downloads/retrains them.
        staging = ""
        arguments = ""
        for number, run_id in CONTROL_RUNS.items():
            staging += f'''mkdir -p "$INPUT_ROOT/exp{number}/workers"
gcloud storage rsync --recursive \\
  --exclude='.*training_states.*|.*continuation_inputs.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \\
  "$BUCKET_ROOT/{run_id}/workers" "$INPUT_ROOT/exp{number}/workers"
'''
            arguments += f' --exp{number}-run "$INPUT_ROOT/exp{number}"'
        script = lifecycle._replace(script, "# Atomic task-result JSON", staging + "# Atomic task-result JSON")
        script = lifecycle._replace(script, '--exp10-run "$INPUT_ROOT/exp8"', '--exp10-run "$INPUT_ROOT/exp8"' + arguments)
    spec["runnables"][0]["script"]["text"] = script
    variables = spec.setdefault("environment", {}).setdefault("variables", {})
    spec["environment"]["variables"] = {k.replace("EXP20", "EXP36"): v for k, v in variables.items()}
    job["labels"] = {"experiment": "exp36-fhp-strength-critic2500", "stage": args.kind}
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("controller", "smoke", "train", "aggregate", "evaluate"), required=True)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "project-id", "region"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repo-url", default=lifecycle.REPO_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--total-hours", type=int, default=24)
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--max-hours", type=int, default=48)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2) + "\n")


if __name__ == "__main__":
    main()
