#!/usr/bin/env python3
"""Shared Batch builder for Experiments 29--33, including automatic evaluation."""
import argparse
from copy import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcp import exp20_half_critic_updates_batch as lifecycle
from experiments.fhp.strategic_feature_suite.registry import SPECS, BASELINE_RUN_ID


def build_job(args):
    number = int(args.experiment)
    slug, kind, _ = SPECS[number]
    if args.source_run_id and not args.source_run_id.startswith(f"exp{number}-"):
        raise ValueError("Continuation source must belong to this experiment")
    if not args.run_id.startswith(f"exp{number}-"):
        raise ValueError("RUN_ID must start with this experiment number")
    inherited = copy(args)
    job = lifecycle.build_job(inherited)
    spec = job["taskGroups"][0]["taskSpec"]
    script = spec["runnables"][0]["script"]["text"]
    for old, new in (
        ("exp20_fhp_half_critic_updates", f"exp{number}_fhp_{slug}"),
        ("gcp/run_exp20_half_critic_updates.sh", f"gcp/run_exp{number}_{slug}.sh"),
        ("half_critic_hand_board_ucv_escher", f"strategic_{kind.lower()}_ucv_escher"),
        ("--exp20-run", "--candidate-run"), ("EXP20", "FEATURE"),
        ("Experiment 20", f"Experiment {number}"), ("exp20-", f"exp{number}-"),
    ):
        script = script.replace(old, new)
    if args.kind == "smoke" and not args.source_run_id:
        module = f"experiments.fhp.exp{number}_fhp_{slug}.run"
        marker = f'python -m {module} capacity-preflight'
        if marker not in script:
            raise RuntimeError("Inherited smoke template changed")
        check = f'''mkdir -p "$WORK_ROOT/baseline/workers"
gcloud storage rsync --recursive \\
  --exclude='.*training_states.*|.*continuation_inputs.*|.*reservoir.*|.*\\.pt$' \\
  "$BUCKET_ROOT/{BASELINE_RUN_ID}/workers" "$WORK_ROOT/baseline/workers"
python -m {module} baseline-preflight --baseline-run "$WORK_ROOT/baseline" \\
  --output-root "$OUTPUT_ROOT/baseline_validation"
'''
        script = script.replace(marker, check + marker)
    spec["runnables"][0]["script"]["text"] = script
    variables = spec.setdefault("environment", {}).setdefault("variables", {})
    spec["environment"]["variables"] = {
        **{k.replace("EXP20", "FEATURE"): v for k, v in variables.items()},
        "FEATURE_NUMBER": str(number), "EVAL_MAX_HOURS": str(args.max_hours)}
    if args.kind == "controller":
        waves = (3 + args.parallelism - 1) // args.parallelism
        training_hours = 72 if args.source_run_id else 36
        spec["maxRunDuration"] = f"{(training_hours * waves + args.max_hours + 12) * 3600}s"
    job["labels"] = {"experiment": f"exp{number}-fhp-{slug.replace('_', '-')}", "stage": args.kind}
    return job


def main(argv=None, *, experiment=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if experiment is None:
        parser.add_argument("--experiment", type=int, choices=tuple(SPECS), required=True)
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
    args = parser.parse_args(argv)
    if experiment is not None:
        args.experiment = experiment
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2) + "\n")


if __name__ == "__main__":
    main()
