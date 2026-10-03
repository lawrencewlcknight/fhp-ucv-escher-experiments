#!/usr/bin/env python3
"""Reuse the tested frozen-audit orchestration, with an Experiment 9 source."""
import argparse
import json
from pathlib import Path

import exp4_average_policy_audit_batch as audit


def build_job(args):
    job = audit.build_job(args)
    spec = job["taskGroups"][0]["taskSpec"]
    script = spec["runnables"][0]["script"]["text"]
    for old, new in (
        ("exp4_fhp_average_policy_audit", "exp15_fhp_policy_learning_rate"),
        ("run_exp4_average_policy_audit.sh", "run_exp15_policy_learning_rate.sh"),
        ("EXP4_REMOTE", "EXP15_REMOTE"), ("EXP2_RUN_ID", "EXP9_RUN_ID"),
        ("exp4-audit", "exp15-policy-lr"), ("exp4-controller", "exp15-controller"),
        ("uv python install 3.11", "uv python install 3.11.16"),
        ("uv venv --python 3.11 ", "uv venv --python 3.11.16 "),
    ):
        script = script.replace(old, new)
    if args.kind == "controller":
        script = script.replace("exec bash gcp/run_exp15_policy_learning_rate.sh",
            f"export EXP15_INCLUDE_LBR={args.include_lbr}\nexec bash gcp/run_exp15_policy_learning_rate.sh")
    elif args.include_lbr:
        script = script.replace("--evaluation-workers 8", "--evaluation-workers 8 --include-lbr")
        script = script.replace('aggregate --output-root "$OUTPUT_ROOT"',
                                'aggregate --output-root "$OUTPUT_ROOT" --include-lbr')
    spec["runnables"][0]["script"]["text"] = script
    job["labels"] = {"experiment": "exp15-fhp-policy-lr", "stage": args.kind}
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("controller", "smoke", "train", "aggregate"))
    parser.add_argument("--output", type=Path, required=True)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "source-run-id", "project-id", "region"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repo-url", default=audit.base.REPO_URL)
    parser.add_argument("--parallelism", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--include-lbr", type=int, choices=(0, 1), default=0)
    parser.add_argument("--controller-action", choices=("orchestrate", "orchestrate-resume"), default="orchestrate")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
