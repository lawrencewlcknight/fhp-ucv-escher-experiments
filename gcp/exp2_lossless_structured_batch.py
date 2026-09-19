#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for FHP Experiment 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from gcp import exp1_grouped_wide_batch as _base
except ModuleNotFoundError:  # Direct execution sets sys.path to gcp/.
    import exp1_grouped_wide_batch as _base


REPO_URL = _base.REPO_URL
MODULE = "experiments.fhp.exp2_fhp_lossless_structured_ucv.run"
TASK_COUNT = 3


def _adapt_script(script: str) -> str:
    replacements = (
        ("exp1-controller", "exp2-controller"),
        ("exp1-fhp", "exp2-fhp"),
        ("EXP1_REMOTE_CONTROLLER", "EXP2_REMOTE_CONTROLLER"),
        ("gcp/run_exp1_grouped_wide.sh", "gcp/run_exp2_lossless_structured.sh"),
        (
            "experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.config",
            "experiments.fhp.exp2_fhp_lossless_structured_ucv.config",
        ),
        (
            "experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.aggregate",
            "experiments.fhp.exp2_fhp_lossless_structured_ucv.aggregate",
        ),
        ("EXP1_TASK_METADATA", "EXP2_TASK_METADATA"),
        ("grouped_wide_ucv_escher", "lossless_structured_ucv_escher"),
        ("Experiment 1", "Experiment 2"),
        ("EXP1_REMOTE_TASK_URI", "EXP2_REMOTE_TASK_URI"),
    )
    for source, target in replacements:
        script = script.replace(source, target)
    return script


def build_job(args) -> dict:
    prior_module = _base.MODULE
    prior_count = _base.TASK_COUNT
    _base.MODULE = MODULE
    _base.TASK_COUNT = TASK_COUNT
    try:
        job = _base.build_job(args)
    finally:
        _base.MODULE = prior_module
        _base.TASK_COUNT = prior_count
    runnable = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]
    runnable["text"] = _adapt_script(runnable["text"])
    job["labels"] = {
        "experiment": "exp2-fhp-lossless-structured",
        "stage": args.kind,
    }
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("controller", "smoke", "train", "aggregate"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--parallelism", type=int, default=TASK_COUNT)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--controller-action",
        choices=("orchestrate", "orchestrate-resume"),
        default="orchestrate",
    )
    args = parser.parse_args()
    if args.parallelism < 1 or args.parallelism > TASK_COUNT:
        parser.error(f"--parallelism must be between 1 and {TASK_COUNT}")
    if args.kind == "controller" and (not args.project_id or not args.region):
        parser.error("controller jobs require --project-id and --region")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
