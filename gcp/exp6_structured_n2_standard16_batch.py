#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for FHP Experiment 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from gcp import exp2_lossless_structured_batch as _base
    from gcp.fhp_checkpoint_retention import apply_retention
except ModuleNotFoundError:  # Direct execution sets sys.path to gcp/.
    import exp2_lossless_structured_batch as _base
    from fhp_checkpoint_retention import apply_retention


REPO_URL = _base.REPO_URL
MODULE = "experiments.fhp.exp6_fhp_structured_n2_standard16.run"
TASK_COUNT = 3


def _adapt_script(script: str) -> str:
    replacements = (
        ("exp2-controller", "exp6-controller"),
        ("exp2-fhp", "exp6-fhp"),
        ("EXP2_REMOTE_CONTROLLER", "EXP6_REMOTE_CONTROLLER"),
        ("gcp/run_exp2_lossless_structured.sh", "gcp/run_exp6_structured_n2_standard16.sh"),
        (
            "experiments.fhp.exp2_fhp_lossless_structured_ucv.config",
            "experiments.fhp.exp6_fhp_structured_n2_standard16.config",
        ),
        (
            "experiments.fhp.exp2_fhp_lossless_structured_ucv.aggregate",
            "experiments.fhp.exp6_fhp_structured_n2_standard16.aggregate",
        ),
        ("EXP2_TASK_METADATA", "EXP6_TASK_METADATA"),
        (
            "lossless_structured_ucv_escher",
            "lossless_structured_ucv_escher_n2_16",
        ),
        ("Experiment 2", "Experiment 6"),
        ("EXP2_REMOTE_TASK_URI", "EXP6_REMOTE_TASK_URI"),
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
    # The inherited raw-input Experiment 1 stress command is not part of the
    # structured CLI. Experiment 6's smoke itself checks checkpoint reload,
    # completed-state resume and aggregation using its real structured worker.
    runnable["text"] = runnable["text"].replace(
        f'python -m {MODULE} checkpoint-smoke --output-root "$OUTPUT_ROOT"\n', "")
    if args.kind in ("train", "smoke"):
        job["taskGroups"][0]["taskSpec"]["computeResource"].update(
            cpuMilli=16_000, memoryMib=62_000)
        job["allocationPolicy"]["instances"][0]["policy"]["machineType"] = "n2-standard-16"
        runnable["text"] = runnable["text"].replace(
            "--requested-memory-mib 30000", "--requested-memory-mib 62000")
    job["labels"] = {
        "experiment": "exp6-fhp-n2-standard16",
        "stage": args.kind,
    }
    return apply_retention(job, kind=args.kind)


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
