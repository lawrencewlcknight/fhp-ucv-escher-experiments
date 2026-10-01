#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for FHP Experiment 11."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from gcp import exp2_lossless_structured_batch as _base
    from gcp.fhp_checkpoint_retention import apply_retention
except ModuleNotFoundError:  # Direct execution sets sys.path to gcp/.
    import exp2_lossless_structured_batch as _base
    from fhp_checkpoint_retention import apply_retention


REPO_URL = _base.REPO_URL
MODULE = "experiments.fhp.exp11_fhp_betting_economics.run"
TASK_COUNT = 3


def _adapt_script(script: str) -> str:
    replacements = (
        ("exp2-controller", "exp11-controller"),
        ("exp2-fhp", "exp11-fhp"),
        ("EXP2_REMOTE_CONTROLLER", "EXP11_REMOTE_CONTROLLER"),
        ("gcp/run_exp2_lossless_structured.sh", "gcp/run_exp11_betting_economics.sh"),
        (
            "experiments.fhp.exp2_fhp_lossless_structured_ucv.config",
            "experiments.fhp.exp11_fhp_betting_economics.config",
        ),
        (
            "experiments.fhp.exp2_fhp_lossless_structured_ucv.aggregate",
            "experiments.fhp.exp11_fhp_betting_economics.aggregate",
        ),
        ("EXP2_TASK_METADATA", "EXP11_TASK_METADATA"),
        (
            "lossless_structured_ucv_escher",
            "betting_economics_cached_parallel_ucv_escher",
        ),
        ("Experiment 2", "Experiment 11"),
        ("EXP2_REMOTE_TASK_URI", "EXP11_REMOTE_TASK_URI"),
    )
    for source, target in replacements:
        script = script.replace(source, target)
    return script


def build_job(args) -> dict:
    total_hours = getattr(args, "total_hours", 24)
    source_run = getattr(args, "source_run_id", "")
    if isinstance(total_hours, bool) or int(total_hours) != total_hours or total_hours < 24 or total_hours % 6:
        raise ValueError("Total hours must be an integer multiple of six, at least 24")
    total_hours = int(total_hours)
    if source_run:
        if (not re.fullmatch(r"[a-z][a-z0-9-]{0,33}[a-z0-9]", source_run)
                or source_run == args.run_id or total_hours <= 24):
            raise ValueError("Extension requires a distinct source run and total hours > 24")
    elif total_hours != 24:
        raise ValueError("A horizon beyond 24 hours requires a source run")
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
    spec = job["taskGroups"][0]["taskSpec"]
    spec.setdefault("environment", {}).setdefault("variables", {}).update(
        EXP11_TOTAL_HOURS=str(total_hours), EXP11_SOURCE_RUN_ID=source_run)
    # Exercise the actual parallel/structured production checkpoint allocation,
    # rather than the inherited raw-input Experiment 1 stress command.
    runnable["text"] = runnable["text"].replace(
        f'python -m {MODULE} checkpoint-smoke --output-root "$OUTPUT_ROOT"\n',
        f'python -m {MODULE} capacity-preflight --output-root "$OUTPUT_ROOT/capacity_preflight"\n')
    if args.kind in ("train", "smoke"):
        job["taskGroups"][0]["taskSpec"]["computeResource"].update(
            cpuMilli=16_000, memoryMib=62_000)
        job["allocationPolicy"]["instances"][0]["policy"]["machineType"] = "n2-standard-16"
        runnable["text"] = runnable["text"].replace(
            "--requested-memory-mib 30000", "--requested-memory-mib 62000")
    if args.kind == "train":
        # Initial run matches Experiment 7's 36h wall ceiling. Extensions may
        # add up to 48 active hours and retain Experiment 8's 72h ceiling.
        spec["maxRunDuration"] = "259200s" if source_run else "129600s"
        command = f'python -m {MODULE} worker --task-index "$TASK_INDEX" --output-root "$OUTPUT_ROOT"'
        stage = f'''if [[ -n "$EXP11_SOURCE_RUN_ID" ]]; then
  python -m {MODULE} stage-source \\
    --source "$BUCKET_ROOT/$EXP11_SOURCE_RUN_ID/workers/$TASK_NAME" \\
    --destination "$OUTPUT_ROOT/workers/$TASK_NAME" \\
    --seed "$SOURCE_SEED" --total-hours "$EXP11_TOTAL_HOURS"
fi
'''
        if command not in runnable["text"]:
            raise RuntimeError("Inherited worker command changed; review Experiment 11 adapter")
        runnable["text"] = runnable["text"].replace(
            command, stage + command + ' --total-hours "$EXP11_TOTAL_HOURS"')
    elif args.kind == "aggregate":
        command = f'python -m {MODULE} aggregate --output-root "$OUTPUT_ROOT"'
        runnable["text"] = runnable["text"].replace(
            command, command + ' --total-hours "$EXP11_TOTAL_HOURS"')
    job["labels"] = {
        "experiment": "exp11-fhp-betting-economics",
        "stage": args.kind,
    }
    return apply_retention(job, kind=args.kind, final_state=True)


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
    parser.add_argument("--total-hours", type=int, default=24)
    parser.add_argument("--source-run-id", default="")
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
