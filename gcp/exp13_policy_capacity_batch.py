#!/usr/bin/env python3
"""Cloud smoke -> parallel screens -> global selection -> parallel deployment -> analysis."""

import argparse
from copy import copy
import json
from pathlib import Path

import exp1_grouped_wide_batch as base
from fhp_checkpoint_retention import apply_retention

MODULE = "experiments.fhp.exp13_fhp_policy_capacity.run"
KINDS = ("controller", "smoke", "screen", "select", "train", "aggregate")


def script(args):
    if args.kind == "controller":
        result = base._controller_script(args).replace("exp1-controller", "exp13-controller")
        result = result.replace("EXP1_REMOTE_CONTROLLER", "EXP13_REMOTE_CONTROLLER")
        return result.replace("exec bash gcp/run_exp1_grouped_wide.sh",
            f"export EXP2_RUN_ID={base._q(args.source_run_id)}\nexec bash gcp/run_exp13_policy_capacity.sh")
    bootstrap = base._bootstrap(args).replace("exp1-fhp", "exp13-capacity")
    bootstrap += f"\nEXP2_RUN_ID={base._q(args.source_run_id)}\nexport OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n"
    if args.kind in ("smoke", "screen", "train"):
        seed = "0" if args.kind == "smoke" else "${BATCH_TASK_INDEX:?Missing task index}"
        command = {"smoke": "smoke", "screen": 'screen --seed "$SEED"', "train": 'deploy --seed "$SEED"'}[args.kind]
        workers = "" if args.kind == "screen" else "--evaluation-workers 8"
        prefix = "$BUCKET_ROOT/$RUN_ID/smoke" if args.kind == "smoke" else "$BUCKET_ROOT/$RUN_ID"
        selection = ('gcloud storage cp "$BUCKET_ROOT/$RUN_ID/selection.json" "$OUTPUT_ROOT/selection.json"'
                     if args.kind == "train" else "")
        action = f"""
SEED="{seed}"
[[ "$SEED" =~ ^[012]$ ]] || {{ echo "Unexpected seed $SEED" >&2; exit 2; }}
SOURCE="$WORK_ROOT/source"
python -m {MODULE} fetch-source --bucket "$BUCKET_ROOT" --run-id "$EXP2_RUN_ID" \
  --seed "$SEED" --directory "$SOURCE"
mkdir -p "$OUTPUT_ROOT/workers/seed_$SEED"
export EXP13_REMOTE_WORKER="{prefix}/workers/seed_$SEED"
if gcloud storage ls "$EXP13_REMOTE_WORKER/manifest.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$EXP13_REMOTE_WORKER" "$OUTPUT_ROOT/workers/seed_$SEED"
fi
{selection}
python -m {MODULE} {command} --source-worker "$SOURCE" --output-root "$OUTPUT_ROOT" {workers}
gcloud storage rsync --recursive "$OUTPUT_ROOT" "{prefix}"
"""
        wrapped = base._diagnostic_wrapper(action, '$OUTPUT_ROOT/diagnostics/' + args.kind + '/seed_${BATCH_TASK_INDEX:-0}',
                                           prefix + '/diagnostics/' + args.kind + '/seed_${BATCH_TASK_INDEX:-0}', 30000)
        wrapped = wrapped.replace('  exit "$exit_code"',
            f'  gcloud storage rsync --recursive "$OUTPUT_ROOT" "{prefix}" || true\n  exit "$exit_code"')
    else:
        wrapped = f"""
mkdir -p "$OUTPUT_ROOT/workers"
gcloud storage rsync --recursive --exclude='(^|/)evaluation_tasks(/|$)|\\.(pt|pkl)$' \
  "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/selection.json" >/dev/null 2>&1; then
  gcloud storage cp "$BUCKET_ROOT/$RUN_ID/selection.json" "$OUTPUT_ROOT/selection.json"
fi
python -m {MODULE} {args.kind} --output-root "$OUTPUT_ROOT"
"""
        if args.kind == "select":
            wrapped += '\ngcloud storage cp "$OUTPUT_ROOT/selection.json" "$BUCKET_ROOT/$RUN_ID/selection.json"\n'
        else:
            wrapped += '\ngcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"\n'
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{wrapped}\n"


def build_job(args):
    mapped = copy(args)
    mapped.kind = {"screen": "train", "select": "aggregate"}.get(args.kind, args.kind)
    job = base.build_job(mapped)
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    spec["runnables"] = [{"script": {"text": script(args)}}]
    spec["maxRetryCount"] = 0
    waves = (3 + args.parallelism - 1) // args.parallelism
    spec["maxRunDuration"] = {"controller": f"{2 * 86400 * waves + 21600}s",
        "smoke": "7200s", "screen": "86400s", "select": "3600s", "train": "86400s", "aggregate": "3600s"}[args.kind]
    job["labels"] = {"experiment": "exp13-fhp-capacity", "stage": args.kind}
    return apply_retention(job, kind=args.kind)


def main(builder=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--output", required=True, type=Path)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "source-run-id", "project-id", "region"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--repo-url", default=base.REPO_URL)
    parser.add_argument("--parallelism", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--controller-action", choices=("orchestrate", "orchestrate-resume"), default="orchestrate")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps((build_job if builder is None else builder)(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
