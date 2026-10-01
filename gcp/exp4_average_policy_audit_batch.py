#!/usr/bin/env python3
"""Cloud-only smoke -> three frozen-replay workers -> small aggregation job."""

import argparse
import json
from pathlib import Path

import exp1_grouped_wide_batch as base
from fhp_checkpoint_retention import apply_retention

MODULE = "experiments.fhp.exp4_fhp_average_policy_audit.run"


def script(args):
    if args.kind == "controller":
        result = base._controller_script(args).replace("exp1-controller", "exp4-controller")
        result = result.replace("EXP1_REMOTE_CONTROLLER", "EXP4_REMOTE_CONTROLLER")
        return result.replace("exec bash gcp/run_exp1_grouped_wide.sh",
                              f"export EXP2_RUN_ID={base._q(args.source_run_id)}\nexec bash gcp/run_exp4_average_policy_audit.sh")
    bootstrap = base._bootstrap(args).replace("exp1-fhp", "exp4-audit")
    bootstrap += f"\nEXP2_RUN_ID={base._q(args.source_run_id)}\n"
    bootstrap += 'export OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n'
    if args.kind in ("smoke", "train"):
        seed = '0' if args.kind == "smoke" else '${BATCH_TASK_INDEX:?Missing task index}'
        command = "smoke" if args.kind == "smoke" else 'worker --seed "$SEED"'
        prefix = '$BUCKET_ROOT/$RUN_ID/smoke' if args.kind == "smoke" else '$BUCKET_ROOT/$RUN_ID'
        action = f"""
SEED="{seed}"
[[ "$SEED" =~ ^[012]$ ]] || {{ echo "Unexpected seed $SEED" >&2; exit 2; }}
SOURCE="$WORK_ROOT/source"
python -m {MODULE} fetch-source --bucket "$BUCKET_ROOT" \
  --run-id "$EXP2_RUN_ID" --seed "$SEED" --directory "$SOURCE"
mkdir -p "$OUTPUT_ROOT/workers/seed_$SEED"
export EXP4_REMOTE_WORKER="{prefix}/workers/seed_$SEED"
if gcloud storage ls "$EXP4_REMOTE_WORKER/manifest.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$EXP4_REMOTE_WORKER" "$OUTPUT_ROOT/workers/seed_$SEED"
fi
python -m {MODULE} {command} --source-worker "$SOURCE" \
  --output-root "$OUTPUT_ROOT" --evaluation-workers 8
gcloud storage rsync --recursive "$OUTPUT_ROOT" "{prefix}"
"""
        wrapped = base._diagnostic_wrapper(action, '$OUTPUT_ROOT/diagnostics/seed_${BATCH_TASK_INDEX:-0}',
                                           f'{prefix}/diagnostics/seed_${{BATCH_TASK_INDEX:-0}}', 30_000)
        # On any failure, preserve completed fit endpoints and evaluation shards.
        wrapped = wrapped.replace('  exit "$exit_code"',
                                  f'  gcloud storage rsync --recursive "$OUTPUT_ROOT" "{prefix}" || true\n  exit "$exit_code"')
    else:
        wrapped = f"""
mkdir -p "$OUTPUT_ROOT/workers"
gcloud storage rsync --recursive \
  --exclude='(^|/)(evaluation_tasks)(/|$)|\\.(pt|pkl)$' \
  "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
python -m {MODULE} aggregate --output-root "$OUTPUT_ROOT"
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
"""
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{wrapped}\n"


def build_job(args):
    # Reuse tested machine/service-account/bootstrap conventions, not training logic.
    result = base.build_job(args)
    group = result["taskGroups"][0]
    spec = group["taskSpec"]
    spec["runnables"] = [{"script": {"text": script(args)}}]
    spec["maxRetryCount"] = 0  # Explicit resume; never silently double the evaluation bill.
    controller_seconds = 86400 * ((3 + args.parallelism - 1) // args.parallelism) + 21600
    spec["maxRunDuration"] = {"controller": f"{controller_seconds}s", "train": "86400s",
                              "smoke": "7200s", "aggregate": "3600s"}[args.kind]
    result["labels"] = {"experiment": "exp4-fhp-policy-audit", "stage": args.kind}
    return apply_retention(result, kind=args.kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("controller", "smoke", "train", "aggregate"))
    parser.add_argument("--output", required=True, type=Path)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "source-run-id", "project-id", "region"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--repo-url", default=base.REPO_URL)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--controller-action", choices=("orchestrate", "orchestrate-resume"), default="orchestrate")
    args = parser.parse_args()
    if args.parallelism not in (1, 2, 3):
        parser.error("parallelism must be 1, 2 or 3")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
