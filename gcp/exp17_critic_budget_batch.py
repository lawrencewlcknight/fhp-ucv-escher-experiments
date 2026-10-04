#!/usr/bin/env python3
"""Batch workflow for the Experiment 10 frozen-data critic audit."""
import argparse
import json
from pathlib import Path

try:
    import exp1_grouped_wide_batch as base
except ModuleNotFoundError:
    from gcp import exp1_grouped_wide_batch as base

MODULE = "experiments.fhp.exp17_fhp_critic_budget.run"


def script(args):
    if args.kind == "controller":
        return (base._controller_script(args).replace("exp1-controller", "exp17-controller")
                .replace("EXP1_REMOTE_CONTROLLER", "EXP17_REMOTE_CONTROLLER")
                .replace("run_exp1_grouped_wide.sh", "run_exp17_critic_budget.sh"))
    bootstrap = (base._bootstrap(args).replace("exp1-fhp", "exp17-critic-budget")
                 .replace("uv python install 3.11", "uv python install 3.11.16")
                 .replace("uv venv --python 3.11 ", "uv venv --python 3.11.16 "))
    if args.kind in ("smoke", "train"):
        seed = "0" if args.kind == "smoke" else "${BATCH_TASK_INDEX:?Missing task index}"
        prefix = "$BUCKET_ROOT/$RUN_ID/smoke" if args.kind == "smoke" else "$BUCKET_ROOT/$RUN_ID"
        command = "smoke" if args.kind == "smoke" else 'worker --seed "$SEED"'
        action = f"""
SEED="{seed}"
[[ "$SEED" =~ ^[012]$ ]] || {{ echo "Invalid seed" >&2; exit 2; }}
SOURCE="$WORK_ROOT/source"
python -m {MODULE} fetch-source --bucket "$BUCKET_ROOT" --seed "$SEED" --directory "$SOURCE"
export EXP17_REMOTE_WORKER="{prefix}/workers/seed_$SEED"
mkdir -p "$OUTPUT_ROOT/workers/seed_$SEED"
if gcloud storage ls "$EXP17_REMOTE_WORKER/manifest.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$EXP17_REMOTE_WORKER" "$OUTPUT_ROOT/workers/seed_$SEED"
fi
python -m {MODULE} {command} --source-worker "$SOURCE" --output-root "$OUTPUT_ROOT"
gcloud storage rsync --recursive "$OUTPUT_ROOT" "{prefix}"
"""
        wrapped = base._diagnostic_wrapper(action, "$OUTPUT_ROOT/diagnostics/seed_${BATCH_TASK_INDEX:-0}",
                                           prefix + "/diagnostics/seed_${BATCH_TASK_INDEX:-0}", 64000)
        wrapped = wrapped.replace('  exit "$exit_code"',
                                  f'  gcloud storage rsync --recursive "$OUTPUT_ROOT" "{prefix}" || true\n  exit "$exit_code"')
    else:
        wrapped = f"""
mkdir -p "$OUTPUT_ROOT/workers"
gcloud storage rsync --recursive --exclude='\\.pt$' \
  "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
python -m {MODULE} aggregate --output-root "$OUTPUT_ROOT"
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
"""
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{wrapped}\n"


def build_job(args):
    result = base.build_job(args)
    spec = result["taskGroups"][0]["taskSpec"]
    spec["runnables"] = [{"script": {"text": script(args)}}]
    spec["maxRetryCount"] = 0
    spec["maxRunDuration"] = {"controller": "172800s", "smoke": "14400s", "train": "43200s", "aggregate": "3600s"}[args.kind]
    if args.kind in ("smoke", "train"):
        spec["computeResource"] = {"cpuMilli": 16000, "memoryMib": 64000}
        policy = result["allocationPolicy"]["instances"][0]["policy"]
        policy.update(machineType="n2-standard-16", bootDisk={"sizeGb": 200, "type": "pd-balanced"})
    result["labels"] = {"experiment": "exp17-fhp-critic-budget", "stage": args.kind}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("controller", "smoke", "train", "aggregate"))
    parser.add_argument("--output", required=True, type=Path)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "project-id", "region"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repo-url", default=base.REPO_URL)
    parser.add_argument("--parallelism", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--controller-action", choices=("orchestrate", "orchestrate-resume"), default="orchestrate")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
