#!/usr/bin/env python3
"""Explicit smoke -> one-seed pilot -> remaining seeds; no automatic promotion."""
import argparse
from copy import copy
import json
from pathlib import Path

try:
    import exp1_grouped_wide_batch as base
except ModuleNotFoundError:
    from gcp import exp1_grouped_wide_batch as base

MODULE = "experiments.fhp.exp22_fhp_critic_objectives.run"


def script(args):
    bootstrap = (base._bootstrap(args).replace("exp1-fhp", "exp22-critic-objectives")
                 .replace("uv python install 3.11", "uv python install 3.11.16")
                 .replace("uv venv --python 3.11 ", "uv venv --python 3.11.16 "))
    if args.kind in ("smoke", "pilot", "full"):
        seed = '"$(( ${BATCH_TASK_INDEX:?Missing index} + 1 ))"' if args.kind == "full" else '"0"'
        action = "smoke" if args.kind == "smoke" else "worker"
        commands = f"""
SEED={seed}
[[ "$SEED" =~ ^[012]$ ]] || exit 2
SOURCE="$WORK_ROOT/source"
export EXP22_REMOTE_WORKER="$BUCKET_ROOT/$RUN_ID/workers/seed_$SEED"
mkdir -p "$OUTPUT_ROOT/diagnostics/task_${{BATCH_TASK_INDEX:-0}}"
FETCH_STARTED=$SECONDS
timeout 1800 python -m {MODULE} fetch-source --bucket "$BUCKET_ROOT" --seed "$SEED" --directory "$SOURCE"
printf '{{"source_fetch_seconds":%s}}\\n' "$(( SECONDS-FETCH_STARTED ))" > "$OUTPUT_ROOT/diagnostics/task_${{BATCH_TASK_INDEX:-0}}/source_fetch.json"
timeout {1800 if args.kind == 'smoke' else 21600} python -m {MODULE} {action} \
  --source-worker "$SOURCE" --seed "$SEED" --output "$OUTPUT_ROOT/workers/seed_$SEED"
"""
    else:
        seeds = (0,) if args.kind == "aggregate-pilot" else (0, 1, 2)
        commands = "mkdir -p \"$OUTPUT_ROOT/workers\"\n"
        for seed in seeds:
            run = args.pilot_run_id if seed == 0 and args.kind == "aggregate" else args.run_id
            commands += f"""
gcloud storage rsync --recursive --exclude='\\.(pt|npz)$' \
  "$BUCKET_ROOT/{run}/workers/seed_{seed}" "$OUTPUT_ROOT/workers/seed_{seed}"
"""
        commands += f"""
python -m {MODULE} aggregate --root "$OUTPUT_ROOT" {'--pilot' if args.kind == 'aggregate-pilot' else ''}
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
"""
    wrapped = base._diagnostic_wrapper(commands, "$OUTPUT_ROOT/diagnostics/task_${BATCH_TASK_INDEX:-0}",
                                      "$BUCKET_ROOT/$RUN_ID/diagnostics/task_${BATCH_TASK_INDEX:-0}", 64000)
    return "#!/usr/bin/env bash\nset -Eeuo pipefail\n"+bootstrap+"\n"+wrapped+"\n"


def build_job(args):
    proxy = copy(args)
    proxy.kind = "train" if args.kind == "full" else "smoke"
    result = base.build_job(proxy)
    group = result["taskGroups"][0]
    group.update(taskCount=2 if args.kind == "full" else 1,
                 parallelism=min(args.parallelism, 2) if args.kind == "full" else 1)
    spec = group["taskSpec"]
    spec.update(maxRetryCount=0, maxRunDuration="28800s" if args.kind in ("pilot", "full") else "7200s",
                computeResource={"cpuMilli": 16000, "memoryMib": 64000})
    # A separate runnable uploads diagnostics even if the main process was killed.
    tail = f"""#!/usr/bin/env bash
set -u
OUT="/workspace/exp22-critic-objectives-${{BATCH_TASK_RETRY_ATTEMPT:-0}}/output"
mkdir -p "$OUT/diagnostics/task_${{BATCH_TASK_INDEX:-0}}"
(date -u; free -h; df -h; dmesg 2>&1 | tail -100) > "$OUT/diagnostics/task_${{BATCH_TASK_INDEX:-0}}/final_system.txt"
if [[ -d "$OUT/workers" ]]; then
  gcloud storage rsync --recursive "$OUT/workers" {base._q(args.bucket_root.rstrip('/')+'/'+args.run_id+'/workers')}
fi
gcloud storage rsync --recursive "$OUT/diagnostics" {base._q(args.bucket_root.rstrip('/')+'/'+args.run_id+'/diagnostics')}
"""
    spec["runnables"] = [{"script": {"text": script(args)}}, {"alwaysRun": True, "script": {"text": tail}}]
    result["allocationPolicy"]["instances"][0]["policy"].update(
        machineType="n2-standard-16", bootDisk={"sizeGb": 200, "type": "pd-balanced"})
    result["labels"] = {"experiment": "exp22-critic-objectives", "stage": args.kind}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=("smoke", "pilot", "full", "aggregate-pilot", "aggregate"))
    p.add_argument("--output", required=True, type=Path)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "project-id", "region"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--repo-url", default=base.REPO_URL)
    p.add_argument("--pilot-run-id", default="")
    p.add_argument("--parallelism", type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    if args.kind == "aggregate" and not args.pilot_run_id:
        p.error("Full aggregation needs --pilot-run-id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2)+"\n")


if __name__ == "__main__":
    main()
