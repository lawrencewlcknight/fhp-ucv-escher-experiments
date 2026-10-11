#!/usr/bin/env python3
"""One CPU Batch job: reference -> late boundary -> three shared audits -> two reports."""
import argparse
from copy import copy
import json
from pathlib import Path

try:
    import exp1_grouped_wide_batch as base
except ModuleNotFoundError:
    from gcp import exp1_grouped_wide_batch as base

MODULE = "experiments.fhp.exp24_fhp_critic_budget_stages.run"


def actions(args):
    flags = "--smoke --ray-smoke" if args.kind == "smoke" else ""
    audit_flags = "--smoke" if args.kind == "smoke" else ""
    return f'''
export EXP24_REMOTE_OUTPUT="$BUCKET_ROOT/$RUN_ID"
unset EXP22_REMOTE_WORKER EXP17_REMOTE_WORKER || true
SOURCE="$WORK_ROOT/source_seed_0"
FETCH_STARTED=$SECONDS
timeout 1800 python -m experiments.fhp.exp22_fhp_critic_objectives.run fetch-source --bucket "$BUCKET_ROOT" --seed 0 --directory "$SOURCE"
printf '{{"source_fetch_seconds":%s}}\\n' "$((SECONDS-FETCH_STARTED))" > "$OUTPUT_ROOT/diagnostics/source_fetch.json"
# Check the real historical source boundary before investing six training hours.
timeout 10800 python -m {MODULE} late --source-worker "$SOURCE" --output "$OUTPUT_ROOT" {audit_flags}
timeout {3600 if args.kind == 'smoke' else 36000} python -m {MODULE} reference --output "$OUTPUT_ROOT" {flags}
timeout 3600 python -m {MODULE} verify-checkpoint --output "$OUTPUT_ROOT" {flags}
for STAGE in one_hour six_hours late_24h_plus_one_iteration; do
  timeout {3600 if args.kind == 'smoke' else 10800} python -m {MODULE} audit --stage "$STAGE" --output "$OUTPUT_ROOT" {audit_flags}
done
timeout 1800 python -m {MODULE} aggregate --output "$OUTPUT_ROOT" {audit_flags}
echo 'Experiments 24 and 25 complete; no downstream training automatically launched.'
'''


def build_job(args):
    proxy = copy(args)
    proxy.kind, proxy.parallelism = "smoke", 1
    result = base.build_job(proxy)
    bootstrap = (base._bootstrap(args).replace("exp1-fhp", "exp24-critic-stages")
        .replace("uv python install 3.11", "uv python install 3.11.16")
        .replace("uv venv --python 3.11 ", "uv venv --python 3.11.16 "))
    script = "#!/usr/bin/env bash\nset -Eeuo pipefail\n" + bootstrap + "\n" + base._diagnostic_wrapper(
        actions(args), "$OUTPUT_ROOT/diagnostics", "$BUCKET_ROOT/$RUN_ID/diagnostics", 64000)
    tail = f'''#!/usr/bin/env bash
set -u
OUT="/workspace/exp24-critic-stages-${{BATCH_TASK_RETRY_ATTEMPT:-0}}/output"
REMOTE={base._q(args.bucket_root.rstrip('/')+'/'+args.run_id)}
mkdir -p "$OUT/diagnostics"
(date -u; free -h; df -h; dmesg 2>&1 | tail -100) > "$OUT/diagnostics/final_system.txt"
# Excludes the downloaded historical source; always preserve diagnostics and
# any complete new checkpoints, even when a later subprocess fails.
gcloud storage rsync --recursive --exclude='(^|/)[^/]+\\.tmp(/|$)' "$OUT" "$REMOTE"
'''
    group = result["taskGroups"][0]
    group.update(taskCount=1, parallelism=1)
    group["taskSpec"].update(maxRetryCount=0, maxRunDuration="21600s" if args.kind == "smoke" else "86400s",
        computeResource=dict(cpuMilli=16000, memoryMib=64000),
        runnables=[{"script": {"text": script}}, {"alwaysRun": True, "script": {"text": tail}}])
    result["allocationPolicy"]["instances"][0]["policy"].update(machineType="n2-standard-16",
        bootDisk={"sizeGb": 250, "type": "pd-balanced"})
    result["labels"] = {"experiment": "exp24-25-critic-stages", "stage": args.kind}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=("smoke", "run"))
    p.add_argument("--output", required=True, type=Path)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref", "project-id", "region"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--repo-url", default=base.REPO_URL)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2)+"\n")


if __name__ == "__main__":
    main()
