#!/usr/bin/env python3
"""One CPU job: seed-0 pilot gate, then remaining seeds; no new full training."""
import argparse
from copy import copy
import json
from pathlib import Path

try:
    import exp1_grouped_wide_batch as base
except ModuleNotFoundError:
    from gcp import exp1_grouped_wide_batch as base

MODULE = "experiments.fhp.exp28_fhp_rare_state_audit.run"


def actions(args):
    flags = "--smoke" if args.kind == "smoke" else ""
    seeds = "0" if args.kind == "smoke" else "0 1 2"
    return f'''
# Inputs stay outside OUTPUT_ROOT and are never re-uploaded as results.
AUDIT_ROOT="$OUTPUT_ROOT/audit"
timeout 3600 python -m {MODULE} panel --bucket "$BUCKET_ROOT" --output "$AUDIT_ROOT" {flags}
timeout 600 python -m {MODULE} fetch-opponent --bucket "$BUCKET_ROOT" --output "$WORK_ROOT/frozen_exp9.pkl"
for AUDIT_SEED in {seeds}; do
  for AUDIT_HOURS in 24 48; do
    SOURCE_DIR="$WORK_ROOT/sources/seed_${{AUDIT_SEED}}_${{AUDIT_HOURS}}h"
    timeout 1800 python -m {MODULE} fetch --bucket "$BUCKET_ROOT" --seed "$AUDIT_SEED" --hours "$AUDIT_HOURS" --output "$SOURCE_DIR"
    timeout {7200 if args.kind == 'smoke' else 21600} python -m {MODULE} offline --seed "$AUDIT_SEED" --hours "$AUDIT_HOURS" \
      --source "$SOURCE_DIR" --root "$AUDIT_ROOT" --output "$AUDIT_ROOT/workers/seed_$AUDIT_SEED/offline_${{AUDIT_HOURS}}h" {flags}
    gcloud storage rsync --recursive --exclude='.*\\.tmp$' "$AUDIT_ROOT" "$BUCKET_ROOT/$RUN_ID/audit"
  done
  SOURCE_DIR="$WORK_ROOT/sources/seed_${{AUDIT_SEED}}_48h"
  timeout 9000 python -m {MODULE} continue --seed "$AUDIT_SEED" --source "$SOURCE_DIR" --root "$AUDIT_ROOT" \
    --output "$AUDIT_ROOT/workers/seed_$AUDIT_SEED/continuation" {flags}
  timeout {1800 if args.kind == 'smoke' else 14400} python -m {MODULE} gameplay --seed "$AUDIT_SEED" --source "$SOURCE_DIR" --root "$AUDIT_ROOT" \
    --output "$AUDIT_ROOT/workers/seed_$AUDIT_SEED" --opponent "$WORK_ROOT/frozen_exp9.pkl" {flags}
  if [[ "$AUDIT_SEED" == 0 ]]; then
    # Integrity/runtime pilot, NOT a gate requiring positive scientific results.
    timeout 1800 python -m {MODULE} aggregate --output "$AUDIT_ROOT" --pilot {flags}
  fi
  gcloud storage rsync --recursive --exclude='.*\\.tmp$' "$AUDIT_ROOT" "$BUCKET_ROOT/$RUN_ID/audit"
done
timeout 1800 python -m {MODULE} aggregate --output "$AUDIT_ROOT" {flags}
# A small analysis-only download contains reports, per-seed summaries, traces
# and source/candidate provenance; no 5GB source states or model weights.
python -m experiments.fhp.exp28_fhp_rare_state_audit.package --root "$AUDIT_ROOT" --output "$OUTPUT_ROOT/analysis"
echo 'Experiment 28 complete; no downstream training submitted.'
'''


def build_job(args):
    proxy = copy(args)
    proxy.kind, proxy.parallelism = "smoke", 1
    result = base.build_job(proxy)
    bootstrap = (base._bootstrap(args).replace("exp1-fhp", "exp28-rare-state")
        .replace("uv python install 3.11", "uv python install 3.11.16")
        .replace("uv venv --python 3.11 ", "uv venv --python 3.11.16 "))
    script = "#!/usr/bin/env bash\nset -Eeuo pipefail\n"+bootstrap+"\n"+base._diagnostic_wrapper(
        actions(args), "$OUTPUT_ROOT/diagnostics", "$BUCKET_ROOT/$RUN_ID/diagnostics", 64000)
    tail = f'''#!/usr/bin/env bash
set -u
OUT="/workspace/exp28-rare-state-${{BATCH_TASK_RETRY_ATTEMPT:-0}}/output"
REMOTE={base._q(args.bucket_root.rstrip('/')+'/'+args.run_id)}
mkdir -p "$OUT/diagnostics"
(date -u; free -h; df -h; dmesg 2>&1 | tail -100) > "$OUT/diagnostics/final_system.txt"
gcloud storage rsync --recursive --exclude='.*\\.tmp$' "$OUT" "$REMOTE"
'''
    group = result["taskGroups"][0]
    group.update(taskCount=1, parallelism=1)
    group["taskSpec"].update(maxRetryCount=0, maxRunDuration="21600s" if args.kind=="smoke" else "86400s",
        computeResource=dict(cpuMilli=16000,memoryMib=64000),
        runnables=[{"script":{"text":script}},{"alwaysRun":True,"script":{"text":tail}}])
    result["allocationPolicy"]["instances"][0]["policy"].update(machineType="n2-standard-16",
        bootDisk={"sizeGb":250,"type":"pd-balanced"})
    result["labels"]={"experiment":"exp28-rare-state-audit","stage":args.kind}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",required=True,choices=("smoke","run"))
    p.add_argument("--output",type=Path,required=True)
    for name in ("run-id","bucket-root","service-account","repo-ref","project-id","region"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--repo-url",default=base.REPO_URL)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(build_job(args),indent=2)+"\n")


if __name__=="__main__":
    main()
