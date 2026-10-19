#!/usr/bin/env python3
"""Standard-library-only builder: one CPU VM, smoke then frozen-policy evaluation."""
import argparse
import json
from pathlib import Path
import re
import shlex

REPO_URL = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
SOURCES = {10: "exp10-features-20261001-161740", 20: "exp20-critic24-20261006-164357",
           36: "exp36-rich2500-20261018-205125"}


def validate(args):
    ids = [args.run_id, *(getattr(args, f"exp{n}_run_id") for n in SOURCES)]
    if any(not re.fullmatch(r"[a-z][a-z0-9-]{0,43}[a-z0-9]", v) for v in ids):
        raise ValueError("Run IDs must be 2-45 lowercase letters, digits or hyphens (leave room for retry suffix)")
    if len(set(ids)) != 4:
        raise ValueError("Source and evaluation run IDs must be distinct")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("Pin REPO_REF to a full pushed commit SHA")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", args.bucket_root):
        raise ValueError("bucket-root must be a gs:// bucket without a subdirectory")
    if not 1 <= args.max_hours <= 72:
        raise ValueError("max-hours must be 1-72")


def script(args):
    q = shlex.quote
    source_downloads = "\n".join(f'''gcloud storage rsync --recursive \\
  --exclude='.*training_states.*|.*continuation_inputs.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \\
  "$BUCKET_ROOT/{getattr(args, f'exp{n}_run_id')}/workers" "$INPUT_ROOT/exp{n}/workers"''' for n in SOURCES)
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/fhp-exp37-matplotlib UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
REPO_REF={q(args.repo_ref)}
BUCKET_ROOT={q(args.bucket_root)}
RUN_ID={q(args.run_id)}
WORK_ROOT=/workspace/fhp-exp37
REPOSITORY="$WORK_ROOT/repository"
INPUT_ROOT="$WORK_ROOT/input"
OUTPUT_ROOT="$WORK_ROOT/output"
VENV_ROOT=/tmp/fhp-exp37-venv
MONITOR_PID=""
SYNC_PID=""
sync_outputs() {{
  gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID"
}}
cleanup() {{
  exit_code="$?"
  set +e
  for child in "$SYNC_PID" "$MONITOR_PID"; do
    if [[ -n "$child" ]]; then kill "$child" 2>/dev/null; wait "$child" 2>/dev/null; fi
  done
  if [[ -x "$VENV_ROOT/bin/python" && -d "$REPOSITORY/fhp_escher" ]]; then
    (cd "$REPOSITORY" && "$VENV_ROOT/bin/python" -m fhp_escher.batch_diagnostics finalize \\
      --snapshots "$OUTPUT_ROOT/resource_snapshots.jsonl" \\
      --output "$OUTPUT_ROOT/batch_diagnostics.json" --status-output "$OUTPUT_ROOT/batch_status.json" \\
      --failure-root "$OUTPUT_ROOT" --exit-code "$exit_code" --experiment-exit-code "$exit_code" \\
      --requested-memory-mib 62000 --job-name "$RUN_ID" --bucket-destination "$BUCKET_ROOT/$RUN_ID") || true
  fi
  if [[ -d "$OUTPUT_ROOT" ]]; then sync_outputs || true; fi
  exit "$exit_code"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -p "$INPUT_ROOT/exp10" "$INPUT_ROOT/exp20" "$INPUT_ROOT/exp36" "$OUTPUT_ROOT" "$MPLCONFIGDIR"
if [[ {int(args.resume)} == 1 ]]; then
  gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$BUCKET_ROOT/$RUN_ID" "$OUTPUT_ROOT"
fi
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
git clone --filter=blob:none {q(args.repo_url)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
[[ "$(git -C "$REPOSITORY" rev-parse HEAD)" == "$REPO_REF" ]]
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install 3.11.16
uv venv --python 3.11.16 --seed "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r "$REPOSITORY/requirements.txt"
python -m pip install --no-cache-dir --no-build-isolation -e "$REPOSITORY"
python -m pip check
cd "$REPOSITORY"
python -m fhp_escher.batch_diagnostics monitor \\
  --output "$OUTPUT_ROOT/resource_snapshots.jsonl" --interval-seconds 30 --cloud-log-every 4 &
MONITOR_PID="$!"
while sleep 300; do sync_outputs || echo "WARNING: periodic output upload failed" >&2; done &
SYNC_PID="$!"
{source_downloads}
SMOKE_RESUME=()
FULL_RESUME=()
if [[ -f "$OUTPUT_ROOT/smoke/evaluation_manifest.json" ]]; then SMOKE_RESUME=(--resume); fi
if [[ -f "$OUTPUT_ROOT/analysis/evaluation_manifest.json" ]]; then FULL_RESUME=(--resume); fi
python -m experiments.fhp.exp37_fhp_critic_budget_evaluation.run \\
  --exp10-run "$INPUT_ROOT/exp10" --exp20-run "$INPUT_ROOT/exp20" --exp36-run "$INPUT_ROOT/exp36" \\
  --output-dir "$OUTPUT_ROOT/smoke" --workers 2 --rule-deals 4 --lbr-deals 2 --lbr-rollouts 16 \\
  --lbr-shard-deals 2 --crossplay-deals 10 --crossplay-shard-deals 5 --smoke "${{SMOKE_RESUME[@]}}"
sync_outputs
if [[ {int(args.smoke_only)} != 1 ]]; then
  python -m experiments.fhp.exp37_fhp_critic_budget_evaluation.run \\
    --exp10-run "$INPUT_ROOT/exp10" --exp20-run "$INPUT_ROOT/exp20" --exp36-run "$INPUT_ROOT/exp36" \\
    --output-dir "$OUTPUT_ROOT/analysis" --workers 16 --rule-deals 10000 --lbr-deals 1000 \\
    --lbr-rollouts 4096 --lbr-shard-deals 10 --crossplay-deals 50000 --crossplay-shard-deals 5000 \\
    "${{FULL_RESUME[@]}}"
fi
kill "$SYNC_PID" 2>/dev/null || true
wait "$SYNC_PID" 2>/dev/null || true
SYNC_PID=""
sync_outputs
'''


def build_job(args):
    validate(args)
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1,
        taskSpec=dict(runnables=[dict(script=dict(text=script(args)))],
            computeResource=dict(cpuMilli=16000, memoryMib=62000), maxRetryCount=0,
            maxRunDuration=f"{args.max_hours*3600}s"))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
            instances=[dict(policy=dict(machineType="n2-standard-16", provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=100, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"), labels=dict(workload="fhp-exp37", stage="evaluation"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref"):
        p.add_argument("--"+name, required=True)
    for n, run in SOURCES.items():
        p.add_argument(f"--exp{n}-run-id", default=run)
    p.add_argument("--repo-url", default=REPO_URL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-hours", type=int, default=48)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main():
    args = parser().parse_args()
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2)+"\n")


if __name__ == "__main__":
    main()
