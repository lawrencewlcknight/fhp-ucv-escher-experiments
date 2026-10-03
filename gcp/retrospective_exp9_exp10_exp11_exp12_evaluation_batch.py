#!/usr/bin/env python3
"""One-VM, resumable frozen-policy evaluation of FHP Experiments 9--12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex

REPO_URL = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"


def _q(value):
    return shlex.quote(str(value))


def validate(args):
    for name in ("run_id", "exp9_run_id", "exp10_run_id", "exp11_run_id", "exp12_run_id"):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,53}[a-z0-9]", getattr(args, name)):
            raise ValueError(f"Invalid {name}: use 2-55 lowercase letters, digits or hyphens")
    if len({args.run_id, args.exp9_run_id, args.exp10_run_id, args.exp11_run_id, args.exp12_run_id}) != 5:
        raise ValueError("Evaluation and source run IDs must be distinct")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("Pin REPO_REF to a full 40-character commit SHA")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", args.bucket_root):
        raise ValueError("bucket-root must be a gs:// bucket without a subdirectory")
    if not 1 <= args.max_hours <= 72:
        raise ValueError("max-hours must be between 1 and 72")


def _script(args):
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/fhp-eval9to12-matplotlib UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
REPO_REF={_q(args.repo_ref)}
BUCKET_ROOT={_q(args.bucket_root)}
RUN_ID={_q(args.run_id)}
EXP9_RUN_ID={_q(args.exp9_run_id)}
EXP10_RUN_ID={_q(args.exp10_run_id)}
EXP11_RUN_ID={_q(args.exp11_run_id)}
EXP12_RUN_ID={_q(args.exp12_run_id)}
SMOKE_ONLY={1 if args.smoke_only else 0}
RESUME={1 if args.resume else 0}
WORK_ROOT=/workspace/fhp-eval9to12
REPOSITORY="$WORK_ROOT/repository"
INPUT_ROOT="$WORK_ROOT/input"
OUTPUT_ROOT="$WORK_ROOT/output"
VENV_ROOT=/tmp/fhp-eval9to12-venv
MONITOR_PID=""
SYNC_PID=""

sync_outputs() {{
  gcloud storage rsync --recursive --exclude='.*[.]tmp$' \
    "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID"
}}
cleanup() {{
  exit_code="$?"
  set +e
  for child in "$SYNC_PID" "$MONITOR_PID"; do
    if [[ -n "$child" ]]; then kill "$child" 2>/dev/null; wait "$child" 2>/dev/null; fi
  done
  if [[ -x "$VENV_ROOT/bin/python" && -d "$REPOSITORY/fhp_escher" ]]; then
    (cd "$REPOSITORY" && "$VENV_ROOT/bin/python" -m fhp_escher.batch_diagnostics finalize \
      --snapshots "$OUTPUT_ROOT/resource_snapshots.jsonl" \
      --output "$OUTPUT_ROOT/batch_diagnostics.json" --status-output "$OUTPUT_ROOT/batch_status.json" \
      --failure-root "$OUTPUT_ROOT" --exit-code "$exit_code" --experiment-exit-code "$exit_code" \
      --requested-memory-mib 62000 --job-name "$RUN_ID" --bucket-destination "$BUCKET_ROOT/$RUN_ID") || true
  fi
  if [[ -d "$OUTPUT_ROOT" ]]; then sync_outputs || true; fi
  exit "$exit_code"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p "$INPUT_ROOT/exp9" "$INPUT_ROOT/exp10" "$INPUT_ROOT/exp11" "$INPUT_ROOT/exp12" "$OUTPUT_ROOT" "$MPLCONFIGDIR"
if [[ "$RESUME" == 1 ]]; then
  gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$BUCKET_ROOT/$RUN_ID" "$OUTPUT_ROOT"
fi
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
git clone --filter=blob:none {_q(args.repo_url)} "$REPOSITORY"
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
python -m fhp_escher.batch_diagnostics monitor \
  --output "$OUTPUT_ROOT/resource_snapshots.jsonl" --interval-seconds 30 --cloud-log-every 4 &
MONITOR_PID="$!"
# Exclude continuation states, replay and the duplicate final-policy copy.
gcloud storage rsync --recursive \
  --exclude='.*training_states.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \
  "$BUCKET_ROOT/$EXP9_RUN_ID/workers" "$INPUT_ROOT/exp9/workers"
gcloud storage rsync --recursive \
  --exclude='.*training_states.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \
  "$BUCKET_ROOT/$EXP10_RUN_ID/workers" "$INPUT_ROOT/exp10/workers"
gcloud storage rsync --recursive \
  --exclude='.*training_states.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \
  "$BUCKET_ROOT/$EXP11_RUN_ID/workers" "$INPUT_ROOT/exp11/workers"
gcloud storage rsync --recursive \
  --exclude='.*training_states.*|.*reservoir.*|.*final_policy_checkpoint[.]pkl$' \
  "$BUCKET_ROOT/$EXP12_RUN_ID/workers" "$INPUT_ROOT/exp12/workers"
# Atomic task-result JSON files are uploaded periodically and on exit.
while sleep 300; do sync_outputs || echo "WARNING: periodic output upload failed" >&2; done &
SYNC_PID="$!"
SMOKE_RESUME=()
FULL_RESUME=()
if [[ -f "$OUTPUT_ROOT/smoke/evaluation_manifest.json" ]]; then SMOKE_RESUME=(--resume); fi
if [[ -f "$OUTPUT_ROOT/analysis/evaluation_manifest.json" ]]; then FULL_RESUME=(--resume); fi
python -m experiments.fhp.retrospective_exp9_exp10_exp11_exp12_evaluation.run \
  --exp9-run "$INPUT_ROOT/exp9" --exp10-run "$INPUT_ROOT/exp10" \
  --exp11-run "$INPUT_ROOT/exp11" --exp12-run "$INPUT_ROOT/exp12" --output-dir "$OUTPUT_ROOT/smoke" \
  --workers 2 --rule-deals 4 --lbr-deals 2 --lbr-rollouts 16 --lbr-shard-deals 2 \
  --crossplay-deals 10 --smoke "${{SMOKE_RESUME[@]}}"
if [[ "$SMOKE_ONLY" != 1 ]]; then
python -m experiments.fhp.retrospective_exp9_exp10_exp11_exp12_evaluation.run \
  --exp9-run "$INPUT_ROOT/exp9" --exp10-run "$INPUT_ROOT/exp10" \
  --exp11-run "$INPUT_ROOT/exp11" --exp12-run "$INPUT_ROOT/exp12" --output-dir "$OUTPUT_ROOT/analysis" \
  --workers 16 --rule-deals 10000 --lbr-deals 1000 --lbr-rollouts 4096 --lbr-shard-deals 10 \
  --crossplay-deals 50000 --base-seed 20260922 "${{FULL_RESUME[@]}}"
fi
# Serialize final upload with the background uploader.
kill "$SYNC_PID" 2>/dev/null || true
wait "$SYNC_PID" 2>/dev/null || true
SYNC_PID=""
sync_outputs
'''


def build_job(args):
    validate(args)
    return {
        "taskGroups": [{"taskCount": 1, "parallelism": 1, "taskCountPerNode": 1,
                        "taskSpec": {"runnables": [{"script": {"text": _script(args)}}],
                                     "computeResource": {"cpuMilli": 16000, "memoryMib": 62000},
                                     "maxRetryCount": 0, "maxRunDuration": f"{args.max_hours * 3600}s"}}],
        "allocationPolicy": {"serviceAccount": {"email": args.service_account},
                             "instances": [{"policy": {"machineType": "n2-standard-16",
                                                        "provisioningModel": "STANDARD",
                                                        "bootDisk": {"sizeGb": 100, "type": "pd-balanced"}}}]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"workload": "fhp-eval9to12", "stage": "evaluation"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "exp9-run-id", "exp10-run-id", "exp11-run-id", "exp12-run-id",
                 "bucket-root", "service-account", "repo-ref"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=int, default=48)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    job = build_job(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(job, indent=2) + "\n")


if __name__ == "__main__":
    main()
