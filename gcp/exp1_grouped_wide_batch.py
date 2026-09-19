#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for the new FHP Experiment 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


REPO_URL = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
MODULE = "experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.run"
TASK_COUNT = 3


def _q(value) -> str:
    return shlex.quote(str(value))


def _bootstrap(args) -> str:
    return f"""
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
UV_INSTALL_DIR=/tmp/uv-bin

REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
BUCKET_ROOT={_q(args.bucket_root.rstrip('/'))}
RUN_ID={_q(args.run_id)}
RETRY_ATTEMPT="${{BATCH_TASK_RETRY_ATTEMPT:-0}}"
WORK_ROOT="/workspace/exp1-fhp-$RETRY_ATTEMPT"
REPOSITORY="$WORK_ROOT/repository"
OUTPUT_ROOT="$WORK_ROOT/output"
VENV_ROOT="/tmp/exp1-fhp-venv-$RETRY_ATTEMPT"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
mkdir -p "$WORK_ROOT" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_INSTALL_DIR"
git clone --filter=blob:none "$REPO_URL" "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
curl -LsSf https://astral.sh/uv/install.sh | \
  env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
export PATH="$UV_INSTALL_DIR:$PATH"
uv python install 3.11
uv venv --python 3.11 --seed "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r "$REPOSITORY/requirements.txt"
python -m pip install --no-cache-dir --no-build-isolation -e "$REPOSITORY"
python -m pip check
cd "$REPOSITORY"
""".strip()


def _controller_script(args) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
CONTROLLER_ACTION={_q(args.controller_action)}
WORK_ROOT="/workspace/exp1-controller-${{BATCH_TASK_RETRY_ATTEMPT:-0}}"
REPOSITORY="$WORK_ROOT/repository"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git ca-certificates python3
mkdir -p "$WORK_ROOT"
git clone --filter=blob:none "$REPO_URL" "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
cd "$REPOSITORY"

export PROJECT_ID={_q(args.project_id)}
export REGION={_q(args.region)}
export BUCKET={_q(args.bucket_root.rstrip('/'))}
export SA_EMAIL={_q(args.service_account)}
export REPO_REF={_q(args.repo_ref)}
export RUN_ID={_q(args.run_id)}
export PARALLELISM={_q(args.parallelism)}
export EXP1_REMOTE_CONTROLLER=1
exec bash gcp/run_exp1_grouped_wide.sh "$CONTROLLER_ACTION"
"""


def _diagnostic_wrapper(action: str, output_dir: str, remote: str, memory_mib: int) -> str:
    return f"""
DIAGNOSTIC_DIR="{output_dir}"
REMOTE_DIAGNOSTIC_DIR="{remote}"
RESOURCE_LOG="$DIAGNOSTIC_DIR/resource_snapshots.jsonl"
DIAGNOSTICS_JSON="$DIAGNOSTIC_DIR/batch_diagnostics.json"
MONITOR_PID=""
mkdir -p "$DIAGNOSTIC_DIR"

cleanup() {{
  exit_code="$?"
  set +e
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  python -m fhp_escher.batch_diagnostics finalize \
    --snapshots "$RESOURCE_LOG" --output "$DIAGNOSTICS_JSON" \
    --status-output "$DIAGNOSTIC_DIR/batch_status.json" \
    --failure-root "$OUTPUT_ROOT" --exit-code "$exit_code" \
    --experiment-exit-code "$exit_code" \
    --requested-memory-mib {memory_mib} --job-name "$RUN_ID" \
    --bucket-destination "$REMOTE_DIAGNOSTIC_DIR" || true
  free -h || true
  df -h || true
  ps -eo pid,ppid,pcpu,pmem,rss,vsz,comm --sort=-rss | head -25 || true
  dmesg 2>&1 | grep -E -i 'out of memory|oom-kill|killed process' | tail -100 || true
  gcloud storage rsync --recursive "$DIAGNOSTIC_DIR" "$REMOTE_DIAGNOSTIC_DIR" || true
  exit "$exit_code"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -m fhp_escher.batch_diagnostics monitor \
  --output "$RESOURCE_LOG" --interval-seconds 15 --cloud-log-every 4 &
MONITOR_PID="$!"
{action}
""".strip()


def _script(args) -> str:
    if args.kind == "controller":
        return _controller_script(args)
    bootstrap = _bootstrap(args)
    if args.kind == "smoke":
        action = f"python -m {MODULE} smoke --output-root \"$OUTPUT_ROOT\" --no-resume"
        wrapped = _diagnostic_wrapper(
            action,
            "$OUTPUT_ROOT/smoke_diagnostics",
            "$BUCKET_ROOT/$RUN_ID/smoke/diagnostics",
            30_000,
        )
        tail = 'gcloud storage rsync --recursive "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID/smoke"'
    elif args.kind == "train":
        action = f"""
TASK_INDEX="${{BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}}"
TASK_METADATA="$(python - "$TASK_INDEX" <<'PY' | sed -n 's/^EXP1_TASK_METADATA //p' | tail -n 1
import sys
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.config import (
    PRODUCTION_SEEDS, task_schedule,
)
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.aggregate import task_name
index = int(sys.argv[1])
_, seed = task_schedule(PRODUCTION_SEEDS)[index]
print("EXP1_TASK_METADATA", seed, task_name(index, seed))
PY
)"
read -r SOURCE_SEED TASK_NAME EXTRA_METADATA <<< "$TASK_METADATA"
EXPECTED_TASK_NAME="task_$(printf '%03d' "$TASK_INDEX")_grouped_wide_ucv_escher_seed_$SOURCE_SEED"
if [[ ! "$SOURCE_SEED" =~ ^[0-9]+$ \
      || "$TASK_NAME" != "$EXPECTED_TASK_NAME" \
      || -n "$EXTRA_METADATA" ]]; then
  echo "Invalid Experiment 1 task metadata: $TASK_METADATA" >&2
  exit 2
fi
REMOTE_TASK="$BUCKET_ROOT/$RUN_ID/workers/$TASK_NAME"
export EXP1_REMOTE_TASK_URI="$REMOTE_TASK"
mkdir -p "$OUTPUT_ROOT/workers/$TASK_NAME"
if gcloud storage ls "$REMOTE_TASK/SUCCESS.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$REMOTE_TASK" "$OUTPUT_ROOT/workers/$TASK_NAME"
elif gcloud storage ls "$REMOTE_TASK/training_states/**" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$REMOTE_TASK" "$OUTPUT_ROOT/workers/$TASK_NAME"
fi
python -m {MODULE} worker --task-index "$TASK_INDEX" --output-root "$OUTPUT_ROOT"
""".strip()
        wrapped = _diagnostic_wrapper(
            action,
            "$OUTPUT_ROOT/task_diagnostics/task_${BATCH_TASK_INDEX:-0}",
            "$BUCKET_ROOT/$RUN_ID/task_diagnostics/task_${BATCH_TASK_INDEX:-0}",
            30_000,
        )
        tail = ":"
    elif args.kind == "aggregate":
        action = f"""
mkdir -p "$OUTPUT_ROOT/workers"
gcloud storage rsync --recursive \
  --exclude='(^|/)training_states(/|$)' \
  "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
python -m {MODULE} aggregate --output-root "$OUTPUT_ROOT"
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
""".strip()
        wrapped = action
        tail = ":"
    else:  # pragma: no cover
        raise ValueError(args.kind)
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{wrapped}\n{tail}\n"


def build_job(args) -> dict:
    task_count = TASK_COUNT if args.kind == "train" else 1
    parallelism = min(task_count, args.parallelism)
    if args.kind == "train":
        duration, cpu, memory, machine, disk = "129600s", 8_000, 30_000, "n2-standard-8", 200
    elif args.kind == "smoke":
        duration, cpu, memory, machine, disk = "7200s", 8_000, 30_000, "n2-standard-8", 100
    elif args.kind == "controller":
        duration, cpu, memory, machine, disk = "604800s", 1_000, 1_500, "e2-small", 30
    else:
        duration, cpu, memory, machine, disk = "14400s", 8_000, 30_000, "n2-standard-8", 100
    return {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [{"script": {"text": _script(args)}}],
                    "computeResource": {"cpuMilli": cpu, "memoryMib": memory},
                    "maxRetryCount": (
                        2
                        if args.kind == "controller"
                        else (1 if args.kind == "train" else 0)
                    ),
                    "maxRunDuration": duration,
                },
                "taskCount": task_count,
                "parallelism": parallelism,
                "taskCountPerNode": 1,
            }
        ],
        "allocationPolicy": {
            "serviceAccount": {"email": args.service_account},
            "instances": [
                {
                    "policy": {
                        "machineType": machine,
                        "provisioningModel": "STANDARD",
                        "bootDisk": {"sizeGb": disk, "type": "pd-balanced"},
                    }
                }
            ],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": "exp1-fhp-grouped-wide", "stage": args.kind},
    }


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
