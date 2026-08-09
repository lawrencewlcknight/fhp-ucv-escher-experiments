#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

# Usage:
#   ./gcp/submit_batch_experiment.sh \
#     JOB_NAME \
#     "PYTHON_EXPERIMENT_COMMAND" \
#     MACHINE_TYPE \
#     MAX_RUN_SECONDS \
#     CPU_MILLI \
#     MEMORY_MIB \
#     BOOT_DISK_SIZE_GB
#
# Examples:
#   n2-standard-2: CPU_MILLI=2000 MEMORY_MIB=8000
#   n2-standard-4: CPU_MILLI=4000 MEMORY_MIB=16000
#   n2-standard-8: CPU_MILLI=8000 MEMORY_MIB=32000

JOB_NAME="$1"
EXPERIMENT_COMMAND="$2"
MACHINE_TYPE="${3:-n2-standard-4}"
MAX_RUN_SECONDS="${4:-50400}"
CPU_MILLI="${5:-4000}"
MEMORY_MIB="${6:-16000}"
BOOT_DISK_SIZE_GB="${7:-100}"
REPO_URL="${REPO_URL:-https://github.com/lawrencewlcknight/fhp-poker-escher-architecture-experiments.git}"

: "${PROJECT_ID:?Set PROJECT_ID first}"
: "${REGION:?Set REGION first}"
: "${BUCKET:?Set BUCKET first}"
: "${SA_EMAIL:?Set SA_EMAIL first}"

JOB_JSON="$(mktemp "/tmp/${JOB_NAME}.XXXXXX.json")"

export JOB_NAME
export EXPERIMENT_COMMAND
export MACHINE_TYPE
export MAX_RUN_SECONDS
export CPU_MILLI
export MEMORY_MIB
export BOOT_DISK_SIZE_GB
export BUCKET
export SA_EMAIL
export JOB_JSON
export REPO_URL

python3 <<'PY'
import json
import os
import shlex

job_json_path = os.environ["JOB_JSON"]
job_name = os.environ["JOB_NAME"]
experiment_command = os.environ["EXPERIMENT_COMMAND"]
experiment_command_literal = shlex.quote(experiment_command)
machine_type = os.environ["MACHINE_TYPE"]
max_run_seconds = os.environ["MAX_RUN_SECONDS"]
cpu_milli = int(os.environ["CPU_MILLI"])
memory_mib = int(os.environ["MEMORY_MIB"])
boot_disk_size_gb = int(os.environ["BOOT_DISK_SIZE_GB"])
bucket = os.environ["BUCKET"]
service_account = os.environ["SA_EMAIL"]
repo_url_literal = shlex.quote(os.environ["REPO_URL"])

script = f"""#!/usr/bin/env bash
set -Euxo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TF_CPP_MIN_LOG_LEVEL=1
EXPERIMENT_COMMAND={experiment_command_literal}
REPO_URL={repo_url_literal}

WORKDIR=/workspace
REPO_DIR="$WORKDIR/fhp-poker-escher-architecture-experiments"
JOB_OUTPUT_DIR="$REPO_DIR/outputs/cloud/{job_name}"
RUN_LOG="$JOB_OUTPUT_DIR/batch_run.log"
RESOURCE_LOG="$JOB_OUTPUT_DIR/resource_snapshots.log"
BOOT_LOG="/tmp/{job_name}_batch_boot.log"
BUCKET_DEST="{bucket}/{job_name}/"
RESOURCE_MONITOR_PID=""

exec > >(tee -a "$BOOT_LOG") 2>&1

echo "Starting job: {job_name}"
echo "Experiment command: $EXPERIMENT_COMMAND"
echo "Requested CPU milli: {cpu_milli}"
echo "Requested memory MiB: {memory_mib}"
echo "Requested boot disk GiB: {boot_disk_size_gb}"

cleanup() {{
  local exit_code="$?"
  set +e

  echo "Batch cleanup trap running with exit code $exit_code"

  if [[ -n "$RESOURCE_MONITOR_PID" ]]; then
    kill "$RESOURCE_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$RESOURCE_MONITOR_PID" >/dev/null 2>&1 || true
  fi

  if [[ -d "$JOB_OUTPUT_DIR" ]]; then
    {{
      echo "Cleanup timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "Job name: {job_name}"
      echo "Exit code: $exit_code"
      echo "Disk usage at cleanup:"
      df -h || true
      echo "Memory at cleanup:"
      free -h || true
      echo "Largest processes at cleanup:"
      ps -eo pid,ppid,pcpu,pmem,rss,vsz,comm --sort=-rss | head -25 || true
    }} | tee -a "$RUN_LOG"

    cat > "$JOB_OUTPUT_DIR/batch_status.json" <<STATUS_JSON
{{
  "job_name": "{job_name}",
  "exit_code": $exit_code,
  "cleanup_timestamp_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "bucket_destination": "$BUCKET_DEST"
}}
STATUS_JSON
  fi

  local upload_code=0
  if [[ -d "$REPO_DIR/outputs" ]]; then
    echo "Uploading outputs to Cloud Storage: $BUCKET_DEST"
    if command -v gcloud >/dev/null 2>&1; then
      gcloud storage cp --recursive "$REPO_DIR/outputs" "$BUCKET_DEST"
      upload_code="$?"
    else
      echo "gcloud command is unavailable; cannot upload outputs."
      upload_code=1
    fi
    echo "Upload exit code: $upload_code"
  else
    echo "No outputs directory found at cleanup; nothing to upload."
  fi

  if [[ "$exit_code" -eq 0 && "$upload_code" -ne 0 ]]; then
    exit_code="$upload_code"
  fi

  echo "Cleanup complete. Exiting with code $exit_code"
  exit "$exit_code"
}}
trap cleanup EXIT
trap 'echo "Received SIGTERM"; exit 143' TERM
trap 'echo "Received SIGINT"; exit 130' INT

start_resource_monitor() {{
  (
    while true; do
      {{
        echo "==== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ===="
        echo "Disk:"
        df -h || true
        echo "Memory:"
        free -h || true
        echo "Largest processes:"
        ps -eo pid,ppid,pcpu,pmem,rss,vsz,comm --sort=-rss | head -25 || true
        echo
      }} >> "$RESOURCE_LOG" 2>&1
      sleep 60
    done
  ) &
  RESOURCE_MONITOR_PID="$!"
  echo "Started resource monitor with PID $RESOURCE_MONITOR_PID"
}}

run_experiment() {{
  local command_exit=0

  echo "Starting experiment command at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  set +e
  bash -o pipefail -c "$EXPERIMENT_COMMAND"
  command_exit="$?"
  set -e
  echo "Experiment command exit code: $command_exit"
  return "$command_exit"
}}

if command -v sudo >/dev/null 2>&1; then
  SUDO=sudo
else
  SUDO=
fi

$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-pip python3-venv python3-dev build-essential time

mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone --depth 1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

mkdir -p "$JOB_OUTPUT_DIR"
cp "$BOOT_LOG" "$RUN_LOG" || true
exec > >(tee -a "$RUN_LOG") 2>&1

echo "Repository commit:"
git rev-parse HEAD || true

export HOME="${{HOME:-/root}}"
export TMPDIR="/tmp"
export PIP_CACHE_DIR="/tmp/pip-cache"
export UV_CACHE_DIR="/tmp/uv-cache"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Log basic machine information for later VM right-sizing.
echo "Machine information:"
nproc || true
free -h || true
df -h || true
lscpu | head -30 || true

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Keep Google Cloud CLI on a runtime it supports. This is separate from the
# ESCHER experiment environment below, which remains Python 3.9.
uv python install 3.10
export CLOUDSDK_PYTHON="$(uv python find 3.10)"
echo "Configured Cloud SDK Python:"
"$CLOUDSDK_PYTHON" --version

# Use Python 3.11 for the FHP PyTorch/OpenSpiel environment.
uv python install 3.11
uv venv --python 3.11 --seed /tmp/fhp-escher-venv
source /tmp/fhp-escher-venv/bin/activate
python --version

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip check || true

mkdir -p "$JOB_OUTPUT_DIR"
start_resource_monitor

if run_experiment; then
  experiment_exit=0
else
  experiment_exit="$?"
fi

deactivate || true

if [[ "$experiment_exit" -ne 0 ]]; then
  echo "Experiment failed with exit code $experiment_exit"
  exit "$experiment_exit"
fi

echo "Experiment completed successfully."
"""

job = {
    "taskGroups": [
        {
            "taskSpec": {
                "runnables": [
                    {
                        "script": {
                            "text": script
                        }
                    }
                ],
                "computeResource": {
                    "cpuMilli": cpu_milli,
                    "memoryMib": memory_mib,
                },
                "maxRetryCount": 0,
                "maxRunDuration": f"{max_run_seconds}s",
            },
            "taskCount": 1,
            "parallelism": 1,
        }
    ],
    "allocationPolicy": {
        "serviceAccount": {
            "email": service_account
        },
        "instances": [
            {
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": "STANDARD",
                    "bootDisk": {
                        "sizeGb": boot_disk_size_gb,
                        "type": "pd-balanced",
                    },
                }
            }
        ],
    },
    "logsPolicy": {
        "destination": "CLOUD_LOGGING"
    },
}

with open(job_json_path, "w", encoding="utf-8") as f:
    json.dump(job, f, indent=2)
PY

echo "Submitting Batch job: ${JOB_NAME}"
echo "Machine type: ${MACHINE_TYPE}"
echo "Max run duration: ${MAX_RUN_SECONDS}s"
echo "CPU milli: ${CPU_MILLI}"
echo "Memory MiB: ${MEMORY_MIB}"
echo "Boot disk GiB: ${BOOT_DISK_SIZE_GB}"
echo "Job config: ${JOB_JSON}"

echo
echo "Script that will run inside Batch:"
echo "-----------------------------------"
python3 - "$JOB_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    job = json.load(f)

print(job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"])
PY
echo "-----------------------------------"
echo

gcloud batch jobs submit "${JOB_NAME}" \
  --location "${REGION}" \
  --config "${JOB_JSON}"

echo "Submitted."
echo "Monitor with:"
echo "  gcloud batch jobs describe ${JOB_NAME} --location ${REGION}"
echo "Outputs will be copied to:"
echo "  ${BUCKET}/${JOB_NAME}/"
