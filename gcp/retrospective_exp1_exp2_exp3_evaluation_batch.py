#!/usr/bin/env python3
"""Build the single-VM Batch job for the FHP Experiment 1--3 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


REPO_URL = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
MAX_RUN_DURATION_SECONDS = 18 * 60 * 60


def _q(value) -> str:
    return shlex.quote(str(value))


def _script(args) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/fhp-evaluation-matplotlib
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
UV_INSTALL_DIR=/tmp/uv-bin

REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
BUCKET_ROOT={_q(args.bucket_root.rstrip('/'))}
RUN_ID={_q(args.run_id)}
EXP1_RUN_ID={_q(args.exp1_run_id)}
EXP2_RUN_ID={_q(args.exp2_run_id)}
EXP3_RUN_ID={_q(args.exp3_run_id)}
REFERENCE_RUN_ID={_q(args.reference_run_id)}
WORK_ROOT=/workspace/fhp-retrospective-evaluation-123
REPOSITORY="$WORK_ROOT/repository"
INPUT_ROOT="$WORK_ROOT/input"
OUTPUT_ROOT="$WORK_ROOT/output"
VENV_ROOT=/tmp/fhp-retrospective-evaluation-123-venv
RESOURCE_LOG="$OUTPUT_ROOT/resource_snapshots.jsonl"
MONITOR_PID=""

cleanup() {{
  exit_code="$?"
  set +e
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
  fi
  if [[ -x "$VENV_ROOT/bin/python" && -d "$REPOSITORY/fhp_escher" ]]; then
    "$VENV_ROOT/bin/python" -m fhp_escher.batch_diagnostics finalize \
      --snapshots "$RESOURCE_LOG" \
      --output "$OUTPUT_ROOT/batch_diagnostics.json" \
      --status-output "$OUTPUT_ROOT/batch_status.json" \
      --failure-root "$OUTPUT_ROOT" --exit-code "$exit_code" \
      --experiment-exit-code "$exit_code" --requested-memory-mib 30000 \
      --job-name "$RUN_ID" --bucket-destination "$BUCKET_ROOT/$RUN_ID" || true
  fi
  if [[ -d "$OUTPUT_ROOT" ]]; then
    gcloud storage rsync --recursive "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID" || true
  fi
  exit "$exit_code"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
mkdir -p "$WORK_ROOT" "$INPUT_ROOT/exp1" "$INPUT_ROOT/exp2" "$INPUT_ROOT/exp3" \
  "$INPUT_ROOT/reference" "$OUTPUT_ROOT" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
  "$UV_INSTALL_DIR" "$MPLCONFIGDIR"
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

python -m fhp_escher.batch_diagnostics monitor \
  --output "$RESOURCE_LOG" --interval-seconds 30 --cloud-log-every 4 &
MONITOR_PID="$!"

gcloud storage rsync --recursive --exclude='.*training_states.*' \
  "$BUCKET_ROOT/$EXP1_RUN_ID/workers" "$INPUT_ROOT/exp1/workers"
gcloud storage rsync --recursive --exclude='.*training_states.*' \
  "$BUCKET_ROOT/$EXP2_RUN_ID/workers" "$INPUT_ROOT/exp2/workers"
gcloud storage rsync --recursive --exclude='.*training_states.*' \
  "$BUCKET_ROOT/$EXP3_RUN_ID/workers" "$INPUT_ROOT/exp3/workers"
gcloud storage rsync --recursive \
  "$BUCKET_ROOT/$REFERENCE_RUN_ID/analysis" "$INPUT_ROOT/reference"

python -m experiments.fhp.retrospective_exp1_exp2_exp3_evaluation.run \
  --exp1-run "$INPUT_ROOT/exp1" --exp2-run "$INPUT_ROOT/exp2" \
  --exp3-run "$INPUT_ROOT/exp3" --reference-analysis "$INPUT_ROOT/reference" \
  --output-dir "$OUTPUT_ROOT/smoke" --workers 2 --rule-deals 4 \
  --lbr-deals 2 --lbr-rollouts 16 --lbr-shard-deals 2 \
  --crossplay-deals 10 --smoke

python -m experiments.fhp.retrospective_exp1_exp2_exp3_evaluation.run \
  --exp1-run "$INPUT_ROOT/exp1" --exp2-run "$INPUT_ROOT/exp2" \
  --exp3-run "$INPUT_ROOT/exp3" --reference-analysis "$INPUT_ROOT/reference" \
  --output-dir "$OUTPUT_ROOT/analysis" --workers 8 \
  --rule-deals {int(args.rule_deals)} --lbr-deals {int(args.lbr_deals)} \
  --lbr-rollouts {int(args.lbr_rollouts)} \
  --lbr-shard-deals {int(args.lbr_shard_deals)} \
  --crossplay-deals {int(args.crossplay_deals)} --base-seed {int(args.base_seed)}

gcloud storage rsync --recursive "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID"
"""


def build_job(args) -> dict:
    return {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"script": {"text": _script(args)}}],
                "computeResource": {"cpuMilli": 8_000, "memoryMib": 30_000},
                "maxRetryCount": 0,
                "maxRunDuration": f"{MAX_RUN_DURATION_SECONDS}s",
            },
            "taskCount": 1,
            "parallelism": 1,
            "taskCountPerNode": 1,
        }],
        "allocationPolicy": {
            "serviceAccount": {"email": args.service_account},
            "instances": [{"policy": {
                "machineType": "n2-standard-8",
                "provisioningModel": "STANDARD",
                "bootDisk": {"sizeGb": 150, "type": "pd-balanced"},
            }}],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"workload": "fhp-retrospective-eval-123", "stage": "analysis"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exp1-run-id", required=True)
    parser.add_argument("--exp2-run-id", required=True)
    parser.add_argument("--exp3-run-id", required=True)
    parser.add_argument("--reference-run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--rule-deals", type=int, default=10_000)
    parser.add_argument("--lbr-deals", type=int, default=1_000)
    parser.add_argument("--lbr-rollouts", type=int, default=4_096)
    parser.add_argument("--lbr-shard-deals", type=int, default=10)
    parser.add_argument("--crossplay-deals", type=int, default=50_000)
    parser.add_argument("--base-seed", type=int, default=20_260_922)
    args = parser.parse_args()
    for name in ("rule_deals", "lbr_deals", "lbr_rollouts", "lbr_shard_deals", "crossplay_deals"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
