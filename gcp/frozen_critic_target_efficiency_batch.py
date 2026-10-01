#!/usr/bin/env python3
"""Build bounded, remote-only Batch jobs for frozen critic-cache benchmarks."""
import argparse
import json
from pathlib import Path
import shlex

MODULE = "experiments.fhp.exp5_fhp_critic_target_cache.run"
REPO_URL = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
PYTHON_VERSION = "3.11"
EXPERIMENT_ID = 5
LAUNCHER = "gcp/run_frozen_critic_target_efficiency.sh"


def q(value):
    return shlex.quote(str(value))


def bootstrap(a, controller=False):
    common = f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/cache
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
WORK=/workspace/critic-cache
REPOSITORY="$WORK/repository"
OUTPUT="$WORK/output"
SOURCE="$WORK/source"
BUCKET_ROOT={q(a.bucket)}
RUN_ID={q(a.run_id)}
SOURCE_RUN_ID={q(a.source_run_id)}
mkdir -p "$WORK" "$OUTPUT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if command -v sudo >/dev/null; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
git clone --filter=blob:none {q(a.repo_url)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach {q(a.repo_ref)}
cd "$REPOSITORY"
"""
    if controller:
        return common
    return common + f"""
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install {PYTHON_VERSION}
uv venv --python {PYTHON_VERSION} --seed /tmp/critic-cache-venv
source /tmp/critic-cache-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip check
"""


def script(a):
    if a.kind == "controller":
        return bootstrap(a, True) + f"""
export PROJECT_ID={q(a.project)} REGION={q(a.region)} BUCKET={q(a.bucket)}
export SA_EMAIL={q(a.service_account)} REPO_REF={q(a.repo_ref)}
export SOURCE_RUN_ID={q(a.source_run_id)} RUN_ID={q(a.run_id)}
export CRITIC_CACHE_REMOTE_CONTROLLER=1
exec bash {LAUNCHER} {q(a.controller_action)}
"""
    text = bootstrap(a)
    if a.kind == "aggregate":
        return text + f"""
mkdir -p "$OUTPUT/workers"
gcloud storage rsync --recursive "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT/workers"
python -m {MODULE} aggregate --output-root "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
"""
    index = "0" if a.kind == "smoke" else "${BATCH_TASK_INDEX:?Missing task index}"
    remote = "$BUCKET_ROOT/$RUN_ID/smoke" if a.kind == "smoke" else "$BUCKET_ROOT/$RUN_ID"
    action = "smoke" if a.kind == "smoke" else "worker"
    return text + f"""
INDEX="{index}"
[[ "$INDEX" =~ ^[012]$ ]] || exit 2
REMOTE="{remote}"
finish() {{
  status="$?"
  set +e
  gcloud storage rsync --recursive "$OUTPUT" "$REMOTE"
  exit "$status"
}}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -m {MODULE} fetch-source --bucket "$BUCKET_ROOT" --source-run-id "$SOURCE_RUN_ID" \\
  --task-index "$INDEX" --directory "$SOURCE"
SEED="$(python -c 'from {MODULE} import SPEC; import sys; print(SPEC["seeds"][int(sys.argv[1])])' "$INDEX" | tail -n 1)"
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 2
export CRITIC_CACHE_REMOTE_WORKER="$REMOTE/workers/seed_$SEED"
mkdir -p "$OUTPUT/workers/seed_$SEED"
if gcloud storage ls "$CRITIC_CACHE_REMOTE_WORKER/manifest.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$CRITIC_CACHE_REMOTE_WORKER" "$OUTPUT/workers/seed_$SEED"
fi
python -m {MODULE} {action} --source-worker "$SOURCE" --task-index "$INDEX" --output-root "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT" "$REMOTE"
"""


def build_job(a):
    count = 3 if a.kind == "worker" else 1
    controller = a.kind == "controller"
    return {
        "taskGroups": [{"taskCount": count, "parallelism": count, "taskCountPerNode": 1,
                        "taskSpec": {"runnables": [{"script": {"text": script(a)}}],
                                     "computeResource": {"cpuMilli": 1000 if controller else 8000,
                                                         "memoryMib": 1500 if controller else 30000},
                                     "maxRetryCount": 0,
                                     "maxRunDuration": {"controller": "64800s", "smoke": "7200s",
                                                        "worker": "43200s", "aggregate": "3600s"}[a.kind]}}],
        "allocationPolicy": {"serviceAccount": {"email": a.service_account},
                             "instances": [{"policy": {"machineType": "e2-small" if controller else "n2-standard-8",
                                                       "provisioningModel": "STANDARD",
                                                       "bootDisk": {"type": "pd-balanced", "sizeGb": 30 if controller else 100}}}]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": f"exp{EXPERIMENT_ID}-critic-cache", "stage": a.kind},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("controller", "smoke", "worker", "aggregate"), required=True)
    for name in ("project", "region", "bucket", "run-id", "source-run-id", "service-account", "repo-ref"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--repo-url", default=REPO_URL)
    p.add_argument("--controller-action", choices=("orchestrate", "orchestrate-resume"), default="orchestrate")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(build_job(a), indent=2) + "\n")


if __name__ == "__main__":
    main()
