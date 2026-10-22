#!/usr/bin/env python3
"""Dependency-free builder for the selected 48h three-algorithm league."""
import argparse
import json
from pathlib import Path
import re
import shlex

UCV_URL = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
VR_URL = "https://github.com/lawrencewlcknight/fhp-vr-deep-experiments.git"
SD_URL = "https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git"
VR_REF = "236bf0c5e94ae90ce57545d82bbabc8bec9b7f07"
SD_REF = "bb689252d6b322adb3de6102d095a49c3ed87250"
RUNS = {"ucv": "exp38-rich48-20261019-230016", "ucv_source36": "exp36-rich2500-20261018-205125",
        "sd": "sdcfr6-48h-20261002-161544", "vr": "vr6-ray48-20261003-221425"}
ALGORITHMS = {"ucv": "strategic_ae_critic2500_ucv_escher",
              "ucv_source36": "strategic_ae_critic2500_ucv_escher",
              "sd": "parallel_structured_uniform_sd_cfr_48h", "vr": "lossless_vr_deep_pdcfr_plus_ray8_48h"}


def validate(args):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,43}[a-z0-9]", args.run_id) or args.run_id in RUNS.values():
        raise ValueError("Use a fresh 2-45 character Batch run ID, distinct from every source")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("Pin REPO_REF to a full pushed commit SHA")
    for key in ("bucket", "ucv_bucket", "sd_bucket", "vr_bucket"):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", getattr(args, key)):
            raise ValueError(f"{key} must be a gs:// bucket without a subdirectory")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+[.]iam[.]gserviceaccount[.]com", args.service_account):
        raise ValueError("Invalid service account")


def source_uris(args):
    return {key: f"{getattr(args, ('ucv' if key == 'ucv_source36' else key)+'_bucket')}/{run}"
            for key, run in RUNS.items()}


def upload_function():
    # Snapshot eligible completion markers before transferring data. A marker
    # cannot overtake the files it certifies if scoring finishes mid-upload.
    return '''sync_outputs() {
  (
    flock -w 180 9 || exit 1
    markers="$(mktemp -d "$WORK/markers.XXXXXX")" || exit 1
    for relative in analysis/SUCCESS.json analysis/smoke/SUCCESS.json; do
      if [[ -f "$OUTPUT/$relative" ]]; then
        mkdir -p "$markers/$(dirname "$relative")"
        cp "$OUTPUT/$relative" "$markers/$relative" || exit 1
      fi
    done
    timeout 600 gcloud storage rsync --recursive --exclude='.*[.]tmp$|(^|/)SUCCESS[.]json$' \\
      "$OUTPUT" "$DESTINATION" || exit 1
    for relative in analysis/smoke/SUCCESS.json analysis/SUCCESS.json; do
      [[ -f "$markers/$relative" ]] || continue
      timeout 60 gcloud storage cp "$markers/$relative" "$DESTINATION/$relative" || exit 1
    done
  ) 9>"$WORK/upload.lock"
}
'''


def download_script(args):
    q, commands = shlex.quote, []
    for key, uri in source_uris(args).items():
        commands.append(f'SOURCE={q(uri)}\nALGORITHM={q(ALGORITHMS[key])}')
        metadata = "run_manifest.json checkpoint_manifest.json SUCCESS.json"
        if key != "sd":
            metadata += " summary.json"
        commands.append(f'''for seed in 0 1 2; do
  WORKER="task_00${{seed}}_${{ALGORITHM}}_seed_${{seed}}"
  LOCAL="$INPUT/{key}/workers/$WORKER"
  mkdir -p "$LOCAL"
  for name in {metadata}; do
    gcloud storage cp "$SOURCE/workers/$WORKER/$name" "$LOCAL/$name"
  done
''')
        if key.startswith("ucv"):
            commands.append('  gcloud storage cp "$SOURCE/workers/$WORKER/runtime_manifest.json" "$LOCAL/runtime_manifest.json"\n')
        if key == "ucv":
            commands.append('  gcloud storage cp "$SOURCE/workers/$WORKER/continuation_source.json" "$LOCAL/continuation_source.json"\n')
        commands.append("done\n")
    # Fail missing metadata/permissions before installing large Python wheels.
    return "\n".join(commands)


def worker_script(args):
    q = shlex.quote
    uris = source_uris(args)
    source_args = " ".join("--source-uri "+q(f"{k}={v}") for k,v in uris.items())
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/exp40-mpl UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
WORK=/workspace/fhp-exp40
REPOSITORY="$WORK/ucv"
VR_REPOSITORY="$WORK/vr"
SD_REPOSITORY="$WORK/sd"
INPUT="$WORK/input"
OUTPUT="$WORK/output"
DESTINATION={q(args.bucket+'/'+args.run_id)}
VENV_ROOT="$WORK/venv"
EVALUATOR_DEADLINE=$(( $(date +%s) + 81900 ))
mkdir -p "$INPUT" "$OUTPUT" "$MPLCONFIGDIR"
SYNC_PID=""
MONITOR_PID=""
{upload_function()}
cleanup() {{
  result=$?
  trap - EXIT
  set +e
  for child in "$SYNC_PID" "$MONITOR_PID"; do
    if [[ -n "$child" ]]; then kill "$child" 2>/dev/null; wait "$child" 2>/dev/null; fi
  done
  if [[ -x "$VENV_ROOT/bin/python" && -d "$REPOSITORY/fhp_escher" ]]; then
    (cd "$REPOSITORY" && "$VENV_ROOT/bin/python" -m fhp_escher.batch_diagnostics finalize \\
      --snapshots "$OUTPUT/resource_snapshots.jsonl" --output "$OUTPUT/batch_diagnostics.json" \\
      --status-output "$OUTPUT/batch_status.json" --failure-root "$OUTPUT" \\
      --exit-code "$result" --experiment-exit-code "$result" --requested-memory-mib 62000 \\
      --job-name {q(args.run_id)} --bucket-destination "$DESTINATION") || true
  fi
  printf '{{"exit_code":%s}}\\n' "$result" > "$OUTPUT/process_exit.json"
  sync_outputs || true
  exit "$result"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ {int(args.resume)} == 1 ]]; then
  gcloud storage rsync --recursive --exclude='.*[.]tmp$' "$DESTINATION" "$OUTPUT"
fi
{download_script(args)}
apt_with_retry() {{
  local attempt result
  for attempt in {{1..30}}; do
    if apt-get -o DPkg::Lock::Timeout=10 "$@"; then return 0; else result=$?; fi
    if [[ "$attempt" == 30 ]]; then return "$result"; fi
    sleep 10
  done
}}
apt_with_retry update -qq
apt_with_retry install -y -qq git curl ca-certificates util-linux python3-dev build-essential
git clone --filter=blob:none {q(UCV_URL)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach {q(args.repo_ref)}
[[ "$(git -C "$REPOSITORY" rev-parse HEAD)" == {q(args.repo_ref)} ]]
git clone --filter=blob:none {q(VR_URL)} "$VR_REPOSITORY"
git -C "$VR_REPOSITORY" checkout --detach {VR_REF}
[[ "$(git -C "$VR_REPOSITORY" rev-parse HEAD)" == {VR_REF} ]]
git clone --filter=blob:none {q(SD_URL)} "$SD_REPOSITORY"
git -C "$SD_REPOSITORY" checkout --detach {SD_REF}
[[ "$(git -C "$SD_REPOSITORY" rev-parse HEAD)" == {SD_REF} ]]
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install 3.11.16
uv venv --python 3.11.16 --seed "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
uv pip install --python "$VENV_ROOT/bin/python" -r "$REPOSITORY/requirements.txt"
python -m pip check
cd "$REPOSITORY"
python -m pip freeze > "$OUTPUT/pip_freeze.txt"
python -m fhp_escher.batch_diagnostics monitor --output "$OUTPUT/resource_snapshots.jsonl" \\
  --interval-seconds 30 --cloud-log-every 4 &
MONITOR_PID=$!
(while sleep 300; do sync_outputs || echo "WARNING: periodic upload failed; will retry" >&2; done) &
SYNC_PID=$!
for seed in 0 1 2; do
  SD_WORKER="task_00${{seed}}_{ALGORITHMS['sd']}_seed_${{seed}}"
  UCV_WORKER="task_00${{seed}}_{ALGORITHMS['ucv']}_seed_${{seed}}"
  VR_WORKER="task_00${{seed}}_{ALGORITHMS['vr']}_seed_${{seed}}"
  mkdir -p "$INPUT/sd/workers/$SD_WORKER/archive" "$INPUT/ucv/workers/$UCV_WORKER/checkpoints" "$INPUT/vr/workers/$VR_WORKER/checkpoints"
  gcloud storage rsync --recursive {q(uris['sd'])}/workers/"$SD_WORKER"/archive "$INPUT/sd/workers/$SD_WORKER/archive"
  gcloud storage cp {q(uris['ucv'])}/workers/"$UCV_WORKER"/checkpoints/{ALGORITHMS['ucv']}_seed_${{seed}}_time_48h.pkl \\
    "$INPUT/ucv/workers/$UCV_WORKER/checkpoints/"
  gcloud storage cp {q(uris['vr'])}/workers/"$VR_WORKER"/checkpoints/{ALGORITHMS['vr']}_seed_${{seed}}_time_48h.pt \\
    "$INPUT/vr/workers/$VR_WORKER/checkpoints/"
done
python -m experiments.fhp.exp40_fhp_threeway_48h.run \\
  --source-root "$INPUT" --vr-repo "$VR_REPOSITORY" --sd-repo "$SD_REPOSITORY" \\
  --output "$OUTPUT/analysis" --workers 16 --deadline-unix "$EVALUATOR_DEADLINE" \\
  {source_args} {'--smoke-only' if args.smoke_only else ''}
kill "$SYNC_PID" 2>/dev/null || true
wait "$SYNC_PID" 2>/dev/null || true
SYNC_PID=""
sync_outputs
'''


def finalizer(args):
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
WORK=/workspace/fhp-exp40
OUTPUT="$WORK/output"
DESTINATION={shlex.quote(args.bucket+'/'+args.run_id)}
mkdir -p "$OUTPUT"
df -h > "$OUTPUT/disk_final.txt" || true
free -m > "$OUTPUT/memory_final.txt" || true
{upload_function()}
sync_outputs
'''


def build_job(args):
    validate(args)
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskSpec=dict(
        runnables=[dict(script=dict(text=worker_script(args)), timeout="82800s"),
                   dict(alwaysRun=True, script=dict(text=finalizer(args)), timeout="1800s")],
        computeResource=dict(cpuMilli=16000, memoryMib=62000), maxRetryCount=0, maxRunDuration="86400s"))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]),
            instances=[dict(policy=dict(machineType="n2-standard-16", provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=200, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="fhp-exp40", run=args.run_id, stage="smoke" if args.smoke_only else "evaluation"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "bucket", "service-account", "repo-ref"):
        p.add_argument("--"+name, required=True)
    for name, bucket in (("ucv", "fhp-escher"), ("vr", "fhp-vr-deep"), ("sd", "fhp-deep-cfr")):
        p.add_argument(f"--{name}-bucket", default=f"gs://clever-overview-399515-{bucket}-results")
    p.add_argument("--output", type=Path, required=True)
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
