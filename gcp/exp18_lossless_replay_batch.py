#!/usr/bin/env python3
"""Single n2-standard-16 Batch job; smoke gates the memory/throughput audit."""
import argparse
import json
from pathlib import Path
try:
    import exp1_grouped_wide_batch as base
except ModuleNotFoundError:
    from gcp import exp1_grouped_wide_batch as base


def build_job(args):
    bootstrap = (base._bootstrap(args).replace("exp1-fhp", "exp18-memory")
                 .replace("uv python install 3.11", "uv python install 3.11.16")
                 .replace("uv venv --python 3.11 ", "uv venv --python 3.11.16 "))
    action = """
mkdir -p "$OUTPUT_ROOT/analysis"
python -m experiments.fhp.exp18_fhp_lossless_replay.run run --output-root "$WORK_ROOT/audit"
"""
    wrapped = base._diagnostic_wrapper(action, "$OUTPUT_ROOT/diagnostics",
                                      "$BUCKET_ROOT/$RUN_ID/diagnostics", 64000)
    # Copy partial metrics on failure too; large temporary states live outside
    # the output tree and must never be uploaded by this engineering audit.
    wrapped = wrapped.replace('  exit "$exit_code"', '''  if [[ -d "$WORK_ROOT/audit/analysis" ]]; then
    gcloud storage rsync --recursive "$WORK_ROOT/audit/analysis" "$BUCKET_ROOT/$RUN_ID/analysis" || exit_code=1
  fi
  exit "$exit_code"''')
    return {
        "taskGroups": [{"taskCount": 1, "parallelism": 1, "taskCountPerNode": 1,
                        "taskSpec": {"runnables": [{"script": {"text":
                            f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{wrapped}\n"}}],
                        "computeResource": {"cpuMilli": 16000, "memoryMib": 64000},
                        "maxRetryCount": 0, "maxRunDuration": "43200s"}}],
        "allocationPolicy": {"serviceAccount": {"email": args.service_account},
            "instances": [{"policy": {"machineType": "n2-standard-16", "provisioningModel": "STANDARD",
                            "bootDisk": {"sizeGb": 200, "type": "pd-balanced"}}}]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": "exp18-fhp-memory", "stage": "validation"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "bucket-root", "service-account", "repo-ref"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repo-url", default=base.REPO_URL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_job(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
