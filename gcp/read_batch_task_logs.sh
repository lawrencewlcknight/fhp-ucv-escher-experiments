#!/usr/bin/env bash
set -euo pipefail

# Read Google Batch task logs for exactly one job.
#
# Usage:
#   ./gcp/read_batch_task_logs.sh JOB_NAME [TEXT_FILTER ...]
#
# Examples:
#   ./gcp/read_batch_task_logs.sh escher-architecture-exp28-20260716-120000
#   ./gcp/read_batch_task_logs.sh escher-architecture-exp28-20260716-120000 ERROR Traceback Killed
#
# The task-log query is always scoped by labels.job_uid before text filters are
# appended, so generic ESCHER terms cannot pull in logs from neighbouring jobs.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 JOB_NAME [TEXT_FILTER ...]" >&2
  exit 2
fi

JOB_NAME="$1"
shift
TEXT_FILTERS=("$@")

: "${PROJECT_ID:?Set PROJECT_ID first}"
: "${REGION:?Set REGION first}"

LIMIT="${LIMIT:-500}"
FORMAT="${FORMAT:-value(timestamp,severity,textPayload,jsonPayload.message)}"

JOB_UID="$(
  gcloud batch jobs describe "${JOB_NAME}" \
    --location "${REGION}" \
    --format='value(uid)'
)"

if [[ -z "${JOB_UID}" ]]; then
  echo "Could not resolve Batch uid for job: ${JOB_NAME}" >&2
  exit 1
fi

PROJECT_LOG_NAME="projects/${PROJECT_ID}/logs/batch_task_logs"
FILTER="logName=\"${PROJECT_LOG_NAME}\" AND labels.job_uid=\"${JOB_UID}\""

if [[ ${#TEXT_FILTERS[@]} -gt 0 ]]; then
  TEXT_FILTER=""
  for token in "${TEXT_FILTERS[@]}"; do
    escaped_token="${token//\\/\\\\}"
    escaped_token="${escaped_token//\"/\\\"}"
    clause="textPayload:\"${escaped_token}\" OR jsonPayload.message:\"${escaped_token}\""
    if [[ -z "${TEXT_FILTER}" ]]; then
      TEXT_FILTER="${clause}"
    else
      TEXT_FILTER="${TEXT_FILTER} OR ${clause}"
    fi
  done
  FILTER="${FILTER} AND (${TEXT_FILTER})"
fi

LOG_OUTPUT="$(mktemp "/tmp/${JOB_NAME}.task-logs.XXXXXX.txt")"

echo "Job: ${JOB_NAME}" >&2
echo "Job UID: ${JOB_UID}" >&2
echo "Filter: ${FILTER}" >&2

gcloud logging read "${FILTER}" \
  --limit="${LIMIT}" \
  --format="${FORMAT}" | tee "${LOG_OUTPUT}"

python3 - "${JOB_NAME}" "${LOG_OUTPUT}" <<'PY'
import re
import sys
from pathlib import Path

target_job = sys.argv[1]
log_path = Path(sys.argv[2])

job_name_pattern = re.compile(r"\bescher-exp[A-Za-z0-9_.-]*\b")
contaminated = []

for line_number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
    for candidate in sorted(set(job_name_pattern.findall(line))):
        if candidate == target_job:
            continue
        if target_job.startswith(f"{candidate}-"):
            # Correct-job logs often contain output prefixes such as
            # outputs/cloud/escher-architecture-exp28 while the Batch job name has a
            # timestamp suffix. Treat those as provenance, not contamination.
            continue
        contaminated.append((line_number, candidate, line))

if contaminated:
    print(
        "\nWARNING: possible cross-job log contamination detected.",
        file=sys.stderr,
    )
    print(
        f"Target job: {target_job}",
        file=sys.stderr,
    )
    for line_number, candidate, line in contaminated[:20]:
        print(
            f"  line {line_number}: saw {candidate!r}: {line}",
            file=sys.stderr,
        )
    if len(contaminated) > 20:
        print(f"  ... {len(contaminated) - 20} more contaminated lines omitted", file=sys.stderr)
    sys.exit(3)
PY
