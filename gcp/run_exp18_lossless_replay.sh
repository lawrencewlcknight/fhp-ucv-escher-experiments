#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ "$ACTION" == smoke-local ]]; then
  exec "${PYTHON:-python3}" -m experiments.fhp.exp18_fhp_lossless_replay.run smoke \
    --output-root "${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-exp18-smoke.XXXXXX)}"
fi
[[ "$ACTION" == run || "$ACTION" == dry-run ]] || { echo "Usage: $0 [run|dry-run|smoke-local]" >&2; exit 2; }
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 18 commit}"
RUN_ID="${RUN_ID:-exp18-memory-$(date -u '+%Y%m%d-%H%M%S')}"
[[ ${#RUN_ID} -le 60 && "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "$BUCKET" == gs://* ]] && BUCKET_ROOT="${BUCKET%/}" || BUCKET_ROOT="gs://${BUCKET%/}"
if [[ "$ACTION" == run ]]; then
  git cat-file -e "$REPO_REF:experiments/fhp/exp18_fhp_lossless_replay/run.py" || {
    echo "REPO_REF lacks Experiment 18; push/pull the new code and reset REPO_REF." >&2; exit 2;
  }
  REPO_REF="$(git rev-parse "$REPO_REF^{commit}")"
  email="$(gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)')"
  [[ "$email" == "$SA_EMAIL" ]] || { echo "Service-account identity mismatch" >&2; exit 2; }
fi
TEMP_DIR="$(mktemp -d /tmp/fhp-exp18-job.XXXXXX)"
python3 "$SCRIPT_DIR/exp18_lossless_replay_batch.py" --run-id "$RUN_ID" \
  --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
  --output "$TEMP_DIR/job.json"
if [[ "$ACTION" == dry-run ]]; then
  echo "Job specification: $TEMP_DIR/job.json"
else
  gcloud batch jobs submit "$RUN_ID" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/job.json"
  echo "Submitted one VM: cloud smoke, paired timings and 10m capacity stress. Outputs: $BUCKET_ROOT/$RUN_ID/analysis"
fi
