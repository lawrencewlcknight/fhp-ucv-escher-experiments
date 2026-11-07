#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${RUN_ID:?Set a fresh Experiment 42 RUN_ID (reuse only for resume/status)}"
BUCKET="gs://${BUCKET#gs://}"
BUCKET="${BUCKET%/}"
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="labels.workload=fhp-exp42 AND labels.run=$RUN_ID" --format='table(name.basename(),status.state,createTime)'
  echo "Completion requires $BUCKET/$RUN_ID/analysis/SUCCESS.json with smoke=false."
  exit 0
fi
case "$ACTION" in run|resume|smoke-cloud|preflight|dry-run) ;; *) echo "Usage: $0 [run|resume|smoke-cloud|preflight|dry-run|status]" >&2; exit 2 ;; esac
: "${REPO_REF:?Set REPO_REF to the full pushed Experiment 42 commit}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || ! git -C "$REPO_DIR" cat-file -e "$REPO_REF^{commit}"; then
  echo "REPO_REF must be a full commit available in this checkout." >&2; exit 2
fi
for file in engine.py sources.py source_pins.json reporting.py run.py; do
  if ! git -C "$REPO_DIR" cat-file -e "$REPO_REF:experiments/fhp/representation_head_to_head/$file"; then
    echo "REPO_REF predates Experiment 42. Commit and push the implementation first." >&2; exit 2
  fi
done
VR_BUCKET="${VR_BUCKET:-gs://clever-overview-399515-fhp-vr-deep-results}"
SD_BUCKET="${SD_BUCKET:-gs://clever-overview-399515-fhp-deep-cfr-results}"
TEMP_DIR="$(mktemp -d /tmp/fhp-exp42.XXXXXX)"
git -C "$REPO_DIR" show "$REPO_REF:gcp/representation_head_to_head_batch.py" > "$TEMP_DIR/builder.py"
EXTRA_ARGS=()
JOB_NAME="$RUN_ID"
if [[ "$ACTION" == resume ]]; then
  EXTRA_ARGS+=(--resume)
  JOB_NAME="${RUN_ID}-r$(date -u '+%H%M%S')"
fi
if [[ "$ACTION" == smoke-cloud || "${EXP42_SMOKE_ONLY:-0}" == 1 ]]; then EXTRA_ARGS+=(--smoke-only); fi
BUILDER_ARGS=(--algorithm sd --output "$TEMP_DIR/job.json" --run-id "$RUN_ID"
  --bucket "$BUCKET" --vr-bucket "$VR_BUCKET" --sd-bucket "$SD_BUCKET"
  --service-account "$SA_EMAIL" --repo-ref "$REPO_REF")
python3 "$TEMP_DIR/builder.py" "${BUILDER_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
if [[ "$ACTION" == dry-run ]]; then
  echo "Prepared $TEMP_DIR/job.json; no cloud job submitted."; exit 0
fi
gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)'
python3 "$SCRIPT_DIR/representation_access_preflight.py" --builder "$TEMP_DIR/builder.py" \
  --project "$PROJECT_ID" --receipt "$TEMP_DIR/access_preflight.json" "${BUILDER_ARGS[@]}"
if [[ "$ACTION" == preflight ]]; then
  echo "Read-only preflight complete; no cloud job submitted."; exit 0
fi
ACTIVE_STATES="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
  --filter="labels.workload=fhp-exp42 AND labels.run=$RUN_ID" --format='value(status.state)')"
for state in $ACTIVE_STATES; do
  case "$state" in SUCCEEDED|FAILED) ;; *) echo "An active job uses this output namespace." >&2; exit 2 ;; esac
done
if [[ "$ACTION" == resume ]]; then
  gcloud storage ls "$BUCKET/$RUN_ID/analysis/evaluation_manifest.json" >/dev/null
else
  if gcloud storage ls "$BUCKET/$RUN_ID/**" >"$TEMP_DIR/objects.txt" 2>"$TEMP_DIR/storage_error.txt"; then
    echo "Output prefix exists; use a fresh RUN_ID or resume." >&2; exit 2
  fi
  if [[ "$( < "$TEMP_DIR/storage_error.txt")" != *"matched no objects"* ]]; then
    cat "$TEMP_DIR/storage_error.txt" >&2
    echo "Cannot confirm unused output prefix; refusing submission." >&2; exit 2
  fi
fi
gcloud batch jobs submit "$JOB_NAME" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/job.json"
echo "Submitted $JOB_NAME. Laptop may disconnect. Results: $BUCKET/$RUN_ID/analysis/"
