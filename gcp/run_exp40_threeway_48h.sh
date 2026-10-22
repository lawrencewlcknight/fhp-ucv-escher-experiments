#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${RUN_ID:?Set a fresh Experiment 40 RUN_ID (reuse only for resume/status)}"
BUCKET="gs://${BUCKET#gs://}"
BUCKET="${BUCKET%/}"
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="labels.workload=fhp-exp40 AND labels.run=$RUN_ID" --format='table(name.basename(),status.state,createTime)'
  echo "Completion requires $BUCKET/$RUN_ID/analysis/SUCCESS.json with smoke=false."
  exit 0
fi
case "$ACTION" in run|resume|smoke-cloud|dry-run) ;; *) echo "Usage: $0 [run|resume|smoke-cloud|dry-run|status]" >&2; exit 2 ;; esac
: "${REPO_REF:?Set REPO_REF to the full pushed Experiment 40 commit}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || ! git -C "$REPO_DIR" cat-file -e "$REPO_REF^{commit}"; then
  echo "REPO_REF must be a full commit available in this checkout." >&2; exit 2
fi
for file in engine.py sources.py run.py; do
  if ! git -C "$REPO_DIR" cat-file -e "$REPO_REF:experiments/fhp/exp40_fhp_threeway_48h/$file"; then
    echo "REPO_REF predates Experiment 40. Commit and push the implementation first." >&2; exit 2
  fi
done
UCV_BUCKET="${UCV_BUCKET:-gs://clever-overview-399515-fhp-escher-results}"
VR_BUCKET="${VR_BUCKET:-gs://clever-overview-399515-fhp-vr-deep-results}"
SD_BUCKET="${SD_BUCKET:-gs://clever-overview-399515-fhp-deep-cfr-results}"
TEMP_DIR="$(mktemp -d /tmp/fhp-exp40.XXXXXX)"
git -C "$REPO_DIR" show "$REPO_REF:gcp/exp40_threeway_48h_batch.py" > "$TEMP_DIR/builder.py"
EXTRA_ARGS=()
JOB_NAME="$RUN_ID"
if [[ "$ACTION" == resume ]]; then
  EXTRA_ARGS+=(--resume)
  JOB_NAME="${RUN_ID}-r$(date -u '+%H%M%S')"
fi
if [[ "$ACTION" == smoke-cloud || "${EXP40_SMOKE_ONLY:-0}" == 1 ]]; then EXTRA_ARGS+=(--smoke-only); fi
python3 "$TEMP_DIR/builder.py" --output "$TEMP_DIR/job.json" --run-id "$RUN_ID" \
  --bucket "$BUCKET" --ucv-bucket "$UCV_BUCKET" --vr-bucket "$VR_BUCKET" --sd-bucket "$SD_BUCKET" \
  --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
if [[ "$ACTION" == dry-run ]]; then
  echo "Prepared $TEMP_DIR/job.json; no cloud job submitted."; exit 0
fi
gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)'
for source in \
  "$UCV_BUCKET/exp38-rich48-20261019-230016" \
  "$UCV_BUCKET/exp36-rich2500-20261018-205125" \
  "$VR_BUCKET/vr6-ray48-20261003-221425" \
  "$SD_BUCKET/sdcfr6-48h-20261002-161544"; do
  for file in run_manifest.json checkpoint_manifest.json SUCCESS.json; do
    gcloud storage ls "$source/workers/**/$file" >/dev/null
  done
  if [[ "$source" != "$SD_BUCKET/sdcfr6-48h-20261002-161544" ]]; then
    gcloud storage ls "$source/workers/**/summary.json" >/dev/null
  fi
done
ACTIVE_STATES="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
  --filter="labels.workload=fhp-exp40 AND labels.run=$RUN_ID" --format='value(status.state)')"
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
