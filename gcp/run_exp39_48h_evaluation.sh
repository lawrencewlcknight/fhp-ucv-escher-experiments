#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${RUN_ID:?Set RUN_ID to a new exp39 comparison ID (reuse only for resume/status)}"
BUCKET_ROOT="gs://${BUCKET#gs://}"
BUCKET_ROOT="${BUCKET_ROOT%/}"
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="labels.workload=fhp-exp39 AND name:${RUN_ID}" --format='table(name.basename(),status.state,createTime)'
  echo "Outputs: $BUCKET_ROOT/$RUN_ID/analysis/"
  exit 0
fi
case "$ACTION" in run|resume|smoke-cloud|dry-run) ;; *) echo "Usage: $0 [run|resume|smoke-cloud|dry-run|status]" >&2; exit 2 ;; esac
: "${REPO_REF:?Set REPO_REF to the full pushed experiment commit SHA}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || ! git -C "$REPO_DIR" cat-file -e "$REPO_REF^{commit}"; then
  echo "REPO_REF must be a full commit SHA available in this checkout." >&2; exit 2
fi
for file in experiments/fhp/exp39_fhp_48h_evaluation/run.py \
            experiments/fhp/exp39_fhp_48h_evaluation/statistics.py \
            gcp/exp39_48h_evaluation_batch.py gcp/run_exp39_48h_evaluation.sh; do
  if ! git -C "$REPO_DIR" cat-file -e "$REPO_REF:$file"; then
    echo "REPO_REF predates Experiment 39. Commit and push the implementation first." >&2; exit 2
  fi
done
EXP10_RUN_ID="${EXP10_RUN_ID:-exp10-features-20261001-161740}"
EXP16_RUN_ID="${EXP16_RUN_ID:-exp16-feat48-20261004-182051}"
EXP36_RUN_ID="${EXP36_RUN_ID:-exp36-rich2500-20261018-205125}"
EXP38_RUN_ID="${EXP38_RUN_ID:-exp38-rich48-20261019-230016}"
JOB_NAME="$RUN_ID"
EXTRA_ARGS=()
if [[ "$ACTION" == resume ]]; then
  JOB_NAME="${RUN_ID}-r$(date -u '+%H%M%S')"
  EXTRA_ARGS+=(--resume)
fi
if [[ "$ACTION" == smoke-cloud ]]; then EXTRA_ARGS+=(--smoke-only); fi
TEMP_DIR="$(mktemp -d /tmp/fhp-exp39.XXXXXX)"
git -C "$REPO_DIR" show "$REPO_REF:gcp/exp39_48h_evaluation_batch.py" > "$TEMP_DIR/builder.py"
python3 "$TEMP_DIR/builder.py" --output "$TEMP_DIR/job.json" --run-id "$RUN_ID" \
  --exp10-run-id "$EXP10_RUN_ID" --exp16-run-id "$EXP16_RUN_ID" --exp36-run-id "$EXP36_RUN_ID" --exp38-run-id "$EXP38_RUN_ID" \
  --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
  --max-hours "${EXP39_MAX_HOURS:-48}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
if [[ "$ACTION" == dry-run ]]; then
  echo "Prepared $TEMP_DIR/job.json; no cloud job submitted."; exit 0
fi
gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)'
gcloud storage ls "$BUCKET_ROOT/exp37-critic-compare-20261019-222256/analysis/rare_panel.json" >/dev/null
for source_run in "$EXP10_RUN_ID" "$EXP16_RUN_ID" "$EXP36_RUN_ID" "$EXP38_RUN_ID"; do
  for file in run_manifest.json checkpoint_manifest.json runtime_manifest.json summary.json SUCCESS.json; do
    gcloud storage ls "$BUCKET_ROOT/$source_run/workers/**/$file" >/dev/null
  done
done
ACTIVE_STATES="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
  --filter="labels.workload=fhp-exp39 AND name:${RUN_ID}" --format='value(status.state)')"
for state in $ACTIVE_STATES; do
  case "$state" in SUCCEEDED|FAILED) ;; *) echo "This output namespace has an active job. Wait before resuming." >&2; exit 2 ;; esac
done
if [[ "$ACTION" == resume ]]; then
  gcloud storage ls "$BUCKET_ROOT/$RUN_ID/**/evaluation_manifest.json" >/dev/null
else
  if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/**" >"$TEMP_DIR/objects.txt" 2>"$TEMP_DIR/storage_error.txt"; then
    echo "Output prefix already exists. Use a new RUN_ID or resume." >&2; exit 2
  fi
  if [[ "$( < "$TEMP_DIR/storage_error.txt")" != *"matched no objects"* ]]; then
    cat "$TEMP_DIR/storage_error.txt" >&2
    echo "Cannot confirm unused output prefix; refusing submission." >&2; exit 2
  fi
fi
gcloud batch jobs submit "$JOB_NAME" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/job.json"
echo "Submitted $JOB_NAME. Laptop may disconnect. Results: $BUCKET_ROOT/$RUN_ID/analysis/"
