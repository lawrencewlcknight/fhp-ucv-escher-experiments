#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET to the FHP UCV-ESCHER results bucket}"
: "${SA_EMAIL:?Set SA_EMAIL to the FHP UCV-ESCHER runner service account}"
: "${RUN_ID:?Set RUN_ID to a new fhp-eval79 run ID (reuse it only for resume/status)}"
EXP7_RUN_ID="${EXP7_RUN_ID:-exp7-par8-20261001-005151}"
EXP9_RUN_ID="${EXP9_RUN_ID:-exp9-cache24-20261001-132550}"
BUCKET_ROOT="gs://${BUCKET#gs://}"
BUCKET_ROOT="${BUCKET_ROOT%/}"
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="labels.workload=fhp-eval79 AND name:${RUN_ID}" \
    --format='table(name.basename(),status.state,createTime)'
  echo "Outputs: $BUCKET_ROOT/$RUN_ID/analysis/"
  exit 0
fi
case "$ACTION" in run|resume|dry-run) ;; *) echo "Usage: $0 [run|resume|status|dry-run]" >&2; exit 2;; esac
: "${REPO_REF:?Set REPO_REF to the full pushed evaluation commit SHA}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || ! git -C "$REPO_DIR" cat-file -e "$REPO_REF^{commit}"; then
  echo "REPO_REF must identify a full commit SHA in this checkout." >&2
  exit 2
fi
for file in experiments/fhp/retrospective_exp7_exp9_evaluation/run.py \
            gcp/retrospective_exp7_exp9_evaluation_batch.py \
            gcp/run_retrospective_exp7_exp9_evaluation.sh; do
  if ! git -C "$REPO_DIR" cat-file -e "$REPO_REF:$file"; then
    echo "REPO_REF predates the Exp7/9 evaluation. Use the new pushed commit." >&2
    exit 2
  fi
done
JOB_NAME="$RUN_ID"
RESUME_ARGS=()
if [[ "$ACTION" == resume ]]; then
  JOB_NAME="${RUN_ID}-r$(date -u '+%H%M%S')"
  RESUME_ARGS=(--resume)
fi
TEMP_DIR="$(mktemp -d /tmp/fhp-eval79.XXXXXX)"
# Remove only these two known temporary files and their now-empty directory.
trap 'rm -f "$TEMP_DIR/builder.py" "$TEMP_DIR/job.json"; rmdir "$TEMP_DIR"' EXIT
git -C "$REPO_DIR" show "$REPO_REF:gcp/retrospective_exp7_exp9_evaluation_batch.py" > "$TEMP_DIR/builder.py"
python3 "$TEMP_DIR/builder.py" --output "$TEMP_DIR/job.json" --run-id "$RUN_ID" \
  --exp7-run-id "$EXP7_RUN_ID" --exp9-run-id "$EXP9_RUN_ID" --bucket-root "$BUCKET_ROOT" \
  --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" --max-hours "${EVAL_MAX_HOURS:-36}" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
if [[ "$ACTION" == dry-run ]]; then
  cp "$TEMP_DIR/job.json" "$REPO_DIR/retrospective_exp7_exp9_evaluation_job.json"
  echo "Wrote retrospective_exp7_exp9_evaluation_job.json; no cloud job submitted."
  exit 0
fi
gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)'
for source_run in "$EXP7_RUN_ID" "$EXP9_RUN_ID"; do
  gcloud storage ls "$BUCKET_ROOT/$source_run/workers/**/checkpoint_manifest.json" >/dev/null
  gcloud storage ls "$BUCKET_ROOT/$source_run/workers/**/runtime_manifest.json" >/dev/null
done
if [[ "$ACTION" == resume ]]; then
  # Fail closed if a job still uses this output namespace.
  ACTIVE_JOBS="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="labels.workload=fhp-eval79 AND name:${RUN_ID}" --format='value(status.state)')"
  for state in $ACTIVE_JOBS; do
    case "$state" in
      SUCCEEDED|FAILED) ;;
      *) echo "An evaluation job is still active for RUN_ID; do not resume concurrently." >&2; exit 2 ;;
    esac
  done
  gcloud storage ls "$BUCKET_ROOT/$RUN_ID/**/evaluation_manifest.json" >/dev/null
else
  if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/**" >/dev/null 2>&1; then
    echo "Outputs already exist. Use a new RUN_ID or resume the same pinned evaluation." >&2
    exit 2
  fi
fi
gcloud batch jobs submit "$JOB_NAME" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/job.json"
echo "Submitted $JOB_NAME: cloud smoke test then full evaluation on one n2-standard-8."
echo "You can disconnect your laptop. Outputs: $BUCKET_ROOT/$RUN_ID/analysis/"
