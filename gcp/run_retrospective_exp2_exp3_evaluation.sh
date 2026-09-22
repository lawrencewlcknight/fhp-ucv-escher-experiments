#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILDER="$SCRIPT_DIR/retrospective_exp2_exp3_evaluation_batch.py"

: "${EXP2_RUN_ID:?Set EXP2_RUN_ID to the completed Experiment 2 run ID}"
: "${EXP3_RUN_ID:?Set EXP3_RUN_ID to the completed Experiment 3 run ID}"

if [[ "$ACTION" == "smoke-local" ]]; then
  EXP2_LOCAL_RUN="${EXP2_LOCAL_RUN:-$REPO_DIR/cloud_outputs/$EXP2_RUN_ID}"
  EXP3_LOCAL_RUN="${EXP3_LOCAL_RUN:-$REPO_DIR/cloud_outputs/$EXP3_RUN_ID}"
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-/tmp/fhp-exp2-exp3-retrospective-smoke}"
  cd "$REPO_DIR"
  exec python3 -m experiments.fhp.retrospective_exp2_exp3_evaluation.run \
    --exp2-run "$EXP2_LOCAL_RUN" --exp3-run "$EXP3_LOCAL_RUN" \
    --output-dir "$SMOKE_OUTPUT" --workers 2 --rule-deals 4 \
    --lbr-deals 2 --lbr-rollouts 16 --lbr-shard-deals 2 \
    --crossplay-deals 10 --smoke
fi

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed evaluation commit SHA}"

RUN_ID="${RUN_ID:-fhp-eval23-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ ${#RUN_ID} -gt 55 || ! "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "RUN_ID must be 2-55 lowercase letters, digits or hyphens" >&2
  exit 2
fi
if [[ "$BUCKET" == gs://* ]]; then
  BUCKET_ROOT="${BUCKET%/}"
else
  BUCKET_ROOT="gs://${BUCKET%/}"
fi
JOB_NAME="$RUN_ID"
TEMP_DIR="$(mktemp -d /tmp/fhp-retrospective-eval.XXXXXX)"
trap 'rm -rf "$TEMP_DIR"' EXIT
JOB_JSON="$TEMP_DIR/job.json"

python3 "$BUILDER" --output "$JOB_JSON" --run-id "$RUN_ID" \
  --exp2-run-id "$EXP2_RUN_ID" --exp3-run-id "$EXP3_RUN_ID" \
  --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" \
  --repo-ref "$REPO_REF"

preflight_service_account() {
  local described_email
  if ! described_email="$(
    gcloud iam service-accounts describe "$SA_EMAIL" \
      --project "$PROJECT_ID" --format='value(email)' 2>&1
  )"; then
    echo "Configured Batch service account does not exist or is inaccessible: $SA_EMAIL" >&2
    echo "$described_email" >&2
    return 2
  fi
  if [[ "$described_email" != "$SA_EMAIL" ]]; then
    echo "Service-account preflight returned an unexpected identity: $described_email" >&2
    return 2
  fi
}

preflight_inputs() {
  local experiment run_id
  for experiment in exp2 exp3; do
    if [[ "$experiment" == "exp2" ]]; then run_id="$EXP2_RUN_ID"; else run_id="$EXP3_RUN_ID"; fi
    if ! gcloud storage ls "$BUCKET_ROOT/$run_id/workers/**/checkpoint_manifest.json" \
      >/dev/null 2>&1; then
      echo "No checkpoint manifests found for $experiment at $BUCKET_ROOT/$run_id/workers" >&2
      return 2
    fi
  done
}

case "$ACTION" in
  run)
    preflight_service_account
    preflight_inputs
    gcloud batch jobs submit "$JOB_NAME" --project "$PROJECT_ID" \
      --location "$REGION" --config "$JOB_JSON"
    echo "Submitted single-VM retrospective evaluation: $JOB_NAME"
    echo "The laptop may now be disconnected or switched off."
    ;;
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
      --filter="name:${RUN_ID}" --format='table(name.basename(),status.state,createTime)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"
    ;;
  dry-run)
    cp "$JOB_JSON" "$REPO_DIR/retrospective_exp2_exp3_evaluation_job.json"
    echo "Wrote $REPO_DIR/retrospective_exp2_exp3_evaluation_job.json"
    ;;
  *)
    echo "Usage: $0 [run|smoke-local|status|dry-run]" >&2
    exit 2
    ;;
esac
