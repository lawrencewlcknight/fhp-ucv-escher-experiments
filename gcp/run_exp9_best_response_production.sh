#!/usr/bin/env bash
# Evaluate the frozen Experiment 9 models; no UCV training is launched.
set -Eeuo pipefail
usage() {
  echo "Usage: bash gcp/run_exp9_best_response_production.sh [prepare|prepare-smoke|run|smoke|aggregate|aggregate-smoke|recover|recover-smoke|status]"
  echo "prepare variants are offline. run/smoke/recover/aggregate submit paid GCP jobs."
  echo "Set PROJECT_ID, REGION, BUCKET, SA_EMAIL and a fresh fhp-br-exp9-prod-* RUN_ID."
  echo "Optional: FHP_EVAL_REPO, FHP_BATCH_OUTPUT_DIR, PYTHON; recover requires RECOVERY_TAG."
}
ACTION="${1:-help}"
if [[ "$#" -gt 1 ]]; then usage >&2; exit 2; fi
case "$ACTION" in
  help|-h|--help) usage; exit 0 ;;
  prepare|prepare-smoke|run|smoke|aggregate|aggregate-smoke|recover|recover-smoke|status) ;;
  *) usage >&2; exit 2 ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${RUN_ID:?Set a fresh fhp-br-exp9-prod-* RUN_ID}"
if [[ ! "$RUN_ID" =~ ^fhp-br-exp9-prod-[a-z0-9]([a-z0-9-]{0,32}[a-z0-9])?$ ]]; then
  echo "RUN_ID must start fhp-br-exp9-prod-, end in a letter/digit, and be at most 51 characters." >&2
  exit 2
fi
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="name:${RUN_ID}-" --format='table(name.basename(),status.state)'
fi
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVALUATION_DIR="${FHP_EVAL_REPO:-$REPO_DIR/../../fhp-evaluation-suite}"
for required in gcp/exp5_ucv_exp9_br_production_batch.py \
                gcp/run_exp5_ucv_exp9_br_production.sh \
                gcp/finalize_exp5_ucv_exp9_br_production.sh \
                gcp/aggregate_exp5_ucv_exp9_br_production.sh \
                gcp/requirements-br-pilot.txt \
                fhp_evaluation/best_response/production.py pyproject.toml; do
  if [[ ! -f "$EVALUATION_DIR/$required" ]]; then
    echo "Missing shared evaluation-suite file: $EVALUATION_DIR/$required" >&2
    echo "Set FHP_EVAL_REPO to an up-to-date fhp-evaluation-suite checkout." >&2
    exit 2
  fi
done
EVALUATION_DIR="$(cd "$EVALUATION_DIR" && pwd)"
OUTPUT_DIR="${FHP_BATCH_OUTPUT_DIR:-$REPO_DIR/outputs/batch/$RUN_ID}"
EXTRA_ARGS=()
case "$ACTION" in
  prepare) MODE=prepare ;;
  prepare-smoke) MODE=prepare; EXTRA_ARGS=(--smoke) ;;
  run) MODE=submit-workers ;;
  smoke) MODE=submit-workers; EXTRA_ARGS=(--smoke) ;;
  aggregate) MODE=submit-aggregate ;;
  aggregate-smoke) MODE=submit-aggregate; EXTRA_ARGS=(--smoke) ;;
  recover|recover-smoke)
    MODE=submit-recovery
    : "${RECOVERY_TAG:?Set a unique RECOVERY_TAG after diagnosing the stopped attempt}"
    EXTRA_ARGS=(--recovery-tag "$RECOVERY_TAG")
    if [[ "$ACTION" == recover-smoke ]]; then EXTRA_ARGS+=(--smoke); fi
    ;;
esac
if [[ "$MODE" != prepare && ! -f "$OUTPUT_DIR/request.json" ]]; then
  echo "No prepared request: $OUTPUT_DIR/request.json. Run prepare or prepare-smoke first." >&2
  exit 2
fi
exec "${PYTHON:-python3}" "$EVALUATION_DIR/gcp/exp5_ucv_exp9_br_production_batch.py" "$MODE" \
  --native-repo "$REPO_DIR" --run-id "$RUN_ID" --run-dir "$OUTPUT_DIR" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
