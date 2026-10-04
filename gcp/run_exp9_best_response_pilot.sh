#!/usr/bin/env bash
# UCV-root convenience launcher; the shared suite owns the evaluation contract.
set -Eeuo pipefail

usage() {
  echo "Usage: bash gcp/run_exp9_best_response_pilot.sh [dry-run|dry-run-smoke|smoke|run|status]"
  echo "smoke/run submit paid GCP Batch jobs; dry-run variants make no cloud calls."
  echo "Set PROJECT_ID, REGION, BUCKET, SA_EMAIL and a fresh fhp-br-exp9-* RUN_ID."
  echo "Optional: FHP_EVAL_REPO, FHP_BATCH_OUTPUT_DIR, PYTHON."
}

ACTION="${1:-help}"
if [[ "$#" -gt 1 ]]; then usage >&2; exit 2; fi
case "$ACTION" in
  help|-h|--help) usage; exit 0 ;;
  dry-run|dry-run-smoke|smoke|run|status) ;;
  *) usage >&2; exit 2 ;;
esac

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${RUN_ID:?Set RUN_ID to a fresh fhp-br-exp9-* ID (keep it for status)}"
if [[ ! "$RUN_ID" =~ ^fhp-br-exp9-[a-z0-9]([a-z0-9-]{0,38}[a-z0-9])?$ ]]; then
  echo "RUN_ID must start fhp-br-exp9-, end with a letter/digit, and be at most 52 characters." >&2
  exit 2
fi
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs describe "$RUN_ID" --project "$PROJECT_ID" \
    --location "$REGION" --format='value(status.state)'
fi

: "${BUCKET:?Set BUCKET to the existing UCV results bucket}"
: "${SA_EMAIL:?Set SA_EMAIL to the Batch runner service account}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVALUATION_DIR="${FHP_EVAL_REPO:-$REPO_DIR/../../fhp-evaluation-suite}"
for required in gcp/exp4_ucv_exp9_br_pilot_batch.py \
                gcp/run_exp4_ucv_exp9_br_pilot.sh \
                gcp/finalize_exp4_ucv_exp9_br_pilot.sh \
                gcp/requirements-br-pilot.txt \
                fhp_evaluation/best_response/pilot.py pyproject.toml; do
  if [[ ! -f "$EVALUATION_DIR/$required" ]]; then
    echo "Missing shared evaluation-suite file: $EVALUATION_DIR/$required" >&2
    echo "Set FHP_EVAL_REPO to an up-to-date fhp-evaluation-suite checkout." >&2
    echo "Repository: https://github.com/lawrencewlcknight/fhp-evaluation-suite" >&2
    exit 2
  fi
done
EVALUATION_DIR="$(cd "$EVALUATION_DIR" && pwd)"

MODE=submit
SUFFIX=""
EXTRA_ARGS=()
case "$ACTION" in
  dry-run) MODE=dry-run; SUFFIX=-preview ;;
  dry-run-smoke) MODE=dry-run; SUFFIX=-smoke-preview; EXTRA_ARGS=(--smoke) ;;
  smoke) EXTRA_ARGS=(--smoke) ;;
esac
OUTPUT_DIR="${FHP_BATCH_OUTPUT_DIR:-$REPO_DIR/outputs/batch/${RUN_ID}${SUFFIX}}"
exec "${PYTHON:-python3}" "$EVALUATION_DIR/gcp/exp4_ucv_exp9_br_pilot_batch.py" "$MODE" \
  --native-repo "$REPO_DIR" --run-id "$RUN_ID" --output-dir "$OUTPUT_DIR" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
