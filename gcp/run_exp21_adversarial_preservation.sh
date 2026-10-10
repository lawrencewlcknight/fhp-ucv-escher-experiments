#!/usr/bin/env bash
# UCV-root entry point; implementation lives in the shared evaluation suite.
set -Eeuo pipefail
ACTION="${1:-help}"
case "$ACTION" in
  help|-h|--help)
    echo "Usage: bash gcp/run_exp21_adversarial_preservation.sh [prepare|qualify|train|evaluate|aggregate|status]"
    echo "Add -smoke to any non-status action for the execution-only smoke."
    echo "prepare is offline; qualify/train/evaluate/aggregate each submit one paid Batch job."
    echo "Full qualify requires SMOKE_RUN_ID; full train requires all three attacker gates to pass."
    exit 0 ;;
esac
if [[ "$#" -gt 1 ]]; then echo 'Expected one action' >&2; exit 2; fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${RUN_ID:?Set a fresh exp21-adv-* RUN_ID}"
if [[ ! "$RUN_ID" =~ ^exp21-adv-[a-z0-9]([a-z0-9-]{0,35}[a-z0-9])?$ ]]; then
  echo 'Invalid RUN_ID; use exp21-adv-* (at most 47 lowercase characters)' >&2; exit 2
fi
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="name:${RUN_ID}-" --format='table(name.basename(),status.state)'
fi
ARGS=()
if [[ "$ACTION" == *-smoke ]]; then ACTION="${ACTION%-smoke}"; ARGS+=(--smoke); fi
case "$ACTION" in prepare|qualify|train|evaluate|aggregate) ;; *) echo 'Unknown action' >&2; exit 2 ;; esac
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVALUATION_DIR="${FHP_EVAL_REPO:-$REPO_DIR/../../fhp-evaluation-suite}"
if [[ ! -f "$EVALUATION_DIR/gcp/exp21_adversarial_preservation_batch.py" ]]; then
  echo 'Set FHP_EVAL_REPO to an up-to-date fhp-evaluation-suite checkout' >&2; exit 2
fi
exec "${PYTHON:-python3}" "$EVALUATION_DIR/gcp/exp21_adversarial_preservation_batch.py" "$ACTION" \
  --native-repo "$REPO_DIR" --run-dir "${FHP_BATCH_OUTPUT_DIR:-$REPO_DIR/outputs/batch/$RUN_ID}" \
  ${ARGS[@]+"${ARGS[@]}"}
