#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUITE_ROOT="${FHP_EVALUATION_SUITE:-$REPO_ROOT/../../fhp-evaluation-suite}"
ACTION="${1:-help}"
case "$ACTION" in
  help|-h|--help)
    echo 'Experiment 23: [prepare|run|submit|recover|status]'
    echo 'Default: two-second feasibility probes ONLY on one n2-standard-16 VM; no gameplay or training.'
    echo 'run prepares and submits; submit uses an existing prepared bundle; recover resumes shards.'
    echo 'Set PROJECT_ID REGION BUCKET SA_EMAIL RUN_ID. REPO_REF is deliberately unused.'
    echo 'EXP23_MODE=gameplay runs the expanded 2s / 180,000-hand study with a 24h Batch cap.'
    echo 'Both modes use 2s. Gameplay includes preflight, paired smoke, all fixed samples and aggregation.'
    echo 'Use a new run ID and prepare the new bundle; submit/recover cannot update an old bundle.'
    exit 0 ;;
  prepare|run|submit|recover|status) ;;
  *) echo 'Unknown action; use help' >&2; exit 2 ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${RUN_ID:?Use a fresh exp23-search-* RUN_ID}"
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}"
fi
: "${BUCKET:?Set BUCKET to the FHP UCV results bucket}"
: "${SA_EMAIL:?Set the existing FHP runner service-account email}"
LAUNCHER="$SUITE_ROOT/gcp/exp23_ucv_flop_resolving_batch.py"
[[ -f "$LAUNCHER" ]] || { echo "Missing shared evaluator: $LAUNCHER" >&2; exit 2; }
RUN_DIR="${EXP23_RUN_DIR:-$REPO_ROOT/outputs/batch/$RUN_ID}"
COMMON=(--native-repo "$REPO_ROOT" --run-dir "$RUN_DIR")
if [[ "$ACTION" == run ]]; then
  "${PYTHON:-python3}" "$LAUNCHER" prepare "${COMMON[@]}"
  ACTION=submit
fi
if [[ "$ACTION" == recover ]]; then
  : "${RECOVERY_TAG:?Set a new recovery tag (1-10 lowercase characters)}"
  COMMON+=(--recovery-tag "$RECOVERY_TAG")
fi
exec "${PYTHON:-python3}" "$LAUNCHER" "$ACTION" "${COMMON[@]}"
