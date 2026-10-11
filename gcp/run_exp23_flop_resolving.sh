#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUITE_ROOT="${FHP_EVALUATION_SUITE:-$REPO_ROOT/../../fhp-evaluation-suite}"
ACTION="${1:-help}"
case "$ACTION" in
  help|-h|--help)
    echo 'Experiment 23: [prepare|run|submit|recover|status]'
    echo 'One n2-standard-16 VM; mandatory correctness/timing checks, then paired gameplay.'
    echo 'run prepares and submits; submit uses an existing prepared bundle; recover resumes shards.'
    echo 'Set PROJECT_ID REGION BUCKET SA_EMAIL RUN_ID. REPO_REF is deliberately unused.'
    echo 'Set EXP23_MODE=feasibility for profiling and timed probes ONLY (recommended next run).'
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
