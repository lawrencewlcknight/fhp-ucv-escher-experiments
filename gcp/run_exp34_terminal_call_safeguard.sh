#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUITE_ROOT="${FHP_EVALUATION_SUITE:-$REPO_ROOT/../../fhp-evaluation-suite}"
ACTION="${1:-help}"
case "$ACTION" in
  help|-h|--help)
    echo 'Experiment 34: [prepare|run|submit|recover|status]'
    echo 'Default EXP34_MODE=smoke: small end-to-end evaluation, no training.'
    echo 'EXP34_MODE=full: all 3 frozen Exp16 48h seeds, paired games and full evaluation suite.'
    echo 'Set PROJECT_ID REGION BUCKET SA_EMAIL RUN_ID (fresh exp34-safeguard-*).'
    echo 'One n2-standard-16 VM, 48h hard cap in full mode; timing gate before large gameplay.'
    echo 'Both repos are source-bundled by hash. REPO_REF is deliberately unused.'
    echo 'run prepares and submits; recover uses the original immutable bundle and RECOVERY_TAG.'
    exit 0 ;;
  prepare|run|submit|recover|status) ;;
  *) echo 'Unknown action; use help' >&2; exit 2 ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${RUN_ID:?Use a fresh exp34-safeguard-* RUN_ID}"
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}"
fi
: "${BUCKET:?Set the FHP UCV results bucket}"
: "${SA_EMAIL:?Set the existing FHP runner service-account email}"
LAUNCHER="$SUITE_ROOT/gcp/exp34_ucv_terminal_safeguard_batch.py"
[[ -f "$LAUNCHER" ]] || { echo "Missing shared evaluator: $LAUNCHER" >&2; exit 2; }
RUN_DIR="${EXP34_RUN_DIR:-$REPO_ROOT/outputs/batch/$RUN_ID}"
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
