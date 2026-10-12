#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-help}"
case "$ACTION" in
  help|-h|--help)
    echo 'Experiment 27: prepare | run | submit | prepare-smoke | run-smoke | submit-smoke | status'
    echo 'One n2-standard-8: learning controls, loss ablation, conditional three-seed qualification.'
    echo 'Set PROJECT_ID REGION BUCKET SA_EMAIL RUN_ID (fresh exp27-attack-*). No REPO_REF needed.'
    exit 0 ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}" "${REGION:?Set REGION}" "${RUN_ID:?Set a fresh exp27-attack-* RUN_ID}"
[[ "$RUN_ID" =~ ^exp27-attack-[a-z0-9][a-z0-9-]*$ && ${#RUN_ID} -le 46 ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}-"
fi
: "${BUCKET:?Set BUCKET}" "${SA_EMAIL:?Set SA_EMAIL}"
ARGS=()
if [[ "$ACTION" == *-smoke ]]; then ACTION="${ACTION%-smoke}"; ARGS+=(--smoke); fi
case "$ACTION" in prepare|run|submit) ;; *) echo 'Unknown action; use help' >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVALUATION_DIR="${FHP_EVAL_REPO:-$REPO_DIR/../../fhp-evaluation-suite}"
LAUNCHER="$EVALUATION_DIR/gcp/exp27_attacker_development_batch.py"
[[ -f "$LAUNCHER" ]] || { echo 'Update the sibling evaluation-suite checkout or set FHP_EVAL_REPO' >&2; exit 2; }
exec "${PYTHON:-python3}" "$LAUNCHER" "$ACTION" --native-repo "$REPO_DIR" \
  --run-dir "${EXP27_RUN_DIR:-$REPO_DIR/outputs/batch/$RUN_ID}" ${ARGS[@]+"${ARGS[@]}"}
