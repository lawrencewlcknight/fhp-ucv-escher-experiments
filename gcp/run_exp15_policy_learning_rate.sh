#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
EXP15_INCLUDE_LBR="${EXP15_INCLUDE_LBR:-0}"
[[ "$EXP15_INCLUDE_LBR" =~ ^[01]$ ]] || { echo "EXP15_INCLUDE_LBR must be 0 or 1" >&2; exit 2; }
LBR_ARGS=()
[[ "$EXP15_INCLUDE_LBR" == 0 ]] || LBR_ARGS+=(--include-lbr)
if [[ "$ACTION" == smoke-local ]]; then
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-exp15-smoke.XXXXXX)}"
  exec "${PYTHON:-python3}" -m experiments.fhp.exp15_fhp_policy_learning_rate.run \
    smoke --output-root "$SMOKE_OUTPUT" --evaluation-workers 1 "${LBR_ARGS[@]}"
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to a pushed commit containing Experiment 15}"
: "${EXP9_RUN_ID:?Set EXP9_RUN_ID to the completed original 24-hour Experiment 9 run}"
RUN_ID="${RUN_ID:-exp15-lr-$(date -u '+%Y%m%d-%H%M%S')}"
PARALLELISM="${PARALLELISM:-3}"
if [[ ${#RUN_ID} -gt 35 || ! "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "RUN_ID must be 2-35 lowercase letters, digits or hyphens" >&2; exit 2
fi
[[ "$EXP9_RUN_ID" =~ ^[a-z][a-z0-9-]+$ && "$EXP9_RUN_ID" != "$RUN_ID" ]] || { echo "Invalid/distinct source run required" >&2; exit 2; }
[[ "$PARALLELISM" =~ ^[123]$ ]] || { echo "PARALLELISM must be 1, 2 or 3" >&2; exit 2; }
[[ "$BUCKET" == gs://* ]] && BUCKET_ROOT="${BUCKET%/}" || BUCKET_ROOT="gs://${BUCKET%/}"
TEMP_DIR="$(mktemp -d /tmp/fhp-exp15-jobs.XXXXXX)"
CONTROLLER_ACTION=orchestrate
TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
SMOKE_JOB="$RUN_ID-smoke"
TRAIN_JOB="$RUN_ID-train"
AGGREGATE_JOB="$RUN_ID-aggregate"
CONTROLLER_JOB="$RUN_ID-controller"
if [[ "$ACTION" == resume || "$ACTION" == orchestrate-resume ]]; then
  CONTROLLER_ACTION=orchestrate-resume
  CONTROLLER_JOB="$RUN_ID-controller-resume-$TAG"
  SMOKE_JOB="$RUN_ID-smoke-retry-$TAG"
  TRAIN_JOB="$RUN_ID-retry-$TAG"
  AGGREGATE_JOB="$RUN_ID-reaggregate-$TAG"
fi
build() {
  python3 "$SCRIPT_DIR/exp15_policy_learning_rate_batch.py" --kind "$1" \
    --output "$TEMP_DIR/$1.json" --run-id "$RUN_ID" --bucket-root "$BUCKET_ROOT" \
    --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" --source-run-id "$EXP9_RUN_ID" \
    --project-id "$PROJECT_ID" --region "$REGION" --parallelism "$PARALLELISM" \
    --include-lbr "$EXP15_INCLUDE_LBR" --controller-action "$CONTROLLER_ACTION"
}
state() {
  gcloud batch jobs describe "$1" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)'
}
submit() {
  gcloud batch jobs submit "$1" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/$2.json"
}
wait_job() {
  local status
  while true; do
    status="$(state "$1")" || return 2
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $1: $status"
    case "$status" in
      SUCCEEDED) return 0 ;;
      FAILED|DELETION_IN_PROGRESS) return 1 ;;
    esac
    sleep 30
  done
}
ensure() {
  local status
  if status="$(state "$1" 2>/dev/null)"; then
    [[ "$status" == SUCCEEDED ]] && return 0
    [[ "$status" == FAILED || "$status" == DELETION_IN_PROGRESS ]] && return 1
  else
    submit "$1" "$2" || return $?
  fi
  wait_job "$1"
}
retry() {
  local status
  if status="$(state "$1" 2>/dev/null)"; then
    [[ "$status" == SUCCEEDED ]] && return 0
    if [[ "$status" != FAILED && "$status" != DELETION_IN_PROGRESS ]]; then
      wait_job "$1" && return 0
    fi
  fi
  ensure "$2" "$3"
}
preflight() {
  local email seed source
  git cat-file -e "$REPO_REF^{commit}"
  for path in experiments/fhp/exp15_fhp_policy_learning_rate/run.py gcp/run_exp15_policy_learning_rate.sh; do
    if ! git cat-file -e "$REPO_REF:$path"; then
      echo "REPO_REF lacks Experiment 15. Pull the pushed code and reset REPO_REF." >&2; return 2
    fi
  done
  email="$(gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)')"
  [[ "$email" == "$SA_EMAIL" ]] || { echo "Service-account identity mismatch" >&2; return 2; }
  for seed in 0 1 2; do
    source="$BUCKET_ROOT/$EXP9_RUN_ID/workers/task_$(printf '%03d' "$seed")_cached_parallel_structured_ucv_escher_seed_$seed"
    gcloud storage ls "$source/SUCCESS.json" "$source/checkpoint_manifest.json" \
      "$source/training_states/cached_parallel_structured_ucv_escher_seed_${seed}_time_24h.pt" >/dev/null
  done
}
case "$ACTION" in
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}" \
      --format='table(name.basename(),status.state,createTime)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"; exit 0 ;;
  run|resume|smoke-cloud) preflight ;;
  dry-run|orchestrate|orchestrate-resume) ;;
  *) echo "Usage: $0 [run|resume|smoke-local|smoke-cloud|status|dry-run]" >&2; exit 2 ;;
esac
for kind in controller smoke train aggregate; do build "$kind"; done
case "$ACTION" in
  dry-run) echo "Job specifications: $TEMP_DIR" ;;
  smoke-cloud) submit "$SMOKE_JOB" smoke ;;
  run|resume)
    submit "$CONTROLLER_JOB" controller
    echo "Submitted $CONTROLLER_JOB; your laptop can disconnect. Results: $BUCKET_ROOT/$RUN_ID/analysis/" ;;
  orchestrate|orchestrate-resume)
    [[ "${EXP15_REMOTE_CONTROLLER:-}" == 1 ]] || { echo "Internal action" >&2; exit 2; }
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --limit=1 >/dev/null
    if [[ "$ACTION" == orchestrate-resume ]]; then
      retry "$RUN_ID-smoke" "$SMOKE_JOB" smoke
      retry "$RUN_ID-train" "$TRAIN_JOB" train
      retry "$RUN_ID-aggregate" "$AGGREGATE_JOB" aggregate
    else
      ensure "$SMOKE_JOB" smoke
      ensure "$TRAIN_JOB" train
      ensure "$AGGREGATE_JOB" aggregate
    fi ;;
esac
