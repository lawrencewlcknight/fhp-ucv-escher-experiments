#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILDER="$SCRIPT_DIR/exp9_cached_parallel_24h_batch.py"

if [[ "$ACTION" == "smoke-local" ]]; then
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/exp9-fhp-smoke.XXXXXX)}"
  cd "$REPO_DIR"
  exec "${PYTHON:-python3}" -m experiments.fhp.exp9_fhp_cached_parallel_24h.run \
    smoke --output-root "$SMOKE_OUTPUT" --no-resume
fi

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 9 commit SHA}"

RUN_ID="${RUN_ID:-exp9-cache24-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ ${#RUN_ID} -gt 35 || ! "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "RUN_ID must be 2-35 lowercase letters, digits or hyphens" >&2
  exit 2
fi
PARALLELISM="${PARALLELISM:-3}"
EXP9_TOTAL_HOURS="${EXP9_TOTAL_HOURS:-24}"
EXP9_SOURCE_RUN_ID="${EXP9_SOURCE_RUN_ID:-}"
if [[ "$ACTION" == "extend" && -z "$EXP9_SOURCE_RUN_ID" ]]; then
  echo "Set EXP9_SOURCE_RUN_ID to the completed Experiment 9 run; use a NEW RUN_ID." >&2
  exit 2
fi
if [[ "$BUCKET" == gs://* ]]; then
  BUCKET_ROOT="${BUCKET%/}"
else
  BUCKET_ROOT="gs://${BUCKET%/}"
fi

SMOKE_JOB="${RUN_ID}-smoke"
TRAIN_JOB="${RUN_ID}-train"
AGGREGATE_JOB="${RUN_ID}-aggregate"
CONTROLLER_JOB="${RUN_ID}-controller"
CONTROLLER_ACTION="orchestrate"
if [[ "$ACTION" == "resume" ]]; then
  RESUME_TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
  CONTROLLER_JOB="${RUN_ID}-controller-resume-${RESUME_TAG}"
  CONTROLLER_ACTION="orchestrate-resume"
elif [[ "$ACTION" == "orchestrate-resume" ]]; then
  RESUME_TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
  SMOKE_JOB="${RUN_ID}-smoke-retry-${RESUME_TAG}"
  TRAIN_JOB="${RUN_ID}-retry-${RESUME_TAG}"
  AGGREGATE_JOB="${RUN_ID}-reaggregate-${RESUME_TAG}"
  CONTROLLER_ACTION="orchestrate-resume"
fi

TEMP_DIR="$(mktemp -d /tmp/exp9-fhp-batch.XXXXXX)"
# Keep the small generated specifications for inspection.

build_json() {
  python3 "$BUILDER" --kind "$1" --output "$2" --run-id "$RUN_ID" \
    --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" \
    --repo-ref "$REPO_REF" --parallelism "$PARALLELISM" \
    --project-id "$PROJECT_ID" --region "$REGION" \
    --controller-action "$CONTROLLER_ACTION" \
    --total-hours "$EXP9_TOTAL_HOURS" --source-run-id "$EXP9_SOURCE_RUN_ID"
}
submit_job() {
  gcloud batch jobs submit "$1" --project "$PROJECT_ID" \
    --location "$REGION" --config "$2"
}
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
preflight_remote_controller() {
  local error_output
  if ! error_output="$(
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
      --limit=1 --format='value(name)' 2>&1
  )"; then
    echo "Remote controller cannot inspect Batch jobs using $SA_EMAIL." >&2
    echo "Grant roles/batch.jobsEditor to the service account before retrying." >&2
    echo "$error_output" >&2
    return 2
  fi
}
job_state() {
  gcloud batch jobs describe "$1" --project "$PROJECT_ID" \
    --location "$REGION" --format='value(status.state)'
}
wait_for_job() {
  local state
  while true; do
    if ! state="$(job_state "$1")"; then
      echo "Unable to inspect Batch job $1; aborting controller." >&2
      return 2
    fi
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $1: $state"
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|DELETION_IN_PROGRESS) return 1 ;;
    esac
    sleep 30
  done
}
ensure_job_succeeds() {
  local state
  if state="$(job_state "$1" 2>/dev/null)"; then
    [[ "$state" == "SUCCEEDED" ]] && return 0
    [[ "$state" == "FAILED" || "$state" == "DELETION_IN_PROGRESS" ]] && return 1
    wait_for_job "$1"
    return
  fi
  submit_job "$1" "$2" || return $?
  wait_for_job "$1"
}
complete_or_retry() {
  local state
  if state="$(job_state "$1" 2>/dev/null)"; then
    [[ "$state" == "SUCCEEDED" ]] && return 0
    if [[ "$state" != "FAILED" && "$state" != "DELETION_IN_PROGRESS" ]]; then
      wait_for_job "$1" && return 0
    fi
  fi
  submit_job "$2" "$3" || return $?
  wait_for_job "$2"
}

case "$ACTION" in
  run|extend|resume|smoke-cloud)
    preflight_service_account
    git -C "$REPO_DIR" cat-file -e "$REPO_REF^{commit}"
    git -C "$REPO_DIR" show "$REPO_REF:experiments/fhp/exp9_fhp_cached_parallel_24h/run.py" >/dev/null
    ;;
esac

build_json controller "$TEMP_DIR/controller.json"
build_json smoke "$TEMP_DIR/smoke.json"
build_json train "$TEMP_DIR/train.json"
build_json aggregate "$TEMP_DIR/aggregate.json"

case "$ACTION" in
  dry-run)
    echo "Experiment 9 job specifications: $TEMP_DIR"
    ;;
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
      --filter="name:${RUN_ID}" --format='table(name.basename(),status.state,createTime)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"
    ;;
  smoke-cloud)
    submit_job "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    ;;
  run|extend|resume)
    submit_job "$CONTROLLER_JOB" "$TEMP_DIR/controller.json"
    echo "Remote Experiment 9 controller submitted: $CONTROLLER_JOB"
    echo "The laptop may now be disconnected or switched off."
    ;;
  orchestrate)
    [[ "${EXP9_REMOTE_CONTROLLER:-}" == "1" ]] || {
      echo "Internal action" >&2
      exit 2
    }
    preflight_remote_controller
    ensure_job_succeeds "$SMOKE_JOB" "$TEMP_DIR/smoke.json" || {
      echo "Cloud smoke failed; production was not submitted." >&2
      exit 1
    }
    ensure_job_succeeds "$TRAIN_JOB" "$TEMP_DIR/train.json"
    ensure_job_succeeds "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    ;;
  orchestrate-resume)
    [[ "${EXP9_REMOTE_CONTROLLER:-}" == "1" ]] || {
      echo "Internal action" >&2
      exit 2
    }
    preflight_remote_controller
    complete_or_retry "${RUN_ID}-smoke" "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    complete_or_retry "${RUN_ID}-train" "$TRAIN_JOB" "$TEMP_DIR/train.json"
    complete_or_retry "${RUN_ID}-aggregate" "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    ;;
  *)
    echo "Usage: $0 [run|extend|resume|smoke-local|smoke-cloud|status|dry-run]" >&2
    exit 2
    ;;
esac
