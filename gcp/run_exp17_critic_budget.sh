#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ "$ACTION" == smoke-local ]]; then
  exec "${PYTHON:-python3}" -m experiments.fhp.exp17_fhp_critic_budget.run smoke \
    --output-root "${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-exp17-smoke.XXXXXX)}"
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 17 commit}"
export CLOUDSDK_CORE_PROJECT="$PROJECT_ID"
RUN_ID="${RUN_ID:-exp17-critic-$(date -u '+%Y%m%d-%H%M%S')}"
PARALLELISM="${PARALLELISM:-3}"
[[ "$PARALLELISM" =~ ^[123]$ ]] || { echo "PARALLELISM must be 1, 2 or 3" >&2; exit 2; }
[[ ${#RUN_ID} -le 35 && "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "$RUN_ID" != exp10-features-20261001-161740 ]] || { echo "Output RUN_ID must differ from the immutable source" >&2; exit 2; }
[[ "$BUCKET" == gs://* ]] && BUCKET_ROOT="${BUCKET%/}" || BUCKET_ROOT="gs://${BUCKET%/}"
TEMP_DIR="$(mktemp -d /tmp/fhp-exp17-jobs.XXXXXX)"
TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
CONTROLLER_ACTION=orchestrate
SUFFIX=""
if [[ "$ACTION" == resume || "$ACTION" == orchestrate-resume ]]; then
  CONTROLLER_ACTION=orchestrate-resume
  SUFFIX="-retry-$TAG"
fi
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
  local name="$RUN_ID-$1$SUFFIX" status
  if status="$(state "$name" 2>/dev/null)"; then
    [[ "$status" == SUCCEEDED ]] && return 0
    [[ "$status" == FAILED || "$status" == DELETION_IN_PROGRESS ]] && return 1
  else
    submit "$name" "$1" || return $?
  fi
  wait_job "$name"
}
case "$ACTION" in
  run|resume|smoke-cloud)
    git cat-file -e "$REPO_REF:experiments/fhp/exp17_fhp_critic_budget/run.py" || {
      echo "REPO_REF lacks Experiment 17. Pull the pushed code and reset REPO_REF." >&2; exit 2;
    }
    REPO_REF="$(git rev-parse "$REPO_REF^{commit}")"
    email="$(gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)')"
    [[ "$email" == "$SA_EMAIL" ]] || { echo "Service-account identity mismatch" >&2; exit 2; }
    python3 -m experiments.fhp.exp17_fhp_critic_budget.source --bucket "$BUCKET_ROOT" ;;
  dry-run) ;;
  orchestrate|orchestrate-resume)
    [[ "${EXP17_REMOTE_CONTROLLER:-}" == 1 ]] || { echo "Internal action" >&2; exit 2; } ;;
  status)
    exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}" ;;
  *) echo "Usage: $0 [run|resume|smoke-local|smoke-cloud|status|dry-run]" >&2; exit 2 ;;
esac
for kind in controller smoke train aggregate; do
  python3 "$SCRIPT_DIR/exp17_critic_budget_batch.py" --kind "$kind" --output "$TEMP_DIR/$kind.json" \
    --run-id "$RUN_ID" --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
    --project-id "$PROJECT_ID" --region "$REGION" --parallelism "$PARALLELISM" --controller-action "$CONTROLLER_ACTION"
done
case "$ACTION" in
  dry-run) echo "Job specifications: $TEMP_DIR" ;;
  smoke-cloud) submit "$RUN_ID-smoke$SUFFIX" smoke ;;
  run|resume)
    submit "$RUN_ID-controller$SUFFIX" controller
    echo "Submitted; cloud smoke, three audit workers and aggregation run remotely. Results: $BUCKET_ROOT/$RUN_ID/analysis/" ;;
  orchestrate|orchestrate-resume)
    ensure smoke
    ensure train
    ensure aggregate ;;
esac
