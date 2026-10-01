#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
MODULE="experiments.fhp.exp5_fhp_critic_target_cache.run"
if [[ "$ACTION" == smoke-local ]]; then
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/critic-cache-smoke.XXXXXX)}"
  exec "${PYTHON:-python3}" -m "$MODULE" smoke --output-root "$SMOKE_OUTPUT"
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to a pushed commit}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-exp2-fhp-20260921-093839}"
RUN_ID="${RUN_ID:-exp5-cache-$(date -u '+%Y%m%d-%H%M%S')}"
[[ "$BUCKET" == gs://* ]] && BUCKET_ROOT="${BUCKET%/}" || BUCKET_ROOT="gs://${BUCKET%/}"
if [[ ${#RUN_ID} -gt 35 || ! "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "Use a 2-35 character lowercase RUN_ID" >&2; exit 2
fi
if [[ ! "$SOURCE_RUN_ID" =~ ^[a-z][a-z0-9-]+$ ]]; then
  echo "Invalid source run ID" >&2; exit 2
fi
TEMP_DIR="$(mktemp -d /tmp/critic-cache-jobs.XXXXXX)"
CONTROLLER_ACTION=orchestrate
CONTROLLER="$RUN_ID-controller"
SMOKE="$RUN_ID-smoke"
WORKER="$RUN_ID-benchmark"
AGGREGATE="$RUN_ID-aggregate"
if [[ "$ACTION" == resume || "$ACTION" == orchestrate-resume ]]; then
  TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
  CONTROLLER_ACTION=orchestrate-resume
  CONTROLLER="$RUN_ID-controller-resume-$TAG"
  SMOKE="$RUN_ID-smoke-retry-$TAG"
  WORKER="$RUN_ID-benchmark-retry-$TAG"
  AGGREGATE="$RUN_ID-reaggregate-$TAG"
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
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $1 $status"
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
case "$ACTION" in
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
      --filter="name:$RUN_ID" --format='table(name.basename(),status.state)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"; exit 0 ;;
  run|resume|smoke-cloud)
    EMAIL="$(gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)')"
    [[ "$EMAIL" == "$SA_EMAIL" ]] || { echo "Service-account preflight failed" >&2; exit 2; }
    git cat-file -e "$REPO_REF^{commit}"
    git show "$REPO_REF:experiments/fhp/exp5_fhp_critic_target_cache/run.py" >/dev/null
    for seed in 0 1 2; do
      source="$BUCKET_ROOT/$SOURCE_RUN_ID/workers/task_$(printf '%03d' "$seed")_lossless_structured_ucv_escher_seed_$seed"
      gcloud storage ls "$source/SUCCESS.json" "$source/checkpoint_manifest.json" "$source/run_manifest.json" \
        "$source/training_states/lossless_structured_ucv_escher_seed_${seed}_time_24h.pt" >/dev/null
    done ;;
  dry-run|orchestrate|orchestrate-resume) ;;
  *) echo "Usage: $0 [run|resume|smoke-local|smoke-cloud|status|dry-run]" >&2; exit 2 ;;
esac
for kind in controller smoke worker aggregate; do
  python3 "$SCRIPT_DIR/frozen_critic_target_efficiency_batch.py" --kind "$kind" \
    --output "$TEMP_DIR/$kind.json" --project "$PROJECT_ID" --region "$REGION" \
    --bucket "$BUCKET_ROOT" --run-id "$RUN_ID" --source-run-id "$SOURCE_RUN_ID" \
    --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" --controller-action "$CONTROLLER_ACTION"
done
case "$ACTION" in
  dry-run) echo "Specifications: $TEMP_DIR" ;;
  smoke-cloud) submit "$SMOKE" smoke ;;
  run|resume)
    submit "$CONTROLLER" controller
    echo "Remote controller submitted. The laptop may disconnect."
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/" ;;
  orchestrate|orchestrate-resume)
    [[ "${CRITIC_CACHE_REMOTE_CONTROLLER:-}" == 1 ]] || { echo "Internal action" >&2; exit 2; }
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --limit=1 >/dev/null
    if [[ "$ACTION" == orchestrate-resume ]]; then
      retry "$RUN_ID-smoke" "$SMOKE" smoke
      retry "$RUN_ID-benchmark" "$WORKER" worker
      retry "$RUN_ID-aggregate" "$AGGREGATE" aggregate
    else
      ensure "$SMOKE" smoke
      ensure "$WORKER" worker
      ensure "$AGGREGATE" aggregate
    fi ;;
esac
