#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
if [[ "$ACTION" == smoke-local ]]; then
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/exp20-smoke.XXXXXX)}"
  exec "${PYTHON:-python3}" -m experiments.fhp.exp20_fhp_half_critic_updates.run smoke --output-root "$SMOKE_OUTPUT" --no-resume
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 20 workflow commit}"
BUCKET_ROOT="${BUCKET%/}"
[[ "$BUCKET_ROOT" == gs://* ]] || BUCKET_ROOT="gs://$BUCKET_ROOT"
RUN_ID="${RUN_ID:-exp20-critic24-$(date -u '+%Y%m%d-%H%M%S')}"
PARALLELISM="${PARALLELISM:-3}"
EXP20_TOTAL_HOURS="${EXP20_TOTAL_HOURS:-24}"
EXP20_SOURCE_RUN_ID="${EXP20_SOURCE_RUN_ID:-}"
if [[ "$ACTION" == run && ( "$EXP20_TOTAL_HOURS" != 24 || -n "$EXP20_SOURCE_RUN_ID" ) ]]; then
  echo "run starts fresh 24h models. Clear EXP20_SOURCE_RUN_ID and set EXP20_TOTAL_HOURS=24, or explicitly use extend." >&2; exit 2
fi
if [[ "$ACTION" == extend && -z "$EXP20_SOURCE_RUN_ID" ]]; then
  echo "extend requires EXP20_SOURCE_RUN_ID and a NEW RUN_ID with a larger cumulative budget." >&2; exit 2
fi
EVAL_MAX_HOURS="${EVAL_MAX_HOURS:-48}"
TEMP_DIR="$(mktemp -d /tmp/exp20-fhp-batch.XXXXXX)"

build() {
  python3 gcp/exp20_half_critic_updates_batch.py --kind "$1" --output "$TEMP_DIR/$1.json" \
    --run-id "$RUN_ID" --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" \
    --repo-ref "$REPO_REF" --project-id "$PROJECT_ID" --region "$REGION" \
    --parallelism "$PARALLELISM" --max-hours "$EVAL_MAX_HOURS" \
    --total-hours "$EXP20_TOTAL_HOURS" --source-run-id "$EXP20_SOURCE_RUN_ID" "${@:2}"
}
for stage in controller smoke train aggregate; do build "$stage"; done

preflight() {
  [[ "$(gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)')" == "$SA_EMAIL" ]]
  git cat-file -e "$REPO_REF^{commit}"
  git show "$REPO_REF:experiments/fhp/exp20_fhp_half_critic_updates/run.py" >/dev/null
  git show "$REPO_REF:gcp/exp20_half_critic_updates_batch.py" >/dev/null
}
submit() {
  gcloud batch jobs submit "$1" --project "$PROJECT_ID" --location "$REGION" --config "$2"
}
state() {
  gcloud batch jobs describe "$1" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)'
}
ensure_stage() {
  local stage="$1" job="$RUN_ID-$1" current
  # Listing errors are fatal; only a successful empty result permits submission.
  current="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="name=projects/$PROJECT_ID/locations/$REGION/jobs/$job" --format='value(status.state)')"
  if [[ -z "$current" ]]; then submit "$job" "$TEMP_DIR/$stage.json"; fi
  while true; do
    current="$(state "$job")"
    echo "$job: $current"
    case "$current" in
      SUCCEEDED) return 0 ;;
      FAILED|DELETION_IN_PROGRESS)
        echo "Stopped at $stage. No training will be restarted automatically." >&2; return 1 ;;
      QUEUED|SCHEDULED|RUNNING) sleep 30 ;;
      *) echo "Unexpected Batch state: $current" >&2; return 1 ;;
    esac
  done
}

case "$ACTION" in
  dry-run) echo "Experiment 20 specifications (no cloud actions): $TEMP_DIR" ;;
  dry-run-evaluate)
    build evaluate
    echo "Experiment 20 evaluation specification (no cloud actions): $TEMP_DIR/evaluate.json"
    ;;
  preflight) preflight ;;
  smoke-cloud)
    preflight
    submit "$RUN_ID-smoke" "$TEMP_DIR/smoke.json"
    echo "Submitted smoke only. Use this RUN_ID for run after it succeeds to reuse the gate."
    ;;
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
      --filter="name:${RUN_ID}" --format='table(name.basename(),status.state)'
    ;;
  run|extend|resume)
    preflight
    controller="$RUN_ID-controller"
    if [[ "$ACTION" == resume ]]; then controller="$RUN_ID-controller-$(date -u '+%H%M%S')"; fi
    submit "$controller" "$TEMP_DIR/controller.json"
    echo "Submitted $controller. You may close your laptop. Outputs: $BUCKET_ROOT/$RUN_ID"
    ;;
  evaluate|evaluate-resume)
    build evaluate
    preflight
    if [[ "$(state "$RUN_ID-train")" != SUCCEEDED || "$(state "$RUN_ID-aggregate")" != SUCCEEDED ]]; then
      echo "Both training and aggregation must succeed before evaluation." >&2; exit 1
    fi
    job="$RUN_ID-evaluate"
    if [[ "$ACTION" == evaluate-resume ]]; then
      active="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
        --filter="name:${RUN_ID}-eval AND (status.state=QUEUED OR status.state=SCHEDULED OR status.state=RUNNING)" \
        --format='value(name)')"
      if [[ -n "$active" ]]; then
        echo "An evaluation is still active; refusing concurrent writes: $active" >&2; exit 1
      fi
      build evaluate --resume
      job="$RUN_ID-eval-retry-$(date -u '+%H%M%S')"
    fi
    submit "$job" "$TEMP_DIR/evaluate.json"
    ;;
  orchestrate)
    [[ "${EXP20_REMOTE_CONTROLLER:-}" == 1 ]] || { echo "Internal controller action" >&2; exit 2; }
    preflight
    for stage in smoke train aggregate; do ensure_stage "$stage"; done
    echo "Experiment 20 training and aggregation complete. Evaluation is separate: use evaluate when ready."
    ;;
  *) echo "Usage: $0 [run|preflight|smoke-local|smoke-cloud|status|dry-run|dry-run-evaluate|resume|extend|evaluate|evaluate-resume]" >&2; exit 2 ;;
esac
