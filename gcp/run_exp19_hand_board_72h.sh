#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 19 workflow commit}"
BUCKET_ROOT="${BUCKET%/}"
[[ "$BUCKET_ROOT" == gs://* ]] || BUCKET_ROOT="gs://$BUCKET_ROOT"
RUN_ID="${RUN_ID:-exp19-feat72-$(date -u '+%Y%m%d-%H%M%S')}"
PARALLELISM="${PARALLELISM:-3}"
EVAL_MAX_HOURS="${EVAL_MAX_HOURS:-48}"
TEMP_DIR="$(mktemp -d /tmp/exp19-fhp-batch.XXXXXX)"

build() {
  python3 gcp/exp19_hand_board_72h_batch.py --kind "$1" --output "$TEMP_DIR/$1.json" \
    --run-id "$RUN_ID" --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" \
    --repo-ref "$REPO_REF" --project-id "$PROJECT_ID" --region "$REGION" \
    --parallelism "$PARALLELISM" --max-hours "$EVAL_MAX_HOURS" "${@:2}"
}
for stage in controller smoke train aggregate evaluate; do build "$stage"; done

preflight() {
  [[ "$(gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)')" == "$SA_EMAIL" ]]
  git cat-file -e "$REPO_REF^{commit}"
  git show "$REPO_REF:experiments/fhp/exp19_fhp_hand_board_72h/contract.py" >/dev/null
  # Read pinned orchestration directly so local edits cannot change preflight semantics.
  git show "$REPO_REF:experiments/fhp/exp19_fhp_hand_board_72h/contract.py" \
    > "$TEMP_DIR/pinned_contract.py"
  python3 "$TEMP_DIR/pinned_contract.py" --bucket "$BUCKET_ROOT" --output "$TEMP_DIR/source_contract.json"
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
  dry-run) echo "Experiment 19 specifications (no cloud actions): $TEMP_DIR" ;;
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
  run|resume)
    preflight
    # Full source/evaluation provenance is uploaded by the pinned controller.
    controller="$RUN_ID-controller"
    if [[ "$ACTION" == resume ]]; then controller="$RUN_ID-controller-$(date -u '+%H%M%S')"; fi
    submit "$controller" "$TEMP_DIR/controller.json"
    echo "Submitted $controller. You may close your laptop. Outputs: $BUCKET_ROOT/$RUN_ID"
    ;;
  evaluate|evaluate-resume)
    preflight
    if [[ "$(state "$RUN_ID-train")" != SUCCEEDED || "$(state "$RUN_ID-aggregate")" != SUCCEEDED ]]; then
      echo "Both training and aggregation must succeed before evaluation." >&2; exit 1
    fi
    job="$RUN_ID-evaluate"
    if [[ "$ACTION" == evaluate-resume ]]; then
      build evaluate --resume
      job="$RUN_ID-eval-retry-$(date -u '+%H%M%S')"
    fi
    submit "$job" "$TEMP_DIR/evaluate.json"
    ;;
  orchestrate)
    [[ "${EXP19_REMOTE_CONTROLLER:-}" == 1 ]] || { echo "Internal controller action" >&2; exit 2; }
    preflight
    gcloud storage cp "$TEMP_DIR/source_contract.json" "$BUCKET_ROOT/$RUN_ID/experiment19_contract.json"
    for stage in smoke train aggregate evaluate; do ensure_stage "$stage"; done
    echo "Experiment 19 training, aggregation and frozen-policy evaluation complete."
    ;;
  *) echo "Usage: $0 [run|preflight|smoke-cloud|status|dry-run|resume|evaluate|evaluate-resume]" >&2; exit 2 ;;
esac
