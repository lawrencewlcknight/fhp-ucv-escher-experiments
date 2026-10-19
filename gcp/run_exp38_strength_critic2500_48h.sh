#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
RUN_ID="${RUN_ID:-exp38-rich48-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="name:${RUN_ID}" --format='table(name.basename(),status.state)'
  exit 0
fi
case "$ACTION" in run|resume|smoke-cloud|aggregate|preflight|dry-run|orchestrate) ;; *)
  echo "Usage: $0 [run|smoke-cloud|preflight|dry-run|status|resume|aggregate]" >&2; exit 2 ;; esac
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 38 WORKFLOW commit}"
BUCKET_ROOT="gs://${BUCKET#gs://}"
BUCKET_ROOT="${BUCKET_ROOT%/}"
EXP38_PARALLELISM="${EXP38_PARALLELISM:-3}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]]; then echo "Use a full workflow commit SHA" >&2; exit 2; fi
git -C "$REPO_DIR" cat-file -e "$REPO_REF:experiments/fhp/exp38_fhp_strength_critic2500_48h/contract.py"
TEMP_DIR="$(mktemp -d /tmp/exp38-batch.XXXXXX)"
mkdir "$TEMP_DIR/pinned"
# All builders, embedded smoke/report code and source checks come from the
# chosen workflow revision, not from uncommitted local changes.
git -C "$REPO_DIR" archive "$REPO_REF" gcp experiments/fhp/exp38_fhp_strength_critic2500_48h \
  | tar -x -C "$TEMP_DIR/pinned"
build() {
  python3 "$TEMP_DIR/pinned/gcp/exp38_strength_critic2500_48h_batch.py" \
    --kind "$1" --output "$TEMP_DIR/$1.json" --run-id "$RUN_ID" \
    --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
    --project-id "$PROJECT_ID" --region "$REGION" --parallelism "$EXP38_PARALLELISM"
}
for stage in controller smoke train aggregate; do build "$stage"; done
preflight() {
  [[ "$(gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)')" == "$SA_EMAIL" ]]
  python3 "$TEMP_DIR/pinned/experiments/fhp/exp38_fhp_strength_critic2500_48h/contract.py" \
    --bucket "$BUCKET_ROOT" --output "$TEMP_DIR/source_contract.json"
}
submit() {
  gcloud batch jobs submit "$1" --project "$PROJECT_ID" --location "$REGION" --config "$2"
}
state() {
  gcloud batch jobs describe "$1" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)'
}
no_active_jobs() {
  local active
  active="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="name:${RUN_ID} AND (status.state=QUEUED OR status.state=SCHEDULED OR status.state=RUNNING)" --format='value(name)')"
  if [[ -n "$active" ]]; then echo "A job is still active for this output prefix: $active" >&2; exit 2; fi
}
new_namespace() {
  # A completed smoke for this RUN_ID is reusable; never overwrite trained data.
  local object
  if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/**" >"$TEMP_DIR/objects.txt" 2>"$TEMP_DIR/storage_error.txt"; then
    while IFS= read -r object; do
      case "$object" in ""|"$BUCKET_ROOT/$RUN_ID/smoke/"*) ;;
        *) echo "Existing non-smoke outputs. Use a new RUN_ID, or resume/aggregate explicitly." >&2; exit 2 ;; esac
    done < "$TEMP_DIR/objects.txt"
  elif [[ "$( < "$TEMP_DIR/storage_error.txt")" != *"matched no objects"* ]]; then
    cat "$TEMP_DIR/storage_error.txt" >&2; exit 2
  fi
}
ensure_stage() {
  local stage="$1" job="$RUN_ID-$1" current
  current="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="name=projects/$PROJECT_ID/locations/$REGION/jobs/$job" --format='value(status.state)')"
  if [[ -z "$current" ]]; then submit "$job" "$TEMP_DIR/$stage.json"; fi
  while true; do
    current="$(state "$job")"
    echo "$job: $current"
    case "$current" in
      SUCCEEDED) return 0 ;;
      FAILED|DELETION_IN_PROGRESS) echo "Stopped at $stage; no automatic paid retraining." >&2; return 1 ;;
      QUEUED|SCHEDULED|RUNNING) sleep 30 ;;
      *) echo "Unexpected Batch state: $current" >&2; return 1 ;; esac
  done
}
case "$ACTION" in
  dry-run) echo "Prepared $TEMP_DIR/{controller,smoke,train,aggregate}.json; no cloud actions." ;;
  preflight) preflight ;;
  smoke-cloud)
    preflight; no_active_jobs; new_namespace
    submit "$RUN_ID-smoke" "$TEMP_DIR/smoke.json"
    echo "After smoke succeeds, retain RUN_ID and use run."
    ;;
  run|resume)
    preflight; no_active_jobs
    controller="$RUN_ID-controller"
    if [[ "$ACTION" == run ]]; then new_namespace
    else
      gcloud storage ls "$BUCKET_ROOT/$RUN_ID/experiment38_contract.json" >/dev/null
      controller="$RUN_ID-controller-$(date -u '+%H%M%S')"
    fi
    submit "$controller" "$TEMP_DIR/controller.json"
    echo "Submitted $controller. You may disconnect. Outputs: $BUCKET_ROOT/$RUN_ID"
    ;;
  aggregate)
    preflight; no_active_jobs
    [[ "$(state "$RUN_ID-train")" == SUCCEEDED ]] || { echo "Training must succeed first" >&2; exit 2; }
    submit "$RUN_ID-aggregate-$(date -u '+%H%M%S')" "$TEMP_DIR/aggregate.json"
    ;;
  orchestrate)
    [[ "${EXP38_REMOTE_CONTROLLER:-}" == 1 ]] || { echo "Internal controller action" >&2; exit 2; }
    preflight
    gcloud storage cp "$TEMP_DIR/source_contract.json" "$BUCKET_ROOT/$RUN_ID/experiment38_contract.json"
    for stage in smoke train aggregate; do ensure_stage "$stage"; done
    echo "Experiment 38 complete: three 48h states saved; no evaluation or further training scheduled."
    ;;
esac
