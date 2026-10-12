#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
case "$ACTION" in
  help|-h|--help)
    echo 'Experiment 28: smoke-local | dry-run | smoke-cloud | run | status'
    echo 'After cloud smoke: one job runs seed 0, checks integrity, repeats seeds 1/2 and aggregates.'
    echo 'PARALLELISM is ignored; one n2-standard-16 VM, eight traversal actors only during continuation.'
    exit 0 ;;
  smoke-local)
    exec "${PYTHON:-python3}" -m experiments.fhp.exp28_fhp_rare_state_audit.run smoke-local \
      --output "${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-exp28-smoke.XXXXXX)}" ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET to gs://bucket}"
: "${RUN_ID:?Set a fresh exp28-rare-* RUN_ID}"
[[ "$RUN_ID" =~ ^exp28-rare-[a-z0-9-]+$ && ${#RUN_ID} -le 43 ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
[[ "$BUCKET" =~ ^gs://[a-z0-9][a-z0-9._-]+/?$ ]] || { echo 'Expected bucket without subdirectories' >&2; exit 2; }
BUCKET_ROOT="${BUCKET%/}"
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}"
fi
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed experiment commit}"
REPO_REF="$(git rev-parse "$REPO_REF^{commit}")"
export CLOUDSDK_CORE_PROJECT="$PROJECT_ID"
case "$ACTION" in
  dry-run) KIND="${JOB_KIND:-run}" ;;
  smoke-cloud) KIND=smoke ;;
  run)
    : "${SMOKE_RUN_ID:?Set the successful exp28 cloud SMOKE_RUN_ID}"
    [[ "$SMOKE_RUN_ID" =~ ^exp28-rare-[a-z0-9-]+$ && "$RUN_ID" != "$SMOKE_RUN_ID" ]] || exit 2
    [[ "$(gcloud batch jobs describe "$SMOKE_RUN_ID-smoke" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)')" == SUCCEEDED ]] || {
      echo 'Required cloud smoke has not succeeded' >&2; exit 2;
    }
    "${PYTHON:-python3}" gcp/exp28_rare_state_gate.py --uri "$BUCKET_ROOT/$SMOKE_RUN_ID" --commit "$REPO_REF"
    KIND=run ;;
  *) echo 'Unknown action; use help' >&2; exit 2 ;;
esac
JOB_DIR="$(mktemp -d /tmp/fhp-exp28-job.XXXXXX)"
"${PYTHON:-python3}" gcp/exp28_rare_state_batch.py --kind "$KIND" --output "$JOB_DIR/job.json" \
  --run-id "$RUN_ID" --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
  --project-id "$PROJECT_ID" --region "$REGION"
if [[ "$ACTION" == dry-run ]]; then echo "Prepared $JOB_DIR/job.json; nothing submitted"; exit 0; fi
git cat-file -e "$REPO_REF:experiments/fhp/exp28_fhp_rare_state_audit/run.py" || {
  echo 'REPO_REF does not contain Experiment 28; commit and push first' >&2; exit 2;
}
# Keep this array nonempty: macOS Bash 3.2 treats an empty array expansion as
# unbound under `set -u`, which would otherwise break only the full-run path.
PREFLIGHT_COMMAND=("${PYTHON:-python3}" -m experiments.fhp.exp28_fhp_rare_state_audit.run preflight
  --bucket "$BUCKET_ROOT" --output "$JOB_DIR/source_preflight.json")
if [[ "$KIND" == smoke ]]; then PREFLIGHT_COMMAND+=(--smoke); fi
"${PREFLIGHT_COMMAND[@]}"
if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/" >/dev/null 2>&1; then
  echo 'Output namespace already exists; use a fresh RUN_ID' >&2; exit 2
fi
gcloud batch jobs submit "$RUN_ID-$KIND" --project "$PROJECT_ID" --location "$REGION" --config "$JOB_DIR/job.json"
echo "Submitted $RUN_ID-$KIND. No further submissions are needed for this run."
