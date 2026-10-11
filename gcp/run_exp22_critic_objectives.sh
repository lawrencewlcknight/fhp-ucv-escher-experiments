#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
case "$ACTION" in
  help|-h|--help)
    echo 'Experiment 22: [smoke-local|dry-run|smoke-cloud|run|status]'
    echo 'After smoke succeeds, run submits ONE job: seed 0, pilot report, seeds 1/2, final report.'
    echo 'Legacy staged actions remain: pilot|aggregate-pilot|full|aggregate.'
    exit 0 ;;
  smoke-local)
    exec "${PYTHON:-python3}" -m experiments.fhp.exp22_fhp_critic_objectives.run smoke \
      --output "${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-exp22-smoke.XXXXXX)}" ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET to gs://bucket}"
: "${RUN_ID:?Set a fresh exp22-critic-* RUN_ID}"
[[ "$RUN_ID" =~ ^exp22-critic-[a-z0-9-]+$ && ${#RUN_ID} -le 40 ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
[[ "$BUCKET" =~ ^gs://[a-z0-9][a-z0-9._-]+/?$ ]] || { echo 'Use a bucket without subdirectories' >&2; exit 2; }
BUCKET_ROOT="${BUCKET%/}"
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}"
fi
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed experiment commit}"
REPO_REF="$(git rev-parse "$REPO_REF^{commit}")"
if [[ "$ACTION" != dry-run ]]; then
  git cat-file -e "$REPO_REF:experiments/fhp/exp22_fhp_critic_objectives/run.py" || {
    echo 'REPO_REF does not contain Experiment 22' >&2; exit 2;
  }
fi
export CLOUDSDK_CORE_PROJECT="$PROJECT_ID"
gate() {
  local id="$1" stage="$2" expected_smoke="$3" state
  [[ "$id" =~ ^exp22-critic-[a-z0-9-]+$ && ${#id} -le 40 ]] || exit 2
  state="$(gcloud batch jobs describe "$id-$stage" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)')"
  [[ "$state" == SUCCEEDED ]] || { echo "Required job $id-$stage has not succeeded" >&2; exit 2; }
  # Standard library only; no local ML install or Python HTTPS requests needed.
  "${PYTHON:-python3}" gcp/exp22_critic_objectives_gate.py \
    --uri "$BUCKET_ROOT/$id/workers/seed_0" --commit "$REPO_REF" --smoke "$expected_smoke"
}
case "$ACTION" in
  smoke-cloud) KIND=smoke ;;
  run|pilot)
    : "${SMOKE_RUN_ID:?Set the successful cloud SMOKE_RUN_ID}"
    gate "$SMOKE_RUN_ID" smoke true
    [[ "$RUN_ID" != "$SMOKE_RUN_ID" ]] || { echo 'Use a fresh post-smoke RUN_ID' >&2; exit 2; }
    KIND="$ACTION" ;;
  full)
    : "${PILOT_RUN_ID:?Set the completed PILOT_RUN_ID}"
    [[ "$RUN_ID" != "$PILOT_RUN_ID" ]] || { echo 'Full run requires a new ID' >&2; exit 2; }
    gate "$PILOT_RUN_ID" pilot false
    KIND=full ;;
  aggregate-pilot)
    gate "$RUN_ID" pilot false
    KIND=aggregate-pilot ;;
  aggregate)
    : "${PILOT_RUN_ID:?Set PILOT_RUN_ID}"
    gate "$PILOT_RUN_ID" pilot false
    [[ "$(gcloud batch jobs describe "$RUN_ID-full" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)')" == SUCCEEDED ]] || exit 2
    KIND=aggregate ;;
  dry-run) KIND="${JOB_KIND:-run}" ;;
  *) echo 'Unknown action; use help' >&2; exit 2 ;;
esac
JOB_DIR="$(mktemp -d /tmp/fhp-exp22-job.XXXXXX)"
"${PYTHON:-python3}" gcp/exp22_critic_objectives_batch.py --kind "$KIND" --output "$JOB_DIR/job.json" \
  --run-id "$RUN_ID" --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
  --project-id "$PROJECT_ID" --region "$REGION" --parallelism "${PARALLELISM:-2}" --pilot-run-id "${PILOT_RUN_ID:-}"
if [[ "$ACTION" == dry-run ]]; then echo "Prepared $JOB_DIR/job.json; nothing submitted"; exit 0; fi
if [[ "$KIND" == smoke || "$KIND" == run || "$KIND" == pilot || "$KIND" == full ]]; then
  # Read-only metadata/source-object checks before buying compute.
  "${PYTHON:-python3}" -m experiments.fhp.exp17_fhp_critic_budget.source --bucket "$BUCKET_ROOT"
  if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/workers/" >/dev/null 2>&1; then
    echo 'Output namespace already exists; use a fresh RUN_ID' >&2; exit 2
  fi
fi
gcloud batch jobs submit "$RUN_ID-$KIND" --project "$PROJECT_ID" --location "$REGION" --config "$JOB_DIR/job.json"
if [[ "$KIND" == run ]]; then
  echo "Submitted $RUN_ID-run. All three seeds and reports run automatically inside this one job."
else
  echo "Submitted $RUN_ID-$KIND. No separate follow-on job will launch automatically."
fi
