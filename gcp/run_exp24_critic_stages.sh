#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
case "$ACTION" in
  help|-h|--help)
    echo 'Experiments 24 + 25: smoke-local | dry-run | smoke-cloud | run | status'
    echo 'After cloud smoke, run submits ONE job for all reference generation, audits and both reports.'
    echo 'PARALLELISM is intentionally ignored: one VM, eight traversal actors, eight learner threads.'
    exit 0 ;;
  smoke-local)
    exec "${PYTHON:-python3}" -m experiments.fhp.exp24_fhp_critic_budget_stages.run smoke-local \
      --output "${SMOKE_OUTPUT:-$(mktemp -d /tmp/fhp-exp24-smoke.XXXXXX)}" ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET to gs://bucket}"
: "${RUN_ID:?Set a fresh exp24-critic-* RUN_ID}"
[[ "$RUN_ID" =~ ^exp24-critic-[a-z0-9-]+$ && ${#RUN_ID} -le 40 ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
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
    : "${SMOKE_RUN_ID:?Set the successful exp24 cloud SMOKE_RUN_ID}"
    [[ "$SMOKE_RUN_ID" =~ ^exp24-critic-[a-z0-9-]+$ && ${#SMOKE_RUN_ID} -le 40 && "$RUN_ID" != "$SMOKE_RUN_ID" ]] || exit 2
    [[ "$(gcloud batch jobs describe "$SMOKE_RUN_ID-smoke" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)')" == SUCCEEDED ]] || {
      echo 'Required cloud smoke has not succeeded' >&2; exit 2;
    }
    "${PYTHON:-python3}" gcp/exp24_critic_stages_gate.py --uri "$BUCKET_ROOT/$SMOKE_RUN_ID" --commit "$REPO_REF"
    KIND=run ;;
  *) echo 'Unknown action; use help' >&2; exit 2 ;;
esac
JOB_DIR="$(mktemp -d /tmp/fhp-exp24-job.XXXXXX)"
"${PYTHON:-python3}" gcp/exp24_critic_stages_batch.py --kind "$KIND" --output "$JOB_DIR/job.json" \
  --run-id "$RUN_ID" --bucket-root "$BUCKET_ROOT" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
  --project-id "$PROJECT_ID" --region "$REGION"
if [[ "$ACTION" == dry-run ]]; then echo "Prepared $JOB_DIR/job.json; nothing submitted"; exit 0; fi
git cat-file -e "$REPO_REF:experiments/fhp/exp24_fhp_critic_budget_stages/run.py" || {
  echo 'REPO_REF does not contain Experiments 24/25; commit and push first' >&2; exit 2;
}
# Existing metadata-only source verification, no local ML installation or urllib TLS.
"${PYTHON:-python3}" -m experiments.fhp.exp17_fhp_critic_budget.source --bucket "$BUCKET_ROOT" --seed 0
if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/" >/dev/null 2>&1; then
  echo 'Output namespace already exists; use a fresh RUN_ID' >&2; exit 2
fi
gcloud batch jobs submit "$RUN_ID-$KIND" --project "$PROJECT_ID" --location "$REGION" --config "$JOB_DIR/job.json"
echo "Submitted $RUN_ID-$KIND. This single job executes the complete paired-study workflow."
