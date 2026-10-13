#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
case "$ACTION" in
  help|-h|--help)
    echo 'Experiment 35: smoke-local | dry-run | smoke-cloud | run | status'
    echo 'Single n2-standard-16; all six configurations; 24h models; no training.'
    echo 'Full run: seeds 0/1/2. Cloud smoke: seed 0, every encoder. Optional RESUME_RUN_ID.'
    exit 0 ;;
  smoke-local)
    exec "${PYTHON:-python3}" -m pytest -q tests/test_exp35_feature_rare_audit.py ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${RUN_ID:?Set a fresh exp35-rare-* RUN_ID}"
[[ "$RUN_ID" =~ ^exp35-rare-[a-z0-9-]+$ && ${#RUN_ID} -le 43 ]] || { echo 'Invalid RUN_ID'; exit 2; }
[[ "$BUCKET" =~ ^gs://[a-z0-9][a-z0-9._-]+/?$ ]] || { echo 'Expected bucket root'; exit 2; }
if [[ "$ACTION" == status ]]; then
  exec gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:${RUN_ID}"
fi
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed commit}"
REPO_REF="$(git rev-parse "$REPO_REF^{commit}")"
case "$ACTION" in
  dry-run) KIND="${JOB_KIND:-run}" ;;
  smoke-cloud) KIND=smoke ;;
  run) KIND=run ;;
  *) echo 'Unknown action'; exit 2 ;;
esac
JOB_DIR="$(mktemp -d /tmp/fhp-exp35-job.XXXXXX)"
EXTRA=(--repo-url 'https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git')
if [[ -n "${RESUME_RUN_ID:-}" ]]; then
  [[ "$RESUME_RUN_ID" =~ ^exp35-rare-[a-z0-9-]+$ && "$RESUME_RUN_ID" != "$RUN_ID" ]] || exit 2
  EXTRA+=(--resume-run-id "$RESUME_RUN_ID")
fi
"${PYTHON:-python3}" gcp/exp35_feature_rare_audit_batch.py --kind "$KIND" --output "$JOB_DIR/job.json" \
  --run-id "$RUN_ID" --bucket-root "${BUCKET%/}" --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" \
  --project-id "$PROJECT_ID" --region "$REGION" "${EXTRA[@]}"
if [[ "$ACTION" == dry-run ]]; then echo "Prepared $JOB_DIR/job.json; nothing submitted"; exit 0; fi
git cat-file -e "$REPO_REF:experiments/fhp/exp35_fhp_feature_rare_audit/run.py"
if [[ "$KIND" == run ]]; then
  : "${SMOKE_RUN_ID:?Set the successful exp35 cloud smoke RUN_ID}"
  [[ "$SMOKE_RUN_ID" =~ ^exp35-rare-[a-z0-9-]+$ && "$SMOKE_RUN_ID" != "$RUN_ID" ]] || exit 2
  [[ "$(gcloud batch jobs describe "$SMOKE_RUN_ID-smoke" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)')" == SUCCEEDED ]] || exit 2
  "${PYTHON:-python3}" - "$BUCKET/$SMOKE_RUN_ID/analysis" "$REPO_REF" <<'PY'
import json,subprocess,sys
from experiments.fhp.exp28_fhp_rare_state_audit.common import digest,file_hash
from experiments.fhp.exp35_fhp_feature_rare_audit.run import HERE
def get(name):return json.loads(subprocess.check_output(['gcloud','storage','cat',sys.argv[1]+'/'+name]))
proof,protocol,panel=get('SUCCESS.json'),get('protocol.json'),get('panel.json')
if (proof.get('status')!='complete' or proof.get('models')!=6 or proof.get('smoke') is not True
    or protocol['revision']!=sys.argv[2] or protocol['source_manifest_sha256']!=file_hash(HERE/'inputs.json')
    or proof['fingerprint']!=digest(dict(contract=protocol,panel=panel))):
    raise SystemExit('Cloud smoke evidence/source revision mismatch')
PY
fi
PREFLIGHT=("${PYTHON:-python3}" -m experiments.fhp.exp35_fhp_feature_rare_audit.run preflight
  --bucket "$BUCKET" --output "$JOB_DIR/sources.json")
if [[ "$KIND" == smoke ]]; then PREFLIGHT+=(--smoke); fi
"${PREFLIGHT[@]}"
if gcloud storage ls "$BUCKET/$RUN_ID/" >/dev/null 2>&1; then
  echo 'Output prefix already exists; use a fresh RUN_ID (RESUME_RUN_ID may reference the old run).'; exit 2
fi
gcloud batch jobs submit "$RUN_ID-$KIND" --project "$PROJECT_ID" --location "$REGION" --config "$JOB_DIR/job.json"
echo "Submitted $RUN_ID-$KIND. No additional submissions needed."
