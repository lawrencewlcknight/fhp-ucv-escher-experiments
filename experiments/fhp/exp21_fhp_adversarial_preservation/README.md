# Experiment 21 — adversarial policy improvement with preservation

Starting from **Experiment 10's three 24-hour policies**, compare unchanged
policies, compute-matched additional grouped fitting, adversarial payoff
fine-tuning, and payoff fine-tuning with replay/blueprint preservation. Student
network size and the lossless hand/board representation remain fixed.

Implementation lives in the sibling **fhp-evaluation-suite**, under
`fhp_evaluation/adversarial/`, with the complete frozen protocol and step-by-step
instructions in `docs/exp21_adversarial_preservation.md`. This repository provides
the root launcher `gcp/run_exp21_adversarial_preservation.sh` and native encoders.
Keep both checkouts up to date. Source bundles capture both repositories.

No original UCV training is repeated. A separate attacker-strength qualification
must pass before repair is allowed. Three repair rounds retain previous opponents;
independent evaluation includes fresh learned attackers, head-to-head play, frozen
UCV reference policies, rule agents, LBR and the specialised full-flop responder.
VR-Deep/SD-CFR cross-family evaluation is a separate follow-up, not part of this
pilot's runtime budget. Results cannot establish exact Nash distance.

## Full pilot: one initiation, one Batch job

From this repository root, with the sibling evaluation-suite checkout updated:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@${PROJECT_ID}.iam.gserviceaccount.com"
export RUN_ID="exp21-adv-full-$(date -u +%Y%m%d-%H%M%S)"
unset FHP_BATCH_OUTPUT_DIR
bash gcp/run_exp21_adversarial_preservation.sh run
```

`run` prepares an immutable local source bundle, checks that the destination is
unused, uploads inputs and submits exactly **one** job, `$RUN_ID-run`. After
submission returns, you can disconnect. No manual follow-up or child Batch jobs
are needed. A Git push, `REPO_REF` and `SMOKE_RUN_ID` are not required: the job
uses the source captured from these two local checkouts, including local edits.
Never reuse the old qualification-only smoke's run ID or prepared directory.

The job runs these stages in order:

1. Integrated smoke: qualification, training, all four evaluation arms and aggregation.
2. Full three-seed attacker qualification.
3. Full training, only if **all three** attacker-strength gates pass.
4. Independent evaluation of all 12 final policies.
5. Aggregation and final result upload.

Each phase waits for all its workers before advancing. Smoke is execution-only
and cannot satisfy the scientific qualification gate. It uses the same immutable
source bundle as the full phases. Historical qualification-only smoke success is
not reused to skip the integrated smoke. Errors, checksum/provenance mismatches
and watchdog limits stop successors. Inadequate attackers stop the entire job
with `STATUS.json` status `inconclusive_qualification` (Batch reports `FAILED`);
this means stop and review, not evidence that the source policy is unexploitable.
There are no automatic retries, resume, or repeated tests until a gate passes.

The single VM is `n2-standard-48`, with a 200-GB balanced disk. It retains three
concurrent qualification/training workers and 12 concurrent evaluation workers,
with two Torch threads per worker. The 48-vCPU VM remains allocated through all
stages, including smoke, aggregation and phase tails; it does not scale down.
Allow roughly **12–24 elapsed hours**, plus provisioning, if qualification passes.
This is a planning estimate, not a benchmark of the combined job. Per-worker
watchdogs retain the original budgets; the whole Batch task has a **26-hour cap**,
including setup and final upload. Failure can end it earlier.

Optional actions: `prepare-run` creates files offline without submitting;
`run` with that same ID submits the frozen bundle. `run-smoke` on a fresh ID runs
only the integrated execution smoke. The old `prepare`, `qualify`, `train`,
`evaluate`, `aggregate` actions remain available for legacy separately prepared
runs; do not mix their layout with `prepare-run`/`run`.

## Progress and results

```bash
bash gcp/run_exp21_adversarial_preservation.sh status
gcloud storage cat "$BUCKET/$RUN_ID/STATUS.json"
```

`STATUS.json` appears after setup and is uploaded periodically; during bootstrap
it may not exist yet. It records the mode, phase, completed tasks, completed
stages and any failure. A complete scientific run requires Batch `SUCCEEDED`,
`STATUS.json` with `scientific_complete: true`, and the final
`$BUCKET/$RUN_ID/analysis/SUCCESS.json` marker. Partial output is diagnostic only.
The final success marker is published only after a successful final sync.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Full outputs retain source identities, final native-playable policies, intermediate
round policies/attackers, optimizer metadata and metrics. These are post-trained
policies, **not resumable full UCV solver states**. Aggregate analysis lives at
`$BUCKET/$RUN_ID/analysis/summary.json`. Worker diagnostics and policies remain
under `{qualify,train,evaluate}/task_N/analysis/`; smoke results are isolated under
`smoke/`. The analysis-only download does not fetch replay caches or checkpoints.
