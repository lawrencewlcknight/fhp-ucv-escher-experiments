# Experiment 24 — critic-update budgets across training stages

Experiment 24 is the primary budget study. [Experiment 25](../exp25_fhp_critic_objective_generalisation/README.md)
is a separately reported objective-generalisation study using the **same states,
panels and candidate fits**. One launch runs both; no duplicated six-hour reference.

## Scientific protocol

- Train one fresh reference seed, **seed 0**, with the unchanged Experiment 10
  learning configuration: eight Ray traversal actors, eight central fitting
  threads, native 10,000-update critic fits, original replay, features and targets.
- Capture a diagnostic snapshot at the first normal critic-ensemble fit boundary
  at/after **one and six active training hours**, before either member fits and
  after normal collection, regret fitting and calibration. Finish that ordinary
  iteration and save a **complete reloadable training checkpoint** for each stage.
  Stop after the six-hour observation's normal iteration completes. Boundary
  granularity means slight overshoot; record actual seconds, iteration and nodes.
- The explicit reference clock is authoritative for active training time. Native
  cumulative subphase timers may include the observational snapshot hook; do not
  use those unadjusted counters to infer fitting speedups.
- Snapshot, checksum and upload time is excluded from active training time.
  No audit fits or probes enter reference training or its RNG streams. Average
  policy extraction and policy evaluation are not performed: the saved full
  states can resume learning but are not newly fitted playable average policies.
- Restore seed 0 of `exp10-features-20261001-161740` at its completed 24-hour
  checkpoint; run exactly one normal iteration and capture the same pre-fit
  boundary. Label it **24-hour-plus-one-iteration**, with actual elapsed time and
  node/iteration counts. Its pre-fit capture precedes the last critic fit within
  that iteration. The original source is read-only. This is historical evidence,
  not the continuation of the new six-hour trajectory.
- On frozen copies compare native replay TD at **0, 100, 500, 1,000, 2,500,
  5,000 and 10,000** updates. Zero keeps the deployed target unchanged: no
  history append, cache construction, optimizer step or deployment refresh.
  Positive candidates are prefixes of one fit, each hypothetically appended
  once to the identical original four-fit parameter-averaging history.
- Restore the original online weights, Adam, replay, frozen policies/targets,
  target histories and replay sampling RNG. A regression test matches positive
  native TD prefixes to the original native cached fit.
- Measure online and deployed corrected-regret variance, per-state covariance,
  p90/p95/p99/max tails, TD errors, conditional means and centring. Report all
  prefixes, not merely selected candidates. Select using independent selection
  data; lock choices before constructing the final-assessment panel.

Panels retain Experiment 22's sizes **per critic member**: 256 fitting roots ×
32 continuations, 64 selection roots ×64, and 128 final roots ×128. Two traversers
are balanced. Split/member histories are disjoint and random streams separate.
This does not guarantee information-set disjointness or exclusion from historical
replay; report that limitation. Root sampling is frozen-behaviour based, not a
global CFR reach-weighted or information-set-balanced evaluation.

Costs include fresh sampling, cached targets, each fit prefix, fixed-panel and
selection diagnostics, assessment, persistence and uploads. Full workflow costs
include reference training, source loading and Batch overhead. Do not sum prefix
times as if they were separate training runs; do not charge the shared reference
twice. Equal updates are not equal computation across objectives.

This is **one-seed development evidence**, not a test of equilibrium quality,
long-run budget equivalence, an adaptive controller, or three independent stages.
If a smaller fixed budget looks sufficient, replicate it before end-to-end
validation. A controller needs additional evidence of varying useful budgets and
comparisons with fixed budgets and a predetermined schedule. No scientific
significance gate automatically authorises longer training.

## Implementation and checkpoint safety

The training hook observes the whole ensemble before either critic member
updates. It does not replace its native fit. Tests verify observation leaves the
trajectory unchanged. Pre-fit states are explicitly **not resumable** through a
normal iteration entry point; they are immutable diagnostic inputs. Completed
iteration states include optimizer, replay, target history, driver RNG and all
eight actor continuation states. `states.restore_reference` is the checked loader;
it rejects pre-fit snapshots. The workflow verifies checkpoint reloadability.

Experiment 22's estimator, split construction and prefix fitting are reused;
old defaults and its strict historical-source validation remain unchanged. Fixed
panel diagnostics and zero/native-RNG options are opt-in. Original training
algorithms and Experiment 10 configuration are unmodified.

## Run locally and on GCP Batch

Run commands from the repository root. The standard `PROJECT_ID`, `REGION`,
`BUCKET` (including `gs://`) and `SA_EMAIL` must already be set. First commit and
push the implementation: the VM clones the exact **pushed** `REPO_REF`.

```bash
# Optional tiny real-game local execution smoke; no cloud job or new full run.
bash gcp/run_exp24_critic_stages.sh smoke-local

# After committing and pushing:
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp24-critic-smoke-$(date -u +%Y%m%d-%H%M%S)"
export SMOKE_RUN_ID="$RUN_ID"
bash gcp/run_exp24_critic_stages.sh smoke-cloud
bash gcp/run_exp24_critic_stages.sh status
```

The cloud smoke is deliberately more substantial than the local smoke. It
restores the **real 24h seed-0 source with all eight actors**, advances one normal
iteration and captures a real pre-fit state. It also runs a tiny new reference
and checkpoint round-trip with all eight actors, then tiny audits of all stages.
No six-hour reference is run during smoke. Allow time for source transfer,
runtime installation and one full native iteration; this is not a seconds-long
test. Its six-hour Batch limit is a safety ceiling, not an expected duration.

Once the cloud smoke reports `SUCCEEDED`, retain its `SMOKE_RUN_ID` and the same
`REPO_REF`. Submit **one** full job:

```bash
export RUN_ID="exp24-critic-stages-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp24_critic_stages.sh run
bash gcp/run_exp24_critic_stages.sh status
```

One `n2-standard-16` (16 vCPUs, 64 GB RAM, 250 GB boot disk), one task, zero
automatic retries. `PARALLELISM` is ignored, so a setting inherited from an older
three-seed job cannot break this launcher. The real late-state boundary is
validated first, before spending six training hours. Reference generation,
checkpoint verification, the three stage audits and both reports then execute
sequentially in fresh subprocesses, releasing memory between phases. No child
jobs or manual intervening submissions. The 24-hour Batch cap leaves headroom;
actual duration includes six active training hours plus fitting-boundary overshoot,
source transfer, snapshots, diagnostics and setup. Use smoke timings to refine
the overhead estimate; no particular end-to-end speed is assumed.

`dry-run` generates job JSON without submitting; `JOB_KIND=smoke` chooses the
smoke definition. Submission checks the existing source metadata first. The smoke
gate verifies real-source coverage, candidate counts, actor reload, artifacts
and matching commit. Resource snapshots, heartbeat logs, exit status and an
always-run upload preserve failure evidence. Failed steps stop subsequent work.

## Outputs and download

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/diagnostics"
gcloud storage rsync --recursive "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive "$BUCKET/$RUN_ID/diagnostics" "cloud_outputs/$RUN_ID/diagnostics"
```

- `analysis/exp24/report.md` and `summary.json`: primary TD budget curves.
- `analysis/exp25/report.md` and `summary.json`: matched-data objective diagnostics.
- `analysis/per_stage.json`: **all** fitting, selection and assessment rows,
  state-level tails, full panel diagnostics, hashes and stage provenance.
- `analysis/summary.json`: reference and audit costs, checkpoint verification,
  scientific caveats. Batch diagnostics add provisioning and end-to-end wall time.
- `reference/checkpoints/{one_hour,six_hours}/`: sharded full training states.
- `reference/pre_fit/...` and `late_reference/pre_fit/...`: diagnostic snapshots.
- `audits/<stage>/`: immutable panels, candidate weights/Adam, selection locks.
  Candidate `.pt` files are critics, not playable policies or full solver states.

Large states and panels remain in GCS unless explicitly downloaded. Each saved
state has per-shard checksums. Do not use arbitrary untrusted pickle checkpoints.
