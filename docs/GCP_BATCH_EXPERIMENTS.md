# Running the FHP UCV-ESCHER experiments on Google Cloud Batch

This guide covers the active experiments and the four archived flop hold'em poker
(FHP) UCV-ESCHER experiments in this repository. Each Batch
job creates a temporary VM, clones the repository, installs an isolated Python
3.11 environment, runs one experiment, uploads the complete `outputs/` tree to
Cloud Storage, and exits. Batch owns the VM lifecycle; there is no persistent VM
to shut down after a completed job.

The four original experiments are archived. Their instructions are retained for
exact historical reproduction; every maintained legacy module and job name
includes `archived` to distinguish a rerun from active Experiments 1, 2, and 3.

The end-to-end workflow is:

1. configure and authenticate the Google Cloud CLI;
2. enable the required APIs;
3. create the results bucket and Batch service account;
4. set the four required environment variables in the current terminal;
5. validate the checked-in submission helper;
6. submit the relevant smoke test and confirm it succeeds;
7. submit the matching full experiment;
8. monitor its status and job-scoped logs;
9. download and verify the uploaded outputs;
10. retain diagnostics and clean up completed Batch job records.

The archived production runs are time-bound. They save reloadable policies at the first
safe trajectory boundary after 6 and 12 effective training hours, then stop.
The 14-hour Batch limit leaves time for provisioning, installation, policy
fitting, checkpoint serialization, diagnostics, and upload.

## Experiment 20: 24 hours with half the critic updates

This is a fresh three-seed run of Experiment 10 with only the per-critic fit
budget changed from 10,000 to 5,000 updates. Each seed runs for 24 active hours
on a Standard `n2-standard-16`, with eight Ray actors and unchanged central
fitting. Four playable policies are saved at 6/12/18/24h; the final checkpoint
also saves complete resumable training state. Training, policy fitting and
uploads can take longer than 24 elapsed hours; the training task cap is 36h.
No evaluation or 48h extension runs automatically. See the
[full protocol](../experiments/fhp/exp20_fhp_half_critic_updates/README.md).

After committing and pushing, with the usual four GCP environment variables:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp20-critic24-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
export EXP20_TOTAL_HOURS=24
unset EXP20_SOURCE_RUN_ID
bash gcp/run_exp20_half_critic_updates.sh dry-run

# Submit smoke only; wait for SUCCEEDED before the next step.
bash gcp/run_exp20_half_critic_updates.sh smoke-cloud
bash gcp/run_exp20_half_critic_updates.sh status

# Same RUN_ID reuses the successful cloud smoke.
bash gcp/run_exp20_half_critic_updates.sh run
```

The controller runs smoke -> training -> aggregation and then stops. You can
close your laptop after submission. Concurrent seeds use 48 N2 vCPUs in total;
reduce `PARALLELISM` to schedule fewer seed VMs at once. Failed stages are not
automatically resubmitted. `resume` restarts only the controller, and refuses to
restart failed training.

When training and aggregation have succeeded, explicitly launch the separate
evaluation if desired:

```bash
bash gcp/run_exp20_half_critic_updates.sh dry-run-evaluate
bash gcp/run_exp20_half_critic_updates.sh evaluate
```

It compares all four checkpoints with the frozen historical Experiment 10
policies using the existing five-agent, LBR and duplicate head-to-head protocol.
The primary endpoint is 24h versus 24h. It runs on one separate `n2-standard-16`
with a 48h elapsed cap. `evaluate-resume` reuses completed scoring tasks after
a stopped evaluation; it does not retrain. Downloads:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
# Run after evaluation completes:
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/evaluation/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
```

Keep the remote full training states. The experiment README documents a later,
explicit `extend` action with a new run ID and the original training commit.

## Experiment 19: continue Experiment 16 from 48 to 72 active hours

This new run restores all three completed 48h full states from
`exp16-feat48-20261004-182051`. It keeps the same learner and `n2-standard-16`
per seed, adds checkpoints at 54/60/66/72h, and saves a final resumable 72h state.
It does not adopt the critic-budget or replay-memory changes of Experiments 17/18.
The original source outputs remain unchanged. See the
[full contract and evaluation protocol](../experiments/fhp/exp19_fhp_hand_board_72h/README.md).

After committing and pushing, from the repo root with the usual four GCP
environment variables set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp19-feat72-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp19_hand_board_72h.sh dry-run
bash gcp/run_exp19_hand_board_72h.sh preflight

# Optional standalone paid smoke; wait for SUCCEEDED.
bash gcp/run_exp19_hand_board_72h.sh smoke-cloud
bash gcp/run_exp19_hand_board_72h.sh status

# Same RUN_ID reuses that successful smoke. Also runs a smoke if none exists.
bash gcp/run_exp19_hand_board_72h.sh run
```

The cloud controller gates training on restart/capacity and second-continuation
smoke tests, then aggregates and evaluates 48/54/60/66/72h frozen policies.
The main contrast is 72h versus 48h; 72h versus 60h and 66h tests late improvement.
The established fixed learned opponents, five rule agents and LBR are retained.
Approximately 24 additional active hours per seed is not 24 elapsed hours:
policy fitting, checkpointing, transfers and evaluation add time. Concurrent
training requires 48 N2 vCPUs; `PARALLELISM=1` or `2` reduces concurrent VM use.
Do not delete the source full states. No stage automatically retries paid work.

You can close the laptop after submitting the controller. If only evaluation
fails, retain the same run ID and workflow commit and use `evaluate-resume`.
`resume` restarts the controller, not failed training. Download training and
evaluation summaries separately:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/evaluation/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
```

## Experiment 9 specialised best-response production follow-up

This implements the fixed-budget follow-up to the 25-pair pilot. The frozen
Experiment 9 24h policies for seeds 0, 1 and 2 each receive **2,000 new duplicate
pairs** against both standard LBR and the specialised exact-flop responder.
Both use the same deals; preflop remains approximate with 4,096 rollouts.
This is evaluation-suite Experiment 5, not a new UCV training experiment.

Four 500-pair shards per model run as twelve parallel Standard `n2-standard-2`
VM tasks (24 vCPUs total), two computation threads per task. Each task has a
four-hour ceiling and no automatic retry. Per-pair progress is saved atomically
and uploaded every minute. Final aggregation is a separate explicitly submitted
job. Earlier pilot outcomes are not pooled into the result.

From this repo root, with `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` set,
first run the cloud execution smoke (two pairs per shard, eight rollouts):

```bash
export RUN_ID="fhp-br-exp9-prod-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp9_best_response_production.sh prepare-smoke
bash gcp/run_exp9_best_response_production.sh smoke
bash gcp/run_exp9_best_response_production.sh status
```

After its workers job reports `SUCCEEDED`, check the complete aggregation path:

```bash
bash gcp/run_exp9_best_response_production.sh aggregate-smoke
bash gcp/run_exp9_best_response_production.sh status
```

Only after both smoke jobs succeed, prepare and launch the full evaluation:

```bash
export RUN_ID="fhp-br-exp9-prod-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp9_best_response_production.sh prepare
# Review outputs/batch/$RUN_ID/request.json and the two job.json files.
bash gcp/run_exp9_best_response_production.sh run
bash gcp/run_exp9_best_response_production.sh status
```

After the full workers job succeeds:

```bash
bash gcp/run_exp9_best_response_production.sh aggregate
bash gcp/run_exp9_best_response_production.sh status
```

`prepare` is offline; all submission actions are paid GCP jobs. Smoke is a manual
prerequisite. There is no automatic full run or aggregate submission. After each
submission the laptop can disconnect. The launcher uses the sibling shared
`../../fhp-evaluation-suite` checkout (`FHP_EVAL_REPO` overrides it), bundles only
allowlisted source, and supplies this checkout for native policy loading. No
`REPO_REF` is required. Preserve `outputs/batch/$RUN_ID`: it contains the immutable
request, checksummed job definitions and source bundle needed for recovery.
Prepared directories are never overwritten.

If a workers job fails, diagnose it and wait until all prior attempts have
stopped. An explicit `RECOVERY_TAG=r1 bash gcp/run_exp9_best_response_production.sh recover`
reuses the same prepared request and uploaded pairs; it does not increase the
sample size. Recovery refuses active attempts and altered source/job identities.
Do not mix partial or smoke outputs with complete production results.

After aggregation succeeds, download the summaries:

```bash
FHP_BUCKET_ROOT="gs://${BUCKET#gs://}"
FHP_BUCKET_ROOT="${FHP_BUCKET_ROOT%/}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$FHP_BUCKET_ROOT/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

`seed_summary.csv` gives each model's response scores and paired improvement;
`aggregate_summary.json` and `report.md` distinguish uncertainty across three
training seeds from Monte Carlo uncertainty conditional on those models. The
latter averages the three models on each shared deal before forming its
interval: the 6,000 model/pair scores are not independent. There are 12,000 hands
per responder, 24,000 total. `shard_timings.csv` records throughput and peak
process memory, with periodic machine/process snapshots in each remote worker
directory. Intervals measure payoff sampling or seed variation, not the gap
between this responder and a true full-game best response. A small response
score does not certify Nash convergence.

## Experiment 9 specialised best-response pilot

This evaluation-only pilot tests the frozen **24-hour, seed-0 Experiment 9** policy.
It profiles 1/2/4/8 CPU computation threads, then compares existing LBR with the
exact-flop response on 25 duplicate pairs using the fastest successful setting.
The machine remains one `n2-standard-8`, the task ceiling is two hours, and no
automatic retries are enabled. The whole-game metric remains an exploitability
lower-bound estimate because preflop is approximate.

Run from the **FHP UCV-ESCHER repository root**, with `PROJECT_ID`, `REGION`,
`BUCKET` and `SA_EMAIL` already exported. The launcher supplies the native repo
path automatically; `FHP_NATIVE_REPO` and `REPO_REF` are not required for this pilot.
It delegates to the shared evaluation suite at `../../fhp-evaluation-suite`.
If that checkout is elsewhere, set `FHP_EVAL_REPO` to its path. An absent or
outdated checkout produces an error before submission; nothing is cloned,
installed or pulled automatically.

Preview the job and source snapshot locally, without cloud calls:

```bash
export RUN_ID="fhp-br-exp9-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp9_best_response_pilot.sh dry-run
```

The preview is saved under `outputs/batch/$RUN_ID-preview`. Use `dry-run-smoke`
for a smoke configuration preview (`$RUN_ID-smoke-preview`). Submission artifacts
go under `outputs/batch/$RUN_ID`, so previewing does not occupy the submit directory.
`FHP_BATCH_OUTPUT_DIR` optionally overrides the destination. Existing output
directories are never overwritten; no-argument invocation only prints help.

First submit the separate cloud execution smoke:

```bash
export RUN_ID="fhp-br-exp9-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp9_best_response_pilot.sh smoke
bash gcp/run_exp9_best_response_pilot.sh status
```

Wait for `SUCCEEDED`, then use a fresh ID for the pilot:

```bash
export RUN_ID="fhp-br-exp9-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp9_best_response_pilot.sh run
bash gcp/run_exp9_best_response_pilot.sh status
```

After the 4 October Python-bootstrap correction, rerun the smoke before the
pilot even if an earlier local test passed. Both this checkout and the shared
`fhp-evaluation-suite` checkout must be up to date. Use fresh IDs: keep the failed
job and its diagnostics intact. The shared bootstrap now exports a Linux PATH,
uses Debian's Python explicitly, and logs the interpreter and failing setup
stage. This smoke-first check is a manual prerequisite; the launcher does not
automatically enforce or submit it.

The launcher uses `python3`, or the executable named by `PYTHON`. No local neural
network dependencies are needed to submit. Unlike the training launchers, this
pilot uploads a hashed, source-only snapshot from the two current checkouts, so
no pushed commit is required. The shared evaluator is not copied into this repo;
the two launch paths use the same implementation and experiment settings.

The VM fetches only the selected policy from the existing source run and verifies
its SHA-256 before loading. Worker logs, memory samples and partial diagnostics
are uploaded periodically and after failure. No training/resumption state is
downloaded. You can disconnect the laptop after submission.

Download to this UCV repository as usual:

```bash
FHP_EVAL_BUCKET="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "${FHP_EVAL_BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Inspect `pilot_manifest.json`, `comparison/result.json` and `SUCCESS.json`, and
also require Batch `SUCCEEDED`. Partial files after a timeout are diagnostics,
not final estimates. The [shared suite protocol](https://github.com/lawrencewlcknight/fhp-evaluation-suite/blob/main/docs/gcp_batch_experiments.md)
describes all stage budgets, source verification and interpretation limits.

## Evaluation of Experiments 7 and 8: longer training and late-stage progress

This evaluation-only job compares Exp7 at 24h with Exp8 at 24/30/36/42/48h,
across all three seeds. All 18 selected policies receive the existing five-rule-agent
and LBR tests. All ten Exp8 checkpoint pairs, plus each Exp8 checkpoint against
the fixed Exp7 24h baseline, receive duplicate-deal head-to-head evaluation.
The [full protocol](../experiments/fhp/retrospective_exp7_exp8_evaluation/README.md)
describes source validation, budgets, seven charts and interpretation.

After committing and pushing the new evaluation code, with the usual GCP variables set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP7_RUN_ID=exp7-par8-20261001-005151
export EXP8_RUN_ID=exp8-par48-20261001-005208
export RUN_ID="fhp-eval78-$(date -u '+%Y%m%d-%H%M%S')"

# One n2-standard-16: real-checkpoint cloud smoke, then full scoring if it passes.
bash gcp/run_retrospective_exp7_exp8_evaluation.sh run
bash gcp/run_retrospective_exp7_exp8_evaluation.sh status
```

No training/refitting occurs. Sixteen single-threaded CPU scoring processes use
a 48-hour elapsed safety cap (`EVAL_MAX_HOURS`, 1–72); the job stops when finished.
The VM downloads policy files automatically; local analysis-only downloads are
sufficient to launch. The laptop can disconnect after submission.

For a **separate smoke-only job**, choose a new run ID; do not reuse the full-run prefix:

```bash
export RUN_ID="fhp-eval78-smoke-$(date -u '+%Y%m%d-%H%M%S')"
EVAL_MAX_HOURS=2 bash gcp/run_retrospective_exp7_exp8_evaluation.sh smoke-cloud
```

Use another new run ID for the full run. Source checks validate all 36 saved
policies, while smoke scores seed 0 at every selected checkpoint with tiny budgets.
After a failed full evaluation, retain its original `RUN_ID`, source IDs and
pinned `REPO_REF`, then run:

```bash
bash gcp/run_retrospective_exp7_exp8_evaluation.sh resume
```

Completed tasks are reused; concurrent or incompatible resumes are rejected.
Task results upload every five minutes and on exit, alongside resource/failure
diagnostics. Download analysis without the recovery cache:

```bash
FHP_EVAL_BUCKET="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "${FHP_EVAL_BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

The primary endpoint is Exp8 48h versus its own 24h policy. Prespecified late-stage
diagnostics include 48h versus 42h and versus 36h, with all consecutive six-hour
gains also reported. Seed-level confidence intervals and rule/LBR trajectories
support decisions about extending training. Flat estimates or non-significant
differences do not establish equivalence or equilibrium convergence.

## Evaluation of Experiments 7 and 9: cached versus uncached critic targets

This evaluation-only job scores all 24 policies (three seeds, four checkpoints
per method) using the existing five-rule-agent/LBR framework and head-to-head
play. The [full protocol](../experiments/fhp/retrospective_exp7_exp9_evaluation/README.md)
documents budgets, interpretation, source checks, recovery and local testing.
The primary comparison is Exp9 versus Exp7 after 24 active training hours;
earlier, temporal and approximate node-matched comparisons are also retained.

After committing and pushing, assuming the usual environment variables are set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP7_RUN_ID="exp7-par8-20261001-005151"
export EXP9_RUN_ID="exp9-cache24-20261001-132550"
export RUN_ID="fhp-eval79-$(date -u '+%Y%m%d-%H%M%S')"

# One n2-standard-8: real-checkpoint cloud smoke, then full evaluation if it passes.
bash gcp/run_retrospective_exp7_exp9_evaluation.sh run
bash gcp/run_retrospective_exp7_exp9_evaluation.sh status
```

There is no training phase. The default elapsed safety ceiling is 36 hours
(`EVAL_MAX_HOURS` can override it); scoring stops as soon as it finishes.
You may disconnect after submission. The VM downloads policy files automatically:
analysis-only local downloads are sufficient for launching this cloud job.

If the evaluation fails, retain the same variables and pinned commit:

```bash
bash gcp/run_retrospective_exp7_exp9_evaluation.sh resume
```

Completed scoring tasks are reused; incompatible or concurrent resumes fail.
Download the final analysis without its per-task recovery files:

```bash
export BUCKET_ROOT="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "${BUCKET_ROOT%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Evaluation of Experiments 9–12: four-way representation comparison

This evaluation-only job compares all 48 saved policies (four experiments,
three seeds, 6/12/18/24h). It uses the established five-rule-agent/LBR suite,
all six matched-time experiment pairings and within-run temporal head-to-head.
The [full protocol](../experiments/fhp/retrospective_exp9_exp10_exp11_exp12_evaluation/README.md)
describes validation, budgets, uncertainty, seven charts and recovery.

After committing and pushing the new code, with the usual GCP variables set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP9_RUN_ID=exp9-cache24-20261001-132550
export EXP10_RUN_ID=exp10-features-20261001-161740
export EXP11_RUN_ID=exp11-econ-20261001-153659
export EXP12_RUN_ID=exp12-showdown-20261001-165650
export RUN_ID="fhp-eval9to12-$(date -u '+%Y%m%d-%H%M%S')"

# One n2-standard-16 VM: cloud smoke, then full scoring if it passes.
bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh run
bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh status
```

There is no training/refitting. Sixteen CPU scoring processes use a 48-hour
elapsed safety cap (override with `EVAL_MAX_HOURS`, 1–72). The job stops when
finished and saves per-task recovery files every five minutes. For a separate
smoke-only test, choose a new `RUN_ID` and run:

```bash
EVAL_MAX_HOURS=2 bash gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh smoke-cloud
```

After failure, retain the original full-run `RUN_ID`, source IDs and pinned
`REPO_REF`, and use `resume` instead of `run`. Completed tasks are reused;
incompatible or concurrent resumes are rejected. Results are under
`$BUCKET/$RUN_ID/analysis/`. The protocol includes download commands. No exact
exploitability is calculated, and node exposure is reported without claiming
equal-node comparisons. The primary comparisons are Exp10/11/12 versus Exp9
after 24 active hours.

## Experiment 14: recovering the runtime-mismatch failure

The failed `exp14-cards-20261001-195256-train` deployment used Python 3.11.17
after screening had used 3.11.16. The corrected launcher pins **3.11.16** in
every scientific stage and reports identity mismatches before loading large
replay files. Recovery reuses the completed screening and locked selection;
it launches only deployment/evaluation and aggregation, not a new screening run.
The [Experiment 14 protocol](../experiments/fhp/exp14_fhp_card_architecture/README.md#recover-the-1-october-runtime-mismatch-failure)
explains the safeguards and remaining runtime-build check.

After committing and pushing the correction, with the usual cloud variables set:

```bash
export REPO_REF="$(git rev-parse HEAD)"  # New pushed launcher commit.
export EXP14_AUDIT_REF=54a3269f62189b8ac7190e59c9a3ea70efb4969c  # Original scientific code.
export EXP14_PYTHON_VERSION=3.11.16
export EXP2_RUN_ID=exp2-fhp-20260921-093839
export RUN_ID=exp14-cards-20261001-195256
export PARALLELISM=3

# Checks saved metadata and active jobs; does not submit anything.
bash gcp/run_exp14_card_architecture.sh check-recovery
# Submit only when the check passes and you intend to start paid work.
bash gcp/run_exp14_card_architecture.sh recover
bash gcp/run_exp14_card_architecture.sh status
```

Keep the original run ID, source run and audit commit. Do not edit the saved
manifests or selection to force compatibility. The controller uses the new
launcher commit; scientific workers use the original audit commit and retain
all code/data/configuration checks. Runtime mismatches are uploaded under
`$BUCKET/$RUN_ID/diagnostics/<stage>/seed_<seed>/failure.json` (no seed component
for selection/aggregation). A metadata-only local check does not prove the
future VM's full interpreter build matches; the VM checks this before replay
download. The launcher refuses recovery while another job for the run is active.

For an entirely new Experiment 14 run, unset `EXP14_AUDIT_REF`, choose a new
`RUN_ID`, keep the explicit Python patch, and use `run` instead of `recover`.

## Active Experiment 1: three-seed grouped-wide baseline

The active `exp1_fhp_grouped_wide_ucv_baseline` uses a remote controller and a
three-task Batch array rather than the archived single-job helper. Each task
runs one of seeds `0`, `1`, and `2` on its own on-demand `n2-standard-8` VM for
24 effective training hours. Reloadable policies and resumable continuation
states are uploaded at 6, 12, 18, and 24 hours.

Experiment 1 stores continuous replay fields as `float32` but deliberately
retains the raw OpenSpiel representation, full-width integer fields, legacy
global-RNG sampling, network definitions, and all optimisation settings. This
prevents checkpoint-time memory pressure without changing VM class or adopting
the bundled structured/compact implementation tested by Experiment 2.

After completing the one-time setup below, test locally:

```bash
./gcp/run_exp1_grouped_wide.sh smoke-local
```

Push the tested commit, then submit the remote controller:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp1-fhp-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_exp1_grouped_wide.sh run
```

The controller submits cloud smoke first. Production is not submitted unless
that smoke succeeds. It then launches all three seeds concurrently, waits for
them, and runs aggregation only when every seed succeeds. The laptop may be
disconnected after the controller job is accepted.

Monitor or resume the same run with:

```bash
./gcp/run_exp1_grouped_wide.sh status
./gcp/run_exp1_grouped_wide.sh resume
```

Keep the original `RUN_ID` and `REPO_REF` when resuming. Each training task has
one automatic retry and restores its newest durable continuation state. The
36-hour task ceiling allows for 24 active hours plus four grouped-policy fits,
large state serialization and upload, setup, and teardown. Resource snapshots,
cgroup memory events, process RSS, disk use, exit codes, Python failures, and
kernel OOM messages are retained beneath `$BUCKET/$RUN_ID/task_diagnostics/`
and each worker directory.

The complete active protocol is in
[`experiments/fhp/exp1_fhp_grouped_wide_ucv_baseline/README.md`](../experiments/fhp/exp1_fhp_grouped_wide_ucv_baseline/README.md).

## Active Experiment 2: lossless structured FHP input

Experiment 2 uses the same controller, cloud-smoke gate, three parallel seed
workers, one-retry policy, 24-hour effective training limit, four six-hourly
checkpoints, durable continuation states, aggregation, and failure diagnostics
as active Experiment 1. It runs one on-demand `n2-standard-8` VM for each seed.

Test the complete worker locally:

```bash
./gcp/run_exp2_lossless_structured.sh smoke-local
```

Push the tested commit and submit the controller:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp2-fhp-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_exp2_lossless_structured.sh run
```

To run only the GCP smoke job, without permitting production submission:

```bash
./gcp/run_exp2_lossless_structured.sh smoke-cloud
```

Monitor or resume a submitted run with the original `RUN_ID` and `REPO_REF`:

```bash
./gcp/run_exp2_lossless_structured.sh status
./gcp/run_exp2_lossless_structured.sh resume
```

Downloaded artifacts use the same layout as Experiment 1 beneath
`$BUCKET/$RUN_ID/`. The complete representation and evaluation contract is in
[`experiments/fhp/exp2_fhp_lossless_structured_ucv/README.md`](../experiments/fhp/exp2_fhp_lossless_structured_ucv/README.md).

## Active Experiment 3: wider lossless structured networks

Experiment 3 changes only Experiment 2's network widths. It retains the same
three-seed task array, one VM per seed, cloud-smoke gate, retry/resume behavior,
24 effective training hours, 6/12/18/24-hour checkpoints, aggregation, and
diagnostics. Each production seed still uses an on-demand `n2-standard-8` VM so
the comparison measures learning per equal wall-clock and hardware budget.

Test the complete worker locally:

```bash
./gcp/run_exp3_wider_structured.sh smoke-local
```

Push the tested commit and submit the controller:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp3-fhp-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_exp3_wider_structured.sh run
```

To run only the GCP smoke job:

```bash
./gcp/run_exp3_wider_structured.sh smoke-cloud
```

Monitor or resume with the original `RUN_ID` and `REPO_REF`:

```bash
./gcp/run_exp3_wider_structured.sh status
./gcp/run_exp3_wider_structured.sh resume
```

Artifacts are written beneath `$BUCKET/$RUN_ID/` using the same layout as
Experiment 2. The frozen capacity comparison is documented in
[`experiments/fhp/exp3_fhp_wider_lossless_structured_ucv/README.md`](../experiments/fhp/exp3_fhp_wider_lossless_structured_ucv/README.md).

## 1. Prerequisites

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI installed locally;
- permission to enable APIs and create service accounts, IAM bindings, buckets,
  and Batch jobs;
- a Batch service account that can write logs and Cloud Storage objects;
- access from the Batch VM to this GitHub repository.

The default repository URL is public HTTPS:

```text
https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git
```

For a private fork, set `REPO_URL` to an authenticated clone URL or use a
pre-built image with credentials configured securely.

## 2. One-time Google Cloud setup

Authenticate, select the project, and choose a region:

```bash
gcloud init
gcloud auth login

export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
gcloud config set project "$PROJECT_ID"
```

Enable the required APIs:

```bash
gcloud services enable \
  compute.googleapis.com \
  batch.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com
```

Create a regional results bucket:

```bash
export BUCKET_NAME="${PROJECT_ID}-fhp-escher-results"
export BUCKET="gs://${BUCKET_NAME}"

gcloud storage buckets create "$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

Confirm that the bucket exists and is in the intended region:

```bash
gcloud storage buckets describe "$BUCKET"
```

Create a dedicated Batch service account:

```bash
export SA_NAME="fhp-escher-runner"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="FHP ESCHER experiment runner" \
  --project="$PROJECT_ID"
```

Grant it the required project and bucket permissions:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.agentReporter"

# The remote controller creates and monitors smoke, training, and aggregation
# child jobs.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.jobsEditor"

gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

# Allow the controller service account to attach itself to its child jobs.
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

The active launchers verify that the configured service account exists before
submitting a cloud job. The remote controller also verifies that it can list
Batch jobs and exits immediately if child-job permissions are missing; it does
not remain alive in a polling loop after an IAM failure.

Allow your user account to submit jobs as the service account and inspect logs:

```bash
export YOUR_EMAIL="your-email@example.com"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/logging.viewer"
```

Confirm that the service account exists:

```bash
gcloud iam service-accounts describe "$SA_EMAIL" \
  --project="$PROJECT_ID"
```

## 3. Environment for each terminal

Set these values before submitting or inspecting jobs:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
```

Check the active values before every submission. An empty or stale value can
send a job or its outputs to the wrong project, region, bucket, or identity:

```bash
echo "$PROJECT_ID"
echo "$REGION"
echo "$BUCKET"
echo "$SA_EMAIL"
gcloud config get-value project
```

## 4. Submission helper

Run all jobs through `gcp/submit_batch_experiment.sh`. Its positional interface
is:

```text
JOB_NAME EXPERIMENT_COMMAND MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB BOOT_DISK_SIZE_GB BOOT_DISK_TYPE
```

Before first use, check the scripts:

```bash
chmod +x gcp/submit_batch_experiment.sh gcp/read_batch_task_logs.sh
bash -n gcp/submit_batch_experiment.sh
bash -n gcp/read_batch_task_logs.sh
```

`BOOT_DISK_TYPE` defaults to `pd-balanced` for backward-compatible N2 use. Pass
it explicitly in maintained commands. The helper rejects a C4-family machine
paired with any `pd-*` Persistent Disk locally, before generating or submitting
a Batch job.

The submission helper deliberately sets `maxRetryCount` to zero so a failed
training run is not silently repeated. It captures stdout and stderr, records
15-second resource snapshots, classifies failures, and uploads outputs from an
exit trap on both success and failure.

## 5. Archived experiment allocations

| Experiment | Backend | Full-run VM | CPU request | Memory request | Boot disk |
|---|---|---:|---:|---:|---:|
| 1 | Sequential baseline | `n2-standard-8` | 8,000 milli | 32,000 MiB | 100 GiB `pd-balanced` |
| 2 | Sequential comparison | `c4-standard-32` | 32,000 milli | 120,000 MiB | 200 GiB `hyperdisk-balanced` |
| 3 | Ray parallel comparison | `c4-standard-32` | 32,000 milli | 120,000 MiB | 200 GiB `hyperdisk-balanced` |
| 4 | CPU-optimized Ray parallel | `c4-standard-32` | 32,000 milli | 120,000 MiB | 200 GiB `hyperdisk-balanced` |

All smoke tests use `n2-standard-4`, 4,000 CPU milli, 16,000 MiB of memory, a
100 GiB `pd-balanced` disk, and a two-hour Batch ceiling. `--smoke` retains the
production orchestration and checkpoint/reload path while shrinking training
work and checkpoint thresholds.

## 6. Submit smoke tests

Run the matching smoke job before each full job.

### Archived Experiment 1

```bash
JOB_NAME="exp1-archived-fhp-ucv-baseline-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_archived_ucv_escher_baseline.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

### Archived Experiment 2

```bash
JOB_NAME="exp2-archived-fhp-ucv-sequential-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_archived_ucv_escher_sequential.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

### Archived Experiment 3

```bash
JOB_NAME="exp3-archived-fhp-ucv-parallel-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_archived_ucv_escher_parallel.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

### Archived Experiment 4

```bash
JOB_NAME="exp4-archived-fhp-ucv-cpu-optimized-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_archived_ucv_escher_cpu_optimized.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

## 7. Submit full runs

### Archived Experiment 1

```bash
JOB_NAME="exp1-archived-fhp-ucv-baseline-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_archived_ucv_escher_baseline.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 50400 8000 32000 100 pd-balanced
```

### Archived Experiment 2

```bash
JOB_NAME="exp2-archived-fhp-ucv-sequential-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_archived_ucv_escher_sequential.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

### Archived Experiment 3

```bash
JOB_NAME="exp3-archived-fhp-ucv-parallel-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_archived_ucv_escher_parallel.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

### Archived Experiment 4

```bash
JOB_NAME="exp4-archived-fhp-ucv-cpu-optimized-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_archived_ucv_escher_cpu_optimized.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

## 8. Monitor a job and inspect logs

List jobs or inspect one job:

```bash
gcloud batch jobs list --location "$REGION"
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
```

The normal state progression is `QUEUED` → `SCHEDULED` → `RUNNING` →
`SUCCEEDED`. Treat `FAILED` as requiring investigation before resubmission. A
job that remains in `SCHEDULED` can be blocked by VM availability, quota, IAM,
or an invalid allocation such as an incompatible machine and disk type.

For a concise status view:

```bash
gcloud batch jobs describe "$JOB_NAME" \
  --location "$REGION" \
  --format="yaml(status.state,status.runDuration,status.statusEvents)"
```

Read only the task logs belonging to one Batch job:

```bash
./gcp/read_batch_task_logs.sh "$JOB_NAME"
```

Filter those logs for likely failure evidence:

```bash
./gcp/read_batch_task_logs.sh \
  "$JOB_NAME" ERROR Traceback Killed OOM "out of memory"
```

The helper resolves the job UID before querying Cloud Logging, preventing logs
from similarly named jobs from being mixed together.

## 9. Retrieve outputs

Inspect uploaded objects:

```bash
gcloud storage ls --recursive "$BUCKET/$JOB_NAME/"
```

Download the job's uploaded output tree into the ignored local
`cloud_outputs/` working area:

```bash
mkdir -p "cloud_outputs/$JOB_NAME"
gcloud storage cp -r \
  "$BUCKET/$JOB_NAME/*" \
  "cloud_outputs/$JOB_NAME/"
```

Keep checkpoint files with their manifests. The final reloadable policy is
`final_policy_checkpoint.pkl`; verify its digest against
`summary.json` or `checkpoint_manifest.json` before head-to-head evaluation.

## 10. Diagnose failures

Batch-level diagnostics are written alongside the experiment run directory:

- `batch_run.log` — complete setup, experiment, cleanup, and upload output;
- `resource_snapshots.jsonl` — 15-second process, memory, disk, load, and CPU
  snapshots;
- `batch_diagnostics.json` — detailed final resource and failure evidence;
- `batch_status.json` — concise classification and exit codes.

An experiment exception also creates `failure.json` inside its timestamped run
directory with the traceback, phase, nodes touched, iteration, and peak RSS when
available. The Batch classifier distinguishes confirmed cgroup OOM, allocator
errors, probable OOM or SIGKILL, timeout or termination, Python exceptions, and
other nonzero exits.

Because output upload runs in the cleanup trap, inspect Cloud Storage even when
the Batch job reports failure. A failure before the repository or output
directory is created can still prevent artifact upload; in that case Cloud
Logging is the primary source of evidence.

## 11. Changing CPU, memory, disk, or VM type

The final six submission-helper arguments control the Batch allocation:

```text
MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB BOOT_DISK_SIZE_GB BOOT_DISK_TYPE
```

CPU is expressed in milli-vCPUs, so `32000` requests 32 vCPUs. Memory is MiB.
The CPU and memory requests must fit the selected machine type.

Keep the disk family compatible with the machine family:

- N2 commands in this repository use `pd-balanced`;
- C4 commands use `hyperdisk-balanced` because C4 does not support Persistent
  Disk;
- the helper rejects a C4-family machine combined with any `pd-*` disk before
  contacting Batch.

The archived production allocations in section 5 are part of the experiment contract.
If you change them for exploratory capacity testing, use a distinct job name
and record the changed allocation with the result. Do not present a modified
allocation as a canonical Experiment 1-4 run.

## 12. Choosing a VM size

Use smoke-test and production diagnostics rather than guessing from model
parameters alone. Review:

- peak and current RSS in `summary.json`;
- cgroup memory peak, limit, and OOM counters in
  `resource_snapshots.jsonl` and `batch_diagnostics.json`;
- CPU utilization, load average, and largest-process snapshots;
- node throughput and learner/worker timings in checkpoint rows;
- disk utilization at cleanup;
- Ray object-store pressure for Experiments 3 and 4.

Experiment 2 intentionally uses the same 32-vCPU production machine as
Experiments 3 and 4 so the sequential/parallel comparison is not confounded by
different hardware. Experiment 4 reserves capacity for 28 traversal actors,
the driver, Ray services, result merging, and concurrent learners.

## 13. Runtime limits and stopping

`MAX_RUN_SECONDS` is the Batch task ceiling, not the solver's effective
training timer. Canonical full runs use `50400` seconds (14 hours), while the
solver saves at 6 and 12 effective training hours and stops after the second
checkpoint. Setup, policy fitting, serialization, cleanup, and upload are
outside effective training time but inside the Batch ceiling.

Do not reduce the full-run Batch limit to 12 wall-clock hours: Batch could
terminate the task before the final policy is serialized and uploaded. A Batch
timeout is a failed run even if it produced a partial checkpoint.

Smoke tests use a two-hour ceiling but normally finish quickly because
`--smoke` replaces the production work and time thresholds with tiny values.

## 14. Adding and running later experiments

New experiments should retain the numbered naming convention:

```text
experiments/fhp/expN_descriptive_name/
```

Their runners should accept `--smoke` and `--output-root`, save reloadable
checkpoints where applicable, and use an `expN-` Batch job prefix. Submit them
through the same checked-in helper:

```bash
JOB_NAME="expN-fhp-description-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.expN_descriptive_name.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

Add both smoke and full commands to this guide when the experiment is added.

## 15. Cleanup

Batch deletes the temporary VM after the job reaches a terminal state, so no
persistent experiment VM needs to be stopped manually. After confirming that
outputs and diagnostics are safely in Cloud Storage, remove an old Batch job
record with:

```bash
gcloud batch jobs delete "$JOB_NAME" \
  --location "$REGION"
```

List bucket contents before deleting any results:

```bash
gcloud storage ls --recursive "$BUCKET/$JOB_NAME/"
```

Keep at least the run manifest, both checkpoint manifests and policy files,
summary, Batch status, detailed diagnostics, resource snapshots, and run log.
Do not delete a failed job's evidence until its cause is understood.

## 16. Dependency installation

Each Batch VM starts clean. The checked-in helper installs system build tools
and `uv`, keeps the Cloud SDK on Python 3.10, and creates an isolated Python
3.11 FHP environment. It then installs the pinned packages in
`requirements.txt`, including CPU-only PyTorch, OpenSpiel, Ray, NumPy, SciPy,
and psutil, followed by an editable install of this repository.

Dependency installation time is part of Batch wall-clock time and cost, but not
effective solver training time. A dependency failure occurs before experiment
outputs may exist, so use Cloud Logging when no `batch_run.log` was uploaded.
The repository commit printed in the run log is the authoritative source
version for reproducibility.
