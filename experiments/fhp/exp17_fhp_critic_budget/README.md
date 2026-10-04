# Experiment 17: frozen-data critic-budget audit

## Question and scope

Does stopping a critic fit at 5,000 rather than 10,000 updates materially
worsen held-out TD prediction or the variance of corrected regret estimates?
This is a **mechanism screen**, not an output-preserving optimisation and not
an end-to-end policy-quality comparison. It does not test increasing collection.

The sources are seeds 0, 1 and 2 of Experiment 10,
`exp10-features-20261001-161740`, at their completed 24-active-hour checkpoints.
Their state hashes, configuration, original commit and runtime are validated.
Each source is restored on its own `n2-standard-16` VM with eight collection
actors. The audit runs three normal consecutive outer iterations, including
10,000 traversals per player, normal regret fitting and calibration.

At each critic-fit boundary, the **native cached 10,000-update fit** executes
once. Its 5,000-update weights are captured without refreshing the TD target,
changing Adam, consuming the replay sampler or splitting the fitting call.
Thus both budgets have identical data, starting weights/Adam, frozen labels
and minibatch prefixes. Only the full-fit branch advances to the next boundary.
This is not three independent seeds per boundary, nor a persistent short-fit arm.

The short candidate's deployed target combines its online weights with the
three most recent prior completed fits. The long candidate uses the native
four-fit target. Appending the short snapshot to the actual learner's target
history would invalidate the experiment and is explicitly avoided.

## Diagnostics

- Per-boundary collection, regret, calibration and native critic-fit timers.
- Per-critic cache construction time/bytes, replay checksum/occupancy, model
  hashes, target versions and training-probe TD errors (8,192 sampled rows).
- 64 fresh diagnostic histories per boundary: 16 per traverser/fold stratum.
  One traverser decision is selected uniformly from each separately generated
  behaviour trajectory. This defines the probe distribution; it is not an
  exhaustive or globally reach-weighted information-set sample.
- 128 paired, recursive corrected rollouts per history. Both budgets use
  identical chance/action draws, the same frozen regret policies and the same
  frozen calibration/sampling policy. Two-fold leave-one-out inference has
  zero ensemble disagreement, so changing the critic cannot change sampling.
  The diagnostic calls the production control-variate/centring algebra at
  every decision, including opponent decisions and recursive bootstrapping.
- Within-history, per-legal-action advantage variance (sample variance with
  `ddof=1`), averaged equally over histories. It is **not** the variance of
  observations pooled across different states, exact exploitability, or a
  complete estimate of global regret-estimator variance.
- Independent one-step transitions from those histories, scored against each
  critic's original frozen TD target. Online and temporally averaged deployed
  models are both evaluated. These are held-out **TD** errors, not errors
  against an exact best response or oracle action values. Fresh trajectories
  are held out; previously visited information sets are not excluded.

Diagnostics never enter any replay. Training RNGs and replay counters are
checked before/after evaluation. Read-only recursive returns are regression
tested against the native replay-writing DFS with identical random draws.
Diagnostic inference uses one thread to avoid overhead on tiny forward passes;
critic fitting retains eight intra-op threads. Diagnostic time is separate.

The short fit's timing is its observed native prefix (including cache
construction) plus the full fit's measured finalisation overhead. It is labelled
an **estimate**, not an independently timed short fit. Snapshot and audit
overheads are excluded from the native fitting timers. The observed whole
iteration duration, which includes instrumentation, is reported separately.

## Interpretation and outputs

`analysis/summary.json`, `per_seed.json`, `detailed_results.json`,
`source_manifests.json` and `report.md` contain the results. Within each seed,
boundaries, folds and histories are averaged first. Paired differences and
95% t intervals use **three source seeds**, not thousands of rollout replicates.
The small sample, adjacent boundaries and late-stage-only scope must accompany
any interpretation. There is no automatic acceptance threshold or promotion:
review variance, held-out errors, tail/state-level diagnostics and cost jointly.
A credible result supports an end-to-end follow-up, not an equivalence claim.

Workers retain JSON, resource diagnostics and small critic-only weight files.
These are not playable policies or resumable full learners. There are no new
policy fits in this experiment, and **no replay or full training states are
uploaded**. The original full state stays read-only outside the output tree.

## Run

After committing and pushing the implementation, from the repository root:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
# Keep the existing FHP experiment runner service account in SA_EMAIL.
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp17-critic-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp17_critic_budget.sh run
```

The launcher reads small source manifests/object metadata before submission.
The cloud controller runs a reduced **real-source** smoke first (full replay
and original eight-actor restore, only shorter diagnostic workloads), then
three audit workers and aggregation. Each worker verifies the large source
state checksum before deserialising it. Three concurrent workers require
three `n2-standard-16` VMs (48 N2 vCPUs). Set `PARALLELISM=1` to run seeds serially.
The 12h worker timeout is a safety ceiling, not a predicted runtime. The real
smoke reports actual resource use; the production variance audit is larger.

No automatic Batch worker retries. `resume` explicitly resubmits the workflow;
completed workers are checksum-verified and reused. An incomplete worker
replays the short audit from the immutable original source, because no new
large continuation state was retained. Do not change REPO_REF for the same
RUN_ID: source/configuration/runtime/code identity mismatches fail closed.

Download the small analysis only:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Local validation uses small real FHP data, **not scientific results**:

```bash
python3 -m pytest -q tests/test_exp17_critic_budget.py
bash gcp/run_exp17_critic_budget.sh smoke-local
# Requires permission to start eight local Ray actors:
RUN_EXP17_RAY_TEST=1 python3 -m pytest -q tests/test_exp17_critic_budget.py -k native_parallel
```
