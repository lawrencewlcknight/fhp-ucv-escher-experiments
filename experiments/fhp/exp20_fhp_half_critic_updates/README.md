# Experiment 20 — 24-hour training with half the critic updates

## Question and controlled change

Does halving the critic-fitting budget improve policy strength at a fixed
24-active-hour training budget? Experiment 17 found similar held-out critic
errors with shorter fits on the full-fit learner's trajectory. It did **not**
establish the effect of persistently training a shortened-budget learner.
Experiment 20 tests that effect through independent, full-length training.

The baseline is Experiment 10, historical run
`exp10-features-20261001-161740`, commit
`e66d4da515eb212e5026a965ac5c39c86144c901`.
The sole production learning-configuration change is
`baseline_network_train_steps: 10000 -> 5000`, for **each of the two critics**.
Training starts from scratch with seeds `0, 1, 2`; it does not load Experiment
10, 16 or 17 models. The strict configuration test enforces this one-key change.

Unchanged: canonical OpenSpiel FHP, hand/board feature encoder (213 policy and
323 critic inputs), all network architectures, learning rates, replay capacities
and sampling, traversal budgets, regret updates, four-fit target averaging,
cached frozen critic targets, and average-policy fitting. Eight Ray traversal
workers and eight central fitting threads run per seed on one Standard
`n2-standard-16`. The optional Experiment 18 replay-memory implementation is
**not** adopted. This is a CPU-only experiment.

## Training, checkpointing and safety

- Three independent seeds run concurrently by default: three VMs / 48 N2 vCPUs.
  `PARALLELISM=1` or `2` changes scheduling, not a seed's resources or algorithm.
- Each seed stops at the first completed outer iteration after **24 active
  training hours**. Policies are fitted and saved at 6, 12, 18 and 24 active
  hours. Boundary overshoot is recorded; checkpoints need not occur at exactly
  those elapsed wall-clock times.
- The active clock excludes checkpoint policy fitting, serialization and
  uploads. Thus 72 seed-VM hours is a lower-bound training budget, not the
  expected billable total. Each initial training task has a **36-hour elapsed
  safety limit**; reaching that limit is a failure, not a valid completed run.
- Only the final 24h checkpoint has a full resumable training state. It includes
  model and target weights, target history, optimizer states, replay buffers and
  counters, driver RNG states, and all eight actors' sampling/RNG state. Earlier
  checkpoints are playable policies, **not** complete training restarts.
- Full states use a distinct Experiment 20 identity. Resume rejects other
  experiment states, incompatible configurations, source commits, or runtimes.
  No 48h extension is automatic.
- The cloud controller runs smoke, training, then aggregation. It stops on
  failure and does not retry paid stages automatically. Once a submitted
  controller is running, the laptop is not needed.
- Cloud smoke checks cached/uncached equivalence, exact next-iteration restart,
  extension of a completed endpoint, and production replay/checkpoint capacity.
  Local smoke uses small replay and shortened thresholds, not full-capacity RAM.
- Resource snapshots, driver/actor RSS, cgroup memory data where available,
  phase/critic-fit/cache timers, tracebacks and Batch diagnostics are retained.
  The final summary records the actual critic budgets for both members and
  aggregation requires `[5000, 5000]`. Host termination can prevent final logs;
  inspect Cloud Logging and periodically uploaded diagnostics as well.

## Run on GCP Batch

First commit and push the implementation. Run these commands from this repo's
root with `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` already configured.
Pin the **full pushed SHA**, and retain it with the run ID for future resumption.

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp20-critic24-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
export EXP20_TOTAL_HOURS=24
unset EXP20_SOURCE_RUN_ID

# Offline job generation; no cloud actions.
bash gcp/run_exp20_half_critic_updates.sh dry-run

# Submit just the paid cloud smoke, then check until SUCCEEDED.
bash gcp/run_exp20_half_critic_updates.sh smoke-cloud
bash gcp/run_exp20_half_critic_updates.sh status

# Retain the same RUN_ID: the controller reuses the successful smoke.
bash gcp/run_exp20_half_critic_updates.sh run
```

`run` also schedules smoke automatically if no standalone smoke exists. It
refuses a source run or non-24h initial budget. `resume` can restart a failed
controller and reuse successful children; it does **not** restart failed
training. Diagnose a failed stage before authorising further compute. The Batch
service account needs the same controller submission, logging and storage
permissions used by the other experiments.

For a small local integration check in an environment with the pinned project
dependencies and Ray available:

```bash
bash gcp/run_exp20_half_critic_updates.sh smoke-local
```

## Separate frozen-policy evaluation

Training contains no rule-agent play, LBR or head-to-head matches, and no exact
exploitability traversal. After training and aggregation succeed, explicitly
launch the independent evaluator using the same run ID and workflow commit:

```bash
bash gcp/run_exp20_half_critic_updates.sh dry-run-evaluate
bash gcp/run_exp20_half_critic_updates.sh evaluate
bash gcp/run_exp20_half_critic_updates.sh status
```

This uses the repository's established FHP evaluation implementation, a separate
Standard `n2-standard-16`, 16 scoring processes with one computation thread
each, and a 48h elapsed ceiling (`EVAL_MAX_HOURS`, maximum 72). It checks a small
smoke panel before the full protocol. Source policies are frozen; no fitting
occurs. Large training states are not downloaded. All three source seeds and
four checkpoints must pass identity, game, configuration, runtime and checksum
validation, including reloading the policies.

The full protocol is the same budget as prior 24h pairwise evaluations:

- Five rule agents: 10,000 duplicate deal pairs per opponent and policy.
- LBR: 1,000 pairs per policy, 4,096 rollouts, 10-pair shards.
- Direct Exp20 versus Exp10: 50,000 pairs per matched seed and training horizon.
- Temporal play: all six checkpoint pairs within each seed/experiment, with
  50,000 pairs per match. Later checkpoints are the first player-policy argument.

There are 24 source policies, 2,568 scoring tasks and 3,624,000 duplicate pairs
(7,248,000 hands), excluding the smoke. The prespecified primary endpoint is
**Exp20 at 24h versus Exp10 at 24h**, averaged over the three matched seed labels.
Positive direct payoff favours Exp20. Earlier and temporal comparisons are
exploratory. Common evaluation deals are used across the compared policies.

Tables report seed-level means and pointwise 95% intervals. The five plots show
rule-agent payoff and LBR payoff against active hours and observed node counts,
plus direct crossplay against hours; plot error bars are standard errors.
Historical matching seed labels do not make learning trajectories identical or
control cloud-host variability. Node-axis plots are descriptive, not a
matched-node experiment. Three seeds give limited precision. An interval
crossing zero is not evidence of equivalence or convergence; LBR is approximate
and restricted, not exact exploitability. There is no automatic best-checkpoint
selection or promotion of the shortened-budget configuration.

If only evaluation fails, first ensure it has stopped, then explicitly resume
with the **same** `RUN_ID`, `REPO_REF` and evaluation budget:

```bash
bash gcp/run_exp20_half_critic_updates.sh evaluate-resume
```

Completed task results are reused only if the policy, protocol, implementation
and dependency fingerprint still matches. A running evaluation blocks a retry.
This action does not rerun training.

## Outputs and interpretation

Training artifacts live at `$BUCKET/$RUN_ID/{workers,analysis}`. Analysis includes
`throughput_summary.json`, `critic_cache_summary.csv`, per-seed/checkpoint
tables, `nodes_by_training_time.png`, `training_time_breakdown.png`, and the
contract and feature specification. The native per-worker checkpoint manifest
identifies the final full-state path and checksum. Preserve these remote states.
Evaluation outputs are under `$BUCKET/$RUN_ID/evaluation/analysis`, including
the direct, rule, LBR and temporal tables, metric differences, figures and
`analysis_summary.md`.

Download small analysis artifacts without downloading replay or individual
evaluation task outputs:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"

# After the separate evaluation has completed:
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/evaluation/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
```

Compare nodes per active second, total nodes at 24h, completed iterations,
critic time and memory with Experiment 10. Experiment 17 motivated a rough
**50% node-throughput uplift forecast**, not a measured Experiment 20 result.
Shorter fits may change trajectories, actor load and model quality. A speed
gain alone does not establish better poker play; use the separate primary
head-to-head endpoint and the supporting diagnostics.

## Later continuation, only if requested

To extend the completed Exp20 states to 48 cumulative active hours, use a new
run ID and the **original Exp20 training commit**. This is not a warm start
from Exp10, and it preserves the original source states and 6–24h policies.

```bash
export EXP20_SOURCE_RUN_ID="$RUN_ID"  # the completed original Exp20 run
export RUN_ID="exp20-critic48-$(date -u '+%Y%m%d-%H%M%S')"
export EXP20_TOTAL_HOURS=48
# Keep REPO_REF pinned to the original Exp20 training commit.
bash gcp/run_exp20_half_critic_updates.sh dry-run
bash gcp/run_exp20_half_critic_updates.sh extend
```

This adds approximately 24 active hours, new 30/36/42/48h policies and one new
final full state. The extension has a 72h elapsed cap. The current evaluation
command deliberately accepts only original 24h runs; an extension's evaluation
should be specified separately.
