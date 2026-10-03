# Experiment 15: Experiment 9 frozen-policy learning-rate audit

Does **Adam 0.0003**, instead of **0.003**, extract a stronger average policy
from the identical final Experiment 9 replay? This is an offline refit, not a
new UCV trajectory. No critic, regret, calibration or traversal training runs.

## Prespecified comparison

- Source: the original **24-hour Experiment 9**, normally run
  `exp9-cache24-20261001-132550`, seeds **0, 1, 2**. Its final full states contain
  the replay; playable policies alone do not. Later extended Experiment 9 runs
  and other experiments are rejected. The VMs fetch just the selected final
  state, playable policy and manifests, not earlier checkpoints.
- Exactly **1,000,000 source replay rows per seed**. Checksums, experiment,
  seed, iteration, encoder, policy/state weights and original fitting settings
  must agree before fitting. Source objects are never changed.
- Same 183-feature canonical representation, 64-wide card/context branches,
  shared **192 × 192** layers, 74,243 parameters and masked three-action output.
- Same grouped soft-target cross-entropy, iteration exponent **2**, uniform
  information-set minibatches of **2,048**, and replay-mass loss multipliers
  `M_g * U / N`. Different hands are not merged into approximate buckets.
- **Reset fitting, 20,000 Adam updates**, with only the learning rate changed:
  `adam_003` versus `adam_0003`. No scheduling or weight decay.
- **Two matched fitting initialisations per source seed**. Each rate uses
  identical initial weights and minibatch identities for a given replicate;
  hashes verify this. Both rates process 40,960,000 group examples per full
  fit. Sampling randomness is independent of network initialisation.
- Retain the **archived 0.003 policy** as an additional control. The primary
  comparison is low-rate versus a fresh matched 0.003 refit, not simply versus
  the historical policy whose reset/sampling randomness differs.

The fixed 20k endpoint answers the narrow equal-budget question. This is not
an optimisation search, and no checkpoint is selected after observing results.
A lower rate still improving at 20k merits a separate budget test; a null here
does not show that the lower rate could never help.

## Fidelity and poker quality

Separate diagnostic paths fit a deterministic **90/10 split by complete
canonical information set**. Repeated observations of the same information
set stay together. Training and held-out CE, KL and probability errors are
recorded at **5k, 10k and 20k** updates, including round/frequency/action
breakdowns. No held-out score changes the rate or stopping rule. Similar card
configurations in different information sets may cross the split; this is not
a card-configuration-disjoint generalisation claim.

Deployment paths reset again and fit **all replay**. Only their fixed 20k
policies receive poker evaluation. Diagnostic held-out scores are not ascribed
to the deployed models, which have seen the whole dataset.

There are eight 20k fitting paths per source (two phases × two rates × two
initialisations), **24 in total**, versus the much larger capacity and card
architecture screens. This deliberately omits the 60k and architecture arms.

Evaluation reuses the existing FHP evaluation suite and units:

- Each deployed fit against the archived policy: **50,000 duplicate deal pairs**.
- Low-rate versus matched fresh 0.003 fit: **50,000 duplicate pairs**, both
  replicates, within each source. This is the primary strategic comparison.
- All four fits and the archived policy against each of the five existing rule
  opponents: **10,000 duplicate pairs** per matchup. Deals and action seeds are
  common across arms and separate from fitting randomness.
- **LBR is off by default**, to keep this follow-up inexpensive. Opt in before
  submission with `EXP15_INCLUDE_LBR=1`: 1,000 pairs, 4,096 preflop rollouts,
  ten-pair resumable shards, using the unchanged evaluator. Omitted LBR is
  absent, not zero. Preserve this setting when resuming a run.

A duplicate pair is two seat-swapped hands. All replicates are retained, not
selected by their poker results. Replicates are averaged **within source**
before means, standard errors, paired differences, exploratory 95% t-intervals
and exact sign-flip tests. Three sources remain three inferential units; the
smallest two-sided sign-flip p-value is 0.25. Lower held-out loss or a direct-play
win does not establish lower exploitability. LBR, if enabled, is restricted
vulnerability rather than exact exploitability. These reused development
sources cannot establish independent confirmation or equilibrium convergence.

## Cloud execution

**Cloud smoke → three policy-fit/evaluation workers → aggregation.**
Each source worker uses one **n2-standard-8** with 30,000 MiB requested memory
and a 200 GB disk. Default parallelism is three (24 regional N2 vCPUs).
Fitting uses one Torch thread for deterministic matched comparisons;
evaluation uses eight spawned processes. Source replay is released before
evaluation starts. No parallel UCV actors are launched.

The real-source cloud smoke first validates and groups the complete source,
then benchmarks production-size fitting before running tiny fits and sampled
gameplay. Local smoke uses real FHP encodings with artificial replay targets;
its results are not evidence about learning performance. Cloud and local smoke
also test reloading fitted policies through the normal playable-policy loader.

Python is pinned to **3.11.16**, and dependency versions come from the existing
requirements. Source, code and full runtime identities are locked per run;
incompatible reuse fails rather than silently mixing results. Completed fit
paths and evaluation tasks are reused on explicit resume. An interrupted fit
restarts from its deterministic reset: no optimiser state is persisted.

Worker jobs have a **24-hour safety ceiling**, not a requested training
duration. Actual completion depends on grouping, fitting and sampled gameplay.
The cloud smoke prints a fitting-only projection; it is not a total-job ETA.
Automatic paid retries are disabled. No cloud jobs are started by tests.

After committing and pushing, from this repository with your existing
`PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL`:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP9_RUN_ID="exp9-cache24-20261001-132550"
export RUN_ID="exp15-lr-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
export EXP15_INCLUDE_LBR=0
bash gcp/run_exp15_policy_learning_rate.sh run
```

The launcher rejects stale commits and missing final source states before
submitting a controller. Your laptop may disconnect after successful
submission. Keep `RUN_ID`, `REPO_REF`, source and LBR setting unchanged for:

```bash
bash gcp/run_exp15_policy_learning_rate.sh status
# Explicit recovery after a failed job:
bash gcp/run_exp15_policy_learning_rate.sh resume
```

For local smoke using a repository-compatible Python environment:

```bash
PYTHON=/path/to/venv/bin/python bash gcp/run_exp15_policy_learning_rate.sh smoke-local
python -m pytest tests/test_exp15_policy_learning_rate.py
```

## Outputs and retention

Retain all small playable policy checkpoints (including diagnostic 5k/10k/20k
checkpoints), metrics, source/code hashes, evaluation tasks and resource logs.
**No new full training, replay or optimiser states are published.** Large
input states live outside the output tree on temporary VMs. Source states in
the original Experiment 9 prefix remain available for future work.

The analysis folder includes per-replicate and per-source metrics, paired
contrasts, complete fitting diagnostics, budget-boundary warnings, source
manifests and two charts: `learning_rate_fidelity.png` and
`learning_rate_policy_quality.png`.

```bash
# BUCKET should include gs://, as in your existing download commands.
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```
