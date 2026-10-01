# Experiment 13: frozen-reservoir policy-capacity audit

## Question and scope

Does increasing only the average-policy network's capacity extract a stronger
policy from the same Experiment 2 replay? This is offline supervised fitting,
not another UCV trajectory. Regret, critic, calibration, traversal and replay
generation are untouched. The Experiment 4 source loader, grouped statistics,
sampler, diagnostic metrics and FHP evaluation protocol are reused.

Inputs are Experiment 2 run `exp2-fhp-20260921-093839`, source seeds **0, 1, 2**,
the first completed 24-hour checkpoint of each, **1,000,000 replay rows per
seed**. The three final states contain approximately 13.4 GB in total. These
are fetched on the VMs, not your laptop; the tiny playable `.pkl` alone does
not contain replay. Only that endpoint and its policy/manifests are fetched,
not intermediate states. The diagnostic and deployment stages each stage
their own inputs. Source objects are never modified or copied to new outputs.
Real inputs are rejected unless each source has exactly one million replay
rows; only explicitly marked synthetic fixtures may use fewer rows, in smoke
mode only.

## Fixed capacity contrast

| Architecture | Card/context branch width | Shared layers | Parameters |
|---|---:|---|---:|
| Experiment 2 (`standard`) | 64 each | 192, 192 | 74,243 |
| Experiment 3 policy only (`wide`) | 96 each | 256, 256 | 133,731 |

Both use the same **183-feature** lossless canonical encoder, three masked
action logits, grouped soft-target CE, iteration exponent 2, uniform group
minibatches of **2,048**, and Adam **0.003**. No new feature/abstraction arms
are included. For group mass `M_g`, group count `U`, replay row count `N`,
the loss multiplier is `M_g * U / N`, preserving Experiment 2's objective
`sum_g M_g * CE_g / N` and its scale. Groups are exact canonical information
sets, not similar hands. No oracle or opponent private cards enter fitting.

Two reset initialisations per source seed are fitted to **20,000 and 60,000
updates**. The earlier endpoint is a prefix of the later fit. Independent
sampler RNGs give both widths and every recipe the same minibatch sequence
for a given source/replicate/partition. Initial weights match across recipes
within one architecture; different widths cannot have identical weights.
Sampler-sequence hashes are checked during selection and deployment aggregation.
Training and validation dataset hashes are checked against the frozen split;
the locked selection binds the complete fitting contracts and diagnostic
results, including checkpoint hashes. Each fit checkpoint is reloaded through
the normal playable-policy loader and its architecture, encoder, weights and
masked action probabilities checked. Regression tests also change hidden
opponent cards while holding the acting player's information fixed, verifying
that neither architecture's policy inputs or probabilities change.

The primary contrast is wide minus standard at **20k**. The **60k** contrast
and capacity-by-budget interaction are secondary. Fixed controls yield
24 deployed endpoints (2 widths x 2 budgets x 3 sources x 2 initialisations).

## Equal optimisation opportunity

Each width receives four diagnostic recipes and two initialisations:

1. Adam, constant learning rate 0.003.
2. Adam, constant learning rate 0.001.
3. Adam, constant learning rate 0.0003.
4. AdamW, constant learning rate 0.001, decoupled weight decay 0.0001 on
   weight matrices only (biases are not decayed).

All run for 60k updates, with diagnostics at **5k, 10k, 20k, 40k, 60k**.
This gives 48 diagnostic fitting paths and 240 small diagnostic policy
checkpoints. These are explicitly labelled `phase=screen`, not deployed
full-replay fits. They are retained so the selected checkpoint can be tested
without retraining or selecting on the held-out test data.

Each source is deterministically split **80/10/10 by canonical group**,
stratified by player, betting round and frequency (singleton, 2--4, 5+).
All observations in a group stay together. Per-stratum rounding and tiny
strata mean fractions can differ slightly; exact counts, replay mass and
split hashes are recorded. Small strata retain training examples rather
than duplicating groups into multiple partitions.

Screening uses training/validation only. After **all three** screens succeed,
a global selection stage chooses one recipe/update budget per architecture:
minimum validation replay-weighted CE, averaging replicas within each source
and then weighting the three sources equally. Exact ties prefer fewer
updates, then the listed recipe order. No poker result or test score enters
selection. `selection.json` is hashed and immutable for a given run.

The deployment stage scores the selected diagnostic models and fixed controls
on the held-out test partition, then performs separate reset fits on **all
replay**. The selected recipe/budget is shared across source seeds, not tuned
individually per source. Tuned/control fits with an identical recipe share
one training path. Up to 12 additional deployed endpoints are produced.
Diagnostic test scores are not attributed to full-replay fits. All optimising
replicas are retained/evaluated, not chosen by their poker results.

Selection of the 60k boundary, or any recipe whose mean validation CE is still
falling from 40k to 60k, raises an optimisation-budget warning. Per-recipe
boundary changes are recorded even if that recipe is not selected. This is a
descriptive diagnostic, not a test of statistically significant improvement. It does
not automatically extend paid jobs or justify a conclusion that capacity is
unhelpful. Any extended screen would be a separately authorised follow-up.

## Strategic evaluation and inference

Each source's full-replay fits plus its archived policy face the existing
five published rule opponents (10,000 duplicate pairs per matchup) and
restricted LBR (1,000 duplicate pairs; 4,096 preflop rollouts per decision).
Every refit plays its archived policy for 50,000 duplicate pairs. Each wide
policy also plays its matched standard policy, same budget/tuned status and
initialisation label, for 50,000 duplicate pairs. One duplicate pair is two
seat-swapped hands. Evaluation seeds/deals are common across arms, separate
from fitting; legal actions, units and evaluator code match Experiment 4.

Evaluation tasks are deterministic, checksummed, sharded and cached. All
source/replica results are retained. Replicas are averaged **within source**
before three-seed means, SEs, paired differences, exploratory t-intervals and
exact sign-flip tests. Replicas and hands are not extra independent sources.
The minimum two-sided exact sign-flip p with three source seeds is **0.25**.
Tuned comparisons reuse development sources and do not isolate width from
optimisation. A null/inconclusive result is not equivalence or evidence that
capacity can never help. LBR is a restricted vulnerability measure, **not
exact exploitability**. Lower CE/KL alone does not establish stronger play.

The held-out targets are sampled replay distributions, not an exact average
policy oracle. Detailed metrics include round/frequency/action breakdowns.
Network-only inference timings use one and up to 2,048 states per call;
feature encoding is unchanged and excluded. Fitting time excludes checkpoint
serialization, inference benchmarks and diagnostic passes, which are separately
recorded. VM-level resource snapshots also include spawned evaluation workers.

## Cloud workflow

`smoke -> screen[3] -> select -> train[3] -> aggregate`

`train` here means **policy refit and evaluation**, not UCV regret training.
Each task array uses one separate **n2-standard-8** per source, 30,000 MiB
requested memory and a 200 GB boot disk. Default parallelism is three:
**24 N2 vCPUs**, not 24 VMs. Screening and deployment arrays run sequentially.
Fitting is single-threaded for reproducibility; evaluation can use eight
spawned processes. A real-source smoke validates million-row loading/grouping
before using a small subset and tiny update/evaluation budgets.
It also times both widths on the full grouped source at the production batch
size of 2,048: ten warm-up steps followed by sixty timed steps. These temporary
models are discarded and never used as candidates. The resulting
`smoke/workers/seed_0/capacity_benchmark.json` and log output report per-update
time and a fitting-only 60k projection, excluding grouping, diagnostic passes,
data transfers, queueing and gameplay. Local synthetic smoke tests do not
provide this production-scale projection.

The provisional estimate is **12--20 elapsed hours** with three workers;
the real-source smoke and recorded timings provide better evidence. Each
screen/deployment task has a 24-hour safety cap (not a training duration).
The controller allows both stages, queued waves and overhead. Automatic
retries are disabled to avoid silently duplicating paid work.

From this repository, with the usual cloud environment variables set, **after
committing and pushing the implementation**:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP2_RUN_ID="exp2-fhp-20260921-093839"
export RUN_ID="exp13-capacity-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_exp13_policy_capacity.sh run
```

Your laptop can disconnect after submission. The account needs existing
Batch child-job and service-account-user permissions used by the other
controllers, plus access to the source/output bucket. Preflight checks the
service-account identity, source states and that `REPO_REF` contains this
experiment. A stale commit is rejected before any job is submitted.

```bash
./gcp/run_exp13_policy_capacity.sh status
# After failure only: preserve RUN_ID, EXP2_RUN_ID and the original REPO_REF.
./gcp/run_exp13_policy_capacity.sh resume
```

Resume reuses completed paths and evaluation shards. An interrupted fitting
path restarts from its deterministic reset (no optimizer state is saved).
Input, code, split or selection changes require a new RUN_ID. Diagnostic
source states remain temporary VM inputs outside the uploaded output root.

Optional local smoke (real FHP encodings with synthetic targets):

```bash
PYTHON=/path/to/venv/bin/python ./gcp/run_exp13_policy_capacity.sh smoke-local
```

## Outputs and retention

Retain all small playable checkpoints, diagnostic metrics, selected recipes,
source/code hashes, evaluation results and resource logs. **No replay or full
training/optimizer states** are written as Experiment 13 outputs.

The complete `analysis/` folder contains JSON, CSV and four charts:

- `capacity_learning_curves.png`: training/validation KL versus updates/time.
- `optimisation_screen.png`: validation trajectories for all four recipes.
- `capacity_policy_quality.png`: held-out fidelity, LBR and direct play.
- `capacity_cost_quality.png`: policy quality versus fitting/inference cost.

Download the analysis without checkpoints or source states:

```bash
AUDIT_BUCKET="${BUCKET%/}"
[[ "$AUDIT_BUCKET" == gs://* ]] || AUDIT_BUCKET="gs://$AUDIT_BUCKET"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$AUDIT_BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```
