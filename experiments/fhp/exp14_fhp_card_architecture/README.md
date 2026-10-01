# Experiment 14: frozen-reservoir card-architecture audit

## Question and scope

Can sharing card-processing weights across suits improve generalisation and
deployed poker play, beyond merely adding parameters to the average-policy
network? This is an **offline policy-fitting audit**, not fresh UCV training.
It uses Experiment 2's final 24-hour reservoirs, seeds **0, 1, 2**, one million
rows each. No critic, regret network, calibration network or replay-generation
process is updated. No exact best-response or exact average-policy target is
used. It does not depend on Experiment 13 having finished.

The existing 183-feature canonical encoding already removes global suit-name
redundancy. This experiment tests a different **inductive bias**, not another
lossless compression factor. Ranks, private/public-card roles, player, round
and exact betting history remain available. No opponent private cards or
Experiment 10--12 feature additions enter any policy.

## Four prespecified arms

| Arm | Architecture | Parameters |
|---|---|---:|
| `standard` | Exact Experiment 2 policy: 64-wide card/context branches, 192 x 192 trunk | 74,243 |
| `dense` | Same baseline plus generic 183 -> 200 -> 110 -> 3 dense residual | 133,486 |
| `deepsets` | Same baseline plus shared-suit encoding and invariant pooling | 133,478 |
| `attention` | Same baseline plus shared-suit encoding, one attention block and invariant pooling | 133,094 |

The three residual alternatives are within **0.3%** in parameter count. The
dense residual is the capacity/control-path comparator; it is intentionally
not simply the Experiment 3 wide trunk tested in Experiment 13. All models
are reset, all baseline parameters remain trainable, and a common seed gives
all arms identical baseline initial weights. Residual output heads start at
zero. There is no pretraining, frozen baseline, dropout or stochastic
augmentation that changes data exposure between arms.

The suit branch forms four tokens. Each token contains **13 ordered hole-card
rank indicators plus 13 ordered public-card rank indicators**. A single
26 -> 64 -> 64 ReLU encoder is applied to every suit, using batched tensor
operations. No suit embedding or positional encoding is added. Mean and max
pooling produce 128 features. These join the 71 suit-invariant context
features; the eight suit-indexed count features are excluded from this branch
but retained in the exact baseline pathway.

Deep Sets uses a 199 -> 184 -> 88 -> 3 residual head. Attention inserts one
64-wide, four-head, pre-normalised self-attention block over the four tokens,
with a 128-wide feed-forward sublayer, then pools and uses a 199 -> 96 -> 3
head. Attention is implemented with batched matrix multiplication, not a
Python loop over suits. Both branches are invariant to a joint permutation
of suits up to floating-point roundoff. The complete model is suit-invariant
through the canonical encoder; the baseline skip path is **not** claimed to
accept arbitrary non-canonical permutations directly.

The methodological motivations are [Deep Sets](https://arxiv.org/abs/1703.06114)
and attention over sets, as in the
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html). This compact
four-suit residual implementation is not a reproduction of those papers and
does not inherit a demonstrated poker-performance benefit from them.

## Stronger held-out test

Diagnostic fits use an **80/10/10 split by exact canonical visible card
configuration**, keeping all betting histories and player labels associated
with the same hole/public cards in one partition. The key is the exact 104
binary card indicators, packed losslessly into 13 bytes. It is not a hand
bucket, equity bin, rank abstraction or approximate hash. Assignment is
deterministic per source and shared by all architectures/recipes/replicates.

Strata use round (encoded by preflop versus postflop category), existing made
hand category and card-configuration replay frequency. Tiny strata retain
training examples rather than duplicating them. Exact card, information-set,
replay-row and replay-mass proportions, player coverage and checksums are
recorded. Fractions need not be exactly 80/10/10 after stratum rounding and
unequal histories per card configuration. Different source reservoirs remain
separate experimental units; they are not pooled into one training set.

**Target grouping remains by complete canonical information set.** The card
key is used only to partition the data. Action targets from different betting
histories or player roles are never averaged together. Replay masses and
iteration weights are retained exactly. Uniform-group minibatches use the
same importance/loss-scale multiplier `M_g * U / N` as Experiment 13.

This is a stricter generalisation test than Experiment 13's information-set
split. Its held-out losses should not be interpreted as directly comparable
scores from the same partition. The targets are still sampled replay targets,
mostly sparse observations, not an exact policy oracle.

## Fitting and selection

All four arms use soft-target cross-entropy, iteration exponent 2,
minibatches of **2,048**, and two reset initialisations per source. Sampler
randomness is independent of network initialization, and minibatch identities
match across all architectures and optimiser recipes for a source/replicate.

Fixed controls use Adam **0.003**, with **20k and 60k** endpoints on one nested
trajectory. The primary architectural contrast is **Deep Sets minus the
parameter-matched dense residual at 20k**. Attention versus dense, comparisons
with the standard policy, 60k and architecture-by-budget interactions are
secondary.

Each architecture receives the same validation-only screen:

1. Adam 0.003.
2. Adam 0.001.
3. Adam 0.0003.
4. AdamW 0.001, weight decay 0.0001 on matrices only (not biases or layer norms).

Every path reaches 60k, with diagnostics at **5k, 10k, 20k, 40k and 60k**.
There are 96 screening paths across three sources and two initialisations.
One recipe/budget per architecture is selected globally by validation CE,
averaging fitting replicates within source then weighting the sources equally.
Ties prefer fewer updates then the listed recipe order. Selection is locked
before any test score or poker result is consulted. A selected 60k endpoint
or any recipe still improving at the boundary flags optimisation as unresolved.

The fixed and selected diagnostic policies are scored once on the held-out
test split. Separate reset fits on **all replay** produce playable policies.
Fixed budgets provide 48 deployed endpoints; tuned arms add up to 24. Shared
recipe paths/checkpoints are reused rather than fitted twice. Diagnostic
test metrics and full-data deployment metrics remain explicitly separate.

## Poker evaluation, cost and inference

Reuse the Experiment 13 evaluator and units:

- Every full-data refit versus its source's archived policy: **50,000 duplicate
  deal pairs**.
- Deep Sets/attention versus matched dense, and dense/Deep Sets/attention
  versus matched standard: **50,000 duplicate pairs** each, matched by source,
  initialisation and fixed-budget/tuned status.
- Five existing rule opponents: **10,000 duplicate pairs** per policy/opponent.
- Restricted LBR: **1,000 duplicate pairs**, with the established **4,096**
  preflop-rollout setting; ten-pair resumable shards.

A pair consists of two seat-swapped hands. Arms share deals and action seeds;
evaluation RNGs are separate from fitting. Byte-identical checkpoints with
identical evaluation settings share their simulated results, but each named
arm retains an independently validated task record. This saves work when a
tuned checkpoint equals a fixed control. Different policies are never deduplicated.

Fitting replicas are averaged within each of the **three source seeds** before
reporting uncertainty. More hands do not turn this into six independent
training runs. Intervals and exact sign-flip tests are exploratory; with three
sources, the minimum two-sided sign-flip p is 0.25. Direct-play performance,
restricted LBR and held-out fidelity are complementary: none alone establishes
equilibrium convergence, and LBR is not exact exploitability.

The analysis includes CE/KL/probability errors by round, frequency and action,
source-level paired contrasts, fit time, processed examples, inference cost,
memory logs and four charts:

- `card_architecture_learning_curves.png`: training/validation KL by updates/time.
- `card_architecture_optimisation.png`: all tuning curves for all four architectures.
- `card_architecture_quality.png`: held-out test KL, LBR and direct play by budget.
- `card_architecture_cost_quality.png`: LBR versus fitting and inference costs.

Only consistent held-out and gameplay gains, with an acceptable inference cost,
would justify a separately approved end-to-end experiment. A negative or
uncertain result only concerns these architectures and frozen data, not the
general value of suit sharing or larger-game network capacity. Exact inputs
remain available, but additional capacity still risks overfitting sparse data.

## Cloud, retention and launch

Workflow: **smoke -> screen[3] -> select -> train[3] -> aggregate**. Here `train`
means policy refitting and evaluation, not UCV regret training. Each source
has a separate **n2-standard-8** VM; default parallelism is three (**24 N2
vCPUs**). Fitting uses one thread for reproducibility; evaluation uses eight
spawned processes. The cloud smoke loads/groups the real million-row source,
benchmarks all four architectures with batch 2,048, and exercises a tiny
end-to-end fit/evaluation. Local smoke uses synthetic targets only.

This is materially larger than Experiment 13: twice as many screening paths
and three times as many nominal direct-play tasks before alias deduplication.
The cloud smoke reports fitting-only timing, not a reliable total completion
estimate. Screening and deployment each have **48-hour safety caps**, not
fixed training durations; the controller covers both stages and queued waves.
Automatic paid retries are disabled. Completed fits/evaluation tasks can be
reused by explicit resume with unchanged source, code and configuration.

Retain playable checkpoints, all analysis/evaluation and provenance. Source
states are temporary VM inputs outside the output root. **No full training,
replay or optimiser states are written to Experiment 14 outputs.**

After the new code has been committed and pushed, from this repository with
`PROJECT_ID`, `REGION`, `BUCKET`, `SA_EMAIL` already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP2_RUN_ID="exp2-fhp-20260921-093839"
export RUN_ID="exp14-cards-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_exp14_card_architecture.sh run
```

The launcher checks that `REPO_REF` contains Experiment 14. The laptop can
disconnect after controller submission. Check status with the same `RUN_ID`:

```bash
./gcp/run_exp14_card_architecture.sh status
# On failure only, preserving RUN_ID, source run and original REPO_REF:
./gcp/run_exp14_card_architecture.sh resume
```

Local verification without a paid cloud job:

```bash
PYTHON=/path/to/venv/bin/python ./gcp/run_exp14_card_architecture.sh smoke-local
```

Results live at `$BUCKET/$RUN_ID/analysis/`; all source/run checksums and
selection locks remain in `workers/`. The existing Experiment 13 launcher,
configuration and default comparisons remain unchanged.
