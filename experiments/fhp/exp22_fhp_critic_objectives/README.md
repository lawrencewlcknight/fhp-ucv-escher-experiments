# Experiment 22 — late-state critic objective audit

This is the **first, frozen-24-hour-state screen only**. It neither resumes UCV
training nor changes existing training defaults. It does not implement the
earlier/middle reference runs, an online adaptive controller or a policy-quality
experiment. Original Experiment 10 objects remain read-only.

## Source and comparisons

Use `exp10-features-20261001-161740`, the `time_24h` full states for source seeds
0, 1 and 2. Source identity, learning configuration, source runtime, metadata and
state checksums use Experiment 17's validator. The historical monolithic files
are about 5.4 GB each. Each worker downloads one, verifies before deserialising,
and reconstructs just the frozen policies/calibration and critic/replay state.
No Ray actors are needed, and no collection iteration is advanced.

| Arm | Data | Objective |
| --- | --- | --- |
| `replay_td` | Original native critic replay | Original cached frozen-target TD MSE |
| `matched_td` | Fresh common paired-continuation panel | Ordinary root sampled-action TD MSE |
| `weighted_td` | Exactly the same fresh panel/minibatches | Sampling-aware weighted root TD MSE |
| `regret_variance` | Exactly the same fresh panel/minibatches | Paired recursive corrected-advantage disagreement |

Each arm starts from identical source online weights, Adam state, original
frozen TD targets and target history **within each critic member**. Fit once to
10,000 updates and capture 2,500/5,000/10,000 prefixes. No reset or target refresh
occurs at an interior prefix. The deployed candidate is the native parameter
average after appending that prefix ONCE to the common history (window four).
Optimiser state and private minibatch RNG state are saved at each prefix.

The extra `matched_td` control is essential: native replay versus fresh panel
changes the data distribution. The comparison of weighted/variance fitting with
matched TD holds that data fixed. Comparisons with native replay instead test
the entire objective-plus-data package. Source replay is available unchanged to
all branches but only `replay_td` consumes it; we do not pretend historical
transitions contain the unrecorded action probabilities or replayable game
histories needed by the other objectives.

Learning rate, Adam state and gradient clipping are inherited unchanged. Native
replay TD keeps its original 2,048-row minibatches. All three matched arms use
32 pairs (64 root transitions) per update, sampled with replacement and identical
minibatch prefixes. The variance arm additionally uses the continuation nodes.
Thus **equal update counts are not equal compute or equal numbers of network
input rows**. Report measured cost alongside the update-budget comparisons.
Prefix records include minibatch-stream hashes and cumulative root/online-network
input counts, so the matched arms' sampling can be independently checked.

## Loss derivation and its limits

At a decision history h, freeze target strategy pi, behaviour proposal xi and
candidate control b. For the sampled action A and recursively corrected return G:

    q_b[a] = b[a] + 1{A=a} / xi[A] * (G - b[A])
    r_b = C_pi q_b,     C_pi = I - 1 pi^T.

The implementation applies this same correction recursively at every decision,
including opponent decisions, and uses the native terminal utility normalisation.
Chance outcomes and sampled actions are recorded independently of candidate
weights. Opponents follow pi; full support is required at traverser decisions.
Opponent zero-probability actions carry zero policy weight in the propagated
value. No action/importance clipping or candidate-dependent sampling is added.

For two **independent** continuations Z1 and Z2 from the same full history:

    E[ ||r_b(h,Z1) - r_b(h,Z2)||^2 / (2 m(h)) | h ]
      = tr Cov(r_b | h) / m(h),

where m is the number of legal actions. Since the estimator's conditional mean
does not depend on a fixed predictable b, this targets noise, not the magnitude
of genuine regrets. Equivalently, its covariance contains
`C_pi Cov(q_b) C_pi^T`, so action covariance is included through centring.
Gradients propagate through the online candidate's predictions at all recorded
decision nodes. Frozen old networks supply TD labels only; they do not replace
the recursive corrected returns in the variance loss.

This is an empirical finite-panel objective, not an exact variance oracle. A
candidate can overfit its fitting continuations; only independent assessment
supports generalisation. Fitting-panel estimates themselves are not unbiased
performance evidence after fitting on those same samples. The deployed average
is evaluated separately; it is not substituted for the online network in the
training objective. An oracle value baseline already has established
variance-reduction properties; this experiment makes no novelty or general
convergence claim merely by using baselines.

The simple weighted-TD comparator multiplies sampled-action TD squared error by
`1 / (m(h) xi[A])`. Its expected root loss is uniform-action TD risk under the
chosen root distribution. This is **not** claimed to be the variance-optimal
loss. There is no batch-dependent normalisation or weight clipping.

Reference theory: [VR-MCCFR](https://arxiv.org/abs/1809.03057) and
[Low-Variance and Zero-Variance Baselines](https://proceedings.mlr.press/v119/davis20a.html).
The paired identity above and its application to this implementation are
documented/tested here, not attributed as a novel result of those papers.

## Independent panels and selection

Per critic member, split roots equally between traversers:

| Split | Full-history roots | Independent continuations per root |
| --- | ---: | ---: |
| Fitting | 256 | 32 |
| Candidate selection | 64 | 64 |
| Final assessment | 128 | 128 |

Choose one traverser decision uniformly from each fresh behaviour trajectory.
This is not an exhaustive, reach-weighted or information-set-balanced sample.
All root hashes are disjoint across splits and members; new RNG streams are
used for every split/root/continuation. Roots may have appeared historically in
source replay, and disjoint full histories do not imply disjoint information sets.

Candidate networks use paired common random numbers across arms, but the two
continuations within each pair are independent. The fixed policies/proposal and
all fresh-data panels are shared. The two original critics remain distinct.

Choose one update budget per objective using mean **deployed** selection-panel
variance averaged over the two members; prefer fewer updates on an exact tie.
Select the stronger native/matched TD control on that same selection panel.
Write and checksum `selection_lock.json` before generating final-assessment
panels. Never use final results to select budgets or rerun until favourable.

Report every candidate, not only winners: online/deployed within-history action
covariance, mean advantages, TD error, mean/median/p90/p95/p99/max state variance,
and maximum absolute sampled advantage. Also assess untouched source critics.
Tail quantiles are descriptive and can be noisy at these panel sizes.

The aggregate gives selected weighted/variance minus matched-TD and best-TD
contrasts with paired t intervals over **three source seed means**, not thousands
of rollouts. Seed 0 alone has no seed-level confidence interval. Intervals are
exploratory, pointwise and not corrected for multiple comparisons. Pilot seed 0
is reused once; continuation to more seeds is a feasibility decision, not a
significance gate. Do not change code/thresholds after inspecting the pilot and
then silently combine it with the remaining seeds.

## Cost and saved artefacts

One `n2-standard-16` per worker: eight fitting threads, one diagnostic thread.
Network prediction and recursive loss inputs are batched; traces use packed
contiguous arrays and only game-depth recurrence loops in Python. Root-TD arms
do not run networks on unused continuation nodes. Large historical replay is
loaded once, not copied for each arm. Diagnostics run outside fitting timers.

The six-hour worker-process cap and eight-hour Batch cap are **safety limits,
not runtime forecasts**. Real-source cloud smoke has reduced panels/three updates
and a 30-minute process cap; setup/download are additional. Inspect pilot timings
before submitting other seeds. No retries or paid follow-on jobs start automatically.

Record source download and loading, panel generation/storage, native target-cache
construction, model/Adam initialisation, each fitting prefix, snapshot/selection cost, final-assessment cost,
audit elapsed time, CPU seconds, peak RSS and external process/cgroup diagnostics.
`standalone_data_plus_fit_seconds` charges each fresh-data method the complete
shared fitting-panel generation/storage cost rather than dividing by the number
of arms. That field excludes selection/assessment/source-loading overhead, which
is reported separately. Audit research cost is not automatically the cost of a
future optimised online controller. Cloud provisioning/uploads remain in job
elapsed time; no unsupported all-in efficiency claim is made from prefix timers.

Saved worker files include candidate `.pt` networks/Adam/RNG states, packed `.npz`
panels, histories/proposals/stream metadata, source hashes, all scores, selection
lock, costs and artifact checksums. These are **critic audit artefacts, not
playable policy or resumable full-solver checkpoints**. Native source states stay
outside upload directories. Pickles are loaded only from trusted hash-verified
sources. Partial failed jobs are diagnostic-only, never eligible for aggregation.

## Run from the repository root

Commit/push before cloud submission. Keep `REPO_REF` unchanged across all stages.
Assume `PROJECT_ID`, `REGION`, `BUCKET` (`gs://...`) and `SA_EMAIL` are set.

```bash
# Optional local execution test with tiny real FHP fixture, not scientific data.
bash gcp/run_exp22_critic_objectives.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp22-critic-smoke-$(date -u +%Y%m%d-%H%M%S)"
export SMOKE_RUN_ID="$RUN_ID"
bash gcp/run_exp22_critic_objectives.sh smoke-cloud
bash gcp/run_exp22_critic_objectives.sh status
```

Wait for the cloud smoke to succeed. It uses the **real** seed-0 state and full
source dimensions; local synthetic smoke cannot unlock production.

```bash
export RUN_ID="exp22-critic-pilot-$(date -u +%Y%m%d-%H%M%S)"
export PILOT_RUN_ID="$RUN_ID"
bash gcp/run_exp22_critic_objectives.sh pilot
# Wait for SUCCEEDED, then explicitly create the small pilot report:
bash gcp/run_exp22_critic_objectives.sh aggregate-pilot
```

Review correctness, finite metrics, cost and diagnostics. If feasible, explicitly
run seeds 1 and 2 with exactly the same settings. Reuse completed seed 0 once;
this avoids paying for an identical repeated pilot. A new code revision requires
a new smoke and pilot, rather than mixing revisions.

```bash
export RUN_ID="exp22-critic-full-$(date -u +%Y%m%d-%H%M%S)"
export PARALLELISM=2   # 1 processes the two remaining seeds sequentially
bash gcp/run_exp22_critic_objectives.sh full
# Wait for SUCCEEDED:
bash gcp/run_exp22_critic_objectives.sh aggregate
```

Use `status` at any stage. `dry-run` generates a job specification without cloud
calls; `JOB_KIND` chooses smoke/pilot/full/aggregate-pilot/aggregate. There is no
controller VM, automatic full experiment, or background polling service.

After the aggregation job succeeds:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Worker panels/weights remain under `workers/seed_N/`. Aggregate downloads exclude
large `.pt` and `.npz` artifacts. Detailed metadata retain their checksums and
source locations. Read `analysis/report.md`, `summary.json`, `per_seed.json` and
`source_manifests.json` together.

**Interpretation:** improvement supports further early-stage/longitudinal study,
not stronger poker play. Failure means no demonstrated benefit in this late-stage
screen, not that the idea cannot help earlier. Nothing is automatically promoted.
