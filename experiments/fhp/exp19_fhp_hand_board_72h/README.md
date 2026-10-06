# Experiment 19: continue Experiment 16 from 48 to 72 hours

Test whether another 24 active hours produces useful further improvement. Resume
the **same three trajectories**, rather than training new 72-hour models or
initialising from policy weights. No changes from Experiments 17 or 18 are used.

## Fixed experiment contract

| Item | Specification |
|---|---|
| Source run | `exp16-feat48-20261004-182051` |
| Source endpoint | Completed 48h full training state, seeds 0, 1 and 2 |
| Original learner commit | `e66d4da515eb212e5026a965ac5c39c86144c901` |
| Learning configuration | Original Experiment 10 hand/board features and cached critic targets, unchanged |
| Resources | One Standard `n2-standard-16` per seed; eight Ray actors and eight fitting threads |
| Python / NumPy / Torch / Ray | 3.11.16 / 1.26.4 / 2.7.0+cpu / 2.51.2 |
| Stop | 72 **cumulative active** training hours per seed |
| New playable policies | 54, 60, 66 and 72h |
| New full resumable states | Final 72h only, one per seed |

All existing 6–48h playable policies are copied byte-for-byte. The full source
states restore networks, optimiser states, replay buffers, counters, learner
RNGs and Ray actor states using the original continuation implementation.
The native training metadata retains Experiment 10's identity for compatibility;
the new run ID, Batch labels and `experiment19_contract.json` identify this
workflow. `REPO_REF` pins the new controller and evaluator, **not** a changed
learner. The learner commit is fixed separately inside the job builder.

Thresholds are serviced at outer-iteration boundaries. The added active time is
approximately 24 hours, less any overshoot at the existing 48h endpoint, with a
possible final boundary overshoot. The active clock excludes output-policy
fitting, evaluation, checkpoint serialization and transfers. Thus this is not
a promise of finishing within 24 elapsed hours. Nominal added training is
**72 n2-standard-16 VM-hours** across three seeds, plus those overheads and
separate smoke/controller/evaluation resources. Default concurrent training
requires 48 N2 vCPUs. `PARALLELISM=1` or `2` reduces concurrency, not per-seed VM size.

## Safety and verification

Before submission, metadata preflight checks all three completed source schedules,
configuration and commit, pinned runtime, summary checksums, nonempty full-state
objects and the frozen external opponent panel. This does not download the large
states or prove that they can be restored. Each training VM downloads and verifies
its source state and policies against SHA-256 before the learner resumes;
incompatible or corrupt states fail closed, with no fresh-training fallback.

The cloud smoke includes the inherited eight-actor restart/cache equivalence and
production-capacity checks, plus a new **second continuation**: a small completed
24→48 smoke trajectory is resumed to a 72-hour *checkpoint prefix*. Smoke uses
millisecond thresholds and reduced learning workloads, not 72 hours of training.
It checks forward progress, unchanged imported policies and final-state retention.
Actual production-state restoration happens in each production worker.

The original Experiment 16 outputs are never overwritten. Imported full states
are temporary inputs excluded from uploads and deleted after use. Only the final
72h full state is retained in the new run; 54/60/66h policies cannot restart
training. Existing 48h full states remain in the source run. Keep them until the
continuation has completed and been verified. No paid stage automatically retries.

## Prespecified frozen-policy evaluation

The existing evaluation framework scores **48, 54, 60, 66 and 72h** policies for
each continuing seed. It validates the entire 6–72h history and its lineage back
to Experiment 16, including exact equality of the imported policy hashes.

- **Primary temporal comparison:** 72h versus the same seed's 48h policy.
- **Late-stage comparisons:** 72h versus 60h and 66h. All ten chronological
  checkpoint pairs are reported, with 50,000 duplicated deal pairs per contrast.
- **Fixed learned opponents:** each seed's same-label Experiment 9 24h policy
  (`exp9-cache24-20261001-132550`) and Experiment 8 48h policy
  (`exp8-par48-20261001-005208`), 50,000 pairs per matchup and horizon.
- **Rule agents:** the same five agents, 10,000 pairs per opponent and policy.
- **LBR:** 1,000 pairs per policy, 4,096 rollouts, ten-pair shards. Frozen learned
  opponents also receive rule-agent and LBR scoring. LBR is not exact exploitability.

This matches Experiment 16's evaluation budgets and deal-seed schedule: 21 scored
policies, 2,265 tasks, 4,071,000 duplicate pairs (two seat-swapped hands each).
A tiny real-policy evaluation smoke gates full scoring. The evaluation VM downloads
policies and metadata only, not full training states. There is no policy refitting.

Reports contain temporal play, fixed-panel scores, within-seed changes against
unchanged external opponents, per-match Monte Carlo uncertainty, seed-level 95%
intervals and six charts. The inferential units are the **three training seeds**,
not individual hands, checkpoints or LBR shards. Historical same-label opponents
do not create paired training interventions or an all-cross-seed league.

Judge sustained, seed-consistent external improvement alongside temporal play
and LBR. A win against a predecessor need not mean generally stronger play.
Three seeds have limited power, and the intervals are pointwise/exploratory.
No numerical practical-equivalence margin has been agreed: a confidence interval
crossing zero is **inconclusive**, not evidence that further training is useless.
Neither a plateau against this panel nor non-significance establishes Nash
convergence. Report all five horizons, without best-checkpoint selection.
There is **no automatic extension to 96 hours** or automatic convergence verdict.

## Run on GCP (after committing and pushing)

From this repository root, with `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL`
already set and the workflow commit available remotely:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp19-feat72-$(date -u '+%Y%m%d-%H%M%S')"

# Local specification generation only; no cloud calls or paid jobs.
bash gcp/run_exp19_hand_board_72h.sh dry-run

# Source metadata check only; no Batch job.
bash gcp/run_exp19_hand_board_72h.sh preflight

# Optional separately submitted cloud smoke. Wait for SUCCEEDED before run.
bash gcp/run_exp19_hand_board_72h.sh smoke-cloud
bash gcp/run_exp19_hand_board_72h.sh status

# Paid workflow: smoke -> three continuation workers -> aggregate -> evaluation.
# Reuses a successful smoke with the SAME RUN_ID; otherwise runs it automatically.
bash gcp/run_exp19_hand_board_72h.sh run
```

After controller submission the laptop can disconnect. The controller finishes
only after training, aggregation and evaluation succeed. Standard provisioning
and the existing memory/resource/failure logging are retained. Training tasks
have a 72-hour **elapsed safety ceiling**, not 72 additional active hours;
evaluation has its own default 48-hour ceiling (`EVAL_MAX_HOURS`, allowed 1–72).
These limits are not forecasts of duration.

`resume` restarts the controller and reuses successful stages; it stops on a
failed child and does not automatically restart interrupted training. A failed
training seed without its final full state needs an explicit recovery decision
from the original 48h state in a new output run. If only evaluation fails, retain
the original `RUN_ID` and `REPO_REF` and run:

```bash
bash gcp/run_exp19_hand_board_72h.sh evaluate-resume
```

This never invokes training. Completed task results are reused only if policy
hashes, budgets, evaluation implementation and dependency versions match.

## Outputs and download

`workers/` contains the imported and new policies, continuation lineage, final
72h training states and runtime metadata. `analysis/` contains training aggregates;
`evaluation/analysis/` contains scores, contrasts, charts, report and task cache.
Both areas require `SUCCESS.json`; smoke outputs are diagnostics, not results.
`experiment19_contract.json` records source-state checksums, sizes and object
generations, learner runtime and frozen-panel identities. Resource logs are
retained in the existing `task_diagnostics/`, smoke and evaluation diagnostics paths.

Download small result summaries without the large training states or task cache:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/evaluation/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
```
