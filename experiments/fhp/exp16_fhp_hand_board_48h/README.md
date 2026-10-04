# Experiment 16: continue Experiment 10 to 48 active hours

Continue the **existing** Experiment 10 seeds 0, 1 and 2 from their completed
24-hour full states to a cumulative 48-active-hour budget. This is not a fresh
48-hour run, a policy-only warm start, or an additional 48 hours. It changes
only the stopping horizon. It does not adopt Experiment 15's policy learning
rate or combine any other interventions.

## Fixed learning and execution contract

| Item | Specification |
|---|---|
| Source run | `exp10-features-20261001-161740` |
| Original learner commit | `e66d4da515eb212e5026a965ac5c39c86144c901` |
| Learning configuration | Experiment 10, unchanged; additive hand/board features and cached frozen critic targets |
| Seeds | 0, 1, 2; their existing learner and actor trajectories |
| Machine | One `n2-standard-16` per seed, Standard provisioning |
| Parallel collection / fitting | Eight Ray actors; eight learner threads, unchanged |
| Python / NumPy / Torch / Ray | 3.11.16 / 1.26.4 / 2.7.0+cpu / 2.51.2 |
| Final budget | 48 cumulative active training hours per seed |
| Playable checkpoint thresholds | 6, 12, 18, 24, 30, 36, 42, 48 hours |
| Full training states among new outputs | Final 48h only, one per seed |

Thresholds are serviced at completed outer iterations, so actual elapsed
training time can slightly exceed each threshold. The imported 6–24h policies
are preserved byte-for-byte. Only approximately 24 additional active hours are
required, less the source's small 24h overshoot. Fitting the output policy,
evaluation, transfers and checkpoint serialization are excluded from the active
clock. A three-seed concurrent training stage requires **48 N2 vCPUs**, not
three vCPUs; these are three 16-vCPU VMs. `PARALLELISM=1` or `2` reduces concurrent
resource use without changing any seed's machine or scientific configuration.

The complete source state restores networks, optimizers, replay, counters,
training RNGs and Ray actor states using Experiment 10's native continuation
implementation. No learner source is modified. The original state format and
native worker metadata retain `experiment_id=10`; the new run prefix, Batch
labels, `experiment16_contract.json` and evaluator identify Experiment 16.
This deliberately avoids masquerading a new implementation as the original
learner or disabling its exact-commit resume guards. `REPO_REF` pins the new
orchestration/evaluator, whereas the learner commit above is fixed internally.

## Preflight and storage

The source commit, configuration digest, Python/package versions, VM and thread
settings, all three completed source schedules, summary checksums, source state
object sizes, and fixed external opponent availability are checked before job
submission. This reads small metadata, not the approximately 5.4 GB per-seed
training states. Each training VM then downloads and SHA-256 verifies its full
state and playable policies before restoration. Existence/metadata checks alone
do not certify the state contents; failure to restore stops the task.

The source run is never overwritten. Imported source states are private temporary
inputs, excluded from uploads and removed after use. New output retains analysis,
resource/runtime diagnostics, metadata, every playable policy, and exactly one
final 48h resumable state per seed. No 30/36/42h full states are saved. Existing
Experiment 10 24h source states remain in their original run. Do not delete them
before continuation has completed successfully.

## Prespecified evaluation

No policy is refitted during evaluation. The saved policies are replayed in the
canonical OpenSpiel FHP game using the existing evaluation suite:

- Score 24, 30, 36, 42 and 48h continued policies against the five established
  rule agents: 10,000 duplicated deal pairs per opponent and policy.
- Compare all ten later-versus-earlier pairs within each continuing seed:
  50,000 duplicated deal pairs per contrast. Primary temporal contrast: 48h
  versus 24h; late-stage contrasts include 48h versus 36h and 42h.
- Fixed learned opponents: Experiment 9's 24h policies from
  `exp9-cache24-20261001-132550` and Experiment 8's 48h policies from
  `exp8-par48-20261001-005208`. Each candidate seed plays the same-label saved
  opponent at every horizon, 50,000 deal pairs per matchup. This is a modest
  fixed historical panel, not an all-cross-seed league.
- Retain the established LBR diagnostic: 1,000 deal pairs per policy, 4,096
  rollouts, ten-deal shards. It is approximate and restricted, not exact
  exploitability. The fixed learned opponents also receive rule/LBR scoring.
- Use common evaluation deals across horizons against each frozen external
  opponent. Report within-seed changes in external payoff, especially 48h minus
  24h and 36h, separately from temporal direct play.

In total: 21 scored policies, 2,265 tasks and 4,071,000 duplicated deal pairs
(two seat-swapped hands per pair). A small integration smoke is run before the
full evaluator. Training-seed means, not hands/checkpoints/LBR shards, are the
three inferential units. Reports retain per-match Monte Carlo uncertainty and
pointwise seed-level 95% intervals; secondary contrasts are exploratory.
Three seeds provide limited power. Matching seed labels across historical
experiments does not create paired training interventions.

Further 72–96h training requires a separate decision. Look for sustained,
seed-consistent external gains alongside temporal performance and LBR, not
just wins against predecessors. Neither inconclusive differences nor a
restricted-opponent plateau proves equilibrium convergence.

## GCP workflow (after committing and pushing)

Use the same project, region, bucket and runner service account as Experiment 10.
The launcher never grants IAM roles or alters source runs.

```bash
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp16-feat48-$(date -u '+%Y%m%d-%H%M%S')"

# Inspect specifications without cloud calls:
bash gcp/run_exp16_hand_board_48h.sh dry-run

# Metadata-only validation; no training or Batch submission:
bash gcp/run_exp16_hand_board_48h.sh preflight

# Explicitly submit paid work:
bash gcp/run_exp16_hand_board_48h.sh run
bash gcp/run_exp16_hand_board_48h.sh status
```

Also set `PROJECT_ID`, `REGION` and `SA_EMAIL` as for existing experiments.
The service account needs the established controller permissions (Batch job
management and permission to run child jobs as itself), bucket access and
enough VM quota. Workflow order:

**cloud continuation/capacity smoke → three continuation workers → aggregation
→ frozen-policy smoke and full evaluation**.

The laptop can be closed after controller submission. The controller does not
report success until evaluation succeeds. `EVAL_MAX_HOURS` defaults to a 48-hour
evaluation wall ceiling; this is a safety limit, not a duration forecast.
Training tasks have a 72-hour wall ceiling for approximately 24 additional
active hours plus checkpoint overhead. No stage automatically retries paid work.

`resume` restarts only the controller and reuses successful child stages. It
stops on a failed child; it never silently restarts interrupted training. With
final-state-only retention, an interrupted seed cannot continue from an
intermediate playable policy. Any recovery from the original 24h state needs
an explicit new-run decision. If only evaluation fails, keep the same `RUN_ID`
and `REPO_REF` and use:

```bash
bash gcp/run_exp16_hand_board_48h.sh evaluate-resume
```

This submits only an evaluation job, reuses checksum-validated task results,
and never invokes training. Changed evaluation code, policy hashes, budgets
or dependencies invalidate the evaluation cache.

## Outputs

- `workers/`: native Experiment 10 checkpoint histories continued to 48h,
  `continuation_source.json`, final full states, summaries and runtime metadata.
- `experiment16_contract.json`: workflow identity, source commit/runtime,
  source state checksums/sizes and external-panel run IDs.
- `analysis/`: native training aggregates, node/time trajectories, throughput,
  cache diagnostics, feature specification and continuation lineage.
- `evaluation/analysis/`: checkpoint index, rule/LBR summaries, temporal
  crossplay, fixed-panel crossplay, external-improvement contrasts, six charts,
  a written report and a checksum-guarded task cache.
- `smoke/`, `task_diagnostics/`, `evaluation/smoke/`: diagnostics only; never
  mix smoke data with scientific results.

Small analysis-only download, excluding caches and all large training states:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/evaluation/analysis" "cloud_outputs/$RUN_ID/evaluation/analysis"
```
