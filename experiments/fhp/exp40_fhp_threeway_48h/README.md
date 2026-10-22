# Experiment 40 — selected 48-hour models across three algorithms

Compare the selected completed UCV-Escher, VR-DeepPDCFR+ and SD-CFR implementations.
This is evaluation only: it neither retrains nor changes a deployed policy.

## Frozen sources

| Algorithm | Source run | Endpoint | Deployment |
|---|---|---|---|
| UCV-Escher, Experiment 38 | `exp38-rich48-20261019-230016` | 48 active hours, seeds 0/1/2 | Rich AE features; 2,500 critic updates; saved average-policy network |
| VR-DeepPDCFR+, Experiment 6 | `vr6-ray48-20261003-221425` | 48 active hours, seeds 0/1/2 | Eight Ray traversal workers; saved average-policy network |
| SD-CFR, Experiment 6 | `sdcfr6-48h-20261002-161544` | 48 active hours, seeds 0/1/2 | Complete uniform historical strategy archive |

The UCV continuation is checked against
`exp36-rich2500-20261018-205125`: original configuration, runtime, seed,
restart-state checksum, imported checkpoint history and additional progress.
Only metadata and playable final policies are downloaded; large UCV/VR restart
states and replay reservoirs are unnecessary. SD's complete archive is required.

Native SD/VR source validation, duplicate play, reporting and the five corrected
rule agents reuse the previously run three-way evaluator. The evaluator is pinned
to VR repository commit `236bf0c5e94ae90ce57545d82bbabc8bec9b7f07`, including a
file checksum, and SD's loader/evaluation-suite copy to
`bb689252d6b322adb3de6102d095a49c3ed87250`. The current UCV commit supplies its
native rich-feature loader and the Experiment 40 wrapper. The previous experiment
and both other repositories are left unchanged.

SD samples its historical network once per hand and retains it throughout that
hand, with independent, controlled model-mixture randomness. Both seat-swapped
hands receive their own draw. This implements the full uniform historical
strategy in ordinary play; it does not average network outputs, select only the
last network, thin the archive, or sample a different model at every decision.

## Prespecified evaluation

- Three direct comparisons: SD minus UCV, SD minus VR, UCV minus VR.
- Each comparison includes all nine cross-seed cells, each with 100,000 duplicate
  deal pairs (200,000 hands), split into 5,000-pair shards.
- Each of the nine policies plays the five established rule agents for 10,000
  duplicate pairs per agent, also in 5,000-pair shards.
- Production total: **630 shards, 3,150,000 duplicate pairs, 6,300,000 hands**.
- Fresh deterministic streams; none of the previous league's scores are reused.
  Corresponding cells share chance/action streams across algorithm comparisons.
- Smoke: every one of the 72 matchup cells, two duplicate pairs each.
- Informational profiling: the same 72 cells, 128 pairs each. Smoke/profile deals
  are independent of production deals and excluded from performance estimates.

LBR, specialised best responses and the full strategic-position audit are not
included. In particular, exact SD historical-mixture probability queries would
have a very different cost from native episode-sampled match play.

## Statistical analysis and interpretation

Report per-seat and per-seed-cell results in mbb/hand, the three complete 3x3
head-to-head matrices, rule-agent results and native training-cost context.
Positive values favour the left-hand algorithm, as indicated in every summary.

The direct comparison uses independent row/column resampling of training seeds,
10,000 bootstrap draws, pointwise 95% intervals and Bonferroni-compatible
98.33% intervals for the three co-primary contrasts. Seed number 0 in one
algorithm is **not** treated as a paired trajectory with seed 0 in another.
Conditional Monte Carlo SEs/95% intervals describe deal noise separately;
the two interval types are not a combined confidence region. With only three
training seeds per algorithm, population inference remains exploratory.

These are selected implementations at nominal 48-active-hour endpoints, not a
controlled architecture ablation, equal-total-compute experiment, exploitability
measurement or proof of convergence. Input representations, capacity, tuning
effort and active-clock definitions differ. Fresh match deals do not undo
historical selection of the candidates. Native source summaries, node counts,
policy storage, scoring time, throughput and VM resource diagnostics provide
context without normalising away these differences. Do not force a ranking when
results are inconclusive or cyclic; no automatic winner is promoted.

## One-command GCP run

Commit and push Experiment 40 first. From the **UCV-Escher repository root**:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp40-threeway48-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp40_threeway_48h.sh run
```

The runner service account must be able to read the three source buckets and
write the output bucket. The launcher checks source access for the submitting
user; that does not prove access for the VM service account. No IAM policy is
changed automatically. Optional source bucket overrides are `UCV_BUCKET`,
`VR_BUCKET` and `SD_BUCKET`; the fixed source run IDs cannot be substituted.

One `n2-standard-16` VM (16 vCPUs, 64 GiB, 200 GB disk) validates inputs, runs
the smoke check, profiles and scores automatically. Sixteen evaluation processes
each use one computation thread. The laptop may disconnect after submission.
There is no manual promotion between stages and no automatic paid retry.

The overall Batch ceiling is 24 hours, including setup and final upload. Scoring
stops before that ceiling to retain results and leave upload time. Roughly 3–6
hours is an initial planning estimate based on the previous league, not a measured
guarantee for the richer UCV encoder. Timing forecasts are informational: they
cannot reject the run just because they predict slow scoring. Genuine source,
numerical or contract failures still stop rather than produce misleading results.

Optional standalone cloud smoke (use a separate run ID):

```bash
export RUN_ID="exp40-threeway-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp40_threeway_48h.sh smoke-cloud
```

`dry-run` builds the job without cloud submission; `status` lists matching jobs.
The standard `run` command already includes smoke, so a separate smoke job is
not required. For a standalone smoke, `SUCCESS.json` explicitly has `smoke=true`
and is not performance evidence.

## Recovery and output

Completed shards are written atomically, checksummed and uploaded every five
minutes, then again on exit and by an always-run finaliser. Uploads are locked;
completion markers are published only after the corresponding files. Resource
snapshots and Cloud Logging help distinguish memory pressure, disk usage,
timeouts and ordinary Python exceptions.

To resume an interrupted **full** run, retain its original `RUN_ID`, `REPO_REF`
and buckets, then:

```bash
bash gcp/run_exp40_threeway_48h.sh resume
```

This submits a new job name but restores the same output namespace. Valid cached
shards are reused; changed sources, executable code, dependencies or random
streams are rejected. At worst, unfinished shards and results since the last
successful upload must be repeated. If interrupted before a manifest exists,
use a fresh run ID; no scoring can be recovered from that prefix. To resume an
optional standalone smoke, also set `EXP40_SMOKE_ONLY=1`; do not promote its
namespace into a full run. Unset that variable before creating a full run.

Completion requires `analysis/SUCCESS.json` with `status=complete`, `smoke=false`,
630 tasks and 3,150,000 duplicate pairs. The expected analysis files include:

- `analysis_summary.md`, `summary.json`, `direct_comparison_summary.csv`;
- `matchups.csv`, `rule_agent_summary.csv`, `final_48h_head_to_head.png`;
- `checkpoint_index.json`, `checkpoint_index.csv`, `training_cost_context.json`;
- `evaluation_manifest.json`, `timing_forecast.json`, `STATUS.json`;
- resumable `task_results/`, plus independent smoke/profile results.

Resource diagnostics and pinned dependency inventory are stored at the run root.
Download the analytical results without the large shard cache:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
gcloud storage cp "$BUCKET/$RUN_ID/batch_diagnostics.json" \
  "$BUCKET/$RUN_ID/resource_snapshots.jsonl" "cloud_outputs/$RUN_ID/"
```

Keep the cache in GCS for recovery. Existing training checkpoints are read-only
throughout this evaluation.
