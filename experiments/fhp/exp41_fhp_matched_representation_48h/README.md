# Experiment 41: matched-representation three-way FHP evaluation

Evaluate the fixed 48-hour policies with the same rich AE player inputs. This
experiment performs no training, tuning or checkpoint selection. It lives in the
UCV repository and leaves Experiment 40 and both baseline repositories unchanged.

## Frozen sources

| Algorithm | Run | Loader/training revision | Deployment |
|---|---|---|---|
| UCV-Escher Exp38 | `exp38-rich48-20261019-230016` | Original training `6ab85bbcd03627281e411243419d64ce9e9236d5`; UCV loader from the pinned evaluation commit | Saved average-policy network |
| SD-CFR Exp8 | `sd8-rich48-20261101-234234` | `ac56342ae92824e06b206dd909a99c897529e373` | Full uniform historical strategy mixture |
| VR-DeepPDCFR+ Exp8 | `vr8-rich48-20261101-234215` | `70abec44561c5e141edf870512ec4237cf9fe4ad` | Saved average-policy network |

All use seeds 0/1/2. `source_pins.json` freezes the downloaded SD/VR worker
metadata byte hashes and training-configuration hashes. Validation requires the
full eight-checkpoint schedule, the final 48-hour endpoint and identical AE
encoder metadata (231 player features; 359 full-state features for critics).
SD/VR must be fresh training runs. UCV's original Exp36-to-Exp38 continuation,
imported checkpoint history and additional training are validated as in Exp40.

Native payload checks bind the policies to their source metadata. Every SD
archive chunk is checked for its hash, dimensions, precision and contiguous
coverage from iteration 1 through the final iteration. All historical policies
are retained. SD samples a network once per hand and keeps it for that hand;
each seat-swapped hand has its own controlled mixture draw. There is no archive
thinning, distillation, final-network substitution or averaging of network outputs.

## Evaluation contract

- Three comparisons: SD minus UCV, SD minus VR, and UCV minus VR.
- Nine cross-seed cells per comparison: all combinations of seeds 0/1/2.
- **100,000 duplicate deal pairs per cell**, with each deal played in both seats.
- **27 cells, 2,700,000 duplicate pairs, 5,400,000 hands** in production.
- Each cell has twenty independently seeded 5,000-pair shards: **540 shards**.
- Fresh deterministic streams, distinct from Exp40 and the earlier league.
  Corresponding cells share random streams across algorithm comparisons.
- Smoke: all 27 cells, two pairs each. Profile: all 27 cells, 128 pairs each.
  These use separate streams and do not contribute to production estimates.
- No rule-agent panel, LBR, strategic-position audit or automatic winner selection.

Results include per-seat values, all 27 cell means, three 3x3 matrices and three
algorithm-pair summaries in mbb/hand. Positive values favour the left algorithm.
Independent row/column bootstrap resampling uses 10,000 draws, pointwise 95%
and nominal Bonferroni-compatible 98.33% intervals for the three co-primary
comparisons. Conditional Monte Carlo SEs/95% intervals describe deal noise
separately. Nine cross-seed cells are not nine independent training replications;
only three training seeds per algorithm are available. These uncertainty
estimates remain exploratory and are not combined into one confidence region.

If UCV retains an advantage, exclusive access to AE features cannot explain that
advantage. This does not isolate architecture, optimisation, critic budget or
tuning history. UCV remains a selected historical reference, not a fresh training
replication. The native active-time clocks differ; this is a nominal 48-hour
comparison, not an equal-total-cost claim. Head-to-head value is not exploitability.

## Run on GCP

Commit and push the new implementation before launching. From the UCV repo:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp41-matched48-$(date -u +%Y%m%d-%H%M%S)"

bash gcp/run_exp41_matched_representation_48h.sh dry-run
bash gcp/run_exp41_matched_representation_48h.sh run
```

The default action is `dry-run`. `run` submits one `n2-standard-16` VM with
16 single-threaded scoring processes, 200 GB disk and a 24-hour safety ceiling.
It validates sources, runs smoke and profiling, then scores automatically.
The timing forecast is informational, not an automatic rejection or retry.
No cloud job is started by tests or by building a job JSON.

The launcher first checks runner bucket grants and the existence of source
metadata without allocating a VM. `preflight` runs only those checks. It requires
direct unconditional supported storage roles and reports that limitation; it
does not modify IAM. Optional `UCV_BUCKET`, `SD_BUCKET` and `VR_BUCKET` overrides
change bucket locations, never the frozen run IDs.

The cloud worker downloads metadata, the three UCV final policies, three VR
final policies, and SD's final archive manifests plus their historical chunks.
It does **not** download restart states or replay buffers. SD historical chunks
are essential model inputs for this evaluation and stay on the cloud VM; they
do not need to be downloaded to the laptop. Outputs contain analysis data only.

## Recovery and completion

Atomic, checksummed shard outputs are uploaded every five minutes and at exit.
A resumed run reuses completed compatible shards and rejects changed code,
policies, dependencies or deal streams. Keep `RUN_ID`, `REPO_REF` and buckets:

```bash
bash gcp/run_exp41_matched_representation_48h.sh resume
bash gcp/run_exp41_matched_representation_48h.sh status
```

Resume requires an existing evaluation manifest. A failure before that point
has no scoring to recover; use a fresh run ID. A standalone `smoke-cloud` action
must use a separate run ID; its completion marker is explicitly `smoke=true`.
Use `EXP41_SMOKE_ONLY=1` only when resuming that smoke namespace.

Full completion requires `analysis/SUCCESS.json` with `smoke=false`, `tasks=540`
and `duplicate_pairs=2700000`. Missing, extra, repeated, corrupt or incomplete
production shards cannot be aggregated. The primary outputs are:

- `summary.json`, `analysis_summary.md`, `direct_comparison_summary.csv`;
- `matchups.csv`, `final_48h_head_to_head.png`;
- `checkpoint_index.json`, `training_cost_context.json`;
- `evaluation_manifest.json`, `timing_forecast.json`, `STATUS.json`;
- resumable `task_results/` and separate smoke/profile results.

Download the small analysis outputs, leaving the shard cache in GCS:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='(^|/)task_results/' \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Local verification

The sibling SD and VR repositories must be present at the workspace paths used
by existing integration tests. Tests use tiny synthetic rich-AE policies and
actual OpenSpiel games; they do not download production archives or launch jobs.

```bash
python3 -m pytest -q tests/test_exp41_matched_representation.py \
  tests/test_exp40_threeway_48h.py
```

The integration test checks rich-AE loading for all algorithms, full SD history,
fixed per-hand mixture selection, rejection of missing history, reproducible
spawned scoring and cache reuse. Protocol tests cover all match cells, exact deal
budgets, fresh streams, source identity, uncertainty reporting, interrupted
execution, incompatible resumes and generated shell scripts.
