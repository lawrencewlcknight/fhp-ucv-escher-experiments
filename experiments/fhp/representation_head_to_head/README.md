# Experiments 42 and 43: the value of the rich representation

Two evaluation-only studies reuse the fixed 48-hour FHP policies. There is no new training.

| Experiment | Algorithm | Rich policy cohort (Exp8) | Original cohort (Exp6) |
|---|---|---|---|
| 42 | SD-CFR | `sd8-rich48-20261101-234234` | `sdcfr6-48h-20261002-161544` |
| 43 | VR-DeepPDCFR+ | `vr8-rich48-20261101-234215` | `vr6-ray48-20261003-221425` |

The question is whether adopting the rich input representation improves head-to-head performance against the original version under the algorithm's native 48-hour active-training budget. These are within-algorithm comparisons, independent of the three-way comparison in Experiment 41. Positive winnings always favour rich inputs.

## Prespecified protocol

- Three training seeds per cohort: 0, 1 and 2; all nine cross-seed pairings.
- Each pairing receives 100,000 seat-swapped duplicate deal pairs: 200,000 hands.
- Each experiment totals 900,000 duplicate pairs / **1.8 million hands**. Both together: 3.6 million hands.
- Twenty resumable shards of 5,000 duplicate pairs per cell: 180 production shards per experiment.
- Fresh deterministic RNG base seeds 642048 (SD) and 643048 (VR), with separate smoke, profiling and production streams. Duplicate deals use the existing split deal/action stream layout.
- Automatic smoke (2 pairs per cell) and timing profile (128 per cell) precede production. They do not enter reported strength estimates or select a policy. Timing forecasts are informational, not a performance stopping rule.
- SD-CFR uses **every historical strategy** in each native 48-hour archive, uniformly sampled once per player per hand. No last-network substitution, averaged-network approximation or archive thinning.
- VR-Deep uses the native saved 48-hour average-policy network. Neither algorithm is refitted or tuned.

The native Experiment 41 loader revisions support both original and rich encoders; both sides use those same evaluation implementations. Training revisions remain separately recorded. Original player inputs contain 183 features; rich inputs contain 231. Source metadata bytes, configurations, checkpoint boundaries and checkpoint payloads are verified. All SD archive chunks are checksum-verified, with complete iteration coverage required. Metadata validation runs before the cloud VM downloads the policy archives.

`source_pins.json` freezes all twelve source workers. `training_cost_context.json` records their exact encoder, node count, iteration count, clock and runtime information. Within each algorithm the manifests differ in training configuration only by encoder selection. Python versions differ (3.11.17 original versus 3.11.18 rich); VR also records different Linux/platform versions. Training code revisions differ. Consequently this is a retrospective fixed-budget comparison, not a perfectly isolated causal rerun. Rich features can change training throughput; node counts describe that consequence and do not normalise winnings.

## Analysis

The primary outcome is the equally weighted mean over the 3×3 matrix, in mbb/hand. Reports include all nine cell means, seat-specific outcomes, per-cell deal uncertainty and node counts.

Training-seed sensitivity uses 10,000 independent row/column bootstrap draws. Reports provide 95% pointwise intervals and 97.5% intervals for the prespecified family of these two comparisons (Bonferroni adjustment). The nine cells share six trained policies: they are **not nine independent training replications**. Equal seed numbers are not treated as paired training randomness. Bootstrap intervals with only three seeds per cohort remain exploratory.

Conditional deal Monte Carlo uncertainty is reported separately, using duplicate-pair dispersion and independent cell streams. It includes sampled policy/action randomness conditional on the trained policies. It is not combined with the seed intervals into a joint confidence claim. Opponent-specific head-to-head improvement does not establish reduced exploitability or universal strength.

Incomplete, duplicated or corrupted production shards cannot produce a final aggregate. Resume requires identical policies, implementation, dependency versions, protocol and streams; worker count may change. Completed shards are reused. A successful smoke is explicitly marked as having no strength evidence.

## Cloud execution

Run from the UCV evaluation repository. Each experiment is independently launchable and uses one `n2-standard-16` VM with a 200 GB disk, 16 single-thread evaluation workers, periodic uploads, and a 24-hour Batch ceiling. The ceiling is a safety limit, not a duration forecast. If needed, resume with the same output namespace and pinned commit. The evaluator emits a timing forecast after profiling.

After committing and pushing the implementation, set:

```bash
export PROJECT_ID=clever-overview-399515
export REGION=europe-west1
export BUCKET=gs://clever-overview-399515-fhp-escher-results
export SA_EMAIL='<existing Batch runner service account>'
export REPO_REF='<full pushed commit containing Experiments 42 and 43>'
```

For SD-CFR:

```bash
export RUN_ID="exp42-sd-repr-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp42_sd_representation_48h.sh dry-run
bash gcp/run_exp42_sd_representation_48h.sh run
```

For VR-Deep:

```bash
export RUN_ID="exp43-vr-repr-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp43_vr_representation_48h.sh dry-run
bash gcp/run_exp43_vr_representation_48h.sh run
```

Supported actions: `dry-run`, `preflight`, `smoke-cloud`, `run`, `resume`, `status`. `run` already includes smoke before full scoring. A separate `smoke-cloud` invocation needs a distinct RUN_ID from production. Preflight checks explicit source-bucket read grants and output-bucket read/write grants for the VM runner, plus metadata existence; it does not change IAM. It cannot resolve inherited/custom/conditional IAM grants or organisational denies. Only the selected algorithm's source bucket is needed. `SD_BUCKET` and `VR_BUCKET` can override the documented source bucket roots.

Launchers export the builder from `REPO_REF`, reject an existing output prefix for a fresh launch, and reject concurrently active jobs using the same namespace. A Batch `SUCCEEDED` state alone is insufficient: production completion requires `analysis/SUCCESS.json` with `smoke=false` and 900,000 duplicate pairs.

## Download analysis only

With `RUN_ID` set to the completed evaluation:

```bash
python3 gcp/download_representation_analysis.py \
  --bucket "$BUCKET" --run-id "$RUN_ID" \
  --output "cloud_outputs/$RUN_ID/analysis"
```

The helper requires a complete production receipt, downloads only the ten named analysis artifacts plus the receipt, and verifies their checksums. It does not download model objects, historical archives, training states or match shards. Outputs include `analysis_summary.md`, the 3×3 heatmap, matchup/summary CSVs, source index, timing forecast and training context. Cloud workers download only the final playable policies and metadata; full SD histories are necessary for faithful evaluation and remain on the VM.

## Local execution

Arrange downloaded sources under `INPUT/{original,rich}/{sd,vr}/workers/<native worker name>/`. Only the chosen algorithm is required. Each worker needs pinned metadata plus its final playable policy (all SD archive chunks or the VR 48h snapshot).

```bash
python3 -m experiments.fhp.exp42_fhp_sd_representation_48h.run \
  --source-root INPUT --vr-repo /path/to/fhp-vr-deep-experiments \
  --sd-repo /path/to/fhp-deep-cfr-experiments --output OUTPUT --workers 16
```

Use `experiments.fhp.exp43_fhp_vr_representation_48h.run` for VR. Source/output directories must be disjoint. Use native repository revisions pinned in `engine.py` and the same environment for resume. `--smoke-only` verifies policies and gameplay without producing a strength result.
