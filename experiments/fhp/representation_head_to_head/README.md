# Shared evaluator: rich versus original representation

The algorithm-owned Experiment 9 studies in the SD-CFR and VR-Deep repositories reuse the fixed 48-hour FHP policies. There is no new training.

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

## Experiment ownership and launching

The experiment definitions, launchers and launch documentation have moved:

| Owner repository | Native experiment | Definition | Legacy evaluation ID |
|---|---|---|---|
| `fhp-deep-cfr-experiments` | SD-CFR Experiment 9 | `experiments/fhp/exp9_sd_cfr_representation_48h/README.md` | 42 |
| `fhp-vr-deep-experiments` | VR-Deep Experiment 9 | `experiments/fhp/exp9_vr_deep_representation_48h/README.md` | 43 |

Launch from the respective algorithm repository. Its `REPO_REF` pins the owner definition; its configuration separately pins this shared evaluator to UCV commit `b6eabbeeeca6d8ae8f07f383a5296492c2cd2376`. Existing evaluation identifiers, scoring streams and resume contracts are retained. Previously launched jobs and results remain valid.

This directory and the generic `gcp/representation_*` helpers remain shared infrastructure. There are no active algorithm-specific launchers in this repository. Use the owner's `gcp/download_exp9_representation_analysis.sh` for analysis-only downloads into that repository.
