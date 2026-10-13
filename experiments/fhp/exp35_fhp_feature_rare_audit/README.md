# Experiment 35 — rare-decision audit of strategic features

Read-only comparison of **Experiment 10 and all five Experiments 29–33**, at
24 active training hours, seeds 0/1/2: 18 deployed-policy/current-regret pairs.
No new training, refitting, traversal, LBR or head-to-head games. Existing
whole-game evaluations remain the complementary evidence, not repeated here.

## Scientific contract

- Pin the original source runs, worker names, policy/state SHA-256 hashes,
  iterations and node counts in `inputs.json`, copied from the downloaded
  24h checkpoint indexes. Approximately **106.5 GB** of training states in total.
- Match source state and saved deployed policy exactly, including feature
  metadata, learning configuration and model weights. Instantiate only native
  inference models; do not construct a learner, optimizer, replay or Ray actors.
  Loading the original state necessarily reads its stored replay, but it is not
  used for learning and is released before inference.
- Evaluate the actual checkpoint encoder for each model, the deployed policy's
  masked softmax, and each seat's current regret network with the native regret
  matching rule (including argmax fallback). No current strategy is substituted
  for the historical average teacher.
- Freeze a shared panel before querying any models. The known straight-flush
  and A/5 side-card quads examples are **development**, with all legal capped
  terminal call/fold histories and both acting seats.
- New **assessment**: eight straight-flush-supporting monotone board classes
  and eight paired board classes where a pocket pair can complete quads, with
  all compatible category-7/8 own hands and four ordinary control hands per board.
  Exact suit-equivalent information states are deduplicated. All recorded boards
  from the strategic library and Experiment 28, including reserved boards, are
  excluded. All 13 trips ranks were already inspected, hence paired-board quads
  provide fresh situations. This also changes the quads context: compare methods
  within each panel, not development/assessment averages as identical populations.
- Fresh means absent from those recorded audits, **not absent from training**.
  Selection seed is fixed at 2026101335 and independent of model predictions.
  Controls are descriptive; uniform-range equity is not an optimal-action label.
- Guarantee a win only when the exact five-card rank beats all 1,081 compatible
  opponent holdings. Validate legal terminal transitions/payoffs in OpenSpiel.
  Ties are not silently counted as guaranteed wins.

Report case-level fold/call probabilities, current regret scores, mean fold
probability, fraction folding above 50%, and mean/p95/maximum conditional loss
in BB on guaranteed winners, separately for development/assessment and
quads/straight flush/all. The loss is fold probability times the terminal
call-minus-fold payoff: **not whole-game loss or exploitability**. Aggregation
uses three training seeds, not thousands of positions as independent replicates.
Candidate-minus-Exp10 intervals are exploratory pointwise t intervals with
two degrees of freedom, not multiplicity-adjusted. Matched seed labels are not
matched training trajectories. All results are retained; no automatic winner.

A sensible current strategy with a poor deployed policy motivates a subsequent
replay/teacher audit; it does not establish policy-compression error by itself.
Poor predictions from both motivate upstream data investigation, not a causal
claim that coverage is the only mechanism. This experiment does not measure
retained replay coverage or implement a repair.

## Single-VM runtime and recovery

One `n2-standard-16` (16 vCPUs, 64 GB RAM), CPU only, 100 GB boot disk. Serial
source loading prevents 18 large states being resident together. Sources live
in a temporary input directory, are checksum-verified before deserialisation,
and are removed after their model result is saved. They are never uploaded as
outputs. Inference uses one CPU thread for these small networks; batching is
used for forward passes. Each model runs in a separate process to release RAM.

**Planning estimate: 45–120 minutes for the full job**, including bootstrap,
~107 GB of transfers, checksum checks and loading. Allow up to three hours if
storage/network throughput is poor. This is not a cloud benchmark. Pure panel
construction/inference is much smaller than training or BR evaluation. The
cloud smoke audits seed 0 for all six encoders (~35.5 GB) on a smaller panel;
allow roughly 15–45 minutes. Its result does not establish scientific quality.

There is a generous **12-hour infrastructure cap**, not a scientific runtime
gate or a short per-model deadline. Progress and resource diagnostics are logged.
The frozen panel and each finished model result are uploaded immediately.
Interrupted jobs can resume from verified per-model receipts with the same code
revision, protocol and smoke/full mode. A negative scientific outcome never
fails the job. Integrity errors still stop it rather than producing misleading
results. Download, loading and inference costs are retained per model.

## Run from repository root

First commit and push the implementation; do not submit a reference that lacks it.
The launcher does not push or launch anything until explicitly invoked.

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
# Keep SA_EMAIL set to your existing authorised Batch runner service account.
export REPO_REF="$(git rev-parse HEAD)"

bash gcp/run_exp35_feature_rare_audit.sh smoke-local
export RUN_ID="exp35-rare-smoke-$(date -u +%Y%m%d-%H%M%S)"
export SMOKE_RUN_ID="$RUN_ID"
bash gcp/run_exp35_feature_rare_audit.sh smoke-cloud
```

After that smoke succeeds, one submission completes the entire full audit:

```bash
export RUN_ID="exp35-rare-full-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp35_feature_rare_audit.sh run
```

`dry-run` renders the Batch JSON without submission. `status` lists matching
jobs. Source preflight checks object presence and size before paid VM work.
For recovery, set `RESUME_RUN_ID` to the interrupted full run, use a **new**
`RUN_ID`, keep the same `REPO_REF` and `SMOKE_RUN_ID`, then invoke `run` again.
Do not set a smoke run as the recovery source of a full run.

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Outputs: `protocol.json`, `panel.json`, `models/expN_seedS/result.json` and
checksummed completion receipts, `summary.json`, `report.md`, `SUCCESS.json`.
System/resource/error diagnostics are uploaded separately under `diagnostics/`.
