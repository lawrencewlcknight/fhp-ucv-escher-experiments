# Experiment 1: grouped-wide UCV-ESCHER FHP baseline

This is the active FHP baseline. It transfers the selected training algorithm
from Leduc ESCHER architecture Experiment 35 while retaining the canonical
OpenSpiel FHP game.

Three independent seeds (`0`, `1`, and `2`) train concurrently on separate GCP
Batch VMs. Each receives 24 effective training hours and saves a reloadable
policy plus a full continuation state at the first completed outer iteration
after 6, 12, 18, and 24 hours. Checkpoint fitting, serialization, and upload
are excluded from the effective-training clock.

Playable policy checkpoints are uploaded before the larger continuation
state is created. Continuation replay arrays are streamed to independent
NumPy shards and committed by a final integrity manifest, avoiding the
transient memory amplification of a multi-gigabyte `torch.save`. Only the
latest continuation state is retained; all four lightweight playable policy
checkpoints remain available for analysis.

The transferred candidate uses grouped soft-target cross-entropy, a `3 x 136`
average-policy network, learning rate `0.003`, 20,000 updates per fit, reset
fitting, a two-fold critic, a four-fit averaged critic target, fixed
control-variate beta `1.0`, no instantaneous predictor, and residual
calibration. Exact Leduc tree diagnostics and the paired legacy fit are not
part of the FHP training algorithm.

To fit this raw-input baseline on the same `n2-standard-8` VM class as the two
structured experiments, all continuous replay fields use NumPy `float32`, the
same precision consumed by the networks. Integer fields remain full-width and
the legacy global NumPy/Python RNG calls, Algorithm-R replacement, minibatch
selection, raw features, models, and optimiser settings are unchanged. The
storage contract is recorded in every run manifest.

## Local smoke

```bash
./gcp/run_exp1_grouped_wide.sh smoke-local
```

The ordinary local smoke uses reduced buffers. A production-capacity
serialization stress test is also available, but requires approximately the
same memory and temporary disk space as a production worker:

```bash
./gcp/run_exp1_grouped_wide.sh checkpoint-smoke-local
```

## GCP Batch

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://YOUR_RESULTS_BUCKET"
export SA_EMAIL="YOUR_BATCH_SERVICE_ACCOUNT"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp1-fhp-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_exp1_grouped_wide.sh run
```

The remote controller first runs both the functional smoke and a
production-capacity continuation-state serialization stress test. Synthetic
stress data are deleted before upload. The controller submits the three-worker
training array only after both checks succeed, then validates and uploads
aggregate metadata. The laptop may disconnect after controller submission.

```bash
./gcp/run_exp1_grouped_wide.sh status
./gcp/run_exp1_grouped_wide.sh resume
```

Each production worker is an on-demand `n2-standard-8` VM with one automatic
retry, a 200 GiB `pd-balanced` boot disk, and a 36-hour hard Batch ceiling. A
retry restores the latest durable six-hour continuation state from Cloud
Storage.

## Evaluation

Evaluate a downloaded seed worker separately from training:

```bash
python -m experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.evaluate_checkpoints \
  --source-run cloud_outputs/RUN_ID/workers/TASK_DIRECTORY
```

The evaluator verifies all four hashes, benchmarks each checkpoint against the
shared five-agent suite and LBR, and performs all-pairs checkpoint cross-play.
LBR remains a lower bound; it is not exact exploitability.
