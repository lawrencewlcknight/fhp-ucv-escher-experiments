# Running the FHP UCV-ESCHER experiments on Google Cloud Batch

This guide covers the repeatable Google Cloud Batch workflow for the four
flop hold'em poker (FHP) UCV-ESCHER experiments in this repository. Each Batch
job creates a temporary VM, clones the repository, installs an isolated Python
3.11 environment, runs one experiment, uploads the complete `outputs/` tree to
Cloud Storage, and exits. Batch owns the VM lifecycle; there is no persistent VM
to shut down after a completed job.

The end-to-end workflow is:

1. configure and authenticate the Google Cloud CLI;
2. enable the required APIs;
3. create the results bucket and Batch service account;
4. set the four required environment variables in the current terminal;
5. validate the checked-in submission helper;
6. submit the relevant smoke test and confirm it succeeds;
7. submit the matching full experiment;
8. monitor its status and job-scoped logs;
9. download and verify the uploaded outputs;
10. retain diagnostics and clean up completed Batch job records.

The production runs are time-bound. They save reloadable policies at the first
safe trajectory boundary after 6 and 12 effective training hours, then stop.
The 14-hour Batch limit leaves time for provisioning, installation, policy
fitting, checkpoint serialization, diagnostics, and upload.

## 1. Prerequisites

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI installed locally;
- permission to enable APIs and create service accounts, IAM bindings, buckets,
  and Batch jobs;
- a Batch service account that can write logs and Cloud Storage objects;
- access from the Batch VM to this GitHub repository.

The default repository URL is public HTTPS:

```text
https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git
```

For a private fork, set `REPO_URL` to an authenticated clone URL or use a
pre-built image with credentials configured securely.

## 2. One-time Google Cloud setup

Authenticate, select the project, and choose a region:

```bash
gcloud init
gcloud auth login

export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
gcloud config set project "$PROJECT_ID"
```

Enable the required APIs:

```bash
gcloud services enable \
  compute.googleapis.com \
  batch.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com
```

Create a regional results bucket:

```bash
export BUCKET_NAME="${PROJECT_ID}-fhp-escher-results"
export BUCKET="gs://${BUCKET_NAME}"

gcloud storage buckets create "$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

Confirm that the bucket exists and is in the intended region:

```bash
gcloud storage buckets describe "$BUCKET"
```

Create a dedicated Batch service account:

```bash
export SA_NAME="fhp-escher-runner"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="FHP ESCHER experiment runner" \
  --project="$PROJECT_ID"
```

Grant it the required project and bucket permissions:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.agentReporter"

gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

Allow your user account to submit jobs as the service account and inspect logs:

```bash
export YOUR_EMAIL="your-email@example.com"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/logging.viewer"
```

Confirm that the service account exists:

```bash
gcloud iam service-accounts describe "$SA_EMAIL" \
  --project="$PROJECT_ID"
```

## 3. Environment for each terminal

Set these values before submitting or inspecting jobs:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-fhp-escher-results"
export SA_EMAIL="fhp-escher-runner@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
```

Check the active values before every submission. An empty or stale value can
send a job or its outputs to the wrong project, region, bucket, or identity:

```bash
echo "$PROJECT_ID"
echo "$REGION"
echo "$BUCKET"
echo "$SA_EMAIL"
gcloud config get-value project
```

## 4. Submission helper

Run all jobs through `gcp/submit_batch_experiment.sh`. Its positional interface
is:

```text
JOB_NAME EXPERIMENT_COMMAND MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB BOOT_DISK_SIZE_GB BOOT_DISK_TYPE
```

Before first use, check the scripts:

```bash
chmod +x gcp/submit_batch_experiment.sh gcp/read_batch_task_logs.sh
bash -n gcp/submit_batch_experiment.sh
bash -n gcp/read_batch_task_logs.sh
```

`BOOT_DISK_TYPE` defaults to `pd-balanced` for backward-compatible N2 use. Pass
it explicitly in maintained commands. The helper rejects a C4-family machine
paired with any `pd-*` Persistent Disk locally, before generating or submitting
a Batch job.

The submission helper deliberately sets `maxRetryCount` to zero so a failed
training run is not silently repeated. It captures stdout and stderr, records
15-second resource snapshots, classifies failures, and uploads outputs from an
exit trap on both success and failure.

## 5. Experiment allocations

| Experiment | Backend | Full-run VM | CPU request | Memory request | Boot disk |
|---|---|---:|---:|---:|---:|
| 1 | Sequential baseline | `n2-standard-8` | 8,000 milli | 32,000 MiB | 100 GiB `pd-balanced` |
| 2 | Sequential comparison | `c4-standard-32` | 32,000 milli | 120,000 MiB | 200 GiB `hyperdisk-balanced` |
| 3 | Ray parallel comparison | `c4-standard-32` | 32,000 milli | 120,000 MiB | 200 GiB `hyperdisk-balanced` |
| 4 | CPU-optimized Ray parallel | `c4-standard-32` | 32,000 milli | 120,000 MiB | 200 GiB `hyperdisk-balanced` |

All smoke tests use `n2-standard-4`, 4,000 CPU milli, 16,000 MiB of memory, a
100 GiB `pd-balanced` disk, and a two-hour Batch ceiling. `--smoke` retains the
production orchestration and checkpoint/reload path while shrinking training
work and checkpoint thresholds.

## 6. Submit smoke tests

Run the matching smoke job before each full job.

### Experiment 1

```bash
JOB_NAME="exp1-fhp-ucv-baseline-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_ucv_escher_baseline.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

### Experiment 2

```bash
JOB_NAME="exp2-fhp-ucv-sequential-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_ucv_escher_sequential.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

### Experiment 3

```bash
JOB_NAME="exp3-fhp-ucv-parallel-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_ucv_escher_parallel.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

### Experiment 4

```bash
JOB_NAME="exp4-fhp-ucv-cpu-optimized-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_ucv_escher_cpu_optimized.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

## 7. Submit full runs

### Experiment 1

```bash
JOB_NAME="exp1-fhp-ucv-baseline-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_ucv_escher_baseline.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 50400 8000 32000 100 pd-balanced
```

### Experiment 2

```bash
JOB_NAME="exp2-fhp-ucv-sequential-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp2_ucv_escher_sequential.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

### Experiment 3

```bash
JOB_NAME="exp3-fhp-ucv-parallel-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp3_ucv_escher_parallel.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

### Experiment 4

```bash
JOB_NAME="exp4-fhp-ucv-cpu-optimized-full-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp4_ucv_escher_cpu_optimized.run \
    --output-root outputs/cloud/$JOB_NAME" \
  c4-standard-32 50400 32000 120000 200 hyperdisk-balanced
```

## 8. Monitor a job and inspect logs

List jobs or inspect one job:

```bash
gcloud batch jobs list --location "$REGION"
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
```

The normal state progression is `QUEUED` → `SCHEDULED` → `RUNNING` →
`SUCCEEDED`. Treat `FAILED` as requiring investigation before resubmission. A
job that remains in `SCHEDULED` can be blocked by VM availability, quota, IAM,
or an invalid allocation such as an incompatible machine and disk type.

For a concise status view:

```bash
gcloud batch jobs describe "$JOB_NAME" \
  --location "$REGION" \
  --format="yaml(status.state,status.runDuration,status.statusEvents)"
```

Read only the task logs belonging to one Batch job:

```bash
./gcp/read_batch_task_logs.sh "$JOB_NAME"
```

Filter those logs for likely failure evidence:

```bash
./gcp/read_batch_task_logs.sh \
  "$JOB_NAME" ERROR Traceback Killed OOM "out of memory"
```

The helper resolves the job UID before querying Cloud Logging, preventing logs
from similarly named jobs from being mixed together.

## 9. Retrieve outputs

Inspect uploaded objects:

```bash
gcloud storage ls --recursive "$BUCKET/$JOB_NAME/"
```

Download the job's uploaded output tree into the ignored local
`cloud_outputs/` working area:

```bash
mkdir -p "cloud_outputs/$JOB_NAME"
gcloud storage cp -r \
  "$BUCKET/$JOB_NAME/*" \
  "cloud_outputs/$JOB_NAME/"
```

Keep checkpoint files with their manifests. The final reloadable policy is
`final_policy_checkpoint.pkl`; verify its digest against
`summary.json` or `checkpoint_manifest.json` before head-to-head evaluation.

## 10. Diagnose failures

Batch-level diagnostics are written alongside the experiment run directory:

- `batch_run.log` — complete setup, experiment, cleanup, and upload output;
- `resource_snapshots.jsonl` — 15-second process, memory, disk, load, and CPU
  snapshots;
- `batch_diagnostics.json` — detailed final resource and failure evidence;
- `batch_status.json` — concise classification and exit codes.

An experiment exception also creates `failure.json` inside its timestamped run
directory with the traceback, phase, nodes touched, iteration, and peak RSS when
available. The Batch classifier distinguishes confirmed cgroup OOM, allocator
errors, probable OOM or SIGKILL, timeout or termination, Python exceptions, and
other nonzero exits.

Because output upload runs in the cleanup trap, inspect Cloud Storage even when
the Batch job reports failure. A failure before the repository or output
directory is created can still prevent artifact upload; in that case Cloud
Logging is the primary source of evidence.

## 11. Changing CPU, memory, disk, or VM type

The final six submission-helper arguments control the Batch allocation:

```text
MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB BOOT_DISK_SIZE_GB BOOT_DISK_TYPE
```

CPU is expressed in milli-vCPUs, so `32000` requests 32 vCPUs. Memory is MiB.
The CPU and memory requests must fit the selected machine type.

Keep the disk family compatible with the machine family:

- N2 commands in this repository use `pd-balanced`;
- C4 commands use `hyperdisk-balanced` because C4 does not support Persistent
  Disk;
- the helper rejects a C4-family machine combined with any `pd-*` disk before
  contacting Batch.

The production allocations in section 5 are part of the experiment contract.
If you change them for exploratory capacity testing, use a distinct job name
and record the changed allocation with the result. Do not present a modified
allocation as a canonical Experiment 1-4 run.

## 12. Choosing a VM size

Use smoke-test and production diagnostics rather than guessing from model
parameters alone. Review:

- peak and current RSS in `summary.json`;
- cgroup memory peak, limit, and OOM counters in
  `resource_snapshots.jsonl` and `batch_diagnostics.json`;
- CPU utilization, load average, and largest-process snapshots;
- node throughput and learner/worker timings in checkpoint rows;
- disk utilization at cleanup;
- Ray object-store pressure for Experiments 3 and 4.

Experiment 2 intentionally uses the same 32-vCPU production machine as
Experiments 3 and 4 so the sequential/parallel comparison is not confounded by
different hardware. Experiment 4 reserves capacity for 28 traversal actors,
the driver, Ray services, result merging, and concurrent learners.

## 13. Runtime limits and stopping

`MAX_RUN_SECONDS` is the Batch task ceiling, not the solver's effective
training timer. Canonical full runs use `50400` seconds (14 hours), while the
solver saves at 6 and 12 effective training hours and stops after the second
checkpoint. Setup, policy fitting, serialization, cleanup, and upload are
outside effective training time but inside the Batch ceiling.

Do not reduce the full-run Batch limit to 12 wall-clock hours: Batch could
terminate the task before the final policy is serialized and uploaded. A Batch
timeout is a failed run even if it produced a partial checkpoint.

Smoke tests use a two-hour ceiling but normally finish quickly because
`--smoke` replaces the production work and time thresholds with tiny values.

## 14. Adding and running later experiments

New experiments should retain the numbered naming convention:

```text
experiments/fhp/expN_descriptive_name/
```

Their runners should accept `--smoke` and `--output-root`, save reloadable
checkpoints where applicable, and use an `expN-` Batch job prefix. Submit them
through the same checked-in helper:

```bash
JOB_NAME="expN-fhp-description-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.expN_descriptive_name.run \
    --smoke --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 7200 4000 16000 100 pd-balanced
```

Add both smoke and full commands to this guide when the experiment is added.

## 15. Cleanup

Batch deletes the temporary VM after the job reaches a terminal state, so no
persistent experiment VM needs to be stopped manually. After confirming that
outputs and diagnostics are safely in Cloud Storage, remove an old Batch job
record with:

```bash
gcloud batch jobs delete "$JOB_NAME" \
  --location "$REGION"
```

List bucket contents before deleting any results:

```bash
gcloud storage ls --recursive "$BUCKET/$JOB_NAME/"
```

Keep at least the run manifest, both checkpoint manifests and policy files,
summary, Batch status, detailed diagnostics, resource snapshots, and run log.
Do not delete a failed job's evidence until its cause is understood.

## 16. Dependency installation

Each Batch VM starts clean. The checked-in helper installs system build tools
and `uv`, keeps the Cloud SDK on Python 3.10, and creates an isolated Python
3.11 FHP environment. It then installs the pinned packages in
`requirements.txt`, including CPU-only PyTorch, OpenSpiel, Ray, NumPy, SciPy,
and psutil, followed by an editable install of this repository.

Dependency installation time is part of Batch wall-clock time and cost, but not
effective solver training time. A dependency failure occurs before experiment
outputs may exist, so use Cloud Logging when no `batch_run.log` was uploaded.
The repository commit printed in the run log is the authoritative source
version for reproducibility.
