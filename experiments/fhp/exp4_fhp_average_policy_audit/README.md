# Active FHP Experiment 4: average-policy fitting audit

## Question and frozen inputs

Does the selected Experiment 2 FHP replay already contain a stronger average
policy than its existing neural fit extracts? Does the group sampler or the
optimisation horizon limit that extraction? This is an **offline development
audit**, not another 24-hour training run or untouched-seed confirmation.

Reuse the first completed 24-hour checkpoints of source seeds **0, 1, 2** from
`exp2-fhp-20260921-093839`. The source is configurable by run ID but must be the
selected Experiment 2 algorithm/encoder/network. The source files are:

- `workers/task_00S_lossless_structured_ucv_escher_seed_S/run_manifest.json`
- `checkpoint_manifest.json` and `SUCCESS.json` in that worker directory
- the manifest's 24-hour playable `.pkl` and full-training-state `.pt`.

The average-policy reservoir lives inside that `.pt`, **not** inside the small
playable policy or prior analytical download. The cloud fetcher downloads only
these files, not the 6/12/18-hour states. Checksums, seed, iteration, network,
encoder and saved policy weights are checked before fitting. Only policy
replay is retained after extraction; critics and calibration are released.
Neither the original states nor the source experiment are changed.

## Four deployment arms

| Arm | Group sampler | Updates |
|---|---|---:|
| `uniform_20000` | Current uniform group sampling without replacement within a minibatch | 20,000 |
| `uniform_60000` | Same | 60,000 |
| `mass_20000` | IID replay-mass-proportional sampling with replacement | 20,000 |
| `mass_60000` | Same | 60,000 |

All use the Experiment 2 structured policy model: 64-unit card/context
branches, shared `192 × 192` trunk, three action outputs; Adam, learning rate
0.003, batch size 2,048, masked soft-target cross-entropy, iteration exponent
2. Each sampler starts from identical reset weights for the same source seed.
NumPy/SciPy **and** Torch initialisation RNGs are controlled. The 20,000-update
model is a saved prefix of its 60,000-update trajectory, not an independently
refitted model. One optimiser replicate per source seed is used. The archived
policy is an additional evaluation reference; its original initialisation is
not claimed to match the new paired control.

Each source has one million replay rows. Grouping is by the complete existing
lossless canonical policy feature vector, preserving players, betting history,
round and strategically relevant cards. **No similar but distinct states are
merged**, no new abstraction is introduced, and no best response supplies a
supervised target.

For row weights `w_i = (2 t_i / T)^2`, group mass `M_g = sum_i w_i`, group count
`U` and original row count `N`, the existing objective is
`sum_g M_g CE_g / N`. The uniform sampler retains weights `M_g U / N`.
For proportional sampling `q_g = M_g / sum M`, the correct loss multiplier is
the constant `sum M / N`. This preserves **both the expected objective and its
scale**. Weighting that loss by `M_g` again, or simply using unscaled CE,
would confound the comparison. Proportional draws use a precomputed CDF and
vectorised binary search, not an O(U) probability normalisation per update.

## Validation and diagnostics

Deployment fits use **all** replay. A separate diagnostic fit repeats each
sampler path with 90% of distinct canonical groups for training and 10% held
out. All observations from a group stay on the same side of the split.
These diagnostic networks are not used for the headline gameplay comparison.
This additional fitting avoids either validation leakage or weakening all
deployment arms by discarding source data. The diagnostic split tests
generalisation to unseen groups represented in replay; it is not an estimate
of fidelity on every possible FHP state.

Diagnostics saved at both budgets include:

- Replay-weighted cross-entropy, KL and absolute probability error, separately
  on training and held-out groups.
- Breakdowns by preflop/flop, singleton/repeated-state frequency and legal action.
- Distinct groups, compression, row-count and mass quantiles, top-1% mass share,
  and loss-weight effective-sample fraction.
- Processed examples, fitting time excluding diagnostic passes, diagnostic
  time, grouping time, peak process memory and independent VM resource logs.
- Playable checkpoints for all four full-replay fits, source hashes, pinned
  revision, initial-weight hashes and fitting metadata; no optimizer/replay
  resume states are retained.

The fidelity objective is a replay surrogate. Neither lower KL nor higher
minibatch weight ESS guarantees reduced vulnerability to an opponent.

## Independent gameplay evaluation

Reuse the validated vendored `fhp_evaluation` implementation and its canonical
game, legal-action checks and unit conversion. Per source seed:

- All four refits and the archived policy face each published rule agent over
  **10,000 duplicate deal pairs** (20,000 hands) per matchup.
- Each of those five policies faces restricted LBR over **1,000 duplicate
  pairs**, with **4,096 preflop rollout samples** per LBR decision. This is
  split into deterministic 10-pair shards for parallelism and resumption.
- Each refit faces its own archived source policy over **50,000 duplicate
  pairs**. The three non-control refits also face the new `uniform_20000`
  control over 50,000 pairs each.

Evaluation uses separate seeds, common random numbers across fitting arms
where applicable, both seats and chance/action sampling. There is no training
on evaluated hands or exact best-response target. Positive head-to-head values
favour the named refit; lower LBR gain is better. **LBR is a restricted
exploitation diagnostic, not exact exploitability or proof of convergence.**
These budgets mirror earlier evaluation tiers, but new dealing seeds make the
measurements new sampled evaluations, not byte-identical previous results.

Aggregate means/standard errors, paired source-seed differences and exploratory
t intervals/sign-flip tests use **three training seeds**. Hands, groups,
checkpoints and arms are not additional independent training runs. The minimum
attainable two-sided sign-flip p-value is 0.25. No arm is automatically selected
or promoted; favour improvements supported jointly by fidelity, LBR and play,
then confirm promising choices in subsequent end-to-end training.
The paired table also includes sampler and update main effects and the
sampler-by-update interaction (difference of the two update increments).

## Run and recover

See the root README for copy-paste commands. `smoke-local` needs the repository
dependencies and uses artificial targets with actual FHP encodings. The cloud
smoke loads the **real seed-0 million-row replay**, checks and groups all of it,
then runs two/four updates on a bounded group subset and tiny gameplay budgets.
It exercises the real storage path without paying for production fitting.

The controller submits smoke -> three array workers -> aggregate entirely in
GCP. Each source seed gets its own `n2-standard-8`, 30,000 MiB requested memory
and 200 GB boot disk. Fits run sequentially within a seed with one Torch thread;
gameplay evaluation uses up to eight subprocesses. No advantage, critic or
regret training is performed. Full-data and held-out diagnostic fits together
cost 240,000 policy updates per seed; the four endpoint arms share prefixes.

Hard caps are 2 hours for cloud smoke, **24 hours per audit worker**, 1 hour for
aggregation and 30 hours for the default three-way controller. These are
failure ceilings, not runtime predictions. There are **zero automatic Batch
retries**. The default cap across the three production VMs is 72 VM-hours per
submission; controller/setup, smoke, aggregation, storage and explicitly
requested resumed jobs cost extra. Reducing `PARALLELISM` changes the controller
ceiling to allow sequential waves.

Completed fitting paths preserve all playable endpoints and metrics, not Adam
or sampler RNG state. An interrupted path repeats from its matched reset
initialization, preserving the nested-budget design. Completed paths are reused
after metadata/policy-hash validation. Evaluation results
are cached per task and uploaded every 20 completed tasks, at fit endpoints
and on normal/error cleanup. A hard VM kill can lose work since the last
successful upload. `resume` uses the same run ID/source/code and reuses saved
artifacts. A changed code revision/configuration is rejected for safety; use a
new run ID when changing the scientific implementation. Input full states are
never copied into the audit output bucket.

Existing controller-capable account permissions are sufficient: read/write
the results bucket, logging, `roles/batch.jobsEditor`, and permission to act
as the configured worker service account. The launcher checks that the
account and source files exist. It cannot prove quota or all IAM permissions
in advance; a child-submission error fails the controller promptly. As with
the other remote controllers, if the self-act-as grant is missing:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/batch.jobsEditor"
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"
```

`dry-run` writes all four Batch JSON specifications into a printed temporary
directory without submitting. `status` is read-only. `resume` submits a new
remote controller; it does not rerun the original Experiment 2 training.

## Outputs

`analysis/` is self-contained for thesis interpretation and includes
`policy_fitting_audit.png`, `summary.json`, `aggregate_metrics.csv`,
`paired_contrasts.csv`, `per_seed_metrics.csv`, `source_manifests.json`,
`detailed_diagnostics.json` (all round/action/frequency and per-policy Monte
Carlo summaries), and interpretation notes. Downloading this folder does not
download replay inputs. All four playable policies are under
`workers/seed_S/deployment/{uniform,mass}/{20000,60000}.pkl` if needed later.

## Verification

```bash
python -m pytest tests/test_exp4_average_policy_audit.py
./gcp/run_exp4_average_policy_audit.sh smoke-local
```

Tests cover source integrity, exact grouping agreement with the existing
trainer, no group leakage, gradient/objective equivalence, matched uniform
training, deterministic resumability, shared evaluation seeds, cache invalidation
and all four generated Batch scripts. Local smoke is not a claim that a cloud
production run or a particular policy improvement has succeeded.
