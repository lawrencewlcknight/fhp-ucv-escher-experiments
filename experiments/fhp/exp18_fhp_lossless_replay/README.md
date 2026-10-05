# Experiment 18 — lossless replay memory and speed validation

Baseline: Experiment 10 (Experiment 9 plus explicit hand/board features),
including frozen-critic-target caching. This is an **engineering experiment**,
not an experiment about poker strength. Existing experiments remain unchanged.

## Changes under test

- Opt-in `replay_storage="fhp_feature_codes_v1"`. Store finite encoder values
  as ordered uint8 dictionary codes; decode the **identical float32 bits** on
  sampling. Reject unrecognised values rather than round or approximate them.
- Preserve policy/regret targets, weights, rewards and continuous calibration
  suffixes in float32. Preserve reservoir admissions, all sampler RNG streams,
  row order, masks, network architectures and the loss functions.
- Group policy replay using byte keys and chunked float64 accumulation. Keep
  the existing lexicographic group order and weighting; decode selected
  minibatches, not the complete grouped feature matrix. Sorting still has
  O(N) scratch storage: this is not an out-of-core sort.
- Store feature codes directly in the existing checksummed NumPy-shard
  checkpoint format. Memory-map reads and restore in bounded batches, also
  supporting legacy dense states. Save only occupied rows. No replay-sized
  float decoding during save, and no full float feature copy during restore.
- Separate policy and critic representations remain separate. There is no
  new opponent-card information in policy, regret or calibration inputs.

Currently restricted to the versioned Experiment 10 encoder. This is not a
generic quantiser and does not silently opt Experiment 11/12 into a new codec.

## Cloud protocol

One `n2-standard-16`, 64 GiB, 200 GB boot disk, eight Ray traversal actors,
eight learner threads, seed 0, pinned Python 3.11.16 and requirements. Arms
run in separate subprocesses, never concurrently. A 12-hour **timeout**, not
a 12-hour training budget, bounds the job. No controller/child jobs needed.

1. **Correctness smoke**: tiny replay/updates but real eight-actor collection,
   three completed iterations and policy fitting. Compare full logical learning
   state bitwise: networks, optimizer states, targets, replay, learner/actor
   RNGs and counters. Save and reload in a fresh solver with fresh actors;
   verify a further iteration is identical to uninterrupted training.
2. **Real training timings**: three repeats of the same seed, alternating arm
   order. Each arm performs two iterations with Experiment 10's production
   learning settings and one 20,000-update output-policy fit. This fixed work
   replaces the 24-hour clock for the audit only. Every repeat must pass the
   same logical-state parity and continuation checks. Report training time and
   speed ratio separately from startup, save/load and continuation timings.
   Collection, merge, regret-fitting and critic-cache counters are retained
   separately, with the inherited nested-timer interpretations.
   These are timing repetitions, not independent scientific seeds.
3. **Matched populated-memory comparison**: fill all buffers, then group/fit
   policy, build both frozen-critic caches/fit and calibrate. Both arms use
   1m policy, 1m total critic, 1m calibration and 200k regret rows per player.
   Compare their resulting learning state exactly. This uses deterministic
   high-diversity synthetic feature vectors, not valid poker experience;
   policy fitting uses 100 updates, each critic/calibration 20 updates.
   Before fitting, collect one real 1,200-trajectory Ray chunk per player
   into the full replay to test transfer/admission overhead (eight per player
   in local smoke). Transient regret occupancy resets as in normal collection;
   its full allocation has already been touched. The mixed replay remains a
   synthetic engineering workload, not a candidate for poker evaluation.
4. **Large coded-only stress**: repeat with 10m policy, 10m total critic
   (5m/fold), 10m calibration and 2m regret rows/player. Actually touch every
   row, construct up to 10m groups, build both complete target caches and
   perform the same short fits. Save/reload/checksum the full persistent state
   while the eight actors remain alive. Do **not** run the unsafe dense arm
   at this capacity on the 64-GiB VM. Require at least 8 GiB available-memory
   headroom throughout. A failure blocks promotion, not automatic VM upsizing.

Estimated allocated replay storage at the large setting: **51.77 GiB dense
vs 13.49 GiB coded** (including unchanged float32 masks). These are array
counts, not peak-memory measurements. Models, Ray, grouping scratch, target
caches, I/O and Python overhead are measured by the job. The calibration
suffix retains six float32 columns; it is not rounded to byte codes.

Whole-VM `total - available` and driver RSS are sampled every 0.5 seconds;
VM pressure includes Ray/processes without double-counting shared RSS. Native
driver peak RSS is also recorded. These measures are not interchangeable and
sampled VM peaks can miss sub-second transients. Resource diagnostics remain
available if a subprocess fails or is killed.

Both arms use the same streamed checkpoint container, isolating the payload
storage difference. The measured save/load ratio is not a benchmark of the
legacy monolithic `torch.save` serializer versus sharding.

The test does **not** increase collection to 100k traversals/player or claim
that enlarged replay improves learning. The large-buffer phase is a capacity
gate before that separate experience-allocation experiment. Short, initially
unfilled-buffer timings do not establish long-run speedup; inspect populated
grouping/cache times too. Slower training is reported even if correctness and
memory pass. Review both before adopting the opt-in format in production.

## Run

After committing/pushing (not performed automatically):

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="YOUR_EXISTING_FHP_RUNNER_SERVICE_ACCOUNT"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp18-memory-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp18_lossless_replay.sh run
```

Local bounded check: `bash gcp/run_exp18_lossless_replay.sh smoke-local`
(requires working local Ray; uses only 1,000 synthetic rows). Unit tests:
`python -m pytest -q tests/test_lossless_replay.py tests/test_exp18_lossless_replay.py`.

Outputs: `$BUCKET/$RUN_ID/analysis/summary.json`, per-phase results, exactness
digests, timing/memory logs, runtime/commit/configuration provenance and cloud
diagnostics. Large validation states are temporary and are **not uploaded**.
This audit does not produce a trained poker candidate or reusable final
policy; no head-to-head evaluation is needed to validate storage identity.
