# Parallel UCV-ESCHER efficiency review

Reviewed 30 September 2026. This is an implementation review and local
validation, not a new training experiment or a measured end-to-end speedup.
No production launcher, experiment configuration, saved result, or training
hyperparameter has been changed.

## Architecture compatibility

The existing `ParallelUnbiasedControlVariateEscher` and
`EfficientParallelUnbiasedControlVariateEscher` implement the archived
architecture, not the selected grouped-policy and averaged-critic package.
Using one of those classes directly as the parallel comparator for active
Experiments 1–3 would confound architecture with execution backend.

The new `unbiased_escher/grouped_parallel_solver.py` provides:

| Sequential implementation | Matching parallel adapter |
| --- | --- |
| Experiment 1 raw-input float32 grouped-wide solver | `ParallelGroupedWideUCVEscher` |
| Experiment 2 structured solver | `ParallelStructuredGroupedUCVEscher` |
| Experiment 3 wider structured solver | `ParallelStructuredGroupedUCVEscher`, using Experiment 3's network settings |

Driver and traversal workers now instantiate matching selected solver classes.
They retain fixed beta 1, two cross-fitted critics, four-fit critic-target
averaging, no predictive path, and reset grouped soft-target cross-entropy
policy fitting. The driver owns the learners and persistent replay. Each
traverser phase uses frozen inference weights; the configured traversal total
is partitioned across workers, not multiplied by their number. Results merge
in dispatch order, not completion order. Time checkpoints in these adapters
run after complete outer iterations, matching the active sequential solvers.

These adapters are tested building blocks, **not yet wired into a new GCP
experiment**. In particular, the existing active experiment continuation
serializer saves driver state but not Ray actors' random-number states. A
production parallel experiment must add actor RNG capture/restoration and a
restart-equivalence test before advertising exact automatic resume. Do not
simply replace the solver class in an existing resumable launcher.

## Implemented optimisations

1. **Batch replay writes without changing replacement randomness.** The legacy
   reservoir now supports batch insertion. Direct fills use contiguous copies;
   replacement draws remain the same scalar NumPy calls in the same order.
   When several records choose one slot, the final record wins, exactly as in
   sequential insertion. Raw-input float32 replay inherits this implementation.
   The structured solver retains its existing compact replay and private RNGs.
2. **Contiguous circular-buffer merges.** Critic and calibration batches use at
   most two slice writes per field, avoiding modulo-index arrays and scattered
   assignments. Wrapped and over-capacity batches retain the same records and
   cursor positions.
3. **Cheaper grouped-target construction.** Reuse the first-occurrence indices
   returned by the unique-state operation, eliminating an additional sort.
   Validate legal-action masks in bounded 65,536-row vectorised chunks rather
   than one Python comparison per row. Weighted target sums keep their original
   `np.add.at` order and float64 accumulation: the fitting objective is unchanged.
4. **Remove redundant full-batch tensor copies.** Full-batch policy updates use
   the already-prepared tensors directly. Minibatch selection is unchanged.
   This saves copies when grouped data fit in a batch; it does not remove the
   genuine minibatch cost of larger datasets.
5. **Reuse frozen worker snapshots.** Enable the existing per-traverser snapshot
   token cache by default. Later chunks reuse weights already loaded into the
   actor. Each new traverser phase gets a new token. The explicit
   `parallel_cache_actor_snapshots=False` option remains available as a control.
6. **Broadcast calibration features.** Allocate the all-action feature matrix
   once and broadcast its shared information-state fields, instead of allocating
   and concatenating a separate state for each action. Scalar logarithms and
   float32 conversion are retained exactly.

The neural layers and batched optimisation already use PyTorch matrix
operations. Rewriting them as different kernels was not necessary for these
gains. Tree traversal still branches on sampled game outcomes; replacing that
loop is a separate batched-simulation design problem, not a safe mechanical
loop-to-matrix substitution.

## Reproducibility correction

Concurrent independent-learner fitting previously allowed replay buffers
sharing Python's global RNG. The interleaving of minibatch draws then depended
on thread scheduling. The backend now enables concurrent fitting only when
all Q-fold and calibration replay buffers expose distinct private NumPy
generators. Otherwise it preserves the sequential calibration-then-critics
update order. Active selected solvers currently take this fallback; their
traversals still run in parallel. The existing fully private-RNG efficient
backend can still fit independent learners concurrently.

This safeguard can change results relative to the old shared-RNG threaded
path: that path had schedule-dependent randomness, not a deterministic
reference. It can also reduce learner concurrency. Preserving the selected
serial sampling behaviour takes priority over a nominal concurrency gain.
A future optimisation could generate the exact serial minibatch stream on
the driver before dispatching fits, subject to a separate equivalence test.

## Verification

The checks cover:

- Exact reservoir arrays, counters, sampled tensors and NumPy RNG state versus
  repeated scalar insertion, before and after saturation and with collisions.
- Exact ring/calibration replay versus scalar insertion, including wraparound,
  empty batches and batches exceeding capacity.
- Bitwise-identical grouped targets, losses, gradients, fitted parameters, Adam
  state and Python RNG state against the prior implementation, for full-batch
  and minibatch fits. Inconsistent legal masks still raise an error, including
  across the validation chunk boundary.
- Identical calibration features and predictions to the old feature builder.
- Sequential fallback for global or shared generators, and concurrent fitting
  with distinct private generators while restoring PyTorch's thread count.
- Real Ray integration for raw, structured and wider structured configurations:
  two workers, two outer iterations and multiple chunks per traverser. Cached
  and uncached runs have identical node counts, policy-fitting loss and tested
  model parameters. Snapshot loads fall from 16 to 8 without changing the eight
  dispatches. All 24 requested trajectories are collected, and time hooks run
  once per completed iteration after critic target histories have updated.
- Existing active-experiment and smoke tests, including sequential continuation.

Local verification used Python 3.12.2, NumPy 1.26.4, PyTorch 2.7.0 and Ray
2.51.2, with a FHP-enabled OpenSpiel build: **80 unit/smoke checks and three
real-Ray integration tests passed**. From the repository root, with the
development dependencies installed:

```bash
python -m pytest -q tests --ignore=tests/test_repository_cleanliness.py -m 'not ray'
python -m pytest -q tests/test_grouped_parallel_solver.py
python -m benchmarks.parallel_hotpaths --rows 50000 --repeats 5
```

The first command excludes the repository-cleanliness scan because that test
recursively reads local downloaded cloud artifacts, not just source files.
It is unrelated to solver correctness and can be run separately in a clean
source checkout. Ray integration requires permission to start local processes.

### Local microbenchmark

Median of five timed repetitions after warm-up, one PyTorch thread. Replay
uses 50,000 synthetic rows with 190 input features and three actions; grouping
has 25,000 distinct states. These are small local CPU measurements, not GCP
training-speed or peak-memory guarantees.

| Operation | Previous | Optimised | Ratio |
| --- | ---: | ---: | ---: |
| Reservoir insertion before capacity | 50.35 ms | 1.00 ms | 50.5x |
| Reservoir insertion after capacity | 121.72 ms | 84.64 ms | 1.44x |
| Complete grouped-target preparation | 311.90 ms | 193.37 ms | 1.61x |
| Calibration feature construction, 2,000 states | 15.40 ms | 5.81 ms | 2.65x |

The calibration measurement excludes neural inference. The large initial-fill
ratio applies only to copying records into a not-yet-full reservoir; it must
not be used as an estimate of overall speedup. End-to-end gains depend on the
fraction of runtime spent traversing, fitting networks, merging replay,
constructing targets and saving checkpoints.

## Conditions for the next serial-versus-parallel experiment

Use the same selected architecture, representation, replay precision and
capacity, learner steps, total traversals per iteration, checkpoint fitting
budget and active-time accounting. Keep both backends on the same machine
class; report worker count and learner/intra-op thread allocations. Distinguish
a one-worker collection baseline from an entirely single-threaded baseline:
PyTorch can already use multiple cores inside a serial learner.

Parallel actors use independent random streams, so a serial run and a
parallel run with the same top-level seed are **not bitwise-equivalent
trajectories**. This is distinct from the exact equivalence demonstrated for
the changed hotpaths and cached/uncached parallel executions. Hold the
parallel worker count and chunk size fixed for reproducible comparisons, and
assess quality over multiple seeds at both matched nodes and matched training
time. Record merge, synchronisation, collection, fitting and checkpoint costs
along with memory use; do not infer a training speedup from these component
benchmarks alone.

Before a long cloud run, add actor-aware resume validation and a production-size
memory preflight including driver replay, worker scratch space, Ray object
storage and checkpoint upload overhead. No such cloud run was launched in
this review.
