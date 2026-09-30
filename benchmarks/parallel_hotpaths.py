"""Reproducible local microbenchmarks, not a training-speed experiment.

Run from the repository root: python -m benchmarks.parallel_hotpaths
Requires the test dependencies for the frozen reference implementation.
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch

from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.float32_solver import (
    Float32LegacyReservoirBuffer,
)
from tests.test_parallel_optimisations import reference_grouping
from unbiased_escher.grouped_wide_solver import GroupedSoftTargetCrossEntropyAvePolicyTrainer
from unbiased_escher.solver import ResidualCalibrationTrainer
from vr_deep_cfr.logger import Logger


def timed(fn, repeats):
    fn()  # untimed warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def compare(reference, optimised, repeats):
    before = timed(reference, repeats)
    after = timed(optimised, repeats)
    return dict(reference_seconds=before, optimised_seconds=after,
                speedup=before / after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 100 <= args.rows <= 1_000_000 or args.repeats < 1:
        parser.error("rows must be 100..1000000 and repeats positive")
    torch.set_num_threads(1)
    n = args.rows
    key = (np.arange(n) % (n // 2)).astype(np.float32)
    payload = dict(
        infostates=np.repeat(key[:, None], 190, axis=1),
        values=np.full((n, 3), 1 / 3, dtype=np.float32),
        legal_masks=np.ones((n, 3), dtype=np.float32),
        iterations=np.ones((n, 1), dtype=np.float32),
    )
    reference = Float32LegacyReservoirBuffer(n, 190, 3)
    optimised = Float32LegacyReservoirBuffer(n, 190, 3)

    def scalar_merge():
        reference.reset()
        for row in zip(*payload.values()):
            reference.add(*row)

    def batch_merge():
        optimised.reset()
        optimised.add_batch(payload)

    results = {"reservoir_before_capacity": compare(scalar_merge, batch_merge, args.repeats)}
    for name in ("infostate_buf", "q_value_buf", "q_value_mask_buf", "iteration_buf"):
        np.testing.assert_array_equal(getattr(reference, name), getattr(optimised, name))
    # Include the RNG-dominated regime; speedup is not assumed after saturation.
    def scalar_full():
        reference.cur_id = n
        np.random.seed(41)
        for row in zip(*payload.values()):
            reference.add(*row)

    def batch_full():
        optimised.cur_id = n
        np.random.seed(41)
        optimised.add_batch(payload)

    results["reservoir_after_capacity"] = compare(scalar_full, batch_full, args.repeats)
    trainer = GroupedSoftTargetCrossEntropyAvePolicyTrainer(
        190, 3, [8], .003, 1, 2048, 1, Logger(verbose=False), "cpu", 2,
    )
    trainer.buffer = optimised
    trainer.buffer.cur_id = n
    # Restore the same deterministic repeated-state payload after reservoir tests.
    for name, value in zip(
        ("infostate_buf", "q_value_buf", "q_value_mask_buf", "iteration_buf"),
        payload.values(),
    ):
        getattr(trainer.buffer, name)[:] = value
    results["grouped_preprocessing"] = compare(
        lambda: reference_grouping(trainer, 2),
        lambda: trainer._grouped_training_data(2),
        args.repeats,
    )
    assert all(torch.equal(a, b) for a, b in zip(
        reference_grouping(trainer, 2), trainer._grouped_training_data(2),
    ))

    calibration = ResidualCalibrationTrainer(
        infostate_size=190, action_size=3, hidden_layers=[64, 64, 64],
        learning_rate=.001, buffer_size=1, batch_size=1, train_steps=0,
        device="cpu", minimum_variance=1e-5,
    )
    info, disagreement = payload["infostates"][0], [.1, .2, .3]
    def old_features():
        for _ in range(2_000):
            np.stack([calibration.feature(info, a, 9, disagreement[a], 0) for a in range(3)])

    # Measure complete prediction separately in tests; isolate common-feature
    # construction here to avoid mislabelling a kernel result as training gain.
    def new_features():
        import math
        for _ in range(2_000):
            features = np.empty((3, 196), dtype=np.float32)
            features[:, :190] = info
            features[:, 190:193] = np.eye(3, dtype=np.float32)
            features[:, -3] = math.log1p(9) / math.log(101)
            features[:, -2] = [math.log1p(max(float(x), 0)) for x in disagreement]
            features[:, -1] = 0.

    results["calibration_feature_construction_2000_states"] = compare(
        old_features, new_features, args.repeats,
    )
    print(json.dumps(dict(rows=n, repeats=args.repeats, torch_threads=1,
                         scope="local microbenchmarks; not end-to-end speedup",
                         results=results), indent=2))


if __name__ == "__main__":
    main()
