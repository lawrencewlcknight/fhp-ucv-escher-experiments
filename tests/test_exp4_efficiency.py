import numpy as np
import torch

from experiments.fhp.exp1_ucv_escher_baseline.config import BEST_UCV_CONFIG
from experiments.fhp.exp4_ucv_escher_cpu_optimized import config
from experiments.fhp.exp4_ucv_escher_cpu_optimized.run import (
    _parallel_solver_kwargs,
)
from unbiased_escher.efficient_replay import (
    CompactCalibrationBuffer,
    CompactCircularBuffer,
    CompactReservoirBuffer,
)
from unbiased_escher.parallel_solver import _append_reservoir


def _reservoir_payload(count: int):
    values = np.arange(count, dtype=np.float32)
    return {
        "infostates": np.column_stack((values, values + 100)),
        "values": values[:, None],
        "legal_masks": np.ones((count, 1), dtype=np.float32),
        "iterations": values[:, None],
    }


def test_exp4_keeps_the_algorithm_config_and_checkpoint_contract():
    assert config.EXPERIMENT_ID == 4
    assert config.EXPERIMENT_NAME.startswith("exp4_")
    assert config.DEFAULT_SEED == 0
    assert config.CHECKPOINT_TRAINING_SECONDS == (21_600, 43_200)
    assert config.UCV_CONFIG == BEST_UCV_CONFIG
    assert config.UCV_CONFIG is not BEST_UCV_CONFIG
    assert config.PARALLEL_NUM_WORKERS == 28
    assert config.PARALLEL_COLLECTION_CHUNK_SIZE == 5_000
    assert config.PARALLEL_LEARNER_THREADS == 4
    assert config.PARALLEL_LEARNER_INTRAOP_THREADS == 4


def test_exp4_enables_snapshot_caching_and_explicit_cpu_budgets():
    production = _parallel_solver_kwargs(seed=7)
    assert production == {
        "parallel_num_workers": 28,
        "parallel_run_seed": 7,
        "parallel_log_to_driver": False,
        "parallel_ray_object_store_memory": 8 * 1024**3,
        "parallelize_independent_learners": True,
        "parallel_learner_threads": 4,
        "parallel_learner_intraop_threads": 4,
        "parallel_collection_chunk_size": 5_000,
        "parallel_cache_actor_snapshots": True,
    }
    smoke = _parallel_solver_kwargs(seed=7, smoke=True)
    assert smoke["parallel_num_workers"] == 2
    assert smoke["parallel_learner_intraop_threads"] == 1
    assert smoke["parallel_collection_chunk_size"] == 2


def test_vectorised_reservoir_merge_preserves_sequential_last_write_semantics():
    class FixedRandom:
        @staticmethod
        def integers(*, low, high, dtype):
            assert low == 0
            assert high.tolist() == [4, 5]
            assert dtype == np.int64
            return np.asarray([0, 0], dtype=np.int64)

    buffer = CompactReservoirBuffer(3, 2, 1, seed=0)
    buffer.rng = FixedRandom()
    _append_reservoir(buffer, _reservoir_payload(5))

    assert len(buffer) == 5
    assert buffer.infostate_buf[:, 0].tolist() == [4.0, 1.0, 2.0]
    assert buffer.iteration_buf[:, 0].tolist() == [4.0, 1.0, 2.0]


def test_compact_replay_uses_training_precision_and_without_replacement_samples():
    reservoir = CompactReservoirBuffer(32, 4, 3, seed=1)
    reservoir.add_batch(
        {
            "infostates": np.arange(128, dtype=np.float32).reshape(32, 4),
            "values": np.ones((32, 3), dtype=np.float32),
            "legal_masks": np.ones((32, 3), dtype=np.float32),
            "iterations": np.arange(32, dtype=np.float32)[:, None],
        }
    )
    samples = reservoir.sample(16)
    assert all(tensor.dtype == torch.float32 for tensor in samples)
    assert len(torch.unique(samples[3])) == 16

    circular = CompactCircularBuffer(32, 6, 4, 3, seed=2)
    circular.size = 32
    circular.action_buf[:] = np.arange(32) % 3
    circular.next_player_buf[:] = np.arange(32) % 2
    circular.done_buf[:] = 0
    circular.next_legal_actions_mask_buf[:] = 1
    circular_samples = circular.sample(16)
    assert all(tensor.dtype == torch.float32 for tensor in circular_samples[:4])
    assert all(tensor.dtype == torch.int64 for tensor in circular_samples[4:])
    assert circular.history_buf.dtype == np.float32
    assert circular.action_buf.dtype == np.int16
    assert circular.next_legal_actions_mask_buf.dtype == np.int8

    calibration = CompactCalibrationBuffer(32, 7, seed=3)
    assert calibration.features.dtype == np.float32
    assert calibration.targets.dtype == np.float32


def test_compact_fhp_replay_allocation_is_below_seven_gibibytes():
    capacity = 1_000_000
    info_size = 190
    action_size = 3
    history_size = 380
    reservoir_bytes = 3 * capacity * (info_size + 2 * action_size + 1) * 4
    q_entry_bytes = (
        (2 * history_size + info_size + 1) * 4
        + 2
        + action_size
        + 1
        + 1
    )
    q_bytes = 3 * (capacity // 3) * q_entry_bytes
    calibration_bytes = capacity * (info_size + action_size + 4) * 4
    total_gib = (reservoir_bytes + q_bytes + calibration_bytes) / 1024**3
    assert total_gib < 7.0
