from experiments.fhp.exp1_ucv_escher_baseline.config import BEST_UCV_CONFIG
from experiments.fhp.exp2_ucv_escher_sequential import config as sequential
from experiments.fhp.exp3_ucv_escher_parallel import config as parallel
from experiments.fhp.exp3_ucv_escher_parallel.run import _parallel_solver_kwargs
from unbiased_escher.parallel_utils import partition_total, worker_seed


def test_comparison_arms_are_time_config_and_machine_matched():
    assert sequential.EXPERIMENT_ID == 2
    assert parallel.EXPERIMENT_ID == 3
    assert sequential.EXPERIMENT_NAME.startswith("exp2_")
    assert parallel.EXPERIMENT_NAME.startswith("exp3_")
    assert sequential.DEFAULT_SEED == parallel.DEFAULT_SEED == 0
    assert sequential.CHECKPOINT_TRAINING_SECONDS == (21_600, 43_200)
    assert parallel.CHECKPOINT_TRAINING_SECONDS == (21_600, 43_200)
    assert sequential.TRAINING_DURATION_SECONDS == 43_200
    assert parallel.TRAINING_DURATION_SECONDS == 43_200
    assert sequential.BATCH_TIMEOUT_SECONDS == 50_400
    assert parallel.BATCH_TIMEOUT_SECONDS == 50_400
    assert sequential.REFERENCE_VM == parallel.REFERENCE_VM == {
        "machine_type": "c4-standard-32",
        "cpu_milli": 32_000,
        "memory_mib": 120_000,
        "boot_disk_gib": 200,
        "boot_disk_type": "hyperdisk-balanced",
    }
    assert sequential.UCV_CONFIG == parallel.UCV_CONFIG == BEST_UCV_CONFIG
    assert sequential.UCV_CONFIG is not BEST_UCV_CONFIG
    assert parallel.UCV_CONFIG is not BEST_UCV_CONFIG
    assert (
        sequential.UCV_TRAINING_CONFIG_SHA256
        == parallel.UCV_TRAINING_CONFIG_SHA256
        == "42a1c60051502d7cf44d6a368d144588b7b470910d4a20c602b2667118431846"
    )


def test_parallel_arm_uses_declared_upstream_backend_contract():
    assert parallel.PARALLEL_NUM_WORKERS == 12
    assert parallel.PARALLEL_COLLECTION_CHUNK_SIZE == 1_200
    assert parallel.PARALLEL_LEARNER_THREADS == 4
    assert parallel.PARALLEL_RAY_OBJECT_STORE_MEMORY == 4 * 1024**3
    assert parallel.PARALLEL_IMPLEMENTATION_PROVENANCE["commit"] == (
        "c09bbe5a9adc6b495c9a0f74c2993dcb682ea754"
    )
    assert partition_total(1_200, 12) == [100] * 12
    assert sum(partition_total(10_000, 12)) == 10_000
    assert len({worker_seed(0, index) for index in range(12)}) == 12


def test_parallel_smoke_execution_contract_is_small_but_structurally_identical():
    kwargs = _parallel_solver_kwargs(seed=7, smoke=True)
    assert kwargs["parallel_num_workers"] == 2
    assert kwargs["parallel_run_seed"] == 7
    assert kwargs["parallel_collection_chunk_size"] == 2
    assert kwargs["parallelize_independent_learners"] is True
