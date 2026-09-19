"""Configuration for archived Experiment 3's Ray-parallel comparison arm."""

from __future__ import annotations

from copy import deepcopy

from experiments.fhp.exp1_archived_ucv_escher_baseline import config as exp1_config


EXPERIMENT_ID = 3
EXPERIMENT_NAME = "exp3_archived_fhp_ucv_escher_ray_parallel"
ALGORITHM_ID = "ucv_escher_ray_parallel"
ALGORITHM_LABEL = "UCV-ESCHER Ray parallel (12 workers)"
DEFAULT_SEED = exp1_config.DEFAULT_SEED
CHECKPOINT_TRAINING_SECONDS = exp1_config.CHECKPOINT_TRAINING_SECONDS
TRAINING_DURATION_SECONDS = exp1_config.TRAINING_DURATION_SECONDS
BATCH_TIMEOUT_SECONDS = exp1_config.BATCH_TIMEOUT_SECONDS
UCV_CONFIG = deepcopy(exp1_config.BEST_UCV_CONFIG)
UCV_TRAINING_CONFIG_SHA256 = exp1_config.BEST_UCV_TRAINING_CONFIG_SHA256

PARALLEL_NUM_WORKERS = 12
PARALLEL_COLLECTION_CHUNK_SIZE = 1_200
PARALLEL_LEARNER_THREADS = 4
PARALLEL_RAY_OBJECT_STORE_MEMORY = 4 * 1024 * 1024 * 1024

PARALLEL_IMPLEMENTATION_PROVENANCE = {
    "repository": (
        "https://github.com/lawrencewlcknight/"
        "leduc-poker-escher-architecture-experiments.git"
    ),
    "commit": "c09bbe5a9adc6b495c9a0f74c2993dcb682ea754",
    "source_file": "unbiased_escher/parallel_solver.py",
    "source_experiment": "experiments/leduc_poker/ucv_escher_parallel_equivalence",
    "adaptations": (
        "canonical OpenSpiel FHP loading; 6h/12h training-time checkpoints; "
        "bounded synchronized traversal dispatches; FHP diagnostics"
    ),
}

REFERENCE_VM = {
    "machine_type": "c4-standard-32",
    "cpu_milli": 32_000,
    "memory_mib": 120_000,
    "boot_disk_gib": 200,
    "boot_disk_type": "hyperdisk-balanced",
}


def smoke_config() -> dict:
    return exp1_config.smoke_config()
