"""Configuration for Experiment 4's high-throughput CPU implementation."""

from __future__ import annotations

from copy import deepcopy

from experiments.fhp.exp1_ucv_escher_baseline import config as exp1_config


EXPERIMENT_ID = 4
EXPERIMENT_NAME = "exp4_fhp_ucv_escher_cpu_optimized"
ALGORITHM_ID = "ucv_escher_cpu_optimized"
ALGORITHM_LABEL = "UCV-ESCHER CPU-optimised parallel"
DEFAULT_SEED = exp1_config.DEFAULT_SEED
CHECKPOINT_TRAINING_SECONDS = exp1_config.CHECKPOINT_TRAINING_SECONDS
TRAINING_DURATION_SECONDS = exp1_config.TRAINING_DURATION_SECONDS
BATCH_TIMEOUT_SECONDS = exp1_config.BATCH_TIMEOUT_SECONDS
UCV_CONFIG = deepcopy(exp1_config.BEST_UCV_CONFIG)
UCV_TRAINING_CONFIG_SHA256 = exp1_config.BEST_UCV_TRAINING_CONFIG_SHA256

# Leave four vCPUs outside Ray for the driver, OS, object-store service, and
# result merging. Learner phases use four disjoint tasks with four intra-op
# threads each; this is explicit so the VM can be profiled and retuned cleanly.
PARALLEL_NUM_WORKERS = 28
PARALLEL_COLLECTION_CHUNK_SIZE = 5_000
PARALLEL_LEARNER_THREADS = 4
PARALLEL_LEARNER_INTRAOP_THREADS = 4
PARALLEL_RAY_OBJECT_STORE_MEMORY = 8 * 1024 * 1024 * 1024

EFFICIENCY_CHANGES = {
    "algorithm_unchanged": (
        "UCV estimator, networks, optimiser steps, traversal budget, player "
        "ordering, frozen targets, and checkpoint schedule"
    ),
    "systems_changes": (
        "28 CPU traversal actors; two dispatches per 10,000-trajectory player "
        "phase; cached actor snapshots; compact typed replay; vectorised exact "
        "reservoir ingestion; vectorised without-replacement minibatch indices; "
        "per-buffer RNGs; explicit learner intra-op thread budget"
    ),
}

REFERENCE_VM = {
    "machine_type": "c4-standard-32",
    "cpu_milli": 32_000,
    "memory_mib": 120_000,
    "boot_disk_gib": 200,
}


def smoke_config() -> dict:
    return exp1_config.smoke_config()
