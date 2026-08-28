"""Configuration for Experiment 2's sequential comparison arm."""

from __future__ import annotations

from copy import deepcopy

from experiments.fhp.exp1_ucv_escher_baseline import config as exp1_config


EXPERIMENT_ID = 2
EXPERIMENT_NAME = "exp2_fhp_ucv_escher_sequential"
ALGORITHM_ID = "ucv_escher_sequential"
ALGORITHM_LABEL = "UCV-ESCHER sequential"
DEFAULT_SEED = exp1_config.DEFAULT_SEED
CHECKPOINT_TRAINING_SECONDS = exp1_config.CHECKPOINT_TRAINING_SECONDS
TRAINING_DURATION_SECONDS = exp1_config.TRAINING_DURATION_SECONDS
BATCH_TIMEOUT_SECONDS = exp1_config.BATCH_TIMEOUT_SECONDS
UCV_CONFIG = deepcopy(exp1_config.BEST_UCV_CONFIG)
UCV_TRAINING_CONFIG_SHA256 = exp1_config.BEST_UCV_TRAINING_CONFIG_SHA256

REFERENCE_VM = {
    "machine_type": "c4-standard-32",
    "cpu_milli": 32_000,
    "memory_mib": 120_000,
    "boot_disk_gib": 200,
}


def smoke_config() -> dict:
    return exp1_config.smoke_config()
