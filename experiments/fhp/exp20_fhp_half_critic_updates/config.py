"""One-factor, fresh-start comparison: halve each critic fit's update budget."""
from copy import deepcopy

from experiments.fhp.exp10_fhp_hand_board_features import config as base

EXPERIMENT_ID = 20
EXPERIMENT_NAME = "exp20_fhp_half_critic_updates"
ALGORITHM_ID = "half_critic_hand_board_ucv_escher"
ALGORITHM_LABEL = "UCV-ESCHER (hand/board features, 5,000 critic updates)"
BASELINE_RUN_ID = "exp10-features-20261001-161740"
BASELINE_REF = "e66d4da515eb212e5026a965ac5c39c86144c901"
DEFAULT_TOTAL_HOURS = base.DEFAULT_TOTAL_HOURS
CHECKPOINT_HOURS = base.CHECKPOINT_HOURS
BATCH_TIMEOUT_SECONDS = base.BATCH_TIMEOUT_SECONDS
PRODUCTION_SEEDS = base.PRODUCTION_SEEDS
SMOKE_SEEDS = base.SMOKE_SEEDS
LEARNER_THREADS = base.LEARNER_THREADS
PARALLEL_WORKERS = base.PARALLEL_WORKERS
COLLECTION_CHUNK_SIZE = base.COLLECTION_CHUNK_SIZE
REFERENCE_VM = deepcopy(base.REFERENCE_VM)
EXPERIMENT_CONFIG = deepcopy(base.EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG["baseline_network_train_steps"] = 5_000
parallel_settings = base.parallel_settings
validate_total_hours = base.validate_total_hours
checkpoint_schedule = base.checkpoint_schedule


def smoke_config():
    # More than one update exercises optimiser state; never strength evidence.
    return dict(base.smoke_config(), baseline_network_train_steps=2)


def task_schedule(seeds=PRODUCTION_SEEDS):
    return tuple((ALGORITHM_ID, int(seed)) for seed in seeds)


def validate_contract(*, seeds, schedule, config, smoke,
                      total_hours=DEFAULT_TOTAL_HOURS):
    expected = smoke_config() if smoke else EXPERIMENT_CONFIG
    if dict(config) != expected:
        raise ValueError("Experiment 20 must equal Experiment 10 except for critic updates")
    inherited = dict(config)
    inherited["baseline_network_train_steps"] = (
        base.smoke_config() if smoke else base.EXPERIMENT_CONFIG)["baseline_network_train_steps"]
    base.validate_contract(seeds=seeds, schedule=schedule, config=inherited,
                           smoke=smoke, total_hours=total_hours)


def contract_manifest(*, total_hours=DEFAULT_TOTAL_HOURS):
    manifest = deepcopy(base.contract_manifest(total_hours=total_hours))
    manifest.update(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
        baseline=base.EXPERIMENT_NAME, baseline_run_id=BASELINE_RUN_ID,
        baseline_commit=BASELINE_REF, training_config=deepcopy(EXPERIMENT_CONFIG),
        single_intended_change="baseline_network_train_steps_10000_to_5000",
        affected_networks=["critic"], critic_updates_per_fit=5_000,
        baseline_critic_updates_per_fit=10_000,
        initialisation="fresh seeds; not a warm start from Experiment 10",
        historical_comparison_caveat=(
            "Same active-time budget, unchanged collection and output-policy fitting. "
            "Shorter critic fits change later trajectories and experience. More nodes "
            "do not imply stronger play. Matching historical seed labels is not a "
            "bitwise paired intervention or control for VM performance differences. "
            "Compare 24h policies with frozen Experiment 10 policies using the separate "
            "rule/LBR/head-to-head evaluator; no automatic policy-quality promotion."
        ),
    )
    return manifest
