"""Prespecified offline audit: no updates to the underlying regret learner."""

SEEDS = (0, 1, 2)
SAMPLERS = ("uniform", "mass")
SOURCE_EXPERIMENT = "exp2_fhp_lossless_structured_ucv"
SOURCE_ALGORITHM = "lossless_structured_ucv_escher"
MODULE = "experiments.fhp.exp4_fhp_average_policy_audit.run"


def contract(smoke=False):
    return {
        "experiment_id": 4,
        "experiment_name": "exp4_fhp_average_policy_audit",
        "source_experiment": SOURCE_EXPERIMENT,
        "source_checkpoint": "time_24h",
        "seeds": list(SEEDS),
        "smoke": bool(smoke),
        "samplers": list(SAMPLERS),
        "updates": [2, 4] if smoke else [20_000, 60_000],
        "batch_size": 8 if smoke else 2048,
        "learning_rate": 0.003,
        "gamma": 2.0,
        "network_layers": [192, 192],
        "branch_width": 64,
        "reset_each_path": True,
        "training_state_retention": "none",
        "initialisation_seed_base": 202609300,
        "heldout_group_fraction": 0.1,
        "split_seed": 20260930,
        "diagnostic_fits_separate_from_full_replay_deployment": True,
        "rule_deals": 2 if smoke else 10_000,
        "direct_deals": 2 if smoke else 50_000,
        "lbr_deals": 2 if smoke else 1000,
        "lbr_shard_deals": 2 if smoke else 10,
        "lbr_rollouts": 16 if smoke else 4096,
        "evaluation_seed_base": 202609301,
        "inferential_unit": "source_training_seed",
        "exact_exploitability": False,
    }


def source_worker_name(seed):
    return f"task_{seed:03d}_{SOURCE_ALGORITHM}_seed_{seed}"


def arm_names(config):
    return [f"{sampler}_{updates}" for sampler in SAMPLERS for updates in config["updates"]]
