"""Two prespecified rates, with no architecture or update-budget search."""
from copy import deepcopy

from experiments.fhp.exp13_fhp_policy_capacity.config import ARCHITECTURES, RECIPES

SOURCE_EXPERIMENT = "exp9_fhp_cached_parallel_24h"
SOURCE_ALGORITHM = "cached_parallel_structured_ucv_escher"
SOURCE_STATE_TYPE = "exp9_fhp_cached_parallel_24h_full_training_state"
MODULE = "experiments.fhp.exp15_fhp_policy_learning_rate.run"
RATES = ("adam_003", "adam_0003")


def contract(smoke=False, include_lbr=False):
    return {
        "experiment_id": 15, "experiment_name": "exp15_fhp_policy_learning_rate",
        "policy_id_prefix": "policy_learning_rate",
        "source_experiment": SOURCE_EXPERIMENT, "source_checkpoint": "time_24h",
        "source_replay_rows": 1_000_000,
        "seeds": [0] if smoke else [0, 1, 2], "smoke": bool(smoke),
        "architectures": {"standard": deepcopy(ARCHITECTURES["standard"])},
        "recipes": {name: deepcopy(RECIPES[name]) for name in RATES},
        "replicates": [0, 1], "sampler": "uniform", "gamma": 2.0,
        "batch_size": 8 if smoke else 2048,
        "updates": [1, 2, 4] if smoke else [5000, 10000, 20000],
        "reset_each_path": True, "initialisation_seed_base": 202610150,
        "sampler_seed_base": 202610151, "split_seed": 20261015,
        "heldout_group_fraction": 0.1,
        "selection": "none; fixed final update is evaluated for every arm",
        "primary_contrast": "adam_0003 versus adam_003 in matched direct play at 20000 updates",
        "rule_deals": 2 if smoke else 10000,
        "direct_deals": 2 if smoke else 50000,
        "lbr_deals": (2 if smoke else 1000) if include_lbr else 0,
        "lbr_shard_deals": 2 if smoke else 10, "lbr_rollouts": 16 if smoke else 4096,
        "evaluation_seed_base": 202610152,
        "deduplicate_identical_evaluation": True,
        "remote_worker_env": "EXP15_REMOTE_WORKER",
        "inferential_unit": "source_training_seed", "training_state_retention": "none",
        "exact_exploitability": False,
    }


def source_worker_name(seed):
    return f"task_{seed:03d}_{SOURCE_ALGORITHM}_seed_{seed}"


def arm_name(recipe, replicate):
    return f"{recipe}_rep{replicate}"


def fit_directory(worker, phase, recipe, replicate):
    return worker / phase / recipe / f"rep{replicate}"
