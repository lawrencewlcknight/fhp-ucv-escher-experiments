"""Prespecified capacity comparison and equally budgeted optimisation screen."""

from copy import deepcopy

from experiments.fhp.exp4_fhp_average_policy_audit.config import contract as audit_contract

SEEDS = (0, 1, 2)
ARCHITECTURES = {
    "standard": {"network_layers": [192, 192], "branch_width": 64, "parameters": 74243},
    "wide": {"network_layers": [256, 256], "branch_width": 96, "parameters": 133731},
}
RECIPES = {
    "adam_003": {"optimizer": "adam", "learning_rate": .003, "weight_decay": 0.0},
    "adam_001": {"optimizer": "adam", "learning_rate": .001, "weight_decay": 0.0},
    "adam_0003": {"optimizer": "adam", "learning_rate": .0003, "weight_decay": 0.0},
    "adamw_001": {"optimizer": "adamw", "learning_rate": .001, "weight_decay": .0001},
}


def contract(smoke=False):
    old = audit_contract(smoke)
    return {
        "experiment_id": 13, "experiment_name": "exp13_fhp_policy_capacity",
        "source_experiment": old["source_experiment"], "source_checkpoint": "time_24h",
        "source_replay_rows": 1_000_000,
        "seeds": [0] if smoke else list(SEEDS), "smoke": bool(smoke),
        "architectures": deepcopy(ARCHITECTURES), "recipes": deepcopy(RECIPES),
        "replicates": [0, 1], "sampler": "uniform",
        "control_recipe": "adam_003", "control_updates": [2, 4] if smoke else [20000, 60000],
        "screen_updates": [1, 2, 4] if smoke else [5000, 10000, 20000, 40000, 60000],
        "batch_size": old["batch_size"], "gamma": 2.0, "reset_each_path": True,
        "initialisation_seed_base": 202610130, "sampler_seed_base": 202610131,
        "split_seed": 20261013, "split_fractions": [.8, .1, .1],
        "split_strata": ["player", "round", "frequency_1_2to4_5plus"],
        "selection_metric": "validation.all.weighted_ce",
        "selection_rule": "equal source weight after averaging replicas; min validation CE; ties prefer fewer updates then recipe order",
        "test_access": "only fixed controls and globally locked selections after screen completion",
        "training_state_retention": "none", "inferential_unit": "source_training_seed",
        "exact_exploitability": False,
        **{k: old[k] for k in ("rule_deals", "direct_deals", "lbr_deals", "lbr_shard_deals", "lbr_rollouts")},
        "evaluation_seed_base": 202610132,
    }


def arm_name(architecture, budget, replicate):
    return f"{architecture}_{budget}_rep{replicate}"


def arm_specs(config, selection):
    """Named endpoints; tuned endpoints may alias an identical fixed control."""
    result = {}
    for architecture in config["architectures"]:
        for replicate in config["replicates"]:
            for updates in config["control_updates"]:
                result[arm_name(architecture, updates, replicate)] = {
                    "architecture": architecture, "replicate": replicate,
                    "recipe": config["control_recipe"], "updates": updates,
                }
            result[arm_name(architecture, "tuned", replicate)] = {
                "architecture": architecture, "replicate": replicate,
                **{k: selection["selected"][architecture][k] for k in ("recipe", "updates")},
            }
    return result
