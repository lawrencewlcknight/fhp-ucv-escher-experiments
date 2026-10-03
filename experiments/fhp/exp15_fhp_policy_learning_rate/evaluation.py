"""Existing FHP evaluator and common deals; only the two final fits are compared."""
from pathlib import Path

from experiments.fhp.exp13_fhp_policy_capacity.evaluation import evaluate as evaluate_tasks
from fhp_escher.checkpointing import sha256_file
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES
from .config import arm_name


def policy_names(config):
    return ["archived", *[arm_name(recipe, rep) for recipe in config["recipes"] for rep in config["replicates"]]]


def expected_metrics(config):
    names = policy_names(config)
    expected = {(name, f"rule_{opponent}") for name in names for opponent in PUBLISHED_AGENT_NAMES}
    expected |= {(name, "direct_crossplay_archived") for name in names if name != "archived"}
    expected |= {(arm_name("adam_0003", rep), "direct_crossplay_matched_003") for rep in config["replicates"]}
    if config["lbr_deals"]:
        expected |= {(name, "lbr_mbb_per_hand") for name in names}
    return expected


def build_tasks(policies, config, seed):
    if set(policies) != set(policy_names(config)):
        raise ValueError("Incomplete policy set")
    hashes = {name: sha256_file(path) for name, path in policies.items()}
    tasks = []
    for name, path in policies.items():
        common = {"policy_a_path": str(Path(path).resolve()), "policy_a_name": name,
                  "source_seed": seed, "policy_a_sha256": hashes[name]}
        base = config["evaluation_seed_base"] + seed * 100000
        for j, opponent in enumerate(PUBLISHED_AGENT_NAMES):
            tasks.append({**common, "kind": "rule", "opponent": opponent,
                          "task_id": f"{name}_rule_{opponent}", "evaluation_seed": base + j,
                          "num_deals": config["rule_deals"]})
        for shard, start in enumerate(range(0, config["lbr_deals"], config["lbr_shard_deals"])):
            tasks.append({**common, "kind": "lbr", "task_id": f"{name}_lbr_{shard:04d}",
                          "evaluation_seed": base + 100 + shard,
                          "num_deals": min(config["lbr_shard_deals"], config["lbr_deals"] - start),
                          "lbr_seed": config["evaluation_seed_base"] + seed,
                          "lbr_rollouts": config["lbr_rollouts"]})
        if name == "archived":
            continue
        opponents = {"archived": "archived"}
        if name.startswith("adam_0003_"):
            opponents["matched_003"] = name.replace("adam_0003_", "adam_003_", 1)
        for label, opponent in opponents.items():
            tasks.append({**common, "kind": "direct_crossplay", "opponent": label,
                          "task_id": f"{name}_versus_{label}",
                          "policy_b_path": str(Path(policies[opponent]).resolve()),
                          "policy_b_name": opponent, "policy_b_sha256": hashes[opponent],
                          "evaluation_seed": base + 10000, "num_deals": config["direct_deals"]})
    return tasks


def evaluate(policies, config, seed, directory, workers=8, on_progress=None):
    return evaluate_tasks(policies, config, seed, directory, workers, on_progress, task_builder=build_tasks)
