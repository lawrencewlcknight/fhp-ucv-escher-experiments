"""Resumable sampled FHP evaluation; never reports exact exploitability."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from experiments.fhp.retrospective_exp2_exp3_evaluation.run import (
    _evaluation_worker, _match_summary,
)
from fhp_escher.checkpointing import sha256_file
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES
from .fitting import write_json
from .config import arm_name


def comparisons(config):
    return config.get("comparisons", [["wide", "standard"]])


def matched_opponents(name, config):
    return {f"matched_{b}": b + name[len(a):] for a, b in comparisons(config) if name.startswith(a + "_")}


def expected_metrics(config):
    arms = [arm_name(a, b, r) for a in config["architectures"]
            for b in [*config["control_updates"], "tuned"] for r in config["replicates"]]
    expected = {(arm, "lbr_mbb_per_hand") for arm in [*arms, "archived"]}
    expected |= {(arm, f"rule_{opponent}") for arm in [*arms, "archived"] for opponent in PUBLISHED_AGENT_NAMES}
    expected |= {(arm, "direct_crossplay_archived") for arm in arms}
    expected |= {(arm, f"direct_crossplay_{label}") for arm in arms for label in matched_opponents(arm, config)}
    return expected


def build_tasks(policies, config, seed):
    """Same random deals across arms; direct capacity contrasts match source/replicate."""
    tasks = []
    hashes = {name: sha256_file(path) for name, path in policies.items()}
    expected = {"archived"} | {arm_name(a, b, r) for a in config["architectures"]
        for b in [*config["control_updates"], "tuned"] for r in config["replicates"]}
    if set(policies) != expected:
        raise ValueError("Incomplete deployment policy set")
    for name, path in policies.items():
        common = {"policy_a_path": str(Path(path).resolve()), "policy_a_name": name,
                  "source_seed": seed, "policy_a_sha256": hashes[name]}
        for j, opponent in enumerate(PUBLISHED_AGENT_NAMES):
            tasks.append({**common, "kind": "rule", "opponent": opponent,
                          "task_id": f"{name}_rule_{opponent}",
                          "evaluation_seed": config["evaluation_seed_base"] + seed * 100_000 + j,
                          "num_deals": config["rule_deals"]})
        for shard, start in enumerate(range(0, config["lbr_deals"], config["lbr_shard_deals"])):
            tasks.append({**common, "kind": "lbr", "task_id": f"{name}_lbr_{shard:04d}",
                          "evaluation_seed": config["evaluation_seed_base"] + seed * 100_000 + 100 + shard,
                          "num_deals": min(config["lbr_shard_deals"], config["lbr_deals"] - start),
                          "lbr_seed": config["evaluation_seed_base"] + seed,
                          "lbr_rollouts": config["lbr_rollouts"]})
        if name == "archived":
            continue
        opponents = {"archived": "archived", **matched_opponents(name, config)}
        for label, opponent in opponents.items():
            tasks.append({**common, "kind": "direct_crossplay", "opponent": label,
                          "task_id": f"{name}_versus_{label}",
                          "policy_b_path": str(Path(policies[opponent]).resolve()),
                          "policy_b_name": opponent, "policy_b_sha256": hashes[opponent],
                          "evaluation_seed": config["evaluation_seed_base"] + seed * 100_000 + 10_000,
                          "num_deals": config["direct_deals"]})
    return tasks


def task_fingerprint(task):
    # Relocation is safe only because identities include checkpoint checksums.
    stable = {k: v for k, v in task.items() if not k.endswith("_path")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def validate_result(task, result):
    if (any(result.get(k) != v for k, v in task.items() if not k.endswith("_path"))
            or result.get("num_deal_pairs") != task["num_deals"]
            or result.get("num_games") != 2 * task["num_deals"]
            or not np.isfinite(result.get("mean_mbb_per_hand", np.nan))):
        raise ValueError("Evaluation result does not match its task identity or deal count")
    if task["kind"] == "lbr":
        for name in ("paired", "player_zero", "player_one"):
            values = np.asarray(result.get(f"_{name}_values", []))
            if values.shape != (task["num_deals"],) or not np.isfinite(values).all():
                raise ValueError("Incomplete or invalid LBR shard")


def evaluation_identity(task):
    """Identical policy bytes and randomness can share work despite arm aliases."""
    keys = ("kind", "source_seed", "policy_a_sha256", "policy_b_sha256",
            "evaluation_seed", "num_deals", "lbr_seed", "lbr_rollouts")
    value = {k: task.get(k) for k in keys}
    if task["kind"] == "rule":
        value["opponent"] = task["opponent"]
    return json.dumps(value, sort_keys=True)


def relabel_result(result, task):
    if task["kind"] == "lbr":
        a, b = "local_best_response", task["policy_a_name"]
    else:
        a = task["policy_a_name"]
        b = task["opponent"] if task["kind"] == "rule" else task["policy_b_name"]
    return {**result, **task, "policy_a": a, "policy_b": b}


def evaluate(policies, config, seed, directory, workers=8, on_progress=None, *, task_builder=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pending, rows, reusable = [], [], {}
    for task in (task_builder or build_tasks)(policies, config, seed):
        path = directory / f"{task['task_id']}.json"
        fingerprint = task_fingerprint(task)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError(f"Incompatible cached evaluation: {path}")
            validate_result(task, saved["result"])
            rows.append(saved["result"])
            reusable[evaluation_identity(task)] = saved["result"]
        else:
            pending.append((task, path, fingerprint))

    def accept(result, task, path, fingerprint):
        validate_result(task, result)
        write_json(path, {"fingerprint": fingerprint, "result": result})
        rows.append(result)
        if len(rows) % 20 == 0:
            print(f"Seed {seed}: completed {len(rows)} evaluation tasks", flush=True)
            if on_progress:
                on_progress()

    groups = {}
    cached_tasks = len(rows)
    deduplicate = config.get("deduplicate_identical_evaluation", False)
    for item in pending:
        task, path, fingerprint = item
        key = evaluation_identity(task) if deduplicate else task["task_id"]
        if deduplicate and key in reusable:
            accept(relabel_result(reusable[key], task), task, path, fingerprint)
        else:
            groups.setdefault(key, []).append(item)

    def accept_group(result, members):
        validate_result(members[0][0], result)
        for task, path, fingerprint in members:
            accept(relabel_result(result, task), task, path, fingerprint)

    if workers == 1:
        for members in groups.values():
            accept_group(_evaluation_worker(members[0][0]), members)
    elif groups:
        # Spawn avoids inheriting replay allocations or a threaded Torch runtime.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_evaluation_worker, members[0][0]): members for members in groups.values()}
            for future in as_completed(futures):
                accept_group(future.result(), futures[future])

    write_json(directory.parent / "evaluation_work.json", {
        "requested_tasks": cached_tasks + len(pending), "cached_tasks": cached_tasks,
        "executed_tasks_this_attempt": len(groups),
        "aliased_tasks_this_attempt": len(pending) - len(groups),
        "identical_policy_evaluation_deduplicated": deduplicate,
    })

    summaries = []
    for name in policies:
        lbr = sorted((r for r in rows if r["kind"] == "lbr" and r["policy_b"] == name),
                     key=lambda r: r["task_id"])
        if not lbr and config["lbr_deals"] == 0:
            continue
        values = {key: np.concatenate([r[f"_{key}_values"] for r in lbr])
                  for key in ("paired", "player_zero", "player_one")}
        summaries.append({"arm": name, "metric": "lbr_mbb_per_hand", "seed": seed,
                          **_match_summary(values, policy_a="local_best_response", policy_b=name)})
    for row in rows:
        if row["kind"] != "lbr":
            summaries.append({**row, "arm": row["policy_a_name"], "seed": seed,
                              "metric": f"{row['kind']}_{row['opponent']}"})
    write_json(directory.parent / "evaluation_summary.json", summaries)
    if on_progress:
        on_progress()
    return summaries
