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
        opponents = {"archived": "archived"}
        if name.startswith("wide_"):
            opponents["matched_standard"] = name.replace("wide_", "standard_", 1)
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


def evaluate(policies, config, seed, directory, workers=8, on_progress=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pending, rows = [], []
    for task in build_tasks(policies, config, seed):
        path = directory / f"{task['task_id']}.json"
        fingerprint = task_fingerprint(task)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError(f"Incompatible cached evaluation: {path}")
            validate_result(task, saved["result"])
            rows.append(saved["result"])
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

    if workers == 1:
        for task, path, fingerprint in pending:
            accept(_evaluation_worker(task), task, path, fingerprint)
    elif pending:
        # Spawn avoids inheriting replay allocations or a threaded Torch runtime.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_evaluation_worker, task): (task, path, fp) for task, path, fp in pending}
            for future in as_completed(futures):
                accept(future.result(), *futures[future])

    summaries = []
    for name in policies:
        lbr = sorted((r for r in rows if r["kind"] == "lbr" and r["policy_b"] == name),
                     key=lambda r: r["task_id"])
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
