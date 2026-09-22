"""Evaluate the frozen Experiment 2 and 3 FHP policies on one multi-core VM."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file
from fhp_evaluation.duplicate import play_hand
from fhp_evaluation.game import (
    FHP_GAME_PARAMETERS,
    MILLI_BIG_BLINDS_PER_CHIP,
    load_fhp_game,
)
from fhp_evaluation.lbr import LBRConfig, LocalBestResponsePolicy
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES, published_rule_agents
from fhp_evaluation.statistics import sample_summary


SCHEMA_VERSION = 1
EXPECTED_HOURS = (6, 12, 18, 24)
EXPECTED_SEEDS = (0, 1, 2)
EXPERIMENTS = {
    "exp2": {
        "experiment_name": "exp2_fhp_lossless_structured_ucv",
        "algorithm_id": "lossless_structured_ucv_escher",
        "label": "Experiment 2: structured",
    },
    "exp3": {
        "experiment_name": "exp3_fhp_wider_lossless_structured_ucv",
        "algorithm_id": "wider_lossless_structured_ucv_escher",
        "label": "Experiment 3: wider structured",
    },
}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_path(worker: Path, row: dict) -> Path:
    declared = Path(str(row["path"]))
    candidates = (worker / declared, worker / "checkpoints" / declared.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(candidates[0])


def discover_checkpoints(run_root: Path, experiment: str) -> list[dict]:
    """Validate and return the three-seed, four-checkpoint policy index."""
    contract = EXPERIMENTS[experiment]
    workers_root = Path(run_root).resolve() / "workers"
    if not workers_root.is_dir():
        raise FileNotFoundError(f"Missing workers directory: {workers_root}")
    records = []
    observed_seeds = set()
    workers = sorted(path for path in workers_root.glob("task_*") if path.is_dir())
    for worker in workers:
        run_manifest_path = worker / "run_manifest.json"
        checkpoint_manifest_path = worker / "checkpoint_manifest.json"
        success_path = worker / "SUCCESS.json"
        if not (run_manifest_path.is_file() and checkpoint_manifest_path.is_file()):
            continue
        if not success_path.is_file():
            raise FileNotFoundError(f"Worker is not marked successful: {worker}")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("experiment_name") != contract["experiment_name"]:
            continue
        if run_manifest.get("algorithm_id") != contract["algorithm_id"]:
            raise ValueError(f"Unexpected algorithm in {run_manifest_path}")
        seed = int(run_manifest["seed"])
        if seed in observed_seeds:
            raise ValueError(f"Duplicate {experiment} seed {seed}")
        rows = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if len(rows) != len(EXPECTED_HOURS):
            raise ValueError(f"Expected four checkpoints in {checkpoint_manifest_path}")
        by_hour = {}
        for row in rows:
            target_seconds = float(row["checkpoint_target_seconds"])
            hour = int(round(target_seconds / 3600.0))
            if hour not in EXPECTED_HOURS or not np.isclose(target_seconds, hour * 3600.0):
                raise ValueError(f"Unexpected checkpoint target in {checkpoint_manifest_path}")
            path = _checkpoint_path(worker, row)
            observed_hash = sha256_file(path)
            if observed_hash != row["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {path}")
            by_hour[hour] = {
                "experiment": experiment,
                "experiment_label": contract["label"],
                "seed": seed,
                "training_hours": hour,
                "checkpoint_id": str(row["checkpoint_id"]),
                "checkpoint_path": str(path),
                "checkpoint_sha256": observed_hash,
                "nodes_touched": int(row["nodes_touched"]),
                "outer_iteration": int(row["outer_iteration"]),
                "training_elapsed_seconds": float(
                    row.get("actual_training_elapsed_seconds", target_seconds)
                ),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            }
        if tuple(sorted(by_hour)) != EXPECTED_HOURS:
            raise ValueError(f"Incomplete checkpoint schedule in {checkpoint_manifest_path}")
        records.extend(by_hour[hour] for hour in EXPECTED_HOURS)
        observed_seeds.add(seed)
    if tuple(sorted(observed_seeds)) != EXPECTED_SEEDS:
        raise ValueError(
            f"{experiment} requires seeds {EXPECTED_SEEDS}; found {sorted(observed_seeds)}"
        )
    return sorted(records, key=lambda row: (row["seed"], row["training_hours"]))


def _seed_pairs(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return (
        rng.integers(0, 2**63 - 1, size=int(count), dtype=np.int64),
        rng.integers(0, 2**63 - 1, size=int(count), dtype=np.int64),
    )


def _duplicate_values(game, policy_a, policy_b, chance_seeds, action_seeds) -> dict:
    player_zero = np.empty(len(chance_seeds), dtype=np.float64)
    player_one = np.empty(len(chance_seeds), dtype=np.float64)
    for index, (chance_seed, action_seed) in enumerate(zip(chance_seeds, action_seeds)):
        return_zero, _ = play_hand(
            game,
            (policy_a, policy_b),
            chance_seed=int(chance_seed),
            action_seed=int(action_seed),
        )
        _, return_one = play_hand(
            game,
            (policy_b, policy_a),
            chance_seed=int(chance_seed),
            action_seed=int(action_seed),
        )
        player_zero[index] = return_zero
        player_one[index] = return_one
    return {
        "paired": 0.5 * (player_zero + player_one),
        "player_zero": player_zero,
        "player_one": player_one,
    }


def _match_summary(values: dict, *, policy_a: str, policy_b: str) -> dict:
    summary = sample_summary(values["paired"])
    scale = MILLI_BIG_BLINDS_PER_CHIP
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "num_deal_pairs": int(summary["n"]),
        "num_games": 2 * int(summary["n"]),
        "mean_chips_per_hand": summary["mean"],
        "std_chips_per_pair": summary["std"],
        "se_chips_per_hand": summary["se"],
        "ci95_low_chips_per_hand": summary["ci95_low"],
        "ci95_high_chips_per_hand": summary["ci95_high"],
        "mean_mbb_per_hand": summary["mean"] * scale,
        "se_mbb_per_hand": summary["se"] * scale,
        "ci95_low_mbb_per_hand": summary["ci95_low"] * scale,
        "ci95_high_mbb_per_hand": summary["ci95_high"] * scale,
        "policy_a_player0_mean_chips": float(np.mean(values["player_zero"])),
        "policy_a_player1_mean_chips": float(np.mean(values["player_one"])),
        "common_random_numbers_within_duplicate_pair": True,
    }


def _evaluation_worker(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    game = load_fhp_game()
    policy_a = LoadedFHPPolicy(game, task["policy_a_path"])
    kind = task["kind"]
    if kind == "rule":
        policy_b = published_rule_agents(game)[task["opponent"]]
        policy_a_name = task["policy_a_name"]
        policy_b_name = task["opponent"]
    elif kind == "lbr":
        target = policy_a
        policy_a = LocalBestResponsePolicy(
            game,
            target,
            config=LBRConfig(
                preflop_rollout_samples=int(task["lbr_rollouts"]),
                seed=int(task["lbr_seed"]),
            ),
        )
        policy_b = target
        policy_a_name = "local_best_response"
        policy_b_name = task["policy_a_name"]
    elif kind in {
        "temporal_crossplay",
        "direct_crossplay",
        "node_matched_crossplay",
    }:
        policy_b = LoadedFHPPolicy(game, task["policy_b_path"])
        policy_a_name = task["policy_a_name"]
        policy_b_name = task["policy_b_name"]
    else:  # pragma: no cover
        raise ValueError(f"Unknown evaluation kind: {kind}")
    chance_seeds, action_seeds = _seed_pairs(task["evaluation_seed"], task["num_deals"])
    values = _duplicate_values(game, policy_a, policy_b, chance_seeds, action_seeds)
    result = {
        **task,
        **_match_summary(values, policy_a=policy_a_name, policy_b=policy_b_name),
    }
    if kind == "lbr":
        result["_paired_values"] = values["paired"].tolist()
        result["_player_zero_values"] = values["player_zero"].tolist()
        result["_player_one_values"] = values["player_one"].tolist()
    return result


def _build_tasks(records: list[dict], args) -> list[dict]:
    tasks = []
    by_key = {
        (row["experiment"], row["seed"], row["training_hours"]): row for row in records
    }
    selected = records
    if args.smoke:
        selected = [
            row for row in records if row["seed"] == 0 and row["training_hours"] in (6, 12)
        ]
    for row in selected:
        identity = {
            "experiment": row["experiment"],
            "training_seed": row["seed"],
            "training_hours": row["training_hours"],
            "policy_a_path": row["checkpoint_path"],
            "policy_a_name": (
                f"{row['experiment']}_seed_{row['seed']}_time_{row['training_hours']:02d}h"
            ),
        }
        for opponent_index, opponent in enumerate(PUBLISHED_AGENT_NAMES):
            tasks.append(
                {
                    **identity,
                    "task_id": f"rule_{identity['policy_a_name']}_{opponent}",
                    "kind": "rule",
                    "opponent": opponent,
                    "num_deals": int(args.rule_deals),
                    "evaluation_seed": int(args.base_seed + 100_000 + opponent_index),
                }
            )
        remaining = int(args.lbr_deals)
        shard_index = 0
        while remaining:
            count = min(int(args.lbr_shard_deals), remaining)
            tasks.append(
                {
                    **identity,
                    "task_id": f"lbr_{identity['policy_a_name']}_shard_{shard_index:04d}",
                    "kind": "lbr",
                    "opponent": "local_best_response",
                    "num_deals": count,
                    "shard_index": shard_index,
                    "evaluation_seed": int(
                        args.base_seed + 1_000_000 + shard_index
                    ),
                    "lbr_seed": int(args.base_seed + 1_500_000),
                    "lbr_rollouts": int(args.lbr_rollouts),
                }
            )
            remaining -= count
            shard_index += 1
    active_seeds = (0,) if args.smoke else EXPECTED_SEEDS
    active_hours = (6, 12) if args.smoke else EXPECTED_HOURS
    for experiment in EXPERIMENTS:
        for seed in active_seeds:
            for earlier, later in itertools.combinations(active_hours, 2):
                later_row = by_key[(experiment, seed, later)]
                earlier_row = by_key[(experiment, seed, earlier)]
                tasks.append(
                    {
                        "task_id": f"temporal_{experiment}_seed_{seed}_{later}h_vs_{earlier}h",
                        "kind": "temporal_crossplay",
                        "experiment": experiment,
                        "training_seed": seed,
                        "training_hours": later,
                        "earlier_hours": earlier,
                        "later_hours": later,
                        "policy_a_path": later_row["checkpoint_path"],
                        "policy_b_path": earlier_row["checkpoint_path"],
                        "policy_a_name": f"{experiment}_time_{later:02d}h",
                        "policy_b_name": f"{experiment}_time_{earlier:02d}h",
                        "num_deals": int(args.crossplay_deals),
                        "evaluation_seed": int(
                            args.base_seed + 2_000_000 + earlier * 1_000 + later
                        ),
                    }
                )
    for seed in active_seeds:
        for hour in active_hours:
            left = by_key[("exp2", seed, hour)]
            right = by_key[("exp3", seed, hour)]
            tasks.append(
                {
                    "task_id": f"direct_exp2_vs_exp3_seed_{seed}_{hour}h",
                    "kind": "direct_crossplay",
                    "experiment": "exp2_vs_exp3",
                    "training_seed": seed,
                    "training_hours": hour,
                    "policy_a_path": left["checkpoint_path"],
                    "policy_b_path": right["checkpoint_path"],
                    "policy_a_name": f"exp2_seed_{seed}_time_{hour:02d}h",
                    "policy_b_name": f"exp3_seed_{seed}_time_{hour:02d}h",
                    "num_deals": int(args.crossplay_deals),
                    "evaluation_seed": int(args.base_seed + 3_000_000 + hour),
                }
            )
    if not args.smoke:
        for seed in active_seeds:
            left = by_key[("exp2", seed, 18)]
            right = by_key[("exp3", seed, 24)]
            tasks.append(
                {
                    "task_id": f"node_matched_exp2_18h_vs_exp3_24h_seed_{seed}",
                    "kind": "node_matched_crossplay",
                    "experiment": "exp2_vs_exp3",
                    "training_seed": seed,
                    "training_hours": 18,
                    "exp2_hours": 18,
                    "exp3_hours": 24,
                    "exp2_nodes_touched": left["nodes_touched"],
                    "exp3_nodes_touched": right["nodes_touched"],
                    "policy_a_path": left["checkpoint_path"],
                    "policy_b_path": right["checkpoint_path"],
                    "policy_a_name": f"exp2_seed_{seed}_time_18h",
                    "policy_b_name": f"exp3_seed_{seed}_time_24h",
                    "num_deals": int(args.crossplay_deals),
                    "evaluation_seed": int(args.base_seed + 3_500_000),
                }
            )
    return tasks


def _run_tasks(tasks: list[dict], workers: int) -> list[dict]:
    print(f"Starting {len(tasks)} evaluation tasks on {workers} workers", flush=True)
    started = time.monotonic()
    results = []
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            try:
                results.append(_evaluation_worker(task))
            except Exception as error:
                raise RuntimeError(f"Evaluation task failed: {task['task_id']}") from error
            if completed == 1 or completed % max(1, len(tasks) // 20) == 0:
                print(
                    f"Completed {completed}/{len(tasks)} tasks "
                    f"({time.monotonic() - started:.1f} seconds)",
                    flush=True,
                )
        return sorted(results, key=lambda row: row["task_id"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluation_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                raise RuntimeError(f"Evaluation task failed: {task['task_id']}") from error
            if completed == 1 or completed % max(1, len(tasks) // 20) == 0:
                print(
                    f"Completed {completed}/{len(tasks)} tasks "
                    f"({time.monotonic() - started:.1f} seconds)",
                    flush=True,
                )
    return sorted(results, key=lambda row: row["task_id"])


def _aggregate(values) -> dict:
    summary = sample_summary(values)
    return {
        "num_training_seeds": int(summary["n"]),
        "mean_mbb_per_hand": summary["mean"],
        "training_seed_std_mbb_per_hand": summary["std"],
        "training_seed_se_mbb_per_hand": summary["se"],
        "training_seed_ci95_low_mbb_per_hand": summary["ci95_low"],
        "training_seed_ci95_high_mbb_per_hand": summary["ci95_high"],
    }


def _collapse_lbr_shards(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["training_seed"], row["training_hours"])].append(row)
    collapsed = []
    for key, selected in sorted(grouped.items()):
        selected.sort(key=lambda row: row["shard_index"])
        values = {
            "paired": np.concatenate(
                [np.asarray(row["_paired_values"], dtype=float) for row in selected]
            ),
            "player_zero": np.concatenate(
                [np.asarray(row["_player_zero_values"], dtype=float) for row in selected]
            ),
            "player_one": np.concatenate(
                [np.asarray(row["_player_one_values"], dtype=float) for row in selected]
            ),
        }
        first = selected[0]
        collapsed.append(
            {
                "task_id": f"lbr_{key[0]}_seed_{key[1]}_time_{key[2]:02d}h",
                "kind": "lbr",
                "experiment": key[0],
                "training_seed": key[1],
                "training_hours": key[2],
                "opponent": "local_best_response",
                "evaluation_seed_start": first["evaluation_seed"],
                "num_shards": len(selected),
                "lbr_seed": first["lbr_seed"],
                "lbr_rollouts": first["lbr_rollouts"],
                **_match_summary(
                    values,
                    policy_a="local_best_response",
                    policy_b=first["policy_a_name"],
                ),
            }
        )
    return collapsed


def _group_aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        output.append(
            {
                **dict(zip(keys, key)),
                **_aggregate([row["mean_mbb_per_hand"] for row in selected]),
                "training_seeds": ",".join(
                    str(value) for value in sorted(row["training_seed"] for row in selected)
                ),
                "deal_pairs_per_training_seed": int(selected[0]["num_deal_pairs"]),
            }
        )
    return output


def _average_rule_agents(rule_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rule_rows:
        grouped[(row["experiment"], row["training_seed"], row["training_hours"])].append(row)
    return [
        {
            "experiment": key[0],
            "training_seed": key[1],
            "training_hours": key[2],
            "mean_mbb_per_hand": float(
                np.mean([row["mean_mbb_per_hand"] for row in values])
            ),
            "num_rule_agents": len(values),
            "num_deal_pairs": int(values[0]["num_deal_pairs"]),
        }
        for key, values in sorted(grouped.items())
    ]


def _attach_mean_nodes(rows: list[dict], records: list[dict]) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["experiment"], record["training_hours"])].append(
            record["nodes_touched"]
        )
    for row in rows:
        key = (row["experiment"], row["training_hours"])
        row["mean_nodes_touched"] = float(np.mean(grouped[key]))


def _plot_analysis(
    output_dir: Path, aggregate_rule_mean, aggregate_lbr, aggregate_direct
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fhp-evaluation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"exp2": ("#1f77b4", "o"), "exp3": ("#ff7f0e", "s")}
    metric_plots = (
        (
            "rule_agent_mean_by_time.png",
            aggregate_rule_mean,
            "Mean payoff (mbb/hand)",
            "Mean performance against five fixed rule agents",
        ),
        (
            "lbr_lower_bound_by_time.png",
            aggregate_lbr,
            "LBR payoff (mbb/hand; lower is better)",
            "Local Best Response lower bound",
        ),
    )
    for filename, rows, ylabel, title in metric_plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        for experiment in EXPERIMENTS:
            selected = sorted(
                (row for row in rows if row["experiment"] == experiment),
                key=lambda row: row["training_hours"],
            )
            x = [row["training_hours"] for row in selected]
            y = [row["mean_mbb_per_hand"] for row in selected]
            errors = [row["training_seed_se_mbb_per_hand"] for row in selected]
            colour, marker = style[experiment]
            ax.errorbar(
                x,
                y,
                yerr=errors,
                label=EXPERIMENTS[experiment]["label"],
                color=colour,
                marker=marker,
                capsize=4,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Effective training time (hours)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    for filename, rows, ylabel, title in (
        (
            "rule_agent_mean_by_nodes.png",
            aggregate_rule_mean,
            "Mean payoff (mbb/hand)",
            "Rule-agent performance by nodes touched",
        ),
        (
            "lbr_lower_bound_by_nodes.png",
            aggregate_lbr,
            "LBR payoff (mbb/hand; lower is better)",
            "Local Best Response lower bound by nodes touched",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for experiment in EXPERIMENTS:
            selected = sorted(
                (row for row in rows if row["experiment"] == experiment),
                key=lambda row: row["mean_nodes_touched"],
            )
            colour, marker = style[experiment]
            ax.errorbar(
                [row["mean_nodes_touched"] for row in selected],
                [row["mean_mbb_per_hand"] for row in selected],
                yerr=[row["training_seed_se_mbb_per_hand"] for row in selected],
                label=EXPERIMENTS[experiment]["label"],
                color=colour,
                marker=marker,
                capsize=4,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean nodes touched across training seeds")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = sorted(aggregate_direct, key=lambda row: row["training_hours"])
    ax.errorbar(
        [row["training_hours"] for row in rows],
        [row["mean_mbb_per_hand"] for row in rows],
        yerr=[row["training_seed_se_mbb_per_hand"] for row in rows],
        color="#2ca02c",
        marker="o",
        capsize=4,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Effective training time (hours)")
    ax.set_ylabel("Experiment 2 minus Experiment 3 (mbb/hand)")
    ax.set_title("Direct duplicate cross-play")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "direct_exp2_vs_exp3_by_time.png", dpi=180)
    plt.close(fig)


def _summary_markdown(aggregate_rule_mean, aggregate_lbr, aggregate_direct) -> str:
    latest_rule = {
        row["experiment"]: row
        for row in aggregate_rule_mean
        if row["training_hours"] == 24
    }
    latest_lbr = {row["experiment"]: row for row in aggregate_lbr if row["training_hours"] == 24}
    latest_direct = next((row for row in aggregate_direct if row["training_hours"] == 24), None)
    lines = [
        "# Retrospective evaluation of FHP Experiments 2 and 3",
        "",
        "All policy-strength intervals aggregate the independently trained seed means. "
        "Duplicate-deal intervals within each seed are retained in the detailed tables.",
        "",
        "Local Best Response is a legal one-step approximate response and is reported "
        "only as a lower bound on best-response value, not exact exploitability.",
    ]
    if latest_direct is not None:
        mean = latest_direct["mean_mbb_per_hand"]
        low = latest_direct["training_seed_ci95_low_mbb_per_hand"]
        high = latest_direct["training_seed_ci95_high_mbb_per_hand"]
        direction = "Experiment 2" if mean > 0 else "Experiment 3"
        lines.extend(
            [
                "",
                "## Twenty-four-hour comparison",
                "",
                f"Direct matched-seed cross-play favours {direction} by "
                f"{abs(mean):.1f} mbb/hand (training-seed 95% CI [{low:.1f}, {high:.1f}]; "
                "positive values favour Experiment 2).",
            ]
        )
    if latest_rule and latest_lbr:
        lines.extend(
            [
                "",
                "| Configuration | Rule-agent mean | LBR lower bound |",
                "|---|---:|---:|",
            ]
        )
        for experiment in EXPERIMENTS:
            rule = latest_rule[experiment]["mean_mbb_per_hand"]
            lbr = latest_lbr[experiment]["mean_mbb_per_hand"]
            lines.append(f"| {EXPERIMENTS[experiment]['label']} | {rule:.1f} | {lbr:.1f} |")
    lines.extend(
        [
            "",
            "The rule-agent benchmark, LBR and direct cross-play answer different "
            "questions and should be interpreted jointly. None is an exact "
            "exploitability calculation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(args) -> Path:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    records = discover_checkpoints(args.exp2_run, "exp2") + discover_checkpoints(
        args.exp3_run, "exp3"
    )
    tasks = _build_tasks(records, args)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "started_utc": started.isoformat(),
        "smoke": bool(args.smoke),
        "source_runs": {
            "exp2": str(Path(args.exp2_run).resolve()),
            "exp3": str(Path(args.exp3_run).resolve()),
        },
        "evaluation": {
            "rule_deal_pairs_per_agent_per_checkpoint": int(args.rule_deals),
            "lbr_deal_pairs_per_checkpoint": int(args.lbr_deals),
            "lbr_preflop_rollout_samples": int(args.lbr_rollouts),
            "lbr_shard_deal_pairs": int(args.lbr_shard_deals),
            "crossplay_deal_pairs_per_match": int(args.crossplay_deals),
            "base_seed": int(args.base_seed),
            "workers": int(args.workers),
            "exact_exploitability": False,
            "uncertainty_unit": "independent_training_seed_mean",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "implementation": {
            "repository_commit": _git_value(
                Path(__file__).resolve().parents[3], "rev-parse", "HEAD"
            ),
            "runner_sha256": sha256_file(Path(__file__)),
            "evaluation_suite": "vendored_validated_snapshot",
            "evaluation_suite_source_tree_sha256": _source_tree_sha256(
                Path(__file__).resolve().parents[3] / "fhp_evaluation"
            ),
        },
        "checkpoints": records,
    }
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    results = _run_tasks(tasks, int(args.workers))
    rule_rows = [row for row in results if row["kind"] == "rule"]
    lbr_rows = _collapse_lbr_shards([row for row in results if row["kind"] == "lbr"])
    temporal_rows = [row for row in results if row["kind"] == "temporal_crossplay"]
    direct_rows = [row for row in results if row["kind"] == "direct_crossplay"]
    node_matched_rows = [
        row for row in results if row["kind"] == "node_matched_crossplay"
    ]
    rule_mean = _average_rule_agents(rule_rows)
    aggregate_rule = _group_aggregate(
        rule_rows, ("experiment", "training_hours", "opponent")
    )
    aggregate_rule_mean = _group_aggregate(
        rule_mean, ("experiment", "training_hours")
    )
    aggregate_lbr = _group_aggregate(lbr_rows, ("experiment", "training_hours"))
    _attach_mean_nodes(aggregate_rule_mean, records)
    _attach_mean_nodes(aggregate_lbr, records)
    aggregate_temporal = _group_aggregate(
        temporal_rows, ("experiment", "earlier_hours", "later_hours")
    )
    aggregate_direct = _group_aggregate(direct_rows, ("training_hours",))
    aggregate_node_matched = (
        _group_aggregate(node_matched_rows, ("exp2_hours", "exp3_hours"))
        if node_matched_rows
        else []
    )
    tables = {
        "checkpoint_index.csv": records,
        "rule_agent_by_seed.csv": rule_rows,
        "rule_agent_aggregate.csv": aggregate_rule,
        "rule_agent_mean_by_seed.csv": rule_mean,
        "rule_agent_mean_aggregate.csv": aggregate_rule_mean,
        "lbr_by_seed.csv": lbr_rows,
        "lbr_aggregate.csv": aggregate_lbr,
        "temporal_crossplay_by_seed.csv": temporal_rows,
        "temporal_crossplay_aggregate.csv": aggregate_temporal,
        "direct_exp2_vs_exp3_by_seed.csv": direct_rows,
        "direct_exp2_vs_exp3_aggregate.csv": aggregate_direct,
        "node_matched_exp2_vs_exp3_by_seed.csv": node_matched_rows,
        "node_matched_exp2_vs_exp3_aggregate.csv": aggregate_node_matched,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / name, rows)
    _plot_analysis(output_dir, aggregate_rule_mean, aggregate_lbr, aggregate_direct)
    (output_dir / "analysis_summary.md").write_text(
        _summary_markdown(aggregate_rule_mean, aggregate_lbr, aggregate_direct),
        encoding="utf-8",
    )
    finished = datetime.now(timezone.utc)
    manifest.update(
        status="complete",
        finished_utc=finished.isoformat(),
        elapsed_seconds=(finished - started).total_seconds(),
        num_tasks=len(tasks),
        artifacts=sorted(
            [
                *tables,
                "analysis_summary.md",
                "rule_agent_mean_by_time.png",
                "lbr_lower_bound_by_time.png",
                "rule_agent_mean_by_nodes.png",
                "lbr_lower_bound_by_nodes.png",
                "direct_exp2_vs_exp3_by_time.png",
            ]
        ),
    )
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    _write_json(output_dir / "SUCCESS.json", {"status": "complete", "smoke": bool(args.smoke)})
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp2-run", type=Path, required=True)
    parser.add_argument("--exp3-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rule-deals", type=int, default=10_000)
    parser.add_argument("--lbr-deals", type=int, default=1_000)
    parser.add_argument("--lbr-rollouts", type=int, default=4_096)
    parser.add_argument("--lbr-shard-deals", type=int, default=10)
    parser.add_argument("--crossplay-deals", type=int, default=50_000)
    parser.add_argument("--base-seed", type=int, default=20_260_922)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    for name in (
        "workers",
        "rule_deals",
        "lbr_deals",
        "lbr_rollouts",
        "lbr_shard_deals",
        "crossplay_deals",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    output = run_analysis(args)
    print(f"Retrospective evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
