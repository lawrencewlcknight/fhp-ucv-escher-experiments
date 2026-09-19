"""Compare consistently evaluated archived FHP UCV-ESCHER Experiments 1–4."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from fhp_escher.evaluation_adapter import _import_suite

_import_suite()

from fhp_evaluation.game import MILLI_BIG_BLINDS_PER_CHIP, load_fhp_game  # noqa: E402
from fhp_evaluation.loaders import load_checkpoint_policy, sha256_file  # noqa: E402
from fhp_evaluation.statistics import sample_summary  # noqa: E402

from experiments.fhp.exp1_archived_ucv_escher_baseline.evaluate_checkpoints import (  # noqa: E402
    _duplicate_values,
    _json_safe,
    _seed_pairs,
    _write_json,
)


EXPERIMENT_IDS = (1, 2, 3, 4)
CHECKPOINT_LABELS = ("checkpoint_6h", "checkpoint_12h")
DEFAULT_BASE_SEED = 20_260_830


def _discover_evaluations(root: Path) -> dict[int, dict]:
    found = {}
    for manifest_path in sorted(root.glob("exp*/exp*seed_0/evaluation_manifest.json")):
        evaluation = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_run = Path(evaluation["source_run"])
        run_manifest = json.loads((source_run / "run_manifest.json").read_text(encoding="utf-8"))
        experiment_id = int(run_manifest["experiment_id"])
        if experiment_id not in EXPERIMENT_IDS:
            continue
        if evaluation["status"] != "complete":
            raise ValueError(f"Incomplete evaluation: {manifest_path}")
        if experiment_id in found:
            raise ValueError(f"Multiple evaluated runs found for Experiment {experiment_id}")
        found[experiment_id] = {
            "directory": manifest_path.parent.resolve(),
            "evaluation": evaluation,
            "run_manifest": run_manifest,
            "summary": json.loads((source_run / "summary.json").read_text(encoding="utf-8")),
        }
    missing = set(EXPERIMENT_IDS) - set(found)
    if missing:
        raise ValueError(f"Missing completed evaluations: {sorted(missing)}")
    reference = found[1]["evaluation"]["evaluation"]
    matched = (
        "base_seed",
        "rule_deal_pairs_per_agent_per_checkpoint",
        "lbr_deal_pairs_per_checkpoint",
        "lbr_preflop_rollout_samples",
        "loose_aggressive_bands",
    )
    for experiment_id, item in found.items():
        current = item["evaluation"]["evaluation"]
        for field in matched:
            if current[field] != reference[field]:
                raise ValueError(
                    f"Experiment {experiment_id} evaluation differs in {field}"
                )
        if int(item["run_manifest"]["seed"]) != 0:
            raise ValueError("Cross-experiment comparison expects training seed 0")
    return found


def _cross_worker(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    game = load_fhp_game()
    policy_a = load_checkpoint_policy(game, task["policy_a"])
    policy_b = load_checkpoint_policy(game, task["policy_b"])
    chance_seeds, action_seeds = _seed_pairs(task["seed"], task["num_deals"])
    values = _duplicate_values(game, policy_a, policy_b, chance_seeds, action_seeds)
    return {**task, **values}


def _tasks(evaluations, *, total_deals: int, shard_deals: int, base_seed: int):
    tasks = []
    for checkpoint_index, checkpoint in enumerate(CHECKPOINT_LABELS):
        seed_start = int(base_seed) + 3_000_000 + checkpoint_index * 100_000
        for left in EXPERIMENT_IDS:
            for right in EXPERIMENT_IDS:
                if left >= right:
                    continue
                remaining = int(total_deals)
                shard_index = 0
                while remaining:
                    count = min(int(shard_deals), remaining)
                    tasks.append(
                        {
                            "checkpoint": checkpoint,
                            "experiment_a": left,
                            "experiment_b": right,
                            "policy_a": evaluations[left]["evaluation"]["checkpoints"][checkpoint]["path"],
                            "policy_b": evaluations[right]["evaluation"]["checkpoints"][checkpoint]["path"],
                            "num_deals": count,
                            # Every experiment pair receives the same deal schedule.
                            "seed": seed_start + shard_index,
                        }
                    )
                    remaining -= count
                    shard_index += 1
    return tasks


def _run_tasks(tasks, workers):
    print(
        f"Starting cross-experiment cross-play: {len(tasks)} shards on {workers} workers",
        flush=True,
    )
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_cross_worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed == 1 or completed % max(1, len(tasks) // 10) == 0:
                print(
                    f"Cross-experiment: {completed}/{len(tasks)} shards complete "
                    f"({time.monotonic() - started:.1f} seconds)",
                    flush=True,
                )
    print(f"Cross-experiment completed in {time.monotonic() - started:.1f} seconds")
    return results


def _combine_pair(results, *, checkpoint: str, left: int, right: int) -> dict:
    selected = sorted(
        (
            item
            for item in results
            if item["checkpoint"] == checkpoint
            and item["experiment_a"] == left
            and item["experiment_b"] == right
        ),
        key=lambda item: item["seed"],
    )
    paired = [value for item in selected for value in item["paired"]]
    player_zero = [value for item in selected for value in item["player_zero"]]
    player_one = [value for item in selected for value in item["player_one"]]
    summary = sample_summary(paired)
    return {
        "checkpoint": checkpoint,
        "policy_a": f"experiment_{left}",
        "policy_b": f"experiment_{right}",
        "interpretation": f"positive_favours_experiment_{left}",
        "num_deal_pairs": int(summary["n"]),
        "num_games": 2 * int(summary["n"]),
        "mean_chips_per_hand": summary["mean"],
        "se_chips_per_hand": summary["se"],
        "ci95_low_chips_per_hand": summary["ci95_low"],
        "ci95_high_chips_per_hand": summary["ci95_high"],
        "mean_mbb_per_hand": summary["mean"] * MILLI_BIG_BLINDS_PER_CHIP,
        "se_mbb_per_hand": summary["se"] * MILLI_BIG_BLINDS_PER_CHIP,
        "ci95_low_mbb_per_hand": summary["ci95_low"] * MILLI_BIG_BLINDS_PER_CHIP,
        "ci95_high_mbb_per_hand": summary["ci95_high"] * MILLI_BIG_BLINDS_PER_CHIP,
        "policy_a_player0_mean_chips": float(np.mean(player_zero)),
        "policy_a_player1_mean_chips": float(np.mean(player_one)),
        "shard_seeds": [int(item["seed"]) for item in selected],
    }


def _matrix(pair_results, checkpoint: str) -> list[dict]:
    lookup = {
        (int(item["policy_a"].split("_")[-1]), int(item["policy_b"].split("_")[-1])): item
        for item in pair_results
        if item["checkpoint"] == checkpoint
    }
    rows = []
    for row in EXPERIMENT_IDS:
        values = {"experiment": row}
        for column in EXPERIMENT_IDS:
            if row == column:
                value = 0.0
            elif row < column:
                value = lookup[(row, column)]["mean_mbb_per_hand"]
            else:
                value = -lookup[(column, row)]["mean_mbb_per_hand"]
            values[f"vs_experiment_{column}_mbb_per_hand"] = value
        rows.append(values)
    return rows


def _checkpoint_nodes(item: dict, target_seconds: float) -> tuple[int, int]:
    rows = item["evaluation"]["source_checkpoint_manifest"]["rows"]
    row = next(
        value
        for value in rows
        if np.isclose(float(value["checkpoint_target_seconds"]), target_seconds)
    )
    return int(row["nodes_touched"]), int(row["outer_iteration"])


def _evaluation_summary(evaluations) -> list[dict]:
    rows = []
    for experiment_id in EXPERIMENT_IDS:
        item = evaluations[experiment_id]
        for checkpoint, target_seconds in (("checkpoint_6h", 21600.0), ("checkpoint_12h", 43200.0)):
            directory = item["directory"]
            rules = json.loads((directory / checkpoint / "rule_agents.json").read_text())[
                "results"
            ]
            lbr = json.loads((directory / checkpoint / "lbr.json").read_text())
            nodes, iteration = _checkpoint_nodes(item, target_seconds)
            by_name = {result["policy_b"]: result for result in rules}
            row = {
                "experiment_id": experiment_id,
                "experiment_name": item["run_manifest"]["experiment_name"],
                "execution_backend": item["run_manifest"]["execution_backend"],
                "checkpoint": checkpoint,
                "training_hours": target_seconds / 3600.0,
                "nodes_touched": nodes,
                "outer_iteration": iteration,
            }
            for name, result in by_name.items():
                row[f"vs_{name}_mbb_per_hand"] = result["mean_mbb_per_hand"]
            row["rule_agent_mean_mbb_per_hand"] = float(
                np.mean([result["mean_mbb_per_hand"] for result in rules])
            )
            row["lbr_lower_bound_mbb_per_hand"] = lbr["mean_mbb_per_hand"]
            row["lbr_ci95_low_mbb_per_hand"] = lbr["ci95_low_mbb_per_hand"]
            row["lbr_ci95_high_mbb_per_hand"] = lbr["ci95_high_mbb_per_hand"]
            rows.append(row)
    return rows


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


def run_comparison(args) -> Path:
    root = args.evaluation_root.resolve()
    evaluations = _discover_evaluations(root)
    output = (args.output_dir or root / "cross_experiment_comparison" / "seed_0").resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "started_utc": started.isoformat(),
        "training_seed": 0,
        "base_seed": int(args.base_seed),
        "crossplay_deal_pairs_per_experiment_pair": int(args.deals),
        "checkpoint_labels": list(CHECKPOINT_LABELS),
        "experiments": {
            str(key): {
                "evaluation_directory": str(value["directory"]),
                "experiment_name": value["run_manifest"]["experiment_name"],
                "execution_backend": value["run_manifest"]["execution_backend"],
                "checkpoints": value["evaluation"]["checkpoints"],
            }
            for key, value in evaluations.items()
        },
    }
    _write_json(output / "comparison_manifest.json", manifest)
    tasks = _tasks(
        evaluations,
        total_deals=args.deals,
        shard_deals=args.shard_deals,
        base_seed=args.base_seed,
    )
    raw_results = _run_tasks(tasks, args.workers)
    pairs = [
        _combine_pair(raw_results, checkpoint=checkpoint, left=left, right=right)
        for checkpoint in CHECKPOINT_LABELS
        for left in EXPERIMENT_IDS
        for right in EXPERIMENT_IDS
        if left < right
    ]
    matrices = {checkpoint: _matrix(pairs, checkpoint) for checkpoint in CHECKPOINT_LABELS}
    summaries = _evaluation_summary(evaluations)
    _write_json(output / "crossplay_pair_results.json", {"results": pairs})
    _write_csv(output / "crossplay_pair_results.csv", pairs)
    for checkpoint, rows in matrices.items():
        _write_csv(output / f"{checkpoint}_crossplay_matrix.csv", rows)
    _write_csv(output / "evaluation_summary.csv", summaries)
    manifest.update(
        {
            "status": "complete",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "evaluation_summary": "evaluation_summary.csv",
                "pair_results_json": "crossplay_pair_results.json",
                "pair_results_csv": "crossplay_pair_results.csv",
                "checkpoint_6h_matrix": "checkpoint_6h_crossplay_matrix.csv",
                "checkpoint_12h_matrix": "checkpoint_12h_crossplay_matrix.csv",
            },
        }
    )
    _write_json(output / "comparison_manifest.json", manifest)
    return output


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--deals", type=int, default=50_000)
    parser.add_argument("--shard-deals", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if min(args.deals, args.shard_deals, args.workers) <= 0:
        raise ValueError("Deal counts, shard size, and workers must be positive")
    output = run_comparison(args)
    print(f"Cross-experiment comparison complete: {output}", flush=True)


if __name__ == "__main__":
    main()
