"""Evaluate verified 6 h and 12 h UCV-ESCHER policy checkpoints."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from fhp_escher.evaluation_adapter import _import_suite

_import_suite()

from fhp_evaluation.duplicate import play_hand  # noqa: E402
from fhp_evaluation.game import (  # noqa: E402
    FHP_GAME_PARAMETERS,
    MILLI_BIG_BLINDS_PER_CHIP,
    load_fhp_game,
)
from fhp_evaluation.lbr import LBRConfig, LocalBestResponsePolicy  # noqa: E402
from fhp_evaluation.loaders import load_checkpoint_policy, sha256_file  # noqa: E402
from fhp_evaluation.rule_agents import published_rule_agents  # noqa: E402
from fhp_evaluation.statistics import sample_summary  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_BASE_SEED = 20_260_830
CHECKPOINT_LABELS = ("checkpoint_6h", "checkpoint_12h")


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
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_verified_checkpoints(source_run: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = source_run / "checkpoint_manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(rows) != 2:
        raise ValueError(
            "Archived Experiment 1 evaluation requires exactly two checkpoints"
        )
    rows.sort(key=lambda row: float(row["checkpoint_target_seconds"]))
    expected_targets = (6 * 60 * 60, 12 * 60 * 60)
    checkpoints = {}
    for label, target, row in zip(CHECKPOINT_LABELS, expected_targets, rows):
        if not np.isclose(float(row["checkpoint_target_seconds"]), float(target)):
            raise ValueError(f"{label} has the wrong training-time target")
        path = source_run / "checkpoints" / Path(row["path"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {label}: {observed}")
        checkpoints[label] = path.resolve()
        row["resolved_path"] = str(path.resolve())
        row["verified_sha256"] = observed
    return {"path": str(manifest_path.resolve()), "rows": rows}, checkpoints


def _seed_pairs(seed: int, count: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(int(seed))
    chance = [int(value) for value in rng.integers(0, 2**63 - 1, size=count)]
    action = [int(value) for value in rng.integers(0, 2**63 - 1, size=count)]
    return chance, action


def _duplicate_values(game, policy_a, policy_b, chance_seeds, action_seeds):
    player_zero = []
    player_one = []
    paired = []
    for chance_seed, action_seed in zip(chance_seeds, action_seeds):
        return_zero, _ = play_hand(
            game,
            (policy_a, policy_b),
            chance_seed=chance_seed,
            action_seed=action_seed,
        )
        _, return_one = play_hand(
            game,
            (policy_b, policy_a),
            chance_seed=chance_seed,
            action_seed=action_seed,
        )
        player_zero.append(return_zero)
        player_one.append(return_one)
        paired.append(0.5 * (return_zero + return_one))
    return {"paired": paired, "player_zero": player_zero, "player_one": player_one}


def _worker(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    game = load_fhp_game()
    checkpoint_6h = load_checkpoint_policy(game, task["checkpoint_6h"])
    checkpoint_12h = load_checkpoint_policy(game, task["checkpoint_12h"])
    chance_seeds, action_seeds = _seed_pairs(task["seed"], task["num_deals"])

    if task["kind"] == "rule":
        opponents = published_rule_agents(game)
        name = task["opponent"]
        result_6h = _duplicate_values(
            game, checkpoint_6h, opponents[name], chance_seeds, action_seeds
        )
        result_12h = _duplicate_values(
            game, checkpoint_12h, opponents[name], chance_seeds, action_seeds
        )
    elif task["kind"] == "lbr":
        config = LBRConfig(
            preflop_rollout_samples=task["lbr_rollouts"],
            seed=task["lbr_seed"],
        )
        lbr_6h = LocalBestResponsePolicy(game, checkpoint_6h, config=config)
        lbr_12h = LocalBestResponsePolicy(game, checkpoint_12h, config=config)
        result_6h = _duplicate_values(
            game, lbr_6h, checkpoint_6h, chance_seeds, action_seeds
        )
        result_12h = _duplicate_values(
            game, lbr_12h, checkpoint_12h, chance_seeds, action_seeds
        )
    elif task["kind"] == "crossplay":
        values = _duplicate_values(
            game, checkpoint_12h, checkpoint_6h, chance_seeds, action_seeds
        )
        return {"kind": "crossplay", "seed": task["seed"], **values}
    else:
        raise ValueError(f"Unknown task kind: {task['kind']}")

    delta = [
        value_12h - value_6h
        for value_6h, value_12h in zip(result_6h["paired"], result_12h["paired"])
    ]
    return {
        "kind": task["kind"],
        "opponent": task.get("opponent"),
        "seed": task["seed"],
        "checkpoint_6h": result_6h,
        "checkpoint_12h": result_12h,
        "delta_12h_minus_6h": delta,
    }


def _shard_tasks(
    *,
    kind: str,
    total_deals: int,
    shard_deals: int,
    seed_start: int,
    checkpoint_6h: Path,
    checkpoint_12h: Path,
    opponent: str | None = None,
    lbr_rollouts: int | None = None,
    lbr_seed: int | None = None,
) -> list[dict]:
    tasks = []
    remaining = int(total_deals)
    index = 0
    while remaining:
        count = min(int(shard_deals), remaining)
        tasks.append(
            {
                "kind": kind,
                "num_deals": count,
                "seed": int(seed_start) + index,
                "checkpoint_6h": str(checkpoint_6h),
                "checkpoint_12h": str(checkpoint_12h),
                "opponent": opponent,
                "lbr_rollouts": lbr_rollouts,
                "lbr_seed": lbr_seed,
            }
        )
        remaining -= count
        index += 1
    return tasks


def _run_tasks(tasks: list[dict], workers: int, label: str) -> list[dict]:
    print(f"Starting {label}: {len(tasks)} shards on {workers} workers", flush=True)
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed == 1 or completed % max(1, len(tasks) // 10) == 0:
                elapsed = time.monotonic() - started
                print(
                    f"{label}: {completed}/{len(tasks)} shards complete "
                    f"({elapsed:.1f} seconds)",
                    flush=True,
                )
    print(f"Completed {label} in {time.monotonic() - started:.1f} seconds", flush=True)
    return results


def _concatenate(results, *keys):
    values = []
    for result in sorted(results, key=lambda item: item["seed"]):
        value = result
        for key in keys:
            value = value[key]
        values.extend(value)
    return values


def _duplicate_summary(
    paired,
    player_zero,
    player_one,
    *,
    policy_a: str,
    policy_b: str,
    seeds,
) -> dict:
    summary = sample_summary(paired)
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "num_deal_pairs": int(summary["n"]),
        "num_games": 2 * int(summary["n"]),
        "shard_seeds": sorted(int(seed) for seed in seeds),
        "mean_chips_per_hand": summary["mean"],
        "std_chips_per_pair": summary["std"],
        "se_chips_per_hand": summary["se"],
        "ci95_low_chips_per_hand": summary["ci95_low"],
        "ci95_high_chips_per_hand": summary["ci95_high"],
        "mean_mbb_per_hand": summary["mean"] * MILLI_BIG_BLINDS_PER_CHIP,
        "se_mbb_per_hand": summary["se"] * MILLI_BIG_BLINDS_PER_CHIP,
        "ci95_low_mbb_per_hand": summary["ci95_low"] * MILLI_BIG_BLINDS_PER_CHIP,
        "ci95_high_mbb_per_hand": summary["ci95_high"] * MILLI_BIG_BLINDS_PER_CHIP,
        "policy_a_player0_mean_chips": float(np.mean(player_zero)),
        "policy_a_player1_mean_chips": float(np.mean(player_one)),
        "common_random_numbers_within_duplicate_pair": True,
    }


def _delta_summary(values, *, interpretation: str, seeds) -> dict:
    summary = sample_summary(values)
    return {
        "metric": "paired_12h_minus_6h",
        "interpretation": interpretation,
        "num_paired_deals": int(summary["n"]),
        "shard_seeds": sorted(int(seed) for seed in seeds),
        "mean_delta_chips_per_hand": summary["mean"],
        "se_delta_chips_per_hand": summary["se"],
        "ci95_low_delta_chips_per_hand": summary["ci95_low"],
        "ci95_high_delta_chips_per_hand": summary["ci95_high"],
        "mean_delta_mbb_per_hand": summary["mean"] * MILLI_BIG_BLINDS_PER_CHIP,
        "se_delta_mbb_per_hand": summary["se"] * MILLI_BIG_BLINDS_PER_CHIP,
        "ci95_low_delta_mbb_per_hand": summary["ci95_low"] * MILLI_BIG_BLINDS_PER_CHIP,
        "ci95_high_delta_mbb_per_hand": summary["ci95_high"] * MILLI_BIG_BLINDS_PER_CHIP,
        "common_random_numbers_between_checkpoints": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rule-deals", type=int, default=10_000)
    parser.add_argument("--lbr-deals", type=int, default=1_000)
    parser.add_argument("--crossplay-deals", type=int, default=50_000)
    parser.add_argument("--lbr-rollouts", type=int, default=4_096)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rule-shard-deals", type=int, default=250)
    parser.add_argument("--lbr-shard-deals", type=int, default=10)
    parser.add_argument("--crossplay-shard-deals", type=int, default=1_000)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser


def run_evaluation(args) -> Path:
    source_run = args.source_run.resolve()
    run_manifest_payload = json.loads(
        (source_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    experiment_name = str(run_manifest_payload["experiment_name"])
    checkpoint_manifest, checkpoints = _load_verified_checkpoints(source_run)
    repository = Path(__file__).resolve().parents[3]
    suite_root = next(
        parent / "fhp-evaluation-suite"
        for parent in repository.parents
        if (parent / "fhp-evaluation-suite" / "fhp_evaluation").is_dir()
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            repository
            / "outputs"
            / "evaluation"
            / experiment_name
            / source_run.name
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    started_utc = datetime.now(timezone.utc)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "started_utc": started_utc.isoformat(),
        "experiment_id": int(run_manifest_payload["experiment_id"]),
        "experiment_name": experiment_name,
        "algorithm_id": str(run_manifest_payload["algorithm_id"]),
        "execution_backend": str(run_manifest_payload["execution_backend"]),
        "training_seed": int(run_manifest_payload["seed"]),
        "source_run": str(source_run),
        "source_run_manifest_sha256": sha256_file(source_run / "run_manifest.json"),
        "source_checkpoint_manifest": checkpoint_manifest,
        "checkpoints": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in checkpoints.items()
        },
        "game": {"name": "FHP", "parameters": dict(FHP_GAME_PARAMETERS)},
        "evaluation": {
            "rule_deal_pairs_per_agent_per_checkpoint": int(args.rule_deals),
            "lbr_deal_pairs_per_checkpoint": int(args.lbr_deals),
            "crossplay_deal_pairs": int(args.crossplay_deals),
            "lbr_preflop_rollout_samples": int(args.lbr_rollouts),
            "base_seed": int(args.base_seed),
            "workers": int(args.workers),
            "rule_shard_deals": int(args.rule_shard_deals),
            "lbr_shard_deals": int(args.lbr_shard_deals),
            "crossplay_shard_deals": int(args.crossplay_shard_deals),
            "loose_aggressive_bands": [-300.0, -100.0],
        },
        "implementation": {
            "ucv_repository_commit": _git_value(repository, "rev-parse", "HEAD"),
            "ucv_repository_dirty": bool(_git_value(repository, "status", "--porcelain")),
            "evaluation_suite_commit": _git_value(suite_root, "rev-parse", "HEAD"),
            "evaluation_suite_source_tree_sha256": _source_tree_sha256(suite_root),
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    _write_json(output_dir / "evaluation_manifest.json", manifest)

    opponents = list(published_rule_agents(load_fhp_game()))
    rule_tasks = []
    for index, opponent in enumerate(opponents):
        rule_tasks.extend(
            _shard_tasks(
                kind="rule",
                total_deals=args.rule_deals,
                shard_deals=args.rule_shard_deals,
                seed_start=args.base_seed + 100_000 + index * 10_000,
                checkpoint_6h=checkpoints["checkpoint_6h"],
                checkpoint_12h=checkpoints["checkpoint_12h"],
                opponent=opponent,
            )
        )
    rule_results = _run_tasks(rule_tasks, args.workers, "rule-agent evaluation")

    lbr_tasks = _shard_tasks(
        kind="lbr",
        total_deals=args.lbr_deals,
        shard_deals=args.lbr_shard_deals,
        seed_start=args.base_seed + 1_000_000,
        checkpoint_6h=checkpoints["checkpoint_6h"],
        checkpoint_12h=checkpoints["checkpoint_12h"],
        lbr_rollouts=args.lbr_rollouts,
        lbr_seed=args.base_seed + 1_500_000,
    )
    lbr_results = _run_tasks(lbr_tasks, args.workers, "LBR evaluation")

    cross_tasks = _shard_tasks(
        kind="crossplay",
        total_deals=args.crossplay_deals,
        shard_deals=args.crossplay_shard_deals,
        seed_start=args.base_seed + 2_000_000,
        checkpoint_6h=checkpoints["checkpoint_6h"],
        checkpoint_12h=checkpoints["checkpoint_12h"],
    )
    cross_results = _run_tasks(cross_tasks, args.workers, "checkpoint cross-play")

    rule_payloads = {label: [] for label in CHECKPOINT_LABELS}
    temporal_rule = []
    for opponent in opponents:
        selected = [result for result in rule_results if result["opponent"] == opponent]
        seeds = [result["seed"] for result in selected]
        for label in CHECKPOINT_LABELS:
            rule_payloads[label].append(
                _duplicate_summary(
                    _concatenate(selected, label, "paired"),
                    _concatenate(selected, label, "player_zero"),
                    _concatenate(selected, label, "player_one"),
                    policy_a=label,
                    policy_b=opponent,
                    seeds=seeds,
                )
            )
        temporal_rule.append(
            {
                "opponent": opponent,
                **_delta_summary(
                    _concatenate(selected, "delta_12h_minus_6h"),
                    interpretation="positive_favours_12h_against_this_rule_agent",
                    seeds=seeds,
                ),
            }
        )

    lbr_seeds = [result["seed"] for result in lbr_results]
    lbr_payloads = {}
    for label in CHECKPOINT_LABELS:
        lbr_payloads[label] = {
            "schema_version": SCHEMA_VERSION,
            "metric": "lbr_lower_bound",
            "interpretation": "lower_bound_on_best_response_value_not_exact_exploitability",
            "lbr_config": {
                "preflop_rollout_samples": int(args.lbr_rollouts),
                "seed": int(args.base_seed + 1_500_000),
            },
            **_duplicate_summary(
                _concatenate(lbr_results, label, "paired"),
                _concatenate(lbr_results, label, "player_zero"),
                _concatenate(lbr_results, label, "player_one"),
                policy_a="local_best_response",
                policy_b=label,
                seeds=lbr_seeds,
            ),
        }

    temporal_lbr = _delta_summary(
        _concatenate(lbr_results, "delta_12h_minus_6h"),
        interpretation="negative_favours_12h_because_lower_lbr_payoff_is_better",
        seeds=lbr_seeds,
    )
    cross_seeds = [result["seed"] for result in cross_results]
    cross_payload = _duplicate_summary(
        _concatenate(cross_results, "paired"),
        _concatenate(cross_results, "player_zero"),
        _concatenate(cross_results, "player_one"),
        policy_a="checkpoint_12h",
        policy_b="checkpoint_6h",
        seeds=cross_seeds,
    )
    cross_payload["interpretation"] = "positive_favours_checkpoint_12h"

    for label in CHECKPOINT_LABELS:
        _write_json(
            output_dir / label / "rule_agents.json",
            {
                "schema_version": SCHEMA_VERSION,
                "suite": "five_published_rule_agents_corrected",
                "checkpoint": manifest["checkpoints"][label],
                "results": rule_payloads[label],
            },
        )
        _write_json(output_dir / label / "lbr.json", lbr_payloads[label])
    _write_json(
        output_dir / "checkpoint_6h_vs_12h" / "duplicate_crossplay.json", cross_payload
    )
    _write_json(
        output_dir / "checkpoint_6h_vs_12h" / "paired_temporal_comparison.json",
        {"rule_agents": temporal_rule, "lbr": temporal_lbr},
    )

    summary_rows = []
    for label in CHECKPOINT_LABELS:
        for result in rule_payloads[label]:
            summary_rows.append(
                {
                    "checkpoint": label,
                    "metric": "duplicate_vs_rule_agent",
                    "opponent": result["policy_b"],
                    **{
                        key: result[key]
                        for key in (
                            "num_deal_pairs",
                            "mean_chips_per_hand",
                            "se_chips_per_hand",
                            "ci95_low_chips_per_hand",
                            "ci95_high_chips_per_hand",
                            "mean_mbb_per_hand",
                        )
                    },
                }
            )
        lbr_result = lbr_payloads[label]
        summary_rows.append(
            {
                "checkpoint": label,
                "metric": "lbr_lower_bound",
                "opponent": "local_best_response",
                **{
                    key: lbr_result[key]
                    for key in (
                        "num_deal_pairs",
                        "mean_chips_per_hand",
                        "se_chips_per_hand",
                        "ci95_low_chips_per_hand",
                        "ci95_high_chips_per_hand",
                        "mean_mbb_per_hand",
                    )
                },
            }
        )
    summary_rows.append(
        {
            "checkpoint": "checkpoint_12h_minus_checkpoint_6h",
            "metric": "direct_duplicate_crossplay",
            "opponent": "checkpoint_6h",
            **{
                key: cross_payload[key]
                for key in (
                    "num_deal_pairs",
                    "mean_chips_per_hand",
                    "se_chips_per_hand",
                    "ci95_low_chips_per_hand",
                    "ci95_high_chips_per_hand",
                    "mean_mbb_per_hand",
                )
            },
        }
    )
    _write_csv(output_dir / "checkpoint_summary.csv", summary_rows)

    finished_utc = datetime.now(timezone.utc)
    manifest.update(
        {
            "status": "complete",
            "finished_utc": finished_utc.isoformat(),
            "elapsed_seconds": (finished_utc - started_utc).total_seconds(),
            "artifacts": {
                "checkpoint_summary": "checkpoint_summary.csv",
                "checkpoint_6h_rule_agents": "checkpoint_6h/rule_agents.json",
                "checkpoint_6h_lbr": "checkpoint_6h/lbr.json",
                "checkpoint_12h_rule_agents": "checkpoint_12h/rule_agents.json",
                "checkpoint_12h_lbr": "checkpoint_12h/lbr.json",
                "direct_crossplay": "checkpoint_6h_vs_12h/duplicate_crossplay.json",
                "paired_temporal_comparison": (
                    "checkpoint_6h_vs_12h/paired_temporal_comparison.json"
                ),
            },
        }
    )
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    return output_dir


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    for name in (
        "rule_deals",
        "lbr_deals",
        "crossplay_deals",
        "lbr_rollouts",
        "workers",
        "rule_shard_deals",
        "lbr_shard_deals",
        "crossplay_shard_deals",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    output = run_evaluation(args)
    print(f"Evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
