"""Evaluate Experiment 1 and combine it with the frozen Experiment 2/3 audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform

import numpy as np

from experiments.fhp.retrospective_exp2_exp3_evaluation.run import (
    EXPECTED_HOURS,
    EXPECTED_SEEDS,
    _attach_mean_nodes,
    _average_rule_agents,
    _collapse_lbr_shards,
    _git_value,
    _group_aggregate,
    _run_tasks,
    _source_tree_sha256,
    _write_csv,
    _write_json,
)
from fhp_escher.checkpointing import sha256_file
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES


SCHEMA_VERSION = 1
EXPERIMENTS = {
    "exp1": {
        "experiment_name": "exp1_fhp_grouped_wide_ucv_baseline",
        "algorithm_id": "grouped_wide_ucv_escher",
        "label": "Experiment 1: grouped-wide raw",
    },
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
PAIR_DEFINITIONS = (
    ("exp1_vs_exp2", "exp1", "exp2"),
    ("exp1_vs_exp3", "exp1", "exp3"),
)
NODE_MATCHES = (
    ("exp1_vs_exp2", "exp1", 18, "exp2", 24),
    ("exp1_vs_exp3", "exp1", 12, "exp3", 18),
)
REFERENCE_FILES = (
    "evaluation_manifest.json",
    "rule_agent_by_seed.csv",
    "lbr_by_seed.csv",
    "temporal_crossplay_by_seed.csv",
    "direct_exp2_vs_exp3_by_seed.csv",
    "node_matched_exp2_vs_exp3_by_seed.csv",
)


def _checkpoint_path(worker: Path, row: dict) -> Path:
    declared = Path(str(row["path"]))
    for candidate in (worker / declared, worker / "checkpoints" / declared.name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(worker / declared)


def discover_checkpoints(run_root: Path, experiment: str) -> list[dict]:
    """Verify a complete three-seed, four-checkpoint source run."""
    contract = EXPERIMENTS[experiment]
    workers_root = Path(run_root).resolve() / "workers"
    if not workers_root.is_dir():
        raise FileNotFoundError(f"Missing workers directory: {workers_root}")
    records = []
    observed_seeds = set()
    for worker in sorted(path for path in workers_root.glob("task_*") if path.is_dir()):
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
        by_hour = {}
        for row in json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")):
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


_INT_FIELDS = {
    "training_seed", "training_hours", "earlier_hours", "later_hours",
    "exp1_hours", "exp2_hours", "exp3_hours", "num_deals", "evaluation_seed",
    "evaluation_seed_start", "num_shards", "lbr_seed", "lbr_rollouts",
    "num_deal_pairs", "num_games", "exp1_nodes_touched", "exp2_nodes_touched",
    "exp3_nodes_touched", "nodes_touched", "outer_iteration",
}
_FLOAT_FIELDS = {
    "mean_chips_per_hand", "std_chips_per_pair", "se_chips_per_hand",
    "ci95_low_chips_per_hand", "ci95_high_chips_per_hand", "mean_mbb_per_hand",
    "se_mbb_per_hand", "ci95_low_mbb_per_hand", "ci95_high_mbb_per_hand",
    "policy_a_player0_mean_chips", "policy_a_player1_mean_chips",
}


def _read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in _INT_FIELDS:
                    row[key] = int(value)
                elif key in _FLOAT_FIELDS:
                    row[key] = float(value)
                elif value in {"True", "False"}:
                    row[key] = value == "True"
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _validate_reference(reference: Path, records: list[dict], args) -> tuple[dict, dict]:
    reference = Path(reference).resolve()
    missing = [name for name in REFERENCE_FILES if not (reference / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Reference analysis is missing: {', '.join(missing)}")
    manifest = json.loads((reference / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("smoke"):
        raise ValueError("Experiment 2/3 reference must be a complete production analysis")
    expected = {
        "rule_deal_pairs_per_agent_per_checkpoint": int(args.rule_deals),
        "lbr_deal_pairs_per_checkpoint": int(args.lbr_deals),
        "lbr_preflop_rollout_samples": int(args.lbr_rollouts),
        "lbr_shard_deal_pairs": int(args.lbr_shard_deals),
        "crossplay_deal_pairs_per_match": int(args.crossplay_deals),
        "base_seed": int(args.base_seed),
        "exact_exploitability": False,
        "uncertainty_unit": "independent_training_seed_mean",
    }
    if not args.smoke:
        observed = manifest.get("evaluation", {})
        mismatches = {
            key: (observed.get(key), value)
            for key, value in expected.items()
            if observed.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Reference evaluation protocol mismatch: {mismatches}")
    current = {
        (row["experiment"], row["seed"], row["training_hours"]): (
            row["checkpoint_sha256"], row["nodes_touched"]
        )
        for row in records if row["experiment"] in {"exp2", "exp3"}
    }
    frozen = {
        (row["experiment"], int(row["seed"]), int(row["training_hours"])): (
            row["checkpoint_sha256"], int(row["nodes_touched"])
        )
        for row in manifest.get("checkpoints", [])
    }
    if current != frozen:
        raise ValueError("Current Experiment 2/3 checkpoints do not match the frozen reference")
    provenance = {
        "analysis_root": str(reference),
        "files": {name: sha256_file(reference / name) for name in REFERENCE_FILES},
        "reference_repository_commit": manifest.get("implementation", {}).get(
            "repository_commit"
        ),
    }
    return manifest, provenance


def _build_new_tasks(records: list[dict], args) -> list[dict]:
    tasks = []
    by_key = {
        (row["experiment"], row["seed"], row["training_hours"]): row for row in records
    }
    active_seeds = (0,) if args.smoke else EXPECTED_SEEDS
    active_hours = (6, 12) if args.smoke else EXPECTED_HOURS
    for seed in active_seeds:
        for hour in active_hours:
            row = by_key[("exp1", seed, hour)]
            identity = {
                "experiment": "exp1",
                "training_seed": seed,
                "training_hours": hour,
                "policy_a_path": row["checkpoint_path"],
                "policy_a_name": f"exp1_seed_{seed}_time_{hour:02d}h",
            }
            for opponent_index, opponent in enumerate(PUBLISHED_AGENT_NAMES):
                tasks.append({
                    **identity,
                    "task_id": f"rule_{identity['policy_a_name']}_{opponent}",
                    "kind": "rule",
                    "opponent": opponent,
                    "num_deals": int(args.rule_deals),
                    "evaluation_seed": int(args.base_seed + 100_000 + opponent_index),
                })
            remaining = int(args.lbr_deals)
            shard_index = 0
            while remaining:
                count = min(int(args.lbr_shard_deals), remaining)
                tasks.append({
                    **identity,
                    "task_id": f"lbr_{identity['policy_a_name']}_shard_{shard_index:04d}",
                    "kind": "lbr",
                    "opponent": "local_best_response",
                    "num_deals": count,
                    "shard_index": shard_index,
                    "evaluation_seed": int(args.base_seed + 1_000_000 + shard_index),
                    "lbr_seed": int(args.base_seed + 1_500_000),
                    "lbr_rollouts": int(args.lbr_rollouts),
                })
                remaining -= count
                shard_index += 1
        for earlier, later in itertools.combinations(active_hours, 2):
            earlier_row = by_key[("exp1", seed, earlier)]
            later_row = by_key[("exp1", seed, later)]
            tasks.append({
                "task_id": f"temporal_exp1_seed_{seed}_{later}h_vs_{earlier}h",
                "kind": "temporal_crossplay",
                "experiment": "exp1",
                "training_seed": seed,
                "training_hours": later,
                "earlier_hours": earlier,
                "later_hours": later,
                "policy_a_path": later_row["checkpoint_path"],
                "policy_b_path": earlier_row["checkpoint_path"],
                "policy_a_name": f"exp1_time_{later:02d}h",
                "policy_b_name": f"exp1_time_{earlier:02d}h",
                "num_deals": int(args.crossplay_deals),
                "evaluation_seed": int(args.base_seed + 2_000_000 + earlier * 1_000 + later),
            })
        for pair_id, left_experiment, right_experiment in PAIR_DEFINITIONS:
            for hour in active_hours:
                left = by_key[(left_experiment, seed, hour)]
                right = by_key[(right_experiment, seed, hour)]
                tasks.append({
                    "task_id": f"direct_{pair_id}_seed_{seed}_{hour}h",
                    "kind": "direct_crossplay",
                    "experiment": pair_id,
                    "pair_id": pair_id,
                    "policy_a_experiment": left_experiment,
                    "policy_b_experiment": right_experiment,
                    "training_seed": seed,
                    "training_hours": hour,
                    "policy_a_path": left["checkpoint_path"],
                    "policy_b_path": right["checkpoint_path"],
                    "policy_a_name": f"{left_experiment}_seed_{seed}_time_{hour:02d}h",
                    "policy_b_name": f"{right_experiment}_seed_{seed}_time_{hour:02d}h",
                    "num_deals": int(args.crossplay_deals),
                    # Reuse the frozen Exp2/3 schedule so all three pairs use common deals.
                    "evaluation_seed": int(args.base_seed + 3_000_000 + hour),
                })
        if not args.smoke:
            for match_index, (
                pair_id,
                left_exp,
                left_hour,
                right_exp,
                right_hour,
            ) in enumerate(NODE_MATCHES):
                left = by_key[(left_exp, seed, left_hour)]
                right = by_key[(right_exp, seed, right_hour)]
                tasks.append({
                    "task_id": f"node_matched_{pair_id}_{left_hour}h_vs_{right_hour}h_seed_{seed}",
                    "kind": "node_matched_crossplay",
                    "experiment": pair_id,
                    "pair_id": pair_id,
                    "policy_a_experiment": left_exp,
                    "policy_b_experiment": right_exp,
                    "training_seed": seed,
                    "training_hours": left_hour,
                    "policy_a_hours": left_hour,
                    "policy_b_hours": right_hour,
                    "policy_a_nodes_touched": left["nodes_touched"],
                    "policy_b_nodes_touched": right["nodes_touched"],
                    "policy_a_path": left["checkpoint_path"],
                    "policy_b_path": right["checkpoint_path"],
                    "policy_a_name": f"{left_exp}_seed_{seed}_time_{left_hour:02d}h",
                    "policy_b_name": f"{right_exp}_seed_{seed}_time_{right_hour:02d}h",
                    "num_deals": int(args.crossplay_deals),
                    "evaluation_seed": int(args.base_seed + 3_500_000 + match_index),
                })
    return tasks


def _normalise_reference_direct(rows: list[dict]) -> list[dict]:
    for row in rows:
        row.update(
            pair_id="exp2_vs_exp3",
            policy_a_experiment="exp2",
            policy_b_experiment="exp3",
        )
    return rows


def _normalise_reference_node(rows: list[dict]) -> list[dict]:
    for row in rows:
        row.update(
            pair_id="exp2_vs_exp3",
            policy_a_experiment="exp2",
            policy_b_experiment="exp3",
            policy_a_hours=row.pop("exp2_hours"),
            policy_b_hours=row.pop("exp3_hours"),
            policy_a_nodes_touched=row.pop("exp2_nodes_touched"),
            policy_b_nodes_touched=row.pop("exp3_nodes_touched"),
        )
    return rows


def _plot_analysis(output_dir: Path, rule_rows, lbr_rows, direct_rows, node_rows) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fhp-evaluation-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "exp1": ("#2ca02c", "^"),
        "exp2": ("#1f77b4", "o"),
        "exp3": ("#ff7f0e", "s"),
    }
    artifacts = []
    for filename, rows, xfield, xlabel, ylabel, title in (
        (
            "rule_agent_mean_by_time.png",
            rule_rows,
            "training_hours",
            "Effective training time (hours)",
            "Mean payoff (mbb/hand)",
            "Mean performance against five fixed rule agents",
        ),
        (
            "lbr_lower_bound_by_time.png",
            lbr_rows,
            "training_hours",
            "Effective training time (hours)",
            "LBR payoff (mbb/hand; lower is better)",
            "Local Best Response lower bound",
        ),
        (
            "rule_agent_mean_by_nodes.png",
            rule_rows,
            "mean_nodes_touched",
            "Mean nodes touched across training seeds",
            "Mean payoff (mbb/hand)",
            "Rule-agent performance by nodes touched",
        ),
        (
            "lbr_lower_bound_by_nodes.png",
            lbr_rows,
            "mean_nodes_touched",
            "Mean nodes touched across training seeds",
            "LBR payoff (mbb/hand; lower is better)",
            "Local Best Response lower bound by nodes touched",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for experiment in EXPERIMENTS:
            selected = sorted(
                (row for row in rows if row["experiment"] == experiment),
                key=lambda row: row[xfield],
            )
            colour, marker = style[experiment]
            ax.errorbar(
                [row[xfield] for row in selected],
                [row["mean_mbb_per_hand"] for row in selected],
                yerr=[row["training_seed_se_mbb_per_hand"] for row in selected],
                label=EXPERIMENTS[experiment]["label"], color=colour,
                marker=marker, capsize=4,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
        artifacts.append(filename)

    pair_colours = {
        "exp1_vs_exp2": "#9467bd",
        "exp1_vs_exp3": "#8c564b",
        "exp2_vs_exp3": "#17becf",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for pair_id in pair_colours:
        selected = sorted(
            (row for row in direct_rows if row["pair_id"] == pair_id),
            key=lambda row: row["training_hours"],
        )
        if not selected:
            continue
        label = (
            f"{selected[0]['policy_a_experiment'].upper()} minus "
            f"{selected[0]['policy_b_experiment'].upper()}"
        )
        ax.errorbar(
            [row["training_hours"] for row in selected],
            [row["mean_mbb_per_hand"] for row in selected],
            yerr=[row["training_seed_se_mbb_per_hand"] for row in selected],
            label=label, color=pair_colours[pair_id], marker="o", capsize=4,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Effective training time (hours)")
    ax.set_ylabel("Policy A minus policy B (mbb/hand)")
    ax.set_title("Matched-seed direct duplicate cross-play")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "direct_pairwise_crossplay_by_time.png", dpi=180)
    plt.close(fig)
    artifacts.append("direct_pairwise_crossplay_by_time.png")

    if node_rows:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        labels = [
            f"{row['policy_a_experiment'].upper()} {row['policy_a_hours']}h\n"
            f"vs {row['policy_b_experiment'].upper()} {row['policy_b_hours']}h"
            for row in node_rows
        ]
        x = np.arange(len(node_rows))
        ax.errorbar(
            x, [row["mean_mbb_per_hand"] for row in node_rows],
            yerr=[row["training_seed_se_mbb_per_hand"] for row in node_rows],
            fmt="o", capsize=5, color="#4c4c4c",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Policy A minus policy B (mbb/hand)")
        ax.set_title("Approximately node-matched direct cross-play")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "node_matched_pairwise_crossplay.png", dpi=180)
        plt.close(fig)
        artifacts.append("node_matched_pairwise_crossplay.png")
    return artifacts


def _summary_markdown(rule_rows, lbr_rows, direct_rows, node_rows) -> str:
    latest_rule = {row["experiment"]: row for row in rule_rows if row["training_hours"] == 24}
    latest_lbr = {row["experiment"]: row for row in lbr_rows if row["training_hours"] == 24}
    latest_direct = [row for row in direct_rows if row["training_hours"] == 24]
    lines = [
        "# Retrospective evaluation of FHP Experiments 1, 2 and 3",
        "",
        "Experiment 1 was evaluated under the frozen Experiment 2/3 protocol. "
        "The Experiment 2/3 rows are imported verbatim from the validated reference analysis; "
        "their checkpoint identities and evaluation settings were verified before combination.",
        "",
        "All aggregate confidence intervals use independently trained seed means. LBR is a "
        "legal one-step approximate response and therefore a lower bound on best-response "
        "value, not exact exploitability.",
        "",
        "## Twenty-four-hour policy-strength diagnostics",
        "",
        "| Configuration | Rule-agent mean | LBR lower bound |",
        "|---|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        if experiment in latest_rule and experiment in latest_lbr:
            lines.append(
                f"| {EXPERIMENTS[experiment]['label']} | "
                f"{latest_rule[experiment]['mean_mbb_per_hand']:.1f} | "
                f"{latest_lbr[experiment]['mean_mbb_per_hand']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Twenty-four-hour direct cross-play",
            "",
            "| Pair | Mean (mbb/hand) | Training-seed 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in sorted(latest_direct, key=lambda item: item["pair_id"]):
        lines.append(
            f"| {row['policy_a_experiment'].upper()} minus {row['policy_b_experiment'].upper()} | "
            f"{row['mean_mbb_per_hand']:.1f} | "
            f"[{row['training_seed_ci95_low_mbb_per_hand']:.1f}, "
            f"{row['training_seed_ci95_high_mbb_per_hand']:.1f}] |"
        )
    if node_rows:
        lines.extend(["", "## Approximately node-matched cross-play", ""])
        for row in node_rows:
            lines.append(
                f"- {row['policy_a_experiment'].upper()} at {row['policy_a_hours']} h minus "
                f"{row['policy_b_experiment'].upper()} at {row['policy_b_hours']} h: "
                f"{row['mean_mbb_per_hand']:.1f} mbb/hand."
            )
    lines.extend([
        "",
        "Rule-agent play, LBR and direct cross-play answer different questions and should "
        "be interpreted jointly. None is an exact exploitability calculation.",
        "",
    ])
    return "\n".join(lines)


def run_analysis(args) -> Path:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    records = []
    for experiment in EXPERIMENTS:
        records.extend(discover_checkpoints(getattr(args, f"{experiment}_run"), experiment))
    reference_root = Path(args.reference_analysis).resolve()
    reference_manifest, reference_provenance = _validate_reference(
        reference_root, records, args
    )
    tasks = _build_new_tasks(records, args)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "started_utc": started.isoformat(),
        "smoke": bool(args.smoke),
        "source_runs": {
            experiment: str(Path(getattr(args, f"{experiment}_run")).resolve())
            for experiment in EXPERIMENTS
        },
        "reference_exp2_exp3_analysis": reference_provenance,
        "reuse_policy": "exp2_exp3_results_imported_without_recalculation",
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
            "direct_crossplay_common_deals_across_pairs": True,
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
        "reference_evaluation_manifest_schema_version": reference_manifest.get("schema_version"),
    }
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    results = _run_tasks(tasks, int(args.workers))
    new_rule = [row for row in results if row["kind"] == "rule"]
    new_lbr = _collapse_lbr_shards([row for row in results if row["kind"] == "lbr"])
    new_temporal = [row for row in results if row["kind"] == "temporal_crossplay"]
    new_direct = [row for row in results if row["kind"] == "direct_crossplay"]
    new_node = [row for row in results if row["kind"] == "node_matched_crossplay"]

    rule_rows = _read_csv(reference_root / "rule_agent_by_seed.csv") + new_rule
    lbr_rows = _read_csv(reference_root / "lbr_by_seed.csv") + new_lbr
    temporal_rows = _read_csv(reference_root / "temporal_crossplay_by_seed.csv") + new_temporal
    direct_rows = _normalise_reference_direct(
        _read_csv(reference_root / "direct_exp2_vs_exp3_by_seed.csv")
    ) + new_direct
    node_rows = _normalise_reference_node(
        _read_csv(reference_root / "node_matched_exp2_vs_exp3_by_seed.csv")
    ) + new_node

    rule_mean = _average_rule_agents(rule_rows)
    aggregate_rule = _group_aggregate(rule_rows, ("experiment", "training_hours", "opponent"))
    aggregate_rule_mean = _group_aggregate(rule_mean, ("experiment", "training_hours"))
    aggregate_lbr = _group_aggregate(lbr_rows, ("experiment", "training_hours"))
    _attach_mean_nodes(aggregate_rule_mean, records)
    _attach_mean_nodes(aggregate_lbr, records)
    aggregate_temporal = _group_aggregate(
        temporal_rows, ("experiment", "earlier_hours", "later_hours")
    )
    aggregate_direct = _group_aggregate(
        direct_rows,
        ("pair_id", "policy_a_experiment", "policy_b_experiment", "training_hours"),
    )
    aggregate_node = _group_aggregate(
        node_rows,
        (
            "pair_id",
            "policy_a_experiment",
            "policy_b_experiment",
            "policy_a_hours",
            "policy_b_hours",
        ),
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
        "direct_pairwise_crossplay_by_seed.csv": direct_rows,
        "direct_pairwise_crossplay_aggregate.csv": aggregate_direct,
        "node_matched_pairwise_crossplay_by_seed.csv": node_rows,
        "node_matched_pairwise_crossplay_aggregate.csv": aggregate_node,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / name, rows)
    charts = _plot_analysis(
        output_dir, aggregate_rule_mean, aggregate_lbr, aggregate_direct, aggregate_node
    )
    (output_dir / "analysis_summary.md").write_text(
        _summary_markdown(aggregate_rule_mean, aggregate_lbr, aggregate_direct, aggregate_node),
        encoding="utf-8",
    )
    finished = datetime.now(timezone.utc)
    manifest.update(
        status="complete",
        finished_utc=finished.isoformat(),
        elapsed_seconds=(finished - started).total_seconds(),
        num_new_tasks=len(tasks),
        imported_reference_rows={
            "rule": len(rule_rows) - len(new_rule),
            "lbr": len(lbr_rows) - len(new_lbr),
            "temporal_crossplay": len(temporal_rows) - len(new_temporal),
            "direct_crossplay": len(direct_rows) - len(new_direct),
            "node_matched_crossplay": len(node_rows) - len(new_node),
        },
        artifacts=sorted([*tables, *charts, "analysis_summary.md"]),
    )
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    _write_json(output_dir / "SUCCESS.json", {"status": "complete", "smoke": bool(args.smoke)})
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp1-run", type=Path, required=True)
    parser.add_argument("--exp2-run", type=Path, required=True)
    parser.add_argument("--exp3-run", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
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
    print(f"Three-experiment retrospective evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
