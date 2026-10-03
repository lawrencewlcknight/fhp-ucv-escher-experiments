"""Evaluate Exp9--12 saved policies; never resume or modify model training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import platform
import re

import numpy as np

from experiments.fhp.retrospective_exp2_exp3_evaluation import run as protocol
from experiments.fhp.retrospective_exp6_exp7_evaluation import run as shared
from experiments.fhp.exp9_fhp_cached_parallel_24h import config as exp9_config
from experiments.fhp.exp10_fhp_hand_board_features import config as exp10_config
from experiments.fhp.exp11_fhp_betting_economics import config as exp11_config
from experiments.fhp.exp12_fhp_critic_showdown import config as exp12_config
from fhp_escher.features import ENCODER_ID, make_feature_encoder
from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file

EXPECTED_HOURS = protocol.EXPECTED_HOURS
EXPECTED_SEEDS = protocol.EXPECTED_SEEDS
CONFIGS = {"exp9": exp9_config, "exp10": exp10_config,
           "exp11": exp11_config, "exp12": exp12_config}
LABELS = {"exp9": "Exp9: cached baseline", "exp10": "Exp10: hand/board features",
          "exp11": "Exp11: betting economics", "exp12": "Exp12: critic showdown"}
EXPERIMENTS = {key: {"experiment_name": config.EXPERIMENT_NAME,
                    "algorithm_id": config.ALGORITHM_ID, "label": LABELS[key]}
               for key, config in CONFIGS.items()}
# Newer numbered experiment is policy A: positive direct scores favour A.
DIRECT_PAIRS = tuple((right, left) for left, right in itertools.combinations(CONFIGS, 2))
PRIMARY_PAIRS = tuple(pair for pair in DIRECT_PAIRS if pair[1] == "exp9")


_digest = shared._digest
_write_json = shared._write_json
_portable_task = shared._portable_task
_run_tasks = shared._run_tasks
_temporal_summary = shared._temporal_summary


def discover_checkpoints(root: Path, experiment: str) -> list[dict]:
    """Require complete real runs, contained checkpoint paths and matching payloads."""
    root = Path(root).resolve()
    config = CONFIGS[experiment]
    manifests = {}
    for worker in sorted((root / "workers").glob("task_*")):
        if not worker.resolve().is_relative_to(root):
            raise ValueError(f"Worker escapes source root: {worker}")
        path = worker / "run_manifest.json"
        if not path.exists():
            raise ValueError(f"Incomplete source worker: {worker}")
        manifest = json.loads(path.read_text())
        if manifest.get("experiment_name") != EXPERIMENTS[experiment]["experiment_name"]:
            raise ValueError(f"Wrong experiment in {path}")
        if manifest.get("smoke") is not False:
            raise ValueError(f"Not a production training run: {worker}")
        if protocol._json_safe(manifest.get("training_config")) != protocol._json_safe(config.EXPERIMENT_CONFIG):
            raise ValueError(f"Unexpected {experiment} learning configuration: {worker}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("repository_commit", ""))):
            raise ValueError(f"Missing or invalid source commit: {worker}")
        success = json.loads((worker / "SUCCESS.json").read_text())
        if success.get("status") != "complete":
            raise ValueError(f"Unsuccessful source worker: {worker}")
        runtime = json.loads((worker / "runtime_manifest.json").read_text())
        if runtime["reference_vm"]["machine_type"] != "n2-standard-16":
            raise ValueError(f"Wrong source VM: {worker}")
        if (runtime["torch_intraop_threads"] != 8
                or runtime["frozen_critic_target_cache"] is not True):
            raise ValueError(f"Wrong source runtime: {worker}")
        if runtime["traversal_execution"] != "ray_parallel":
            raise ValueError(f"Wrong source collector: {worker}")
        if runtime["parallel_settings"] != config.parallel_settings():
            raise ValueError(f"Wrong parallel settings: {worker}")
        encoder = make_feature_encoder(config.EXPERIMENT_CONFIG.get("feature_encoder_id", ENCODER_ID))
        if experiment != "exp9" and runtime.get("feature_encoder") != encoder.metadata():
            raise ValueError(f"Wrong runtime feature encoder: {worker}")
        for row in json.loads((worker / "checkpoint_manifest.json").read_text()):
            resolved = protocol._checkpoint_path(worker, row)
            if not resolved.is_relative_to(worker.resolve()):
                raise ValueError(f"Checkpoint escapes source worker: {resolved}")
        manifests[int(manifest["seed"])] = (manifest, runtime)
    records = protocol.discover_checkpoints(root, experiment, contracts=EXPERIMENTS)
    game = protocol.load_fhp_game()
    for record in records:
        manifest, runtime = manifests[record["seed"]]
        restored = LoadedFHPPolicy(game, record["checkpoint_path"])
        payload = restored.checkpoint
        expected = {
            "experiment_name": EXPERIMENTS[experiment]["experiment_name"],
            "algorithm_id": EXPERIMENTS[experiment]["algorithm_id"],
            "seed": record["seed"], "nodes_touched": record["nodes_touched"],
            "outer_iteration": record["outer_iteration"],
            "checkpoint_target_seconds": record["training_hours"] * 3600,
            "training_config": manifest["training_config"],
            "feature_encoder": make_feature_encoder(
                config.EXPERIMENT_CONFIG.get("feature_encoder_id", ENCODER_ID)).metadata(),
        }
        # Pickle retains tuples; JSON manifests encode the same sequences as lists.
        mismatches = [key for key, value in expected.items()
                      if protocol._json_safe(payload.get(key)) != protocol._json_safe(value)]
        if mismatches:
            raise ValueError(f"Checkpoint metadata mismatch ({', '.join(mismatches)}): {record['checkpoint_path']}")
        if (not np.isfinite(record["training_elapsed_seconds"])
                or record["training_elapsed_seconds"] < record["training_hours"] * 3600):
            raise ValueError("Checkpoint precedes its training threshold")
        if record["nodes_touched"] <= 0 or record["outer_iteration"] <= 0:
            raise ValueError("Checkpoint has no completed training")
        record.update(
            training_config_sha256=_digest(manifest["training_config"]),
            learning_config_without_encoder_sha256=_digest({
                k: v for k, v in manifest["training_config"].items() if k != "feature_encoder_id"
            }),
            source_repository_commit=manifest["repository_commit"],
            source_runtime=runtime,
        )
    return records



def _build_tasks(records, args):
    tasks = protocol._build_tasks(records, args, experiments=EXPERIMENTS,
                                  direct_pair=DIRECT_PAIRS[0], node_hours=None)
    by_key = {(r["experiment"], r["seed"], r["training_hours"]): r for r in records}
    # Reuse the established direct-play schedule for every unordered pair.
    templates = [task for task in tasks if task["kind"] == "direct_crossplay"]
    for left, right in DIRECT_PAIRS[1:]:
        for template in templates:
            seed, hour = template["training_seed"], template["training_hours"]
            pair_id = f"{left}_vs_{right}"
            tasks.append({
                **template, "task_id": f"direct_{pair_id}_seed_{seed}_{hour}h",
                "experiment": pair_id,
                "policy_a_path": by_key[(left, seed, hour)]["checkpoint_path"],
                "policy_b_path": by_key[(right, seed, hour)]["checkpoint_path"],
                "policy_a_name": f"{left}_seed_{seed}_time_{hour:02d}h",
                "policy_b_name": f"{right}_seed_{seed}_time_{hour:02d}h",
            })
    by_path = {r["checkpoint_path"]: r for r in records}
    for task in tasks:
        for side in ("a", "b"):
            key = f"policy_{side}_path"
            if key in task:
                record = by_path[task[key]]
                task[f"policy_{side}_sha256"] = record["checkpoint_sha256"]
                task[f"policy_{side}_nodes_touched"] = record["nodes_touched"]
        if task["kind"] == "direct_crossplay":
            left, right = task["experiment"].split("_vs_")
            task.update(pair_id=task["experiment"], left_experiment=left, right_experiment=right,
                        primary_endpoint=task["training_hours"] == 24 and (left, right) in PRIMARY_PAIRS)
    return tasks


def _paired_differences(rows, metric):
    by_key = {(r["experiment"], r["training_seed"], r["training_hours"]): r for r in rows}
    output = []
    for left, right in DIRECT_PAIRS:
        for (experiment, seed, hour), row in sorted(by_key.items()):
            if experiment != left:
                continue
            baseline = by_key[(right, seed, hour)]
            output.append({
                "metric": metric, "pair_id": f"{left}_vs_{right}",
                "left_experiment": left, "right_experiment": right,
                "training_seed": seed, "training_hours": hour,
                "mean_mbb_per_hand": row["mean_mbb_per_hand"] - baseline["mean_mbb_per_hand"],
                "num_deal_pairs": row["num_deal_pairs"],
            })
    return output


def _plot_analysis(output, rule, lbr, direct, temporal, *, smoke=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fhp-eval9to12-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        if smoke:
            fig.suptitle("SMOKE TEST — integration only; not policy-strength evidence", fontsize=11)
        fig.savefig(output / name, dpi=180)
        plt.close(fig)

    def trajectory(ax, selected, xkey, **kwargs):
        selected = sorted(selected, key=lambda r: r[xkey])
        scale = 1e6 if xkey == "mean_nodes_touched" else 1
        x = np.array([r[xkey] / scale for r in selected])
        y = np.array([r["mean_mbb_per_hand"] for r in selected])
        low = np.array([r["training_seed_ci95_low_mbb_per_hand"] for r in selected])
        high = np.array([r["training_seed_ci95_high_mbb_per_hand"] for r in selected])
        # Smoke has one seed: missing uncertainty is not a zero-width interval.
        ax.plot(x, y, marker="o", **kwargs)
        finite = np.isfinite(low) & np.isfinite(high)
        ax.fill_between(x, low, high, where=finite, alpha=.12)
        ax.grid(alpha=.2)

    figures = []
    for name, rows, ylabel in (("rule_agent", rule, "Policy payoff (mbb/hand; higher is better)"),
                                ("lbr", lbr, "LBR payoff (mbb/hand; lower is better)")):
        for axis, xkey in (("time", "training_hours"), ("nodes", "mean_nodes_touched")):
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for experiment in EXPERIMENTS:
                trajectory(ax, [r for r in rows if r["experiment"] == experiment],
                           xkey, label=LABELS[experiment])
            ax.set(xlabel="Active training hours" if axis == "time" else "Mean training nodes (millions)",
                   ylabel=ylabel, title="Frozen-policy evaluation (exploratory seed-level 95% intervals)")
            ax.legend(fontsize=8)
            figures.append(f"{name}_by_{axis}.png")
            save(fig, figures[-1])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (left, right) in zip(axes.flat, DIRECT_PAIRS):
        trajectory(ax, [r for r in direct if r["pair_id"] == f"{left}_vs_{right}"], "training_hours")
        ax.axhline(0, color="grey", linewidth=.8)
        ax.set(title=f"{left} vs {right} (+ favours {left})", xlabel="Active training hours",
               ylabel="mbb/hand")
    figures.append("matched_time_crossplay.png")
    save(fig, figures[-1])
    experiments = list(EXPERIMENTS)
    matrix = np.full((4, 4), np.nan)
    for row in direct:
        if row["training_hours"] == 24:
            left, right = row["pair_id"].split("_vs_")
            i, j = experiments.index(left), experiments.index(right)
            matrix[i, j], matrix[j, i] = row["mean_mbb_per_hand"], -row["mean_mbb_per_hand"]
    limit = max(1, float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 1
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    artist = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
    for i, j in itertools.product(range(4), repeat=2):
        ax.text(j, i, f"{matrix[i, j]:.1f}" if np.isfinite(matrix[i, j]) else "—",
                ha="center", va="center")
    ax.set_xticks(range(4), experiments)
    ax.set_yticks(range(4), experiments)
    ax.set(title="24-hour direct play: row policy vs column policy",
           xlabel="Column opponent", ylabel="Row policy")
    fig.colorbar(artist, ax=ax, label="Row-policy payoff (mbb/hand)")
    figures.append("final_crossplay_matrix.png")
    save(fig, figures[-1])
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    limit = max(1, max(abs(r["mean_mbb_per_hand"]) for r in temporal))
    for ax, experiment in zip(axes.flat, EXPERIMENTS):
        matrix = np.full((4, 4), np.nan)
        for row in temporal:
            if row["experiment"] == experiment:
                i, j = EXPECTED_HOURS.index(row["later_hours"]), EXPECTED_HOURS.index(row["earlier_hours"])
                matrix[i, j] = row["mean_mbb_per_hand"]
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center")
        artist = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_xticks(range(4), EXPECTED_HOURS)
        ax.set_yticks(range(4), EXPECTED_HOURS)
        ax.set(title=LABELS[experiment], xlabel="Earlier checkpoint (hours)", ylabel="Later checkpoint (hours)")
    fig.colorbar(artist, ax=axes, label="Later-policy payoff (mbb/hand)")
    figures.append("temporal_crossplay.png")
    save(fig, figures[-1])
    return figures


def _summarise(output, records, results, *, smoke=False):
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    direct = [r for r in results if r["kind"] == "direct_crossplay"]
    temporal = [r for r in results if r["kind"] == "temporal_crossplay"]
    rule_mean = protocol._average_rule_agents(rules)
    differences = _paired_differences(rule_mean, "rule_agent_mean") + _paired_differences(lbr, "lbr_payoff")
    aggregate_rule = protocol._group_aggregate(rule_mean, ("experiment", "training_hours"))
    aggregate_lbr = protocol._group_aggregate(lbr, ("experiment", "training_hours"))
    aggregate_direct = protocol._group_aggregate(direct, ("pair_id", "training_hours"))
    aggregate_temporal = protocol._group_aggregate(temporal, ("experiment", "earlier_hours", "later_hours"))
    temporal_seed_means = _temporal_summary(temporal)
    for rows in (aggregate_rule, aggregate_lbr):
        protocol._attach_mean_nodes(rows, records)
    for row in aggregate_direct:
        left, right = row["pair_id"].split("_vs_")
        row.update(left_experiment=left, right_experiment=right,
                   primary_endpoint=row["training_hours"] == 24 and (left, right) in PRIMARY_PAIRS)
    tables = {
        "checkpoint_index.csv": records,
        "rule_agent_by_seed.csv": rules,
        "rule_agent_aggregate.csv": protocol._group_aggregate(rules, ("experiment", "training_hours", "opponent")),
        "rule_agent_mean_by_seed.csv": rule_mean, "rule_agent_mean_aggregate.csv": aggregate_rule,
        "lbr_by_seed.csv": lbr, "lbr_aggregate.csv": aggregate_lbr,
        "paired_metric_differences_by_seed.csv": differences,
        "paired_metric_differences_aggregate.csv": protocol._group_aggregate(
            differences, ("metric", "pair_id", "training_hours")),
        "temporal_crossplay_by_seed.csv": temporal, "temporal_crossplay_aggregate.csv": aggregate_temporal,
        "temporal_run_means.csv": temporal_seed_means,
        "temporal_run_mean_aggregate.csv": protocol._group_aggregate(temporal_seed_means, ("experiment",)),
        "direct_crossplay_by_seed.csv": direct, "direct_crossplay_aggregate.csv": aggregate_direct,
        "final_crossplay_aggregate.csv": [r for r in aggregate_direct if r["training_hours"] == 24],
    }
    for name, rows in tables.items():
        protocol._write_csv(output / name, rows)
    figures = _plot_analysis(output, aggregate_rule, aggregate_lbr, aggregate_direct, aggregate_temporal, smoke=smoke)
    lines = ["# Frozen-policy evaluation: FHP Experiments 9–12", "",
             "SMOKE TEST ONLY — not policy-strength evidence." if smoke else "Full-budget evaluation.", "",
             "No solver or policy was retrained. All values are in mbb/hand.", "",
             "The primary comparisons are Exp10, Exp11 and Exp12 against Exp9 at 24 active hours. "
             "All six pairings are reported without selecting a winner from earlier checkpoints.", "",
             "## Final matched-time head-to-head", "",
             "| Policy A vs B | A payoff | Training-seed 95% CI | Primary |",
             "|---|---:|---:|---|"]
    for row in tables["final_crossplay_aggregate.csv"]:
        lines.append(f"| {row['pair_id']} | {row['mean_mbb_per_hand']:.2f} | "
                     f"[{row['training_seed_ci95_low_mbb_per_hand']:.2f}, "
                     f"{row['training_seed_ci95_high_mbb_per_hand']:.2f}] | {row['primary_endpoint']} |")
    lines.extend(["", "Positive direct scores favour policy A (the row policy in the final matrix). "
                  "Positive temporal scores favour the later policy. Higher rule payoff is better; "
                  "lower LBR payoff is better. Paired diagnostic differences are A minus B, so negative "
                  "LBR differences favour A.", "",
                  "Uncertainty uses three matched training-seed means, not individual hands, checkpoints "
                  "or LBR shards. Intervals are exploratory and unadjusted for multiple comparisons; "
                  "the three primary contrasts do not form a single preselected hypothesis. "
                  "Matching seed labels does not imply identical training trajectories.", "",
                  "These are historical runs from separately recorded source commits. Input changes "
                  "can alter model parameter counts, RNG consumption and throughput. Equal active "
                  "training time is not equal node exposure or billable time; node trajectories are "
                  "descriptive, not a matched-node intervention. No arbitrary node-matched crossplay "
                  "or interpolated policy is constructed.", "",
                  "Experiment 12's privileged showdown features are confined to its training critic. "
                  "The deployed policy uses the unchanged visible-information encoding. "
                  "LBR is restricted and approximate, not exact exploitability; direct wins and "
                  "rule-agent scores alone do not establish equilibrium convergence.", ""])
    (output / "analysis_summary.md").write_text("\n".join(lines))
    return sorted([*tables, *figures, "analysis_summary.md"])

def run_analysis(args):
    import torch
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    source_roots = {experiment: getattr(args, experiment + "_run") for experiment in EXPERIMENTS}
    if len({root.resolve() for root in source_roots.values()}) != len(EXPERIMENTS):
        raise ValueError("Source directories must be distinct")
    if any(output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output)
           for root in source_roots.values()):
        raise ValueError("Output and source directories must be disjoint")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Use --resume for an existing output: {output}")
    records = [record for experiment, root in source_roots.items()
               for record in discover_checkpoints(root, experiment)]
    if len({r["learning_config_without_encoder_sha256"] for r in records}) != 1:
        raise ValueError("Experiments differ beyond the prescribed feature encoders")
    # Historical runs need one coherent source commit per experiment.
    for experiment in EXPERIMENTS:
        if len({r["source_repository_commit"] for r in records if r["experiment"] == experiment}) != 1:
            raise ValueError(f"Mixed source commits within {experiment}")
    tasks = _build_tasks(records, args)
    repository = Path(__file__).resolve().parents[3]
    signature = {
        "schema_version": 1, "smoke": args.smoke,
        "tasks": [_portable_task(t) for t in tasks],
        "source_records": [{k: v for k, v in r.items() if k != "checkpoint_path"} for r in records],
        "implementation": {
            "runner_sha256": sha256_file(Path(__file__)),
            "shared_runner_sha256": sha256_file(Path(shared.__file__)),
            "protocol_sha256": sha256_file(Path(protocol.__file__)),
            "evaluation_suite_sha256": protocol._source_tree_sha256(repository / "fhp_evaluation"),
            "policy_loader_source_sha256": protocol._source_tree_sha256(repository / "fhp_escher"),
            "mlp_source_sha256": sha256_file(repository / "vr_deep_cfr" / "solver.py"),
        },
        "versions": {name: version(name) for name in ("numpy", "scipy", "torch", "open_spiel")},
        "python": platform.python_version(),
    }
    fingerprint = _digest(signature)
    manifest_path = output / "evaluation_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old is not None and old["fingerprint"] != fingerprint:
        raise ValueError("Resume rejected: changed policies, protocol, implementation or dependencies")
    if old is None and output.exists() and any(output.iterdir()):
        raise ValueError("Cannot resume outputs without a valid evaluation manifest")
    output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "in_progress", "fingerprint": fingerprint, "signature": signature,
        "started_utc": old["started_utc"] if old else now, "last_invocation_utc": now,
        "source_runs": {key: str(root.resolve()) for key, root in source_roots.items()},
        "repository_commit": protocol._git_value(repository, "rev-parse", "HEAD"),
        "workers": args.workers, "checkpoints": records, "num_tasks": len(tasks),
        "evaluation": {
            "rule_agents": list(protocol.PUBLISHED_AGENT_NAMES), "rule_deals": args.rule_deals,
            "lbr_deals": args.lbr_deals, "lbr_rollouts": args.lbr_rollouts,
            "lbr_shard_deals": args.lbr_shard_deals, "crossplay_deals": args.crossplay_deals,
            "base_seed": args.base_seed, "exact_exploitability": False,
            "uncertainty_unit": "matched_training_seed_mean", "positive_direct_favours": "left_experiment",
            "direct_pairs": [list(pair) for pair in DIRECT_PAIRS],
            "primary_pairs": [list(pair) for pair in PRIMARY_PAIRS],
            "primary_endpoint_hours": 24,
            "training_config_difference": "feature_encoder_only",
            "source_commit_requirement": "one_commit_per_experiment; historical_cross_commit_comparison",
            "node_comparison": "descriptive_trajectories_only; no_matched_node_crossplay",
        },
    }
    _write_json(manifest_path, manifest)
    # Remove only our own stale completion marker before recomputing analysis.
    (output / "SUCCESS.json").unlink(missing_ok=True)
    try:
        results = _run_tasks(tasks, args.workers, output / "task_results", fingerprint)
        artifacts = _summarise(output, records, results, smoke=args.smoke)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(manifest_path, manifest)
        _write_json(output / "failure.json", {"stage": "evaluation", "error_type": type(error).__name__,
                                             "error": str(error)})
        raise
    (output / "failure.json").unlink(missing_ok=True)
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts)
    _write_json(manifest_path, manifest)
    _write_json(output / "SUCCESS.json", {"status": "complete", "smoke": args.smoke,
                                          "fingerprint": fingerprint, "num_tasks": len(tasks)})
    return output


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for experiment in EXPERIMENTS:
        parser.add_argument("--" + experiment + "-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rule-deals", type=int, default=10_000)
    parser.add_argument("--lbr-deals", type=int, default=1_000)
    parser.add_argument("--lbr-rollouts", type=int, default=4096)
    parser.add_argument("--lbr-shard-deals", type=int, default=10)
    parser.add_argument("--crossplay-deals", type=int, default=50_000)
    parser.add_argument("--base-seed", type=int, default=20_260_922)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    for name in ("workers", "rule_deals", "lbr_deals", "lbr_rollouts", "lbr_shard_deals", "crossplay_deals"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    print(f"Evaluation complete: {run_analysis(args)}", flush=True)


if __name__ == "__main__":
    main()
