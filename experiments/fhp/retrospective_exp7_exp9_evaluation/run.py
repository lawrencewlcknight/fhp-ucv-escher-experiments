"""Evaluate Exp7/9 saved policies; never resume or modify model training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re

import numpy as np

from experiments.fhp.retrospective_exp2_exp3_evaluation import run as protocol
from experiments.fhp.retrospective_exp6_exp7_evaluation import run as shared
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16 import config as exp7_config
from experiments.fhp.exp9_fhp_cached_parallel_24h import config as exp9_config
from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file

EXPECTED_HOURS = protocol.EXPECTED_HOURS
EXPECTED_SEEDS = protocol.EXPECTED_SEEDS
EXPERIMENTS = {
    "exp7": {"experiment_name": exp7_config.EXPERIMENT_NAME,
             "algorithm_id": exp7_config.ALGORITHM_ID,
             "label": "Experiment 7: uncached critic targets"},
    "exp9": {"experiment_name": exp9_config.EXPERIMENT_NAME,
             "algorithm_id": exp9_config.ALGORITHM_ID,
             "label": "Experiment 9: cached critic targets"},
}
CONFIGS = {"exp7": exp7_config, "exp9": exp9_config}
DIRECT_PAIR = ("exp9", "exp7")  # Positive head-to-head values favour caching.
# Fixed from observed node counts, before observing any policy-strength result.
# Roughly 85.4m versus 78.2m nodes; explicitly NOT an exact matched-node test.
NODE_HOURS = (12, 18)


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
                or runtime["frozen_critic_target_cache"] is not (experiment == "exp9")):
            raise ValueError(f"Wrong source runtime: {worker}")
        if runtime["traversal_execution"] != "ray_parallel":
            raise ValueError(f"Wrong source collector: {worker}")
        if runtime["parallel_settings"] != config.parallel_settings():
            raise ValueError(f"Wrong parallel settings: {worker}")
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
            learning_config_without_cache_sha256=_digest({
                **manifest["training_config"], "cache_frozen_critic_targets": False,
            }),
            source_repository_commit=manifest["repository_commit"],
            source_runtime=runtime,
        )
    return records


def _build_tasks(records, args):
    tasks = protocol._build_tasks(records, args, experiments=EXPERIMENTS,
                                  direct_pair=DIRECT_PAIR, node_hours=NODE_HOURS)
    by_path = {r["checkpoint_path"]: r for r in records}
    for task in tasks:
        for side in ("a", "b"):
            key = f"policy_{side}_path"
            if key in task:
                task[f"policy_{side}_sha256"] = by_path[task[key]]["checkpoint_sha256"]
                task[f"policy_{side}_nodes_touched"] = by_path[task[key]]["nodes_touched"]
        if task["kind"] == "node_matched_crossplay":
            task["relative_node_excess_exp9"] = (
                task["exp9_nodes_touched"] / task["exp7_nodes_touched"] - 1
            )
            task["exactly_node_matched"] = False
    return tasks


def _paired_differences(rows, metric):
    by_key = {(r["experiment"], r["training_seed"], r["training_hours"]): r for r in rows}
    output = []
    for (experiment, seed, hour), row in sorted(by_key.items()):
        if experiment != "exp9":
            continue
        baseline = by_key[("exp7", seed, hour)]
        output.append({"metric": metric, "training_seed": seed, "training_hours": hour,
                       "mean_mbb_per_hand": row["mean_mbb_per_hand"] - baseline["mean_mbb_per_hand"],
                       "num_deal_pairs": row["num_deal_pairs"]})
    return output


def _temporal_plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    limit = max(1, max(abs(r["mean_mbb_per_hand"]) for r in rows))
    for ax, experiment in zip(axes, EXPERIMENTS):
        matrix = np.full((4, 4), np.nan)
        for row in rows:
            if row["experiment"] == experiment:
                i, j = EXPECTED_HOURS.index(row["later_hours"]), EXPECTED_HOURS.index(row["earlier_hours"])
                matrix[i, j] = row["mean_mbb_per_hand"]
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center")
        artist = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_xticks(range(4), EXPECTED_HOURS)
        ax.set_yticks(range(4), EXPECTED_HOURS)
        ax.set(xlabel="Earlier checkpoint (hours)", ylabel="Later checkpoint (hours)",
               title=EXPERIMENTS[experiment]["label"])
    fig.colorbar(artist, ax=axes, label="Later minus earlier (mbb/hand)")
    fig.savefig(output / "temporal_crossplay.png", dpi=180)
    plt.close(fig)


def _summarise(output, records, results):
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    temporal = [r for r in results if r["kind"] == "temporal_crossplay"]
    direct = [r for r in results if r["kind"] == "direct_crossplay"]
    nodes = [r for r in results if r["kind"] == "node_matched_crossplay"]
    rule_mean = protocol._average_rule_agents(rules)
    differences = _paired_differences(rule_mean, "rule_agent_mean") + _paired_differences(lbr, "lbr_payoff")
    aggregate_rule = protocol._group_aggregate(rule_mean, ("experiment", "training_hours"))
    aggregate_lbr = protocol._group_aggregate(lbr, ("experiment", "training_hours"))
    aggregate_direct = protocol._group_aggregate(direct, ("training_hours",))
    aggregate_temporal = protocol._group_aggregate(temporal, ("experiment", "earlier_hours", "later_hours"))
    temporal_seed_means = _temporal_summary(temporal)
    for rows in (aggregate_rule, aggregate_lbr):
        protocol._attach_mean_nodes(rows, records)
    aggregate_nodes = protocol._group_aggregate(nodes, ("exp9_hours", "exp7_hours")) if nodes else []
    for row in aggregate_nodes:
        row.update(mean_exp9_nodes=float(np.mean([r["exp9_nodes_touched"] for r in nodes])),
                   mean_exp7_nodes=float(np.mean([r["exp7_nodes_touched"] for r in nodes])),
                   mean_relative_node_excess_exp9=float(np.mean([r["relative_node_excess_exp9"] for r in nodes])),
                   exactly_node_matched=False)
    tables = {
        "checkpoint_index.csv": records,
        "rule_agent_by_seed.csv": rules,
        "rule_agent_aggregate.csv": protocol._group_aggregate(rules, ("experiment", "training_hours", "opponent")),
        "rule_agent_mean_by_seed.csv": rule_mean, "rule_agent_mean_aggregate.csv": aggregate_rule,
        "lbr_by_seed.csv": lbr, "lbr_aggregate.csv": aggregate_lbr,
        "paired_metric_differences_by_seed.csv": differences,
        "paired_metric_differences_aggregate.csv": protocol._group_aggregate(differences, ("metric", "training_hours")),
        "temporal_crossplay_by_seed.csv": temporal, "temporal_crossplay_aggregate.csv": aggregate_temporal,
        "temporal_run_means.csv": temporal_seed_means,
        "temporal_run_mean_aggregate.csv": protocol._group_aggregate(temporal_seed_means, ("experiment",)),
        "direct_exp9_vs_exp7_by_seed.csv": direct, "direct_exp9_vs_exp7_aggregate.csv": aggregate_direct,
        "node_matched_exp9_vs_exp7_by_seed.csv": nodes, "node_matched_exp9_vs_exp7_aggregate.csv": aggregate_nodes,
    }
    for name, rows in tables.items():
        protocol._write_csv(output / name, rows)
    protocol._plot_analysis(output, aggregate_rule, aggregate_lbr, aggregate_direct,
                            experiments=EXPERIMENTS, direct_pair=DIRECT_PAIR)
    _temporal_plot(output, aggregate_temporal)
    lines = ["# Frozen-policy evaluation: FHP Experiments 7 and 9", "",
             "No solver or output policy was retrained. Results are in mbb/hand.", "",
             "Positive direct values favour Experiment 9; positive temporal values favour later policies. "
             "Higher rule-agent payoff is better; lower LBR payoff is better. LBR is an approximate "
             "best-response diagnostic, not exact exploitability or proof of convergence.", "",
             "## Matched active-time head-to-head", "",
             "| Hours | Exp9 minus Exp7 | Training-seed 95% CI |", "|---|---:|---:|"]
    for row in aggregate_direct:
        lines.append(f"| {row['training_hours']} | {row['mean_mbb_per_hand']:.2f} | "
                     f"[{row['training_seed_ci95_low_mbb_per_hand']:.2f}, {row['training_seed_ci95_high_mbb_per_hand']:.2f}] |")
    lines.extend(["", "Intervals aggregate matched seed-level means (three seeds in production), "
                  "not individual hands or temporal cells. They are exploratory and not multiplicity-adjusted. "
                  "This is a historical comparison across source commits; matching seed labels does not "
                  "guarantee identical training trajectories or hardware conditions.", "",
                  "At equal time the cached-target model has seen more nodes. The prespecified secondary "
                  "comparison uses Exp9 at 12h versus Exp7 at 18h, selected from throughput metadata "
                  "before strength evaluation. Node counts and their mismatch are reported explicitly; "
                  "no weights or playable policies are interpolated."])
    for row in aggregate_nodes:
        lines.extend(["", f"Approximate node match: Exp9 {row['mean_exp9_nodes']/1e6:.3f}m versus "
                      f"Exp7 {row['mean_exp7_nodes']/1e6:.3f}m; mean per-seed excess "
                      f"{100*row['mean_relative_node_excess_exp9']:.2f}%. Direct value "
                      f"{row['mean_mbb_per_hand']:.2f} mbb/hand."])
    lines.extend(["", "Rule-agent and LBR trajectories by time and nodes, paired metric differences, "
                  "and all later-versus-earlier contrasts are in the companion tables and charts. "
                  "No single diagnostic establishes equilibrium quality in this game.", ""])
    (output / "analysis_summary.md").write_text("\n".join(lines))
    return sorted([*tables, "analysis_summary.md", *[p.name for p in output.glob("*.png")]])


def run_analysis(args):
    import torch
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output)
           for root in (args.exp7_run, args.exp9_run)):
        raise ValueError("Output and source directories must be disjoint")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Use --resume for an existing output: {output}")
    records = discover_checkpoints(args.exp7_run, "exp7") + discover_checkpoints(args.exp9_run, "exp9")
    if len({r["learning_config_without_cache_sha256"] for r in records}) != 1:
        raise ValueError("Experiment 7 and 9 differ beyond critic-target caching")
    # Exp9 adds cache support in a later commit. Require one coherent source
    # commit per experiment, not an impossible shared commit across both runs.
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
        "source_runs": {"exp7": str(args.exp7_run.resolve()), "exp9": str(args.exp9_run.resolve())},
        "repository_commit": protocol._git_value(repository, "rev-parse", "HEAD"),
        "workers": args.workers, "checkpoints": records, "num_tasks": len(tasks),
        "evaluation": {
            "rule_agents": list(protocol.PUBLISHED_AGENT_NAMES), "rule_deals": args.rule_deals,
            "lbr_deals": args.lbr_deals, "lbr_rollouts": args.lbr_rollouts,
            "lbr_shard_deals": args.lbr_shard_deals, "crossplay_deals": args.crossplay_deals,
            "base_seed": args.base_seed, "exact_exploitability": False,
            "uncertainty_unit": "matched_training_seed_mean", "positive_direct_favours": "exp9",
            "primary_endpoint_hours": 24,
            "training_config_difference": "cache_frozen_critic_targets_only",
            "source_commit_requirement": "one_commit_per_experiment; historical_cross_commit_comparison",
            "node_comparison": {"exp9_hours": 12, "exp7_hours": 18, "exactly_matched": False},
        },
    }
    _write_json(manifest_path, manifest)
    # Remove only our own stale completion marker before recomputing analysis.
    (output / "SUCCESS.json").unlink(missing_ok=True)
    try:
        results = _run_tasks(tasks, args.workers, output / "task_results", fingerprint)
        artifacts = _summarise(output, records, results)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(manifest_path, manifest)
        raise
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts)
    _write_json(manifest_path, manifest)
    _write_json(output / "SUCCESS.json", {"status": "complete", "smoke": args.smoke,
                                          "fingerprint": fingerprint, "num_tasks": len(tasks)})
    return output


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp7-run", type=Path, required=True)
    parser.add_argument("--exp9-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
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
