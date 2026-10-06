"""Independent frozen-policy comparison: half versus full critic-update budgets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from experiments.fhp.exp10_fhp_hand_board_features import config as baseline
from experiments.fhp.retrospective_exp7_exp8_evaluation import run as long_eval
from . import config as candidate

protocol = long_eval.protocol
shared = long_eval.shared
sha256_file = long_eval.sha256_file
EXPECTED_HOURS = (6, 12, 18, 24)
EXPECTED_SEEDS = (0, 1, 2)
CONFIGS = {"exp10": baseline, "exp20": candidate}
EXPERIMENTS = {key: {"experiment_name": c.EXPERIMENT_NAME, "algorithm_id": c.ALGORITHM_ID,
                    "label": "Exp10: 10,000 critic updates" if key == "exp10" else "Exp20: 5,000 critic updates"}
               for key, c in CONFIGS.items()}
DIRECT_PAIR = ("exp20", "exp10")
_digest = shared._digest
_write_json = shared._write_json
_portable_task = shared._portable_task
_run_tasks = shared._run_tasks


def discover_checkpoints(root, experiment):
    config = CONFIGS[experiment]
    records = long_eval.discover_checkpoints(
        root, experiment, config=config, hours=EXPECTED_HOURS,
        label=EXPERIMENTS[experiment]["label"], cache=True, allow_continuation=False)
    for worker in sorted((Path(root) / "workers").glob("task_*")):
        manifest = json.loads((worker / "run_manifest.json").read_text())
        summary = json.loads((worker / "summary.json").read_text())
        success = json.loads((worker / "SUCCESS.json").read_text())
        runtime = json.loads((worker / "runtime_manifest.json").read_text())
        if (manifest.get("experiment_id") != config.EXPERIMENT_ID
                or manifest.get("training_state_retention") != "final"
                or manifest.get("training_config_sha256") != _digest(config.EXPERIMENT_CONFIG)
                or summary.get("status") != "complete"
                or summary.get("seed") != manifest["seed"] or summary.get("checkpoint_count") != 4
                or sha256_file(worker / "summary.json") != success.get("summary_sha256")):
            raise ValueError("Incomplete or incompatible source metadata")
        if experiment == "exp10" and manifest["repository_commit"] != candidate.BASELINE_REF:
            raise ValueError("Wrong historical Experiment 10 source commit")
        for key, value in {"python_version": "3.11.16", "torch_version": "2.7.0+cpu",
                           "numpy_version": "1.26.4", "ray_version": "2.51.2"}.items():
            if runtime.get(key) != value:
                raise ValueError(f"Unexpected source runtime at {key}")
        if experiment == "exp20" and summary.get("execution_diagnostics", {}).get(
                "critic_train_steps_per_fit") != [5000, 5000]:
            raise ValueError("Candidate did not report 5,000 updates for both critics")
    normalised = dict(config.EXPERIMENT_CONFIG, baseline_network_train_steps=10_000)
    for record in records:
        record["learning_config_without_critic_budget_sha256"] = _digest(normalised)
    return records


def _build_tasks(records, args):
    # Preserve the established protocol without inventing a matched-node pairing.
    tasks = protocol._build_tasks(records, args, experiments=EXPERIMENTS,
                                  direct_pair=DIRECT_PAIR, node_hours=None)
    by_path = {r["checkpoint_path"]: r for r in records}
    for task in tasks:
        task["primary_endpoint"] = task["kind"] == "direct_crossplay" and task["training_hours"] == 24
        for side in ("a", "b"):
            key = f"policy_{side}_path"
            if key in task:
                task[f"policy_{side}_sha256"] = by_path[task[key]]["checkpoint_sha256"]
                task[f"policy_{side}_nodes_touched"] = by_path[task[key]]["nodes_touched"]
    return tasks


def _differences(rows, metric):
    indexed = {(r["experiment"], r["training_seed"], r["training_hours"]): r for r in rows}
    result = []
    for (experiment, seed, hour), row in indexed.items():
        if experiment == "exp20":
            control = indexed[("exp10", seed, hour)]
            result.append({"metric": metric, "training_seed": seed, "training_hours": hour,
                           "mean_mbb_per_hand": row["mean_mbb_per_hand"] - control["mean_mbb_per_hand"],
                           "num_deal_pairs": row["num_deal_pairs"]})
    return result


def _summarise(output, records, results, *, smoke):
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    direct = [r for r in results if r["kind"] == "direct_crossplay"]
    temporal = [r for r in results if r["kind"] == "temporal_crossplay"]
    means = protocol._average_rule_agents(rules)
    differences = _differences(means, "rule_agent_mean") + _differences(lbr, "lbr_payoff")
    aggregate_rule = protocol._group_aggregate(means, ("experiment", "training_hours"))
    aggregate_lbr = protocol._group_aggregate(lbr, ("experiment", "training_hours"))
    aggregate_direct = protocol._group_aggregate(direct, ("training_hours",))
    for rows in (aggregate_rule, aggregate_lbr):
        protocol._attach_mean_nodes(rows, records)
    tables = {
        "checkpoint_index.csv": records, "rule_agent_by_seed.csv": rules,
        "rule_agent_aggregate.csv": protocol._group_aggregate(rules, ("experiment", "training_hours", "opponent")),
        "rule_agent_mean_by_seed.csv": means, "rule_agent_mean_aggregate.csv": aggregate_rule,
        "lbr_by_seed.csv": lbr, "lbr_aggregate.csv": aggregate_lbr,
        "direct_exp20_vs_exp10_by_seed.csv": direct, "direct_exp20_vs_exp10_aggregate.csv": aggregate_direct,
        "temporal_crossplay_by_seed.csv": temporal,
        "temporal_crossplay_aggregate.csv": protocol._group_aggregate(temporal, ("experiment", "earlier_hours", "later_hours")),
        "metric_differences_by_seed.csv": differences,
        "metric_differences_aggregate.csv": protocol._group_aggregate(differences, ("metric", "training_hours")),
    }
    for name, rows in tables.items():
        protocol._write_csv(output / name, rows)
    protocol._plot_analysis(output, aggregate_rule, aggregate_lbr, aggregate_direct,
                            experiments=EXPERIMENTS, direct_pair=DIRECT_PAIR)
    lines = ["# Experiment 20 versus Experiment 10: half critic updates", "",
             "SMOKE ONLY; not strength evidence." if smoke else "Full-budget frozen-policy evaluation.", "",
             "Primary comparison: 24h versus 24h. Earlier and temporal comparisons are exploratory.",
             "Positive direct payoff favours Experiment 20; higher rule payoff and lower LBR payoff are better.", "",
             "| Hours | Exp20 vs Exp10 (mbb/hand) | Seed-level 95% interval |", "|---|---:|---:|"]
    for row in aggregate_direct:
        low, high = row["training_seed_ci95_low_mbb_per_hand"], row["training_seed_ci95_high_mbb_per_hand"]
        interval = f"[{low:.3f}, {high:.3f}]" if np.isfinite(low) and np.isfinite(high) else "Not estimable (one seed)"
        lines.append(f"| {row['training_hours']} | {row['mean_mbb_per_hand']:.3f} | {interval} |")
    lines.extend(["", "The three training seeds, not hands, shards or checkpoints, are the inferential units. "
                  "Intervals are pointwise; secondary comparisons are exploratory. Historical matching seed "
                  "labels do not imply identical training paths or hardware. Common evaluation deals are used. "
                  "Equal active time is not equal nodes or billable time; node-axis plots are descriptive, "
                  "not matched-node experiments.", "",
                  "The 5,000-update arm trains from scratch throughout. Lower fit cost or more nodes do not "
                  "establish stronger play. LBR is restricted and approximate, not exact exploitability. "
                  "An interval crossing zero does not prove equivalence, non-inferiority or convergence. "
                  "No automatic promotion or best-checkpoint selection is performed.", ""])
    (output / "analysis_summary.md").write_text("\n".join(lines))
    return [*tables, "analysis_summary.md", *sorted(p.name for p in output.glob("*.png"))]


def run_analysis(args):
    import torch
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output)
           for root in (args.exp10_run, args.exp20_run)):
        raise ValueError("Output and source directories must be disjoint")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Use --resume for an existing output: {output}")
    if args.exp10_run.resolve() == args.exp20_run.resolve():
        raise ValueError("Source runs must be distinct")
    records = discover_checkpoints(args.exp10_run, "exp10") + discover_checkpoints(args.exp20_run, "exp20")
    if len({r["learning_config_without_critic_budget_sha256"] for r in records}) != 1:
        raise ValueError("Experiments 10 and 20 differ beyond critic-update budget")
    # Historical baseline and new experiment each require a coherent commit.
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
            "checkpoint_validation_sha256": sha256_file(Path(long_eval.__file__)),
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
        "source_runs": {"exp10": str(args.exp10_run.resolve()), "exp20": str(args.exp20_run.resolve())},
        "repository_commit": protocol._git_value(repository, "rev-parse", "HEAD"),
        "workers": args.workers, "checkpoints": records, "num_tasks": len(tasks),
        "evaluation": {
            "rule_agents": list(protocol.PUBLISHED_AGENT_NAMES), "rule_deals": args.rule_deals,
            "lbr_deals": args.lbr_deals, "lbr_rollouts": args.lbr_rollouts,
            "lbr_shard_deals": args.lbr_shard_deals, "crossplay_deals": args.crossplay_deals,
            "base_seed": args.base_seed, "exact_exploitability": False,
            "uncertainty_unit": "matched_training_seed_mean", "positive_direct_favours": "exp20",
            "primary_endpoint_hours": 24,
            "training_config_difference": "baseline_network_train_steps_10000_to_5000",
            "source_commit_requirement": "one_commit_per_experiment; historical_cross_commit_comparison",
            "node_comparison": None,
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
        _write_json(output / "failure.json", {"error": manifest["error"], "fingerprint": fingerprint})
        raise
    (output / "failure.json").unlink(missing_ok=True)
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts)
    _write_json(manifest_path, manifest)
    _write_json(output / "SUCCESS.json", {"status": "complete", "smoke": args.smoke,
                                          "fingerprint": fingerprint, "num_tasks": len(tasks)})
    return output


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp10-run", type=Path, required=True)
    parser.add_argument("--exp20-run", type=Path, required=True)
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
