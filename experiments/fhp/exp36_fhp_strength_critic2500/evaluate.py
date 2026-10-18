"""Separate, resumable evaluation of frozen 24h policies; never retrain controls."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

from experiments.fhp.exp10_fhp_hand_board_features import config as exp10
from experiments.fhp.exp20_fhp_half_critic_updates import config as exp20
from experiments.fhp.strategic_feature_suite.config import get_experiment
from experiments.fhp.strategic_feature_suite import evaluate as features
from .config import EXPERIMENT
from . import rare

sources, protocol, shared = features.sources, features.protocol, features.shared
CONFIGS = {"exp36": EXPERIMENT, "exp20": exp20, "exp33": get_experiment(33), "exp10": exp10}
CONTROLS = ("exp20", "exp33", "exp10")


def discover(root, key):
    rows = features.discover(root, CONFIGS[key], control=(key == "exp10"))
    if key in ("exp20", "exp36"):
        expected = CONFIGS[key].EXPERIMENT_CONFIG["baseline_network_train_steps"]
        for worker in (Path(root) / "workers").glob("task_*"):
            summary = json.loads((worker / "summary.json").read_text())
            if summary.get("execution_diagnostics", {}).get("critic_train_steps_per_fit") != [expected] * 2:
                raise ValueError(f"Wrong actual critic budget in {key}")
    return rows


def build_tasks(records, args):
    # The established single-comparator task builder evaluates every source
    # against the common panel once; only direct-play tasks need extending.
    tasks = features.build_tasks(records, args, experiment=EXPERIMENT)
    tasks = [t for t in tasks if t["kind"] != "direct_crossplay"]
    index = {(r["experiment"], r["seed"]): r for r in records if r["training_hours"] == 24}
    for control in CONTROLS:
        for seed in ((0,) if args.smoke else (0, 1, 2)):
            tasks.append(dict(task_id=f"direct_exp36_vs_{control}_seed{seed}", kind="direct_crossplay",
                experiment=f"exp36_vs_{control}", comparator=control, training_seed=seed,
                training_hours=24, primary_endpoint=(control == "exp20"),
                num_deals=args.crossplay_deals, evaluation_seed=args.base_seed + 3_000_024,
                **sources._policy_fields(index[("exp36", seed)], "a"),
                **sources._policy_fields(index[(control, seed)], "b")))
    return tasks


def summarise(output, records, results, rare_results, *, smoke):
    direct = [r for r in results if r["kind"] == "direct_crossplay"]
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    mean_rules = protocol._average_rule_agents(rules)
    tables = {
        "checkpoint_index.csv": records,
        "direct_by_seed.csv": direct,
        "direct_aggregate.csv": protocol._group_aggregate(direct, ("experiment", "training_hours")),
        "rule_agent_by_seed.csv": rules,
        "rule_agent_aggregate.csv": protocol._group_aggregate(rules, ("experiment", "training_hours", "opponent")),
        "rule_agent_mean_by_seed.csv": mean_rules,
        "rule_agent_mean_aggregate.csv": protocol._group_aggregate(mean_rules, ("experiment", "training_hours")),
        "lbr_by_seed.csv": lbr,
        "lbr_aggregate.csv": protocol._group_aggregate(lbr, ("experiment", "training_hours")),
    }
    differences = []
    for metric, rows in (("rule_suite_mean", mean_rules), ("lbr_exploiter_payoff", lbr)):
        index = {(r["experiment"], r["training_seed"]): r for r in rows}
        for row in rows:
            if row["experiment"] != "exp36":
                continue
            for control in CONTROLS:
                differences.append(dict(training_seed=row["training_seed"], training_hours=24,
                    comparator=control, metric=metric, num_deal_pairs=row["num_deal_pairs"],
                    mean_mbb_per_hand=row["mean_mbb_per_hand"] - index[(control, row["training_seed"])]["mean_mbb_per_hand"]))
    tables["metric_differences_by_seed.csv"] = differences
    tables["metric_differences_aggregate.csv"] = protocol._group_aggregate(differences, ("metric", "comparator"))
    for name, rows in tables.items():
        protocol._write_csv(output / name, rows)
    summary = dict(experiment_id=36, smoke=smoke, exact_exploitability=False,
        primary_endpoint=[r for r in tables["direct_aggregate.csv"] if r["experiment"] == "exp36_vs_exp20"],
        secondary_endpoints=[r for r in tables["direct_aggregate.csv"] if r["experiment"] != "exp36_vs_exp20"],
        rare_diagnostics=rare_results,
        uncertainty="Three independent training seeds, not hands or diagnostic positions, are inferential units. Pointwise exploratory intervals; no multiplicity adjustment.",
        interpretation="Against Experiment 20, features and critic budget both change. Against Experiment 33, features match but updates fall from 10,000 to 2,500. Historical comparisons are not a factorial interaction test or proof of convergence.")
    shared._write_json(output / "summary.json", summary)
    lines = ["# Experiment 36: separate 24-hour evaluation", "",
        "Positive head-to-head values favour Experiment 36. Experiment 20 is the primary comparator; 33 and 10 are secondary.",
        "LBR is exploiter payoff (lower is better), not exact exploitability. Controls were re-evaluated, not retrained.", "",
        "| Comparison | Mean (mbb/hand) | Training-seed 95% interval |", "|---|---:|---:|"]
    for row in tables["direct_aggregate.csv"]:
        lo, hi = row["training_seed_ci95_low_mbb_per_hand"], row["training_seed_ci95_high_mbb_per_hand"]
        interval = "not estimated (smoke)" if smoke else f"[{lo:.2f}, {hi:.2f}]"
        lines.append(f"| {row['experiment']} | {row['mean_mbb_per_hand']:.2f} | {interval} |")
    lines += ["", summary["uncertainty"], "", summary["interpretation"], "",
        "Rare-decision metrics are in rare_summary.json and rare_by_seed.csv, with individual examples in rare_results/.",
        "Their BB losses are conditional on selected positions, not reach-weighted whole-game losses.",
        "See training analysis for nodes, active time, fitting/checkpoint cost and memory; throughput is not policy quality."]
    (output / "analysis_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def run_analysis(args):
    import torch
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    roots = {key: getattr(args, "candidate_run" if key == "exp36" else key + "_run").resolve() for key in CONFIGS}
    paths = [output, *roots.values()]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError("All inputs and output must be disjoint")
    records = [r for key, root in roots.items() for r in discover(root, key)]
    tasks = build_tasks(records, args)
    repo = Path(__file__).resolve().parents[3]
    signature = dict(schema=1, experiment_id=36, smoke=args.smoke,
        tasks=[shared._portable_task(t) for t in tasks],
        sources=[{k: v for k, v in r.items() if k != "checkpoint_path"} for r in records],
        implementation={
            "runner": sources.sha256_file(Path(__file__)),
            "rare": sources.sha256_file(Path(rare.__file__)),
            "diagnostic_dependencies": rare.implementation_hashes(),
            "feature_evaluation": sources.sha256_file(Path(features.__file__)),
            "protocol": sources.sha256_file(Path(protocol.__file__)),
            "shared": sources.sha256_file(Path(shared.__file__)),
            "source_validation": sources.sha256_file(Path(sources.__file__)),
            "suite": protocol._source_tree_sha256(repo / "fhp_evaluation"),
            "policy": protocol._source_tree_sha256(repo / "fhp_escher"),
            "network": sources.sha256_file(repo / "vr_deep_cfr/solver.py")},
        versions={k: version(k) for k in ("numpy", "scipy", "torch", "open_spiel")},
        python=platform.python_version())
    fingerprint = shared._digest(signature)
    path = output / "evaluation_manifest.json"
    prior = json.loads(path.read_text()) if path.exists() else None
    if output.exists() and not args.resume:
        raise FileExistsError("Use --resume to recover an existing evaluation")
    if prior is not None and prior.get("fingerprint") != fingerprint:
        raise ValueError("Evaluation resume mismatch: policies, runtime or protocol changed")
    if prior is None and output.exists() and any(output.iterdir()):
        raise ValueError("Existing evaluation has no valid manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", fingerprint=fingerprint, signature=signature,
        started_utc=prior["started_utc"] if prior else datetime.now(timezone.utc).isoformat(),
        controls_retrained=False, num_tasks=len(tasks))
    shared._write_json(path, manifest)
    (output / "SUCCESS.json").unlink(missing_ok=True)
    try:
        rare_results = rare.run(output, records, fingerprint, smoke=args.smoke)
        results = shared._run_tasks(tasks, args.workers, output / "task_results", fingerprint)
        summarise(output, records, results, rare_results, smoke=args.smoke)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        shared._write_json(path, manifest)
        raise
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    shared._write_json(path, manifest)
    shared._write_json(output / "SUCCESS.json", dict(status="complete", fingerprint=fingerprint, num_tasks=len(tasks), smoke=args.smoke))
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-run", "exp20-run", "exp33-run", "exp10-run", "output-dir"):
        result.add_argument("--" + name, type=Path, required=True)
    for name, default in (("workers", 16), ("rule-deals", 10000), ("lbr-deals", 1000),
                          ("lbr-rollouts", 4096), ("lbr-shard-deals", 10),
                          ("crossplay-deals", 50000), ("base-seed", 2026101836)):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.lbr_deals < 0 or any(getattr(args, key) <= 0 for key in (
            "workers", "rule_deals", "lbr_rollouts", "lbr_shard_deals", "crossplay_deals")):
        raise ValueError("Budgets must be positive; LBR may explicitly be zero")
    print(run_analysis(args))


if __name__ == "__main__":
    main()
