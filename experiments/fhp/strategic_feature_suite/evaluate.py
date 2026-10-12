"""Read-only 24h comparison with archived Experiment 10; never train a control."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

from experiments.fhp.exp10_fhp_hand_board_features import config as baseline
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _config_sha256
from experiments.fhp.retrospective_exp7_exp8_evaluation import run as sources
from .config import BASELINE_REF

protocol, shared = sources.protocol, sources.shared


def discover(root, experiment, *, control=False):
    cfg = baseline if control else experiment
    key = "exp10" if control else f"exp{experiment.EXPERIMENT_ID}"
    records = sources.discover_checkpoints(root, key, config=cfg, hours=(6, 12, 18, 24),
        label="Archived Experiment 10" if control else experiment.ALGORITHM_LABEL,
        cache=True, allow_continuation=False)
    for worker in sorted((Path(root) / "workers").glob("task_*")):
        manifest = json.loads((worker / "run_manifest.json").read_text())
        summary = json.loads((worker / "summary.json").read_text())
        success = json.loads((worker / "SUCCESS.json").read_text())
        runtime = json.loads((worker / "runtime_manifest.json").read_text())
        if (manifest.get("experiment_id") != cfg.EXPERIMENT_ID
                or manifest.get("training_config_sha256") != _config_sha256(cfg.EXPERIMENT_CONFIG)
                or manifest.get("training_state_retention") != "final"
                or summary.get("status") != "complete" or summary.get("checkpoint_count") != 4
                or summary.get("seed") != manifest["seed"]
                or sources.sha256_file(worker / "summary.json") != success.get("summary_sha256")):
            raise ValueError("Incomplete or incompatible source metadata")
        if control and manifest["repository_commit"] != BASELINE_REF:
            raise ValueError("Unexpected historical Experiment 10 source commit")
        for key, expected in {"python_version": "3.11.16", "torch_version": "2.7.0+cpu",
                              "numpy_version": "1.26.4", "ray_version": "2.51.2"}.items():
            if runtime.get(key) != expected:
                raise ValueError(f"Unexpected source runtime at {key}")
    if len({r["source_repository_commit"] for r in records}) != 1:
        raise ValueError("Mixed source commits within an experiment")
    return records


def build_tasks(records, args, *, experiment):
    candidate = f"exp{experiment.EXPERIMENT_ID}"
    selected = [r for r in records if r["training_hours"] == 24
                and (not args.smoke or r["seed"] == 0)]
    index = {(r["experiment"], r["seed"]): r for r in selected}
    tasks = []
    for row in selected:
        identity = dict(experiment=row["experiment"], training_seed=row["seed"],
                        training_hours=24, **sources._policy_fields(row, "a"))
        for i, opponent in enumerate(protocol.PUBLISHED_AGENT_NAMES):
            tasks.append(dict(identity, task_id=f"rule_{row['experiment']}_{row['seed']}_{opponent}",
                kind="rule", opponent=opponent, num_deals=args.rule_deals,
                evaluation_seed=args.base_seed + 100_000 + i))
        for shard, start in enumerate(range(0, args.lbr_deals, args.lbr_shard_deals)):
            tasks.append(dict(identity, task_id=f"lbr_{row['experiment']}_{row['seed']}_{shard}",
                kind="lbr", opponent="local_best_response", shard_index=shard,
                num_deals=min(args.lbr_shard_deals, args.lbr_deals-start),
                evaluation_seed=args.base_seed+1_000_000+shard,
                lbr_seed=args.base_seed+1_500_000, lbr_rollouts=args.lbr_rollouts))
    for seed in ((0,) if args.smoke else (0, 1, 2)):
        tasks.append(dict(task_id=f"direct_{candidate}_vs_exp10_seed{seed}",
            kind="direct_crossplay", experiment=f"{candidate}_vs_exp10", training_seed=seed,
            training_hours=24, primary_endpoint=True, num_deals=args.crossplay_deals,
            evaluation_seed=args.base_seed+3_000_024,
            **sources._policy_fields(index[(candidate, seed)], "a"),
            **sources._policy_fields(index[("exp10", seed)], "b")))
    return tasks


def summarise(output, records, results, *, experiment, smoke):
    direct = [r for r in results if r["kind"] == "direct_crossplay"]
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    mean_rules = protocol._average_rule_agents(rules)
    tables = {
        "checkpoint_index.csv": records,
        "direct_vs_exp10_by_seed.csv": direct,
        "direct_vs_exp10_aggregate.csv": protocol._group_aggregate(direct, ("training_hours",)),
        "rule_agent_by_seed.csv": rules,
        "rule_agent_aggregate.csv": protocol._group_aggregate(rules, ("experiment", "training_hours", "opponent")),
        "rule_agent_mean_by_seed.csv": mean_rules,
        "rule_agent_mean_aggregate.csv": protocol._group_aggregate(mean_rules, ("experiment", "training_hours")),
        "lbr_by_seed.csv": lbr,
        "lbr_aggregate.csv": protocol._group_aggregate(lbr, ("experiment", "training_hours")),
    }
    candidate = f"exp{experiment.EXPERIMENT_ID}"
    differences = []
    for metric, rows in (("rule_suite_mean", mean_rules), ("lbr_payoff", lbr)):
        baseline_values = {r["training_seed"]: r for r in rows if r["experiment"] == "exp10"}
        for row in rows:
            if row["experiment"] == candidate:
                # Marginal deal-level SEs do not describe a paired difference.
                # Report differences by seed, then infer across training seeds.
                differences.append(dict(training_seed=row["training_seed"],
                    training_hours=24, metric=metric, num_deal_pairs=row["num_deal_pairs"],
                    mean_mbb_per_hand=row["mean_mbb_per_hand"]-baseline_values[row["training_seed"]]["mean_mbb_per_hand"]))
    tables["metric_differences_by_seed.csv"] = differences
    tables["metric_differences_aggregate.csv"] = protocol._group_aggregate(differences, ("metric", "training_hours"))
    for name, rows in tables.items():
        protocol._write_csv(output / name, rows)
    summary = dict(experiment_id=experiment.EXPERIMENT_ID, smoke=smoke,
        primary_endpoint=tables["direct_vs_exp10_aggregate.csv"],
        comparison="candidate 24h versus archived Experiment 10 24h; no retraining",
        positive_direct_favours=candidate, exact_exploitability=False,
        uncertainty="Three source training seeds; intervals are pointwise and exploratory, not adjusted across five experiments",
        interpretation="Historical comparison of full feature packages; different input dimensions, initialization and node counts. No automatic selection; no between-candidate evaluation.")
    shared._write_json(output / "summary.json", summary)
    lines = [f"# Experiment {experiment.EXPERIMENT_ID}: 24h evaluation against Experiment 10", "",
        "Experiment 10 was re-evaluated, not retrained. Positive direct-play value favours the candidate.",
        "The three training seeds, not individual hands, are the inferential units.",
        "LBR reports exploiter payoff (lower is better), not exact exploitability.", "",
        "| Metric | Mean (mbb/hand) | Training-seed 95% interval |", "|---|---:|---:|"]
    for label, rows in (("Direct play", tables["direct_vs_exp10_aggregate.csv"]),
                        ("Candidate minus baseline", tables["metric_differences_aggregate.csv"])):
        for row in rows:
            lo, hi = (row["training_seed_ci95_low_mbb_per_hand"], row["training_seed_ci95_high_mbb_per_hand"])
            interval = "not estimated (smoke)" if smoke else f"[{lo:.2f}, {hi:.2f}]"
            lines.append(f"| {label}: {row.get('metric', 'head-to-head')} | {row['mean_mbb_per_hand']:.2f} | {interval} |")
    lines += ["", summary["uncertainty"], "", summary["interpretation"], "",
        "Inspect individual-seed and individual-opponent results before selecting a configuration.",
        "Feature cost, input parameter count and achieved training nodes are part of the tested package."]
    (output / "analysis_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def run_analysis(args, *, experiment):
    import torch
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    inputs = [args.candidate_run.resolve(), args.exp10_run.resolve()]
    if inputs[0] == inputs[1] or any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError("Evaluation output and both source runs must be disjoint")
    records = discover(inputs[0], experiment) + discover(inputs[1], experiment, control=True)
    tasks = build_tasks(records, args, experiment=experiment)
    repo = Path(__file__).resolve().parents[3]
    signature = dict(schema=1, experiment_id=experiment.EXPERIMENT_ID, smoke=args.smoke,
        tasks=[shared._portable_task(t) for t in tasks],
        sources=[{k: v for k, v in r.items() if k != "checkpoint_path"} for r in records],
        implementation={
            "runner": sources.sha256_file(Path(__file__)),
            "protocol": sources.sha256_file(Path(protocol.__file__)),
            "shared": sources.sha256_file(Path(shared.__file__)),
            "source_validation": sources.sha256_file(Path(sources.__file__)),
            "suite": protocol._source_tree_sha256(repo / "fhp_evaluation"),
            "policy": protocol._source_tree_sha256(repo / "fhp_escher"),
            "network": sources.sha256_file(repo / "vr_deep_cfr/solver.py"),
            "encoder_contract": experiment.encoder().metadata()},
        versions={k: version(k) for k in ("numpy", "scipy", "torch", "open_spiel")},
        python=platform.python_version())
    fingerprint = shared._digest(signature)
    path = output / "evaluation_manifest.json"
    prior = json.loads(path.read_text()) if path.exists() else None
    if output.exists() and not args.resume:
        raise FileExistsError("Use --resume to recover an existing evaluation")
    if prior is not None and prior.get("fingerprint") != fingerprint:
        raise ValueError("Evaluation resume mismatch: policies, features, runtime or protocol changed")
    if prior is None and output.exists() and any(output.iterdir()):
        raise ValueError("Existing evaluation has no valid manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", fingerprint=fingerprint, signature=signature,
        started_utc=prior["started_utc"] if prior else datetime.now(timezone.utc).isoformat(),
        baseline_run_id=experiment.BASELINE_RUN_ID, cross_candidate_evaluation=False,
        baseline_retrained=False, num_tasks=len(tasks))
    shared._write_json(path, manifest)
    (output / "SUCCESS.json").unlink(missing_ok=True)
    try:
        results = shared._run_tasks(tasks, args.workers, output / "task_results", fingerprint)
        summarise(output, records, results, experiment=experiment, smoke=args.smoke)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        shared._write_json(path, manifest)
        raise
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    shared._write_json(path, manifest)
    shared._write_json(output / "SUCCESS.json", dict(status="complete", fingerprint=fingerprint,
        num_tasks=len(tasks), smoke=args.smoke))
    return output


def main(argv=None, *, experiment):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-run", "exp10-run", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name, default in (("workers", 16), ("rule-deals", 10000), ("lbr-deals", 1000),
                          ("lbr-rollouts", 4096), ("lbr-shard-deals", 10),
                          ("crossplay-deals", 50000), ("base-seed", 20260922)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.lbr_deals < 0 or any(getattr(args, k) <= 0 for k in (
            "workers", "rule_deals", "lbr_rollouts", "lbr_shard_deals", "crossplay_deals")):
        raise ValueError("Budgets must be positive (LBR may explicitly be zero)")
    print(run_analysis(args, experiment=experiment))
