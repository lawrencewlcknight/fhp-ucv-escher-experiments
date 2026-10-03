"""Evaluate Exp7 at 24h and Exp8 at 24/30/36/42/48h, without retraining."""
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
from experiments.fhp.exp7_fhp_parallel_structured_n2_standard16 import config as exp7_config
from experiments.fhp.exp8_fhp_parallel_48h import config as exp8_config
from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file
from fhp_escher.features import ENCODER_ID, make_feature_encoder

EXPECTED_SEEDS = (0, 1, 2)
SOURCE_HOURS = {"exp7": (6, 12, 18, 24), "exp8": tuple(range(6, 49, 6))}
EVALUATED_HOURS = {"exp7": (24,), "exp8": (24, 30, 36, 42, 48)}
CONFIGS = {"exp7": exp7_config, "exp8": exp8_config}
EXPERIMENTS = {
    key: {"experiment_name": config.EXPERIMENT_NAME, "algorithm_id": config.ALGORITHM_ID,
          "label": "Exp7: 24-hour baseline" if key == "exp7" else "Exp8: extended training"}
    for key, config in CONFIGS.items()
}
_digest = shared._digest
_write_json = shared._write_json
_portable_task = shared._portable_task
_run_tasks = shared._run_tasks


def discover_checkpoints(root: Path, experiment: str) -> list[dict]:
    """Validate the complete original schedule, including policies not scored."""
    root = Path(root).resolve()
    config = CONFIGS[experiment]
    hours = SOURCE_HOURS[experiment]
    workers_root = root / "workers"
    if not workers_root.is_dir():
        raise FileNotFoundError(f"Missing workers directory: {workers_root}")
    records, seeds = [], set()
    game = protocol.load_fhp_game()
    encoder = make_feature_encoder(config.EXPERIMENT_CONFIG.get("feature_encoder_id", ENCODER_ID))
    for worker in sorted(workers_root.glob("task_*")):
        if not worker.is_dir() or not worker.resolve().is_relative_to(root):
            raise ValueError(f"Invalid or escaping source worker: {worker}")
        manifest_path = worker / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for key in ("experiment_name", "algorithm_id"):
            if manifest.get(key) != EXPERIMENTS[experiment][key]:
                raise ValueError(f"Wrong {key}: {worker}")
        seed = manifest.get("seed")
        if type(seed) is not int or seed not in EXPECTED_SEEDS or seed in seeds:
            raise ValueError(f"Invalid or duplicate {experiment} seed: {seed}")
        seeds.add(seed)
        if manifest.get("smoke") is not False:
            raise ValueError(f"Not a production training run: {worker}")
        if protocol._json_safe(manifest.get("training_config")) != protocol._json_safe(config.EXPERIMENT_CONFIG):
            raise ValueError(f"Unexpected {experiment} learning configuration: {worker}")
        if manifest.get("checkpoint_schedule") != list(config.checkpoint_schedule()):
            raise ValueError(f"Unexpected source checkpoint schedule: {worker}")
        if manifest.get("training_duration_seconds") != hours[-1] * 3600:
            raise ValueError(f"Unexpected source training horizon: {worker}")
        if manifest.get("game", {}).get("parameters") != protocol.FHP_GAME_PARAMETERS:
            raise ValueError(f"Wrong source game: {worker}")
        if (worker / "continuation_source.json").exists():
            raise ValueError(f"Expected an original run, not an imported continuation: {worker}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("repository_commit", ""))):
            raise ValueError(f"Missing or invalid source commit: {worker}")
        if json.loads((worker / "SUCCESS.json").read_text()).get("status") != "complete":
            raise ValueError(f"Unsuccessful source worker: {worker}")
        runtime = json.loads((worker / "runtime_manifest.json").read_text())
        if (runtime.get("reference_vm", {}).get("machine_type") != "n2-standard-16"
                or runtime.get("torch_intraop_threads") != 8
                or runtime.get("torch_interop_threads") != 8
                or runtime.get("frozen_critic_target_cache") is not False
                or runtime.get("traversal_execution") != "ray_parallel"
                or runtime.get("parallel_settings") != config.parallel_settings()):
            raise ValueError(f"Wrong source runtime: {worker}")
        checkpoint_manifest = worker / "checkpoint_manifest.json"
        rows = json.loads(checkpoint_manifest.read_text())
        if len(rows) != len(hours):
            raise ValueError(f"Incomplete checkpoint schedule: {worker}")
        by_hour = {}
        for row in rows:
            target = float(row["checkpoint_target_seconds"])
            if not np.isfinite(target) or target not in {h * 3600 for h in hours}:
                raise ValueError(f"Unexpected checkpoint target: {worker}")
            hour = int(target / 3600)
            if hour in by_hour:
                raise ValueError(f"Duplicate checkpoint hour: {worker}")
            path = protocol._checkpoint_path(worker, row)
            if not path.is_relative_to(worker.resolve()):
                raise ValueError(f"Checkpoint escapes source worker: {path}")
            observed_hash = sha256_file(path)
            if observed_hash != row["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {path}")
            restored = LoadedFHPPolicy(game, path)  # Also validates game and network/encoder layout.
            expected = {
                "experiment_name": config.EXPERIMENT_NAME, "algorithm_id": config.ALGORITHM_ID,
                "seed": seed, "nodes_touched": row["nodes_touched"],
                "outer_iteration": row["outer_iteration"], "checkpoint_target_seconds": target,
                "training_config": manifest["training_config"], "feature_encoder": encoder.metadata(),
            }
            mismatches = [key for key, value in expected.items()
                          if protocol._json_safe(restored.checkpoint.get(key)) != protocol._json_safe(value)]
            if mismatches:
                raise ValueError(f"Checkpoint metadata mismatch ({', '.join(mismatches)}): {path}")
            elapsed = float(row["actual_training_elapsed_seconds"])
            if not np.isfinite(elapsed) or elapsed < target:
                raise ValueError(f"Checkpoint precedes its training threshold: {path}")
            if row["nodes_touched"] <= 0 or row["outer_iteration"] <= 0:
                raise ValueError(f"Checkpoint has no completed training: {path}")
            by_hour[hour] = {
                "experiment": experiment, "experiment_label": EXPERIMENTS[experiment]["label"],
                "seed": seed, "training_hours": hour, "checkpoint_id": row["checkpoint_id"],
                "checkpoint_path": str(path), "checkpoint_sha256": observed_hash,
                "nodes_touched": int(row["nodes_touched"]), "outer_iteration": int(row["outer_iteration"]),
                "training_elapsed_seconds": elapsed,
                "run_manifest_sha256": sha256_file(manifest_path),
                "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
                "training_config_sha256": _digest(manifest["training_config"]),
                "source_repository_commit": manifest["repository_commit"], "source_runtime": runtime,
                "selected_for_evaluation": hour in EVALUATED_HOURS[experiment],
            }
        if tuple(sorted(by_hour)) != hours:
            raise ValueError(f"Incomplete checkpoint schedule: {worker}")
        for earlier, later in zip(hours, hours[1:]):
            if any(by_hour[later][key] < by_hour[earlier][key]
                   for key in ("nodes_touched", "outer_iteration", "training_elapsed_seconds")):
                raise ValueError(f"Non-monotone training history: {worker}")
        records.extend(by_hour[hour] for hour in hours)
    if tuple(sorted(seeds)) != EXPECTED_SEEDS:
        raise ValueError(f"{experiment} requires seeds {EXPECTED_SEEDS}; found {sorted(seeds)}")
    return sorted(records, key=lambda row: (row["seed"], row["training_hours"]))


def _contrasts():
    """Prespecified orientation: A is always Exp8, and later for temporal play."""
    result = []
    for earlier, later in itertools.combinations(EVALUATED_HOURS["exp8"], 2):
        result.append({
            "comparison_id": f"exp8_{later}h_vs_exp8_{earlier}h", "kind": "temporal_crossplay",
            "left_experiment": "exp8", "right_experiment": "exp8",
            "left_hours": later, "right_hours": earlier,
            "primary_endpoint": (later, earlier) == (48, 24),
            "adjacent_checkpoint": later - earlier == 6,
            "late_stage_endpoint": (later, earlier) in ((48, 42), (48, 36)),
        })
    for hour in EVALUATED_HOURS["exp8"]:
        result.append({
            "comparison_id": f"exp8_{hour}h_vs_exp7_24h", "kind": "direct_crossplay",
            "left_experiment": "exp8", "right_experiment": "exp7",
            "left_hours": hour, "right_hours": 24, "primary_endpoint": False,
            "adjacent_checkpoint": False, "late_stage_endpoint": False,
        })
    return result


def _selected_records(records, args):
    return [r for r in records if r["training_hours"] in EVALUATED_HOURS[r["experiment"]]
            and (not args.smoke or r["seed"] == 0)]


def _policy_fields(row, side):
    return {f"policy_{side}_path": row["checkpoint_path"],
            f"policy_{side}_name": f"{row['experiment']}_seed_{row['seed']}_time_{row['training_hours']:02d}h",
            f"policy_{side}_sha256": row["checkpoint_sha256"],
            f"policy_{side}_nodes_touched": row["nodes_touched"]}


def _build_tasks(records, args):
    """Same scoring, random seeds and budgets as earlier evaluations; new schedule."""
    selected = _selected_records(records, args)
    by_key = {(r["experiment"], r["seed"], r["training_hours"]): r for r in selected}
    tasks = []
    for row in selected:
        identity = {"experiment": row["experiment"], "training_seed": row["seed"],
                    "training_hours": row["training_hours"], **_policy_fields(row, "a")}
        for index, opponent in enumerate(protocol.PUBLISHED_AGENT_NAMES):
            tasks.append({**identity, "task_id": f"rule_{identity['policy_a_name']}_{opponent}",
                          "kind": "rule", "opponent": opponent, "num_deals": args.rule_deals,
                          "evaluation_seed": args.base_seed + 100_000 + index})
        for shard, start in enumerate(range(0, args.lbr_deals, args.lbr_shard_deals)):
            tasks.append({**identity, "task_id": f"lbr_{identity['policy_a_name']}_shard_{shard:04d}",
                          "kind": "lbr", "opponent": "local_best_response", "shard_index": shard,
                          "num_deals": min(args.lbr_shard_deals, args.lbr_deals - start),
                          "evaluation_seed": args.base_seed + 1_000_000 + shard,
                          "lbr_seed": args.base_seed + 1_500_000, "lbr_rollouts": args.lbr_rollouts})
    for seed in ((0,) if args.smoke else EXPECTED_SEEDS):
        for contrast in _contrasts():
            left, right = contrast["left_hours"], contrast["right_hours"]
            temporal = contrast["kind"] == "temporal_crossplay"
            tasks.append({
                **contrast, "task_id": f"{contrast['comparison_id']}_seed_{seed}",
                "experiment": "exp8" if temporal else "exp8_vs_exp7",
                "training_seed": seed, "training_hours": left,
                **({"earlier_hours": right, "later_hours": left} if temporal else {}),
                **_policy_fields(by_key[(contrast["left_experiment"], seed, left)], "a"),
                **_policy_fields(by_key[(contrast["right_experiment"], seed, right)], "b"),
                "num_deals": args.crossplay_deals,
                "evaluation_seed": args.base_seed + (2_000_000 + right * 1000 + left
                                                     if temporal else 3_000_000 + left),
            })
    return tasks


def _paired_differences(rows, metric):
    by_key = {(r["experiment"], r["training_seed"], r["training_hours"]): r for r in rows}
    output = []
    for seed in sorted({r["training_seed"] for r in rows}):
        for contrast in _contrasts():
            left = by_key[(contrast["left_experiment"], seed, contrast["left_hours"])]
            right = by_key[(contrast["right_experiment"], seed, contrast["right_hours"])]
            output.append({**contrast, "metric": metric, "training_seed": seed,
                           "mean_mbb_per_hand": left["mean_mbb_per_hand"] - right["mean_mbb_per_hand"],
                           "num_deal_pairs": left["num_deal_pairs"]})
    return output


def _aggregate_contrasts(rows, *, metric=False):
    aggregate = protocol._group_aggregate(rows, ("metric", "comparison_id") if metric else ("comparison_id",))
    metadata = {c["comparison_id"]: c for c in _contrasts()}
    for row in aggregate:
        row.update(metadata[row["comparison_id"]])
    return aggregate


def _plot_analysis(output, rule, lbr, direct, temporal, differences, *, smoke=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fhp-eval78-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []

    def save(fig, name):
        if smoke:
            fig.suptitle("SMOKE TEST — integration only; not policy-strength evidence", fontsize=10)
        fig.savefig(output / name, dpi=180)
        plt.close(fig)
        figures.append(name)

    def trajectory(ax, rows, key, **kwargs):
        rows = sorted(rows, key=lambda row: row[key])
        scale = 1e6 if key == "mean_nodes_touched" else 1
        x = np.array([r[key] / scale for r in rows])
        y = np.array([r["mean_mbb_per_hand"] for r in rows])
        low = np.array([r["training_seed_ci95_low_mbb_per_hand"] for r in rows])
        high = np.array([r["training_seed_ci95_high_mbb_per_hand"] for r in rows])
        line, = ax.plot(x, y, marker="o", **kwargs)
        finite = np.isfinite(low) & np.isfinite(high)
        ax.vlines(x[finite], low[finite], high[finite], color=line.get_color(), alpha=.6)
        ax.grid(alpha=.2)

    for name, rows, ylabel in (("rule_agent", rule, "Policy payoff (mbb/hand; higher is better)"),
                               ("lbr", lbr, "LBR payoff (mbb/hand; lower is better)")):
        for axis, key in (("time", "training_hours"), ("nodes", "mean_nodes_touched")):
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for experiment in EXPERIMENTS:
                trajectory(ax, [r for r in rows if r["experiment"] == experiment], key,
                           label=EXPERIMENTS[experiment]["label"])
            ax.set(xlabel="Active training hours" if axis == "time" else "Mean training nodes (millions)",
                   ylabel=ylabel, title="Frozen-policy diagnostics: seed-level 95% intervals")
            ax.legend(fontsize=8)
            save(fig, f"{name}_by_{axis}.png")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    trajectory(ax, direct, "left_hours")
    ax.axhline(0, color="grey", linewidth=.8)
    ax.set(xlabel="Experiment 8 active training hours", ylabel="Exp8 payoff (mbb/hand)",
           title="Exp8 versus fixed Exp7 24h baseline: seed-level 95% intervals")
    save(fig, "exp8_vs_exp7_24h.png")
    hours = EVALUATED_HOURS["exp8"]
    matrix = np.full((len(hours), len(hours)), np.nan)
    for row in temporal:
        i, j = hours.index(row["left_hours"]), hours.index(row["right_hours"])
        matrix[i, j], matrix[j, i] = row["mean_mbb_per_hand"], -row["mean_mbb_per_hand"]
    limit = max(1, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    artist = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
    for i, j in itertools.product(range(len(hours)), repeat=2):
        ax.text(j, i, f"{matrix[i, j]:.1f}" if np.isfinite(matrix[i, j]) else "—", ha="center", va="center")
    ax.set_xticks(range(len(hours)), hours)
    ax.set_yticks(range(len(hours)), hours)
    ax.set(xlabel="Opponent checkpoint (hours)", ylabel="Row checkpoint (hours)",
           title="Exp8 checkpoint cross-play (upper triangle is sign-reversed)")
    fig.colorbar(artist, ax=ax, label="Row-policy payoff (mbb/hand)")
    save(fig, "exp8_temporal_crossplay.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, rows, title in zip(axes, (temporal,
            [r for r in differences if r["metric"] == "rule_agent_mean"],
            [r for r in differences if r["metric"] == "lbr_payoff"]),
            ("Direct later vs earlier (+ better)", "Rule payoff change (+ better)", "LBR payoff change (− better)")):
        trajectory(ax, [r for r in rows if r["adjacent_checkpoint"]], "left_hours")
        ax.axhline(0, color="grey", linewidth=.8)
        ax.set_xticks((30, 36, 42, 48), ("24→30", "30→36", "36→42", "42→48"), rotation=25)
        ax.set(title=title, xlabel="Consecutive checkpoints (hours)", ylabel="mbb/hand (95% seed interval)")
    save(fig, "exp8_six_hour_gains.png")
    return figures


def _summarise(output, records, results, *, smoke=False):
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    temporal = [r for r in results if r["kind"] == "temporal_crossplay"]
    direct = [r for r in results if r["kind"] == "direct_crossplay"]
    rule_mean = protocol._average_rule_agents(rules)
    differences = _paired_differences(rule_mean, "rule_agent_mean") + _paired_differences(lbr, "lbr_payoff")
    aggregate_rule = protocol._group_aggregate(rule_mean, ("experiment", "training_hours"))
    aggregate_lbr = protocol._group_aggregate(lbr, ("experiment", "training_hours"))
    aggregate_temporal, aggregate_direct = _aggregate_contrasts(temporal), _aggregate_contrasts(direct)
    aggregate_differences = _aggregate_contrasts(differences, metric=True)
    for rows in (aggregate_rule, aggregate_lbr):
        protocol._attach_mean_nodes(rows, records)
    late = [r for r in aggregate_temporal if r["primary_endpoint"] or r["late_stage_endpoint"]]
    tables = {
        "checkpoint_index.csv": records,
        "rule_agent_by_seed.csv": rules,
        "rule_agent_aggregate.csv": protocol._group_aggregate(rules, ("experiment", "training_hours", "opponent")),
        "rule_agent_mean_by_seed.csv": rule_mean, "rule_agent_mean_aggregate.csv": aggregate_rule,
        "lbr_by_seed.csv": lbr, "lbr_aggregate.csv": aggregate_lbr,
        "paired_metric_differences_by_seed.csv": differences,
        "paired_metric_differences_aggregate.csv": aggregate_differences,
        "temporal_crossplay_by_seed.csv": temporal, "temporal_crossplay_aggregate.csv": aggregate_temporal,
        "direct_exp8_vs_exp7_by_seed.csv": direct, "direct_exp8_vs_exp7_aggregate.csv": aggregate_direct,
        "adjacent_checkpoint_gains_by_seed.csv": [r for r in temporal if r["adjacent_checkpoint"]],
        "adjacent_checkpoint_gains_aggregate.csv": [r for r in aggregate_temporal if r["adjacent_checkpoint"]],
        "late_training_endpoints.csv": late,
    }
    for name, rows in tables.items():
        protocol._write_csv(output / name, rows)
    figures = _plot_analysis(output, aggregate_rule, aggregate_lbr, aggregate_direct,
                             aggregate_temporal, aggregate_differences, smoke=smoke)
    lines = ["# Frozen-policy evaluation: FHP Experiments 7 and 8", "",
             "SMOKE TEST ONLY — not policy-strength evidence." if smoke else "Full-budget evaluation.", "",
             "No training or policy refitting. All values are in mbb/hand.", "",
             "Primary endpoint: Exp8 48h versus its own 24h policy, within each training seed. "
             "Prespecified late-stage diagnostics: 48h versus 42h and 48h versus 36h. "
             "All ten within-Exp8 pairings are reported, not just the best checkpoint.", ""]
    for title, rows in (("Primary and late-stage endpoints", late),
                         ("Six-hour incremental gains", tables["adjacent_checkpoint_gains_aggregate.csv"]),
                         ("Historical Exp7 24h baseline", aggregate_direct)):
        lines.extend([f"## {title}", "", "| Comparison (A vs B) | A payoff | Training-seed 95% CI |",
                      "|---|---:|---:|"])
        for row in rows:
            low, high = row["training_seed_ci95_low_mbb_per_hand"], row["training_seed_ci95_high_mbb_per_hand"]
            interval = f"[{low:.2f}, {high:.2f}]" if np.isfinite(low) and np.isfinite(high) else "Not estimable (one seed)"
            lines.append(f"| {row['comparison_id']} | {row['mean_mbb_per_hand']:.2f} | "
                         f"{interval} |")
        lines.append("")
    lines.extend([
        "## Interpretation and limits", "",
        "Positive head-to-head scores favour Exp8/the later checkpoint. Higher rule payoff is better; "
        "lower LBR payoff is better. Diagnostic differences are A minus B: negative LBR differences favour A.", "",
        "Production intervals use the three matched training-seed means, not hands, checkpoints or LBR shards as "
        "independent training replicates. The same deals are reused across seeds and diagnostics as in prior "
        "evaluations. Seed intervals are conditional on that deal schedule, not a full decomposition of "
        "training and Monte Carlo uncertainty; per-match deal uncertainty is also retained. "
        "Secondary contrasts are exploratory with unadjusted intervals, and overlapping checkpoint "
        "comparisons are correlated. Three seeds have limited power.", "",
        "Look for sustained late direct-play gains accompanied by improved rule-agent payoffs and/or "
        "lower LBR payoffs, consistently across seeds. Such evidence would support testing longer training. "
        "A flat point estimate or interval containing zero does not establish equivalence or convergence. "
        "Inspect the upper/lower bounds for still-plausible gains: wide intervals mean inconclusive evidence, "
        "not a plateau. No practical-equivalence margin has been prespecified, so this report makes no "
        "automatic convergence decision or extrapolated prediction beyond 48 hours.", "",
        "Cross-play can be non-transitive and performance against earlier selves is not a global strength "
        "measure. LBR is restricted and approximate, not exact exploitability; even agreement among "
        "these diagnostics only supports a plateau on this evaluation suite, not equilibrium convergence.", "",
        "Exp8 24h versus Exp7 24h checks historical repeatability; Exp8 48h versus Exp7 24h checks the "
        "external baseline. Within-Exp8 comparisons best isolate additional training along the same trajectory. "
        "Matching labels across separately trained runs does not imply identical trajectories. Source "
        "commits, configuration, hashes and node counts are recorded. Active training time excludes policy "
        "checkpoint fitting, serialization and upload; node plots are descriptive, not matched-node tests.", "",
    ])
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
    if len({r["training_config_sha256"] for r in records}) != 1:
        raise ValueError("Experiment 7 and 8 learning configurations differ")
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
            "evaluated_hours": EVALUATED_HOURS,
            "contrasts": _contrasts(),
            "primary_endpoint": "exp8_48h_vs_exp8_24h",
            "late_stage_endpoints": ["exp8_48h_vs_exp8_42h", "exp8_48h_vs_exp8_36h"],
            "training_config_difference": "none; active_training_horizon_only",
            "convergence_claim": "not_established_by_this_evaluation",
            "num_scored_policies": len(_selected_records(records, args)),
            "total_deal_pairs": sum(task["num_deals"] for task in tasks),
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
