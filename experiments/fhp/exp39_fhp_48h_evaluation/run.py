"""Resumable 48-hour comparison and paired 24-to-48-hour continuation evaluation."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from experiments.fhp.exp36_fhp_strength_critic2500 import evaluate as existing
from experiments.fhp.exp37_fhp_critic_budget_evaluation.execution import run_tasks
from . import sources as contracts, rare
from .statistics import COHORTS, PAIRS, collapse_direct, comparisons, decision_screen, paired_changes

sources, protocol, shared = existing.sources, existing.protocol, existing.shared
INTERPRETATION = (
    "Exp38 versus Exp16 compares complete 48h configurations: richer AE input features AND "
    "2,500 rather than 10,000 critic updates. It cannot isolate either change. Exp38 versus Exp36 "
    "tests continuing the same three trajectories from 24 to 48 active hours. A win over an earlier "
    "policy is not a convergence test; corroborate with unchanged external opponents. "
    "No retraining, no safeguard, no exact exploitability.")
UNCERTAINTY = (
    "All nine cross-seed pairings per primary contrast involve only three trajectories per cohort. "
    "Bootstrap resampling shares trajectory weights between Exp36 and Exp38; Exp16 is independent. "
    "The all-pair empirical cohort average includes diagonal and off-diagonal pairings; the three "
    "same-trajectory predecessor matches are also reported separately as a secondary endpoint. "
    "Independent cell-level Gaussian deal-SE perturbations are a conservative sensitivity addition "
    "because empirical model effects already contain deal noise. Pointwise 95% and two-comparison "
    "Bonferroni 97.5% percentile intervals are exploratory: three seeds do not give reliable "
    "confirmatory coverage. Conditional deal intervals describe only these fixed models.")


def build_tasks(records, args):
    seeds = (0,) if args.smoke else (0, 1, 2)
    selected = [r for r in records if r["training_hours"] == contracts.HOURS[r["experiment"]] and r["seed"] in seeds]
    index = {(r["experiment"], r["seed"]): r for r in selected}
    if len(index) != len(selected) or set(index) != {(c, s) for c in COHORTS for s in seeds}:
        raise ValueError("Require exactly one specified endpoint checkpoint per selected cohort/seed")
    tasks = []
    for a, b in PAIRS:
        for i in seeds:
            for j in seeds:
                for shard, start in enumerate(range(0, args.crossplay_deals, args.crossplay_shard_deals)):
                    tasks.append(dict(task_id=f"direct_{a}_{i}_vs_{b}_{j}_{shard}",
                        kind="direct_crossplay", experiment=f"{a}_vs_{b}", cohort_a=a, cohort_b=b,
                        seed_a=i, seed_b=j, training_hours=48, hours_a=contracts.HOURS[a],
                        hours_b=contracts.HOURS[b], shard_index=shard,
                        num_deals=min(args.crossplay_shard_deals, args.crossplay_deals-start),
                        **sources._policy_fields(index[(a, i)], "a"),
                        **sources._policy_fields(index[(b, j)], "b")))
    for r in selected:
        identity = dict(experiment=r["experiment"], training_seed=r["seed"], training_hours=r["training_hours"],
                        **sources._policy_fields(r, "a"))
        for opponent in protocol.PUBLISHED_AGENT_NAMES:
            tasks.append(dict(identity, kind="rule", opponent=opponent, num_deals=args.rule_deals,
                task_id=f"rule_{r['experiment']}_{r['seed']}_{opponent}"))
        for shard, start in enumerate(range(0, args.lbr_deals, args.lbr_shard_deals)):
            tasks.append(dict(identity, kind="lbr", opponent="local_best_response", shard_index=shard,
                task_id=f"lbr_{r['experiment']}_{r['seed']}_{shard}", lbr_rollouts=args.lbr_rollouts,
                num_deals=min(args.lbr_shard_deals, args.lbr_deals-start)))
    # Direct match cells/shards are independent. Common rule-agent tests use
    # matched deal/action streams across the same 36/38 trajectory.
    for task in tasks:
        stream = task["task_id"]
        if task["kind"] == "rule":
            stream = stream.replace("exp36_", "continuation_").replace("exp38_", "continuation_")
        task["random_stream_id"] = stream
        task["evaluation_seed"] = int(shared._digest([args.base_seed, stream, "deal"])[:15], 16)
        if task["kind"] == "lbr":
            task["lbr_seed"] = int(shared._digest([args.base_seed, stream, "lbr"])[:15], 16)
    streams = {t["random_stream_id"]: t["evaluation_seed"] for t in tasks}
    if len(set(streams.values())) != len(streams):
        raise ValueError("Random-stream collision")
    return tasks


def training_context(roots):
    """Read already validated source summaries; never download training states."""
    rows = []
    for cohort, root in roots.items():
        for path in sorted((root / "workers").glob("task_*/summary.json")):
            data = json.loads(path.read_text())
            d = data.get("execution_diagnostics", {})
            active = data.get("final_training_elapsed_seconds")
            nodes = data.get("final_nodes_touched")
            rows.append(dict(experiment=cohort, seed=data["seed"], source_summary_sha256=sources.sha256_file(path),
                critic_updates=contracts.CONFIGS[cohort].EXPERIMENT_CONFIG["baseline_network_train_steps"],
                feature_encoder=contracts.CONFIGS[cohort].EXPERIMENT_CONFIG["feature_encoder_id"],
                nodes=nodes, iterations=data.get("final_iteration"), active_seconds=active,
                counter_scope="cumulative_through_checkpoint_including_restored_source_counters",
                nodes_per_active_second=nodes/active if nodes is not None and active else None,
                started_utc=data.get("started_utc"), finished_utc=data.get("finished_utc"),
                peak_driver_rss_mib=d.get("parallel_driver_peak_rss_mib"),
                critic_fit_seconds=d.get("cumulative_critic_fit_seconds"),
                critic_cache_build_seconds=d.get("cumulative_critic_cache_build_seconds"),
                collection_seconds=d.get("cumulative_experience_collection_seconds"),
                learner_seconds=d.get("cumulative_parallel_learner_seconds"),
                reference_vm=data.get("reference_vm")))
    return rows


def plots(output, direct, cells, rules, lbr, *, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"exp16": "Original / 10,000 / 48h", "exp36": "Richer / 2,500 / 24h", "exp38": "Richer / 2,500 / 48h"}
    suffix = " — SMOKE ONLY" if smoke else ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (a, b) in zip(axes, PAIRS):
        rows = [r for r in cells if r["cohort_a"] == a and r["cohort_b"] == b]
        n = 1 if smoke else 3
        values = np.zeros((n, n))
        for r in rows:
            values[r["seed_a"], r["seed_b"]] = r["mean_mbb_per_hand"]
        bound = max(1., float(np.max(np.abs(values))))
        image = ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound)
        for r in rows:
            ax.text(r["seed_b"], r["seed_a"], f"{r['mean_mbb_per_hand']:.1f}", ha="center", va="center")
        ax.set(xticks=range(n), yticks=range(n), xlabel=f"{b} seed", ylabel=f"{a} seed", title=f"{a} versus {b}")
        fig.colorbar(image, ax=ax, label="mbb/hand, row policy payoff", shrink=.7)
    fig.suptitle("All cross-seed matchups" + suffix)
    fig.savefig(output / "seed_pair_payoffs.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, row in enumerate(direct):
        ax.plot(row["mean_mbb_per_hand"], i, "o", color="navy")
        for prefix, width, color in (("bootstrap_ci95", 5, "steelblue"), ("bootstrap_family_ci", 2, "navy")):
            lo, hi = row[prefix+"_low"], row[prefix+"_high"]
            if lo is not None:
                ax.hlines(i, lo, hi, linewidth=width, color=color)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(yticks=range(2), yticklabels=[r["comparison"] for r in direct],
           xlabel="mbb/hand; positive favours first configuration",
           title="Trajectory-aware intervals: 95% thick / 97.5% thin" + suffix)
    fig.savefig(output / "head_to_head_intervals.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, rows, title in ((axes[0], rules, "Mean rule-panel payoff (higher better)"),
                            (axes[1], lbr, "LBR exploiter payoff (lower better)")):
        for i, c in enumerate(COHORTS):
            values = [r["mean_mbb_per_hand"] for r in rows if r["experiment"] == c]
            if values:
                ax.scatter([i]*len(values), values)
                ax.plot([i-.15, i+.15], [np.mean(values)]*2, color="black")
        ax.set(xticks=range(3), xticklabels=[titles[c] for c in COHORTS], ylabel="mbb/hand", title=title+suffix)
        ax.tick_params(axis="x", rotation=15)
    fig.savefig(output / "common_opponent_scores.png", dpi=180)
    plt.close(fig)


def summarise(output, records, results, context, rare_result, args):
    cells = collapse_direct(results, protocol.MILLI_BIG_BLINDS_PER_CHIP)
    direct, matched = comparisons(cells, smoke=args.smoke)
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    mean_rules = protocol._average_rule_agents(rules)
    changes, change_summary = paired_changes(mean_rules, lbr)
    costs = []
    for c in COHORTS:
        rows = [r for r in context if r["experiment"] == c]
        means = {}
        for key in ("nodes", "iterations", "active_seconds", "nodes_per_active_second", "critic_fit_seconds", "collection_seconds"):
            values = [r[key] for r in rows if r[key] is not None]
            means["mean_"+key] = float(np.mean(values)) if len(values) == len(rows) and values else None
        costs.append(dict(experiment=c, models=len(rows), **means))
    rare_means = []
    for c in COHORTS:
        rows = [r for r in rare_result.get("by_seed", []) if r["experiment"] == c and r["group"] == "assessment/all"]
        if rows:
            rare_means.append(dict(experiment=c, models=len(rows), positions_per_model=rows[0]["positions"],
                mean_fold_probability=float(np.mean([r["mean_fold_probability"] for r in rows])),
                mean_gap_bb=float(np.mean([r["mean_gap_bb"] for r in rows])),
                mean_state_p95_gap_bb=float(np.mean([r["p95_gap_bb"] for r in rows])),
                fraction_fold_over_half=float(np.mean([r["fraction_fold_over_half"] for r in rows]))))
    tables = {"checkpoint_index.csv": records, "training_context.csv": context,
        "training_context_summary.csv": costs, "rare_assessment_summary.csv": rare_means,
        "head_to_head_by_seed_pair.csv": cells, "head_to_head_summary.csv": direct,
        "same_trajectory_head_to_head.csv": [matched],
        "paired_common_opponent_changes.csv": changes, "paired_common_opponent_summary.csv": change_summary,
        "rule_agent_by_seed.csv": rules, "rule_panel_mean_by_seed.csv": mean_rules,
        "lbr_by_seed.csv": lbr,
        "rule_agent_summary.csv": protocol._group_aggregate(rules, ("experiment", "opponent")),
        "rule_panel_summary.csv": protocol._group_aggregate(mean_rules, ("experiment",)),
        "lbr_summary.csv": protocol._group_aggregate(lbr, ("experiment",))}
    for name, rows in tables.items():
        if rows:
            protocol._write_csv(output / name, rows)
    summary = dict(experiment_id=39, smoke=args.smoke, exact_exploitability=False,
        training_hours=contracts.HOURS, head_to_head=direct, decision=decision_screen(direct),
        same_trajectory_head_to_head=matched, paired_common_opponent_changes=change_summary,
        evaluation_task_seconds_sum=sum(r.get("evaluation_task_seconds", 0.) for r in results),
        evaluation_cost_note="Sum of completed task wall times (including model loading), not job elapsed time. Attempts and Batch resource logs cover setup, failed work and other overhead separately.",
        rule_panel=tables["rule_panel_summary.csv"], lbr=tables["lbr_summary.csv"],
        rare_diagnostics=rare_result, rare_assessment_summary=rare_means,
        training_context=context, training_context_summary=costs,
        interpretation=INTERPRETATION, uncertainty=UNCERTAINTY,
        supporting_intervals="Rule/LBR summaries use the existing pointwise three-seed intervals. Diagnostic, not multiplicity-adjusted primary endpoints.")
    shared._write_json(output / "summary.json", summary)
    plots(output, direct, cells, mean_rules, lbr, smoke=args.smoke)
    lines = ["# Experiment 39: 48-hour configurations and continuation benefit", "",
        "**SMOKE TEST: no strength conclusions.**" if args.smoke else "Frozen policies: three training seeds per configuration; all nine cross-seed pairings per contrast.",
        "", INTERPRETATION, "", "## Primary head-to-head results", "",
        "Positive payoff favours the first configuration. A duplicated deal pair includes both seats.", "",
        "| Comparison | mbb/hand | Crossed-model 95% interval | Two-comparison adjusted interval |",
        "|---|---:|---|---|"]
    for row in direct:
        def interval(prefix):
            return "not estimated" if args.smoke else f"[{row[prefix+'_low']:.2f}, {row[prefix+'_high']:.2f}]"
        lines.append(f"| {row['comparison']} | {row['mean_mbb_per_hand']:.2f} | {interval('bootstrap_ci95')} | {interval('bootstrap_family_ci')} |")
    lines += ["", UNCERTAINTY, "", "Primary signals: " + str(summary["decision"]["primary_signals"]),
        summary["decision"]["qualification"], "", "## Same-trajectory predecessor matches", "",
        f"Exp38 versus its own Exp36 predecessor: {matched['mean_mbb_per_hand']:.2f} mbb/hand, "
        f"exploratory 95% interval [{matched['bootstrap_ci95_low']}, {matched['bootstrap_ci95_high']}]. "
        "This is the mean of three diagonal matchups, not the all-nine-pair endpoint.",
        "", "## Supporting evidence", "",
        "| Configuration | Rule-panel mean (mbb/hand; higher better) | LBR payoff (mbb/hand; lower better) |",
        "|---|---:|---:|"]
    for c in COHORTS:
        rule = next(r["mean_mbb_per_hand"] for r in summary["rule_panel"] if r["experiment"] == c)
        attack = next((r["mean_mbb_per_hand"] for r in summary["lbr"] if r["experiment"] == c), None)
        attack_text = f"{attack:.2f}" if attack is not None else "disabled"
        lines.append(f"| {c} | {rule:.2f} | {attack_text} |")
    lines += ["", "LBR is a restricted exploiter, not exact exploitability. The common panel is secondary; no global scalar rank is forced in a non-transitive game.",
        "", "Rare-state folding probabilities and conditional BB losses are in rare_by_seed.csv; rare_results/ contains every example. The exact frozen Exp37 panel is reused: even historically assessment-labelled positions are now known diagnostics, not a fresh holdout. These losses are not reach-weighted match losses.",
        "", "training_context.csv includes nodes, active seconds, critic/collection/learner timings and memory. Critic time is inside learner time; cache construction is inside critic time. Do not add these nested timers. Throughput alone does not establish policy strength. The pinned native restart format restores cumulative critic and parallel counters, so reported counters include the original 24h segment. Wall-time setup, evaluation and infrastructure cost are separate from nominal active training hours."]
    if rare_means:
        lines += ["", "### Previously examined assessment-labelled positions (means across trained seeds)", "",
            "| Configuration | Fold probability | Conditional loss (BB) | Mean state-level p95 loss (BB) |",
            "|---|---:|---:|---:|"]
        for r in rare_means:
            lines.append(f"| {r['experiment']} | {100*r['mean_fold_probability']:.2f}% | {r['mean_gap_bb']:.3f} | {r['mean_state_p95_gap_bb']:.3f} |")
    lines += ["", "### Historical training cost (means across three seeds)", "",
        "| Configuration | Million nodes | Active hours | Critic fit hours | Collection hours |",
        "|---|---:|---:|---:|---:|"]
    for r in costs:
        values = ["unavailable" if r[k] is None else f"{r[k]/scale:.2f}" for k, scale in (
            ("mean_nodes", 1e6), ("mean_active_seconds", 3600),
            ("mean_critic_fit_seconds", 3600), ("mean_collection_seconds", 3600))]
        lines.append("| "+r["experiment"]+" | "+" | ".join(values)+" |")
    lines += ["", "### Paired external changes: 48h minus 24h", "",
        "Rule deals/action streams are matched within each trajectory; LBR streams are independent. "
        "These secondary intervals resample neither hands nor positions as training seeds.", "",
        "| Metric | Change (mbb/hand) | Paired-seed 95% interval | Better |",
        "|---|---:|---|---|"]
    for row in change_summary:
        lines.append(f"| {row['metric']} | {row['mean_change_mbb_per_hand']:.2f} | "
            f"[{row['ci95_low']}, {row['ci95_high']}] | {row['better_direction']} |")
    (output / "analysis_summary.md").write_text("\n".join(lines)+"\n")
    return summary


def run_analysis(args):
    validate(args)
    import torch
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    roots = {c: getattr(args, c+"_run").resolve() for c in COHORTS}
    paths = [output, args.source10_run.resolve(), *roots.values()]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError("All inputs and output must be disjoint")
    rare.load_panel(args.rare_panel)
    records = contracts.discover(roots, args.source10_run.resolve())
    tasks = build_tasks(records, args)
    context = training_context(roots)
    repo = Path(__file__).resolve().parents[3]
    signature = dict(schema=1, experiment_id=39, smoke=args.smoke, rare_panel_sha256=rare.PANEL_SHA256,
        tasks=[shared._portable_task(t) for t in tasks], training_context=context,
        sources=[{k: v for k, v in r.items() if k != "checkpoint_path"} for r in records],
        implementation={"experiment": protocol._source_tree_sha256(Path(__file__).parent),
            "executor": protocol._source_tree_sha256(Path(__file__).parent.parent / "exp37_fhp_critic_budget_evaluation"),
            "old_contract": sources.sha256_file(Path(contracts.old_contract.__file__)),
            "new_contract": sources.sha256_file(Path(contracts.new_contract.__file__)),
            "source_evaluator": sources.sha256_file(Path(existing.__file__)),
            "rare": sources.sha256_file(Path(rare.base.__file__)), "rare_dependencies": rare.implementation_hashes(),
            "feature_evaluation": sources.sha256_file(Path(existing.features.__file__)),
            "protocol": sources.sha256_file(Path(protocol.__file__)),
            "shared": sources.sha256_file(Path(shared.__file__)),
            "source_validation": sources.sha256_file(Path(sources.__file__)),
            "suite": protocol._source_tree_sha256(repo / "fhp_evaluation"),
            "policy": protocol._source_tree_sha256(repo / "fhp_escher"),
            "network": sources.sha256_file(repo / "vr_deep_cfr/solver.py")},
        versions={k: version(k) for k in ("numpy", "scipy", "torch", "open_spiel")}, python=platform.python_version())
    fingerprint = shared._digest(signature)
    path = output / "evaluation_manifest.json"
    prior = json.loads(path.read_text()) if path.exists() else None
    if output.exists() and not args.resume:
        raise FileExistsError("Use --resume for existing output")
    if prior is not None and prior.get("fingerprint") != fingerprint:
        raise ValueError("Evaluation resume mismatch: sources, code, runtime or budgets changed")
    if prior is None and output.exists() and any(output.iterdir()):
        raise ValueError("Existing output has no valid manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", fingerprint=fingerprint, signature=signature, num_tasks=len(tasks),
        controls_retrained=False, started_utc=prior["started_utc"] if prior else datetime.now(timezone.utc).isoformat(),
        attempts=list(prior.get("attempts", [])) if prior else [])
    shared._write_json(path, manifest)
    (output / "SUCCESS.json").unlink(missing_ok=True)
    started = perf_counter()
    try:
        print(f"Validated {len(records)} policies; {len(tasks)} resumable tasks", flush=True)
        results = run_tasks(tasks, args.workers, output / "task_results", fingerprint)
        print("Match tasks complete; assessing terminal-call diagnostic panel", flush=True)
        rare_result = rare.run(output, records, fingerprint, smoke=args.smoke, panel_path=args.rare_panel)
        summarise(output, records, results, context, rare_result, args)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    finally:
        manifest["attempts"].append(dict(seconds=perf_counter()-started, workers=args.workers, status=manifest["status"]))
        shared._write_json(path, manifest)
    shared._write_json(output / "SUCCESS.json", dict(status="complete", fingerprint=fingerprint, num_tasks=len(tasks), smoke=args.smoke))
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source10-run", "exp16-run", "exp36-run", "exp38-run", "rare-panel", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    for name, default in (("workers", 16), ("rule-deals", 10000), ("lbr-deals", 1000),
            ("lbr-rollouts", 4096), ("lbr-shard-deals", 10), ("crossplay-deals", 50000),
            ("crossplay-shard-deals", 5000), ("base-seed", 2026102139)):
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def validate(args):
    if args.lbr_deals < 0 or args.base_seed < 0 or any(getattr(args, k) <= 0 for k in (
            "workers", "rule_deals", "lbr_rollouts", "lbr_shard_deals", "crossplay_deals", "crossplay_shard_deals")):
        raise ValueError("Positive budgets required; LBR may explicitly be zero")
    if min(args.rule_deals, args.crossplay_deals) < 2 or args.lbr_deals == 1:
        raise ValueError("At least two duplicated pairs per endpoint; use zero to disable LBR")


def main(argv=None):
    print(run_analysis(parser().parse_args(argv)))


if __name__ == "__main__":
    main()
