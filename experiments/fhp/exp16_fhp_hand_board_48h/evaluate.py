"""Frozen-policy evaluation of the actual Experiment 10 continuation; no fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform

import numpy as np

from experiments.fhp.exp10_fhp_hand_board_features import config as learner
from experiments.fhp.retrospective_exp7_exp8_evaluation import run as long_eval
from experiments.fhp.retrospective_exp9_exp10_exp11_exp12_evaluation import run as feature_eval
from . import contract as c

protocol = long_eval.protocol
shared = long_eval.shared
sha256_file = long_eval.sha256_file
LABELS = {"exp16": "Exp10 continued to 48h", "exp9": "Fixed Exp9 24h", "exp8": "Fixed Exp8 48h"}
EVALUATED_HOURS = {"exp16": c.EVALUATED_HOURS, "exp9": (24,), "exp8": (48,)}


def discover_sources(roots):
    """Reject mixed cohorts, changed source policies and invalid continuation lineage."""
    continued = long_eval.discover_checkpoints(
        roots["exp16"], "exp16", config=learner, hours=c.HOURS,
        label=LABELS["exp16"], cache=True, allow_continuation=True)
    original = feature_eval.discover_checkpoints(roots["source10"], "exp10")
    original_index = {(r["seed"], r["training_hours"]): r for r in original}
    for seed in c.SEEDS:
        source = roots["source10"] / "workers" / c.worker_name(seed)
        destination = roots["exp16"] / "workers" / c.worker_name(seed)
        source_data = c.validate_metadata({n: (source / n).read_bytes() for n in c.METADATA_FILES}, seed=seed)
        c.validate_metadata({n: (destination / n).read_bytes() for n in c.METADATA_FILES}, seed=seed, hours=48)
        lineage = json.loads((destination / "continuation_source.json").read_text())
        final = source_data["checkpoint_manifest.json"][-1]
        expected = {"seed": seed, "source_total_hours": 24, "total_hours": 48,
                    "source_commit": c.TRAINING_REF, "source_state_path": final["training_state_path"],
                    "source_state_sha256": final["training_state_sha256"],
                    "source_summary_sha256": source_data["SUCCESS.json"]["summary_sha256"]}
        if any(lineage.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Invalid Experiment 10 continuation lineage: seed {seed}")
        if not lineage.get("source_worker", "").endswith(f"/{c.SOURCE_RUN_ID}/workers/{c.worker_name(seed)}"):
            raise ValueError("Continuation used a different source cohort")
        rows = [r for r in continued if r["seed"] == seed]
        for record in rows:
            hour = record["training_hours"]
            if hour <= 24 and any(record[k] != original_index[(seed, hour)][k] for k in
                                  ("checkpoint_sha256", "nodes_touched", "outer_iteration", "training_elapsed_seconds")):
                raise ValueError("Imported 6–24h policy history differs from Experiment 10")
        if rows[-1]["outer_iteration"] <= original_index[(seed, 24)]["outer_iteration"]:
            raise ValueError("No additional training occurred after restore")
    records = (continued + feature_eval.discover_checkpoints(roots["exp9"], "exp9")
               + long_eval.discover_checkpoints(roots["exp8"], "exp8"))
    for experiment in LABELS:
        if len({r["source_repository_commit"] for r in records if r["experiment"] == experiment}) != 1:
            raise ValueError(f"Mixed training commits in {experiment}")
    for r in records:
        r["selected_for_evaluation"] = r["training_hours"] in EVALUATED_HOURS[r["experiment"]]
    return records


def contrasts():
    result = []
    for earlier, later in itertools.combinations(c.EVALUATED_HOURS, 2):
        result.append({"comparison_id": f"exp16_{later}h_vs_exp16_{earlier}h",
                       "kind": "temporal_crossplay", "left_experiment": "exp16", "right_experiment": "exp16",
                       "left_hours": later, "right_hours": earlier,
                       "primary_endpoint": (later, earlier) == (48, 24),
                       "late_stage_endpoint": (later, earlier) in ((48, 36), (48, 42)),
                       "adjacent_checkpoint": later - earlier == 6})
    for opponent, endpoint in (("exp9", 24), ("exp8", 48)):
        for hour in c.EVALUATED_HOURS:
            result.append({"comparison_id": f"exp16_{hour}h_vs_{opponent}_{endpoint}h",
                           "kind": "direct_crossplay", "left_experiment": "exp16", "right_experiment": opponent,
                           "left_hours": hour, "right_hours": endpoint,
                           "primary_endpoint": hour == 48, "late_stage_endpoint": False,
                           "adjacent_checkpoint": False})
    return result


def selected_records(records, args):
    return [r for r in records if r["training_hours"] in EVALUATED_HOURS[r["experiment"]]
            and (not args.smoke or r["seed"] == 0)]


def build_tasks(records, args):
    selected = selected_records(records, args)
    index = {(r["experiment"], r["seed"], r["training_hours"]): r for r in selected}
    tasks = []
    for row in selected:
        identity = {"experiment": row["experiment"], "training_seed": row["seed"],
                    "training_hours": row["training_hours"], **long_eval._policy_fields(row, "a")}
        for number, opponent in enumerate(protocol.PUBLISHED_AGENT_NAMES):
            tasks.append({**identity, "task_id": f"rule_{identity['policy_a_name']}_{opponent}",
                          "kind": "rule", "opponent": opponent, "num_deals": args.rule_deals,
                          "evaluation_seed": args.base_seed + 100_000 + number})
        for shard, start in enumerate(range(0, args.lbr_deals, args.lbr_shard_deals)):
            tasks.append({**identity, "task_id": f"lbr_{identity['policy_a_name']}_shard_{shard:04d}",
                          "kind": "lbr", "opponent": "local_best_response", "shard_index": shard,
                          "num_deals": min(args.lbr_shard_deals, args.lbr_deals - start),
                          "evaluation_seed": args.base_seed + 1_000_000 + shard,
                          "lbr_seed": args.base_seed + 1_500_000, "lbr_rollouts": args.lbr_rollouts})
    for seed in ((0,) if args.smoke else c.SEEDS):
        for contrast in contrasts():
            left, right = contrast["left_hours"], contrast["right_hours"]
            temporal = contrast["kind"] == "temporal_crossplay"
            # Reuse common deals across all candidate horizons against each frozen
            # opponent, enabling paired estimates of *external* improvement.
            evaluation_seed = (args.base_seed + 2_000_000 + right * 1000 + left if temporal
                               else args.base_seed + 3_000_000 + int(contrast["right_experiment"][3:]))
            tasks.append({**contrast, "task_id": f"{contrast['comparison_id']}_seed_{seed}",
                          "experiment": "exp16" if temporal else f"exp16_vs_{contrast['right_experiment']}",
                          "training_seed": seed, "training_hours": left,
                          **({"earlier_hours": right, "later_hours": left} if temporal else {}),
                          **long_eval._policy_fields(index[("exp16", seed, left)], "a"),
                          **long_eval._policy_fields(index[(contrast["right_experiment"], seed, right)], "b"),
                          "num_deals": args.crossplay_deals, "evaluation_seed": evaluation_seed})
    return tasks


def external_changes(rule_mean, lbr, direct):
    """Within-trajectory differences against unchanged external tests; not self-play."""
    metrics = {"rule_agent_mean": [r for r in rule_mean if r["experiment"] == "exp16"],
               "lbr_payoff": [r for r in lbr if r["experiment"] == "exp16"]}
    metrics.update({f"payoff_vs_{opponent}": [r for r in direct if r["right_experiment"] == opponent]
                    for opponent in ("exp9", "exp8")})
    result = []
    for metric, rows in metrics.items():
        index = {(r["training_seed"], r["training_hours"]): r for r in rows}
        for seed in sorted({r["training_seed"] for r in rows}):
            for earlier, later in itertools.combinations(c.EVALUATED_HOURS, 2):
                a, b = index[(seed, later)], index[(seed, earlier)]
                result.append({"metric": metric, "comparison_id": f"{later}h_minus_{earlier}h",
                               "training_seed": seed, "earlier_hours": earlier, "later_hours": later,
                               "mean_mbb_per_hand": a["mean_mbb_per_hand"] - b["mean_mbb_per_hand"],
                               "num_deal_pairs": a["num_deal_pairs"]})
    return result


def _plots(output, rule_mean, lbr, direct, temporal, *, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []

    def plot(rows, xkey, xlabel, ylabel, name, groups):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for label, selected in groups(rows):
            selected = sorted(selected, key=lambda r: r[xkey])
            x = np.array([r[xkey] for r in selected]) / (1e6 if xkey == "mean_nodes_touched" else 1)
            y = np.array([r["mean_mbb_per_hand"] for r in selected])
            low = np.array([r["training_seed_ci95_low_mbb_per_hand"] for r in selected])
            high = np.array([r["training_seed_ci95_high_mbb_per_hand"] for r in selected])
            line, = ax.plot(x, y, "o-", label=label)
            finite = np.isfinite(low) & np.isfinite(high)
            ax.vlines(x[finite], low[finite], high[finite], color=line.get_color(), alpha=.6)
        ax.axhline(0, color="grey", linewidth=.6)
        ax.set(xlabel=xlabel, ylabel=ylabel, title="SMOKE ONLY" if smoke else "Seed-level 95% intervals (exploratory)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.savefig(output / name, dpi=180)
        plt.close(fig)
        figures.append(name)

    cohorts = lambda rows: [(LABELS[e], [r for r in rows if r["experiment"] == e]) for e in LABELS]
    for metric, rows, ylabel in (("rule_agent", rule_mean, "Policy payoff (mbb/hand; higher is better)"),
                                 ("lbr", lbr, "LBR payoff (mbb/hand; lower is better)")):
        for key, axis in (("training_hours", "hours"), ("mean_nodes_touched", "nodes")):
            plot(rows, key, "Active hours" if axis == "hours" else "Mean training nodes (millions)",
                 ylabel, f"{metric}_by_{axis}.png", cohorts)
    plot(direct, "left_hours", "Experiment 10 cumulative active hours", "Candidate payoff (mbb/hand)",
         "fixed_learned_panel.png", lambda rows: [(LABELS[e], [r for r in rows if r["right_experiment"] == e])
                                                  for e in ("exp9", "exp8")])
    plot([r for r in temporal if r["right_hours"] == 24], "left_hours", "Cumulative active hours",
         "Later policy payoff against its own 24h policy (mbb/hand)", "temporal_gains.png",
         lambda rows: [("Own 24h predecessor", rows)])
    return figures


def summarise(output, records, results, *, smoke):
    rules = [r for r in results if r["kind"] == "rule"]
    lbr = protocol._collapse_lbr_shards([r for r in results if r["kind"] == "lbr"])
    temporal = [r for r in results if r["kind"] == "temporal_crossplay"]
    direct = [r for r in results if r["kind"] == "direct_crossplay"]
    rule_mean = protocol._average_rule_agents(rules)
    changes = external_changes(rule_mean, lbr, direct)
    aggregate_rule = protocol._group_aggregate(rule_mean, ("experiment", "training_hours"))
    aggregate_lbr = protocol._group_aggregate(lbr, ("experiment", "training_hours"))
    for rows in (aggregate_rule, aggregate_lbr):
        protocol._attach_mean_nodes(rows, records)
    metadata = {r["comparison_id"]: r for r in contrasts()}

    def aggregate_matches(rows):
        return [dict(r, **{k: v for k, v in metadata[r["comparison_id"]].items() if k != "comparison_id"})
                for r in protocol._group_aggregate(rows, ("comparison_id",))]

    aggregate_temporal, aggregate_direct = aggregate_matches(temporal), aggregate_matches(direct)
    aggregate_changes = protocol._group_aggregate(changes, ("metric", "comparison_id", "earlier_hours", "later_hours"))
    tables = {"checkpoint_index.csv": records, "rule_agent_by_seed.csv": rules,
              "rule_agent_aggregate.csv": protocol._group_aggregate(rules, ("experiment", "training_hours", "opponent")),
              "rule_agent_mean_by_seed.csv": rule_mean, "rule_agent_mean_aggregate.csv": aggregate_rule,
              "lbr_by_seed.csv": lbr, "lbr_aggregate.csv": aggregate_lbr,
              "temporal_crossplay_by_seed.csv": temporal, "temporal_crossplay_aggregate.csv": aggregate_temporal,
              "fixed_panel_crossplay_by_seed.csv": direct, "fixed_panel_crossplay_aggregate.csv": aggregate_direct,
              "external_improvement_by_seed.csv": changes, "external_improvement_aggregate.csv": aggregate_changes,
              "late_training_endpoints.csv": [r for r in aggregate_temporal if r["primary_endpoint"] or r["late_stage_endpoint"]]}
    for name, rows in tables.items():
        protocol._write_csv(output / name, rows)
    figures = _plots(output, aggregate_rule, aggregate_lbr, aggregate_direct, aggregate_temporal, smoke=smoke)
    report = ["# Experiment 16: Experiment 10 continued from 24 to 48 active hours", "",
              "SMOKE TEST ONLY; not strength evidence." if smoke else "Full-budget frozen-policy evaluation.", "",
              "No refitting. Positive direct scores favour the continued policy; lower LBR payoff is better.", "",
              "## Prespecified 48h temporal and external comparisons", "",
              "| Comparison | Mean mbb/hand | Seed-level 95% interval |", "|---|---:|---:|"]
    for r in [*tables["late_training_endpoints.csv"], *[r for r in aggregate_direct if r["left_hours"] == 48]]:
        low, high = r["training_seed_ci95_low_mbb_per_hand"], r["training_seed_ci95_high_mbb_per_hand"]
        interval = f"[{low:.3f}, {high:.3f}]" if np.isfinite(low) and np.isfinite(high) else "Not estimable (one seed)"
        report.append(f"| {r['comparison_id']} | {r['mean_mbb_per_hand']:.3f} | "
                      f"{interval} |")
    report.extend(["", "## Interpretation", "",
        "The experimental units are three continuing training seeds. Deal pairs, checkpoints and LBR shards "
        "are not independent training replications. Intervals are pointwise and exploratory, conditional "
        "on the fixed opponent panel and evaluation deal schedule. Per-match Monte Carlo uncertainty is also saved.", "",
        "Each seed plays its same-label frozen Exp9 24h and Exp8 48h opponent at every candidate horizon. "
        "These are historical controls, not paired training interventions or an all-cross-seed league. "
        "Rule agents and learned opponent hashes are fixed before scoring. External improvement tables "
        "subtract earlier from later payoff against the SAME opponents, including 48h minus 24h and 36h. "
        "Positive direct value alone is not proof of general improvement or transitive strength.", "",
        "Review seed consistency, external-payoff changes and LBR alongside temporal play before considering "
        "72–96 hours. LBR is a restricted approximate exploiter, not exact exploitability. Neither a self-play "
        "win nor an inconclusive interval establishes equilibrium convergence or a plateau. There is no "
        "automatic horizon extension or pass/fail convergence claim. All scheduled checkpoints are reported; "
        "no best-checkpoint selection. Active time excludes policy fitting, evaluation, serialization and upload.", ""])
    (output / "analysis_summary.md").write_text("\n".join(report))
    return [*tables, *figures, "analysis_summary.md"]


def run_analysis(args):
    import torch
    torch.set_num_threads(1)
    roots = {key: getattr(args, key + "_run").resolve() for key in (*LABELS, "source10")}
    output = args.output_dir.resolve()
    if len(set(roots.values())) != 4 or any(output.is_relative_to(r) or r.is_relative_to(output) for r in roots.values()):
        raise ValueError("All input roots and output must be distinct and disjoint")
    if output.exists() and not args.resume:
        raise FileExistsError("Use --resume for an existing evaluation directory")
    records = discover_sources(roots)
    tasks = build_tasks(records, args)
    repo = Path(__file__).resolve().parents[3]
    signature = {"schema_version": 1, "smoke": args.smoke,
                 "tasks": [shared._portable_task(t) for t in tasks],
                 "sources": [{k: v for k, v in r.items() if k != "checkpoint_path"} for r in records],
                 "implementation": {"evaluation": sha256_file(Path(__file__)),
                                    "contract": sha256_file(Path(c.__file__)),
                                    "long_eval": sha256_file(Path(long_eval.__file__)),
                                    "feature_eval": sha256_file(Path(feature_eval.__file__)),
                                    "shared": sha256_file(Path(shared.__file__)),
                                    "protocol": sha256_file(Path(protocol.__file__)),
                                    **{name: protocol._source_tree_sha256(repo / name) for name in
                                       ("fhp_evaluation", "fhp_escher", "vr_deep_cfr")}},
                 "versions": {name: version(name) for name in ("numpy", "scipy", "torch", "open_spiel")},
                 "python": platform.python_version()}
    fingerprint = shared._digest(signature)
    manifest_path = output / "evaluation_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old is not None and old["fingerprint"] != fingerprint:
        raise ValueError("Resume rejected: policies, protocol, code or dependencies changed")
    if old is None and output.exists() and any(output.iterdir()):
        raise ValueError("Cannot resume outputs without a valid manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment_id": 16, "status": "in_progress", "fingerprint": fingerprint,
                "signature": signature, "started_utc": old["started_utc"] if old else datetime.now(timezone.utc).isoformat(),
                "source_runs": {k: str(v) for k, v in roots.items()}, "checkpoints": records,
                "num_tasks": len(tasks), "total_deal_pairs": sum(t["num_deals"] for t in tasks),
                "evaluation": {"contrasts": contrasts(), "exact_exploitability": False,
                               "uncertainty_unit": "training_seed", "automatic_extension": False}}
    shared._write_json(manifest_path, manifest)
    (output / "SUCCESS.json").unlink(missing_ok=True)
    try:
        results = shared._run_tasks(tasks, args.workers, output / "task_results", fingerprint)
        artifacts = summarise(output, records, results, smoke=args.smoke)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        shared._write_json(manifest_path, manifest)
        shared._write_json(output / "failure.json", {"error": manifest["error"]})
        raise
    (output / "failure.json").unlink(missing_ok=True)
    manifest.update(status="complete", artifacts=artifacts, finished_utc=datetime.now(timezone.utc).isoformat())
    shared._write_json(manifest_path, manifest)
    shared._write_json(output / "SUCCESS.json", {"status": "complete", "smoke": args.smoke,
                                               "fingerprint": fingerprint, "num_tasks": len(tasks)})
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key in (*LABELS, "source10"):
        result.add_argument("--" + key + "-run", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    for name, default in (("workers", 16), ("rule-deals", 10_000), ("lbr-deals", 1000),
                          ("lbr-rollouts", 4096), ("lbr-shard-deals", 10), ("crossplay-deals", 50_000),
                          ("base-seed", 20260922)):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def main():
    args = parser().parse_args()
    for name in ("workers", "rule_deals", "lbr_deals", "lbr_rollouts", "lbr_shard_deals", "crossplay_deals"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    print(f"Evaluation complete: {run_analysis(args)}")


if __name__ == "__main__":
    main()
