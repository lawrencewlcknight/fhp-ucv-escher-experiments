"""Paired source-seed inference; fitting replicates are not extra training seeds."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.fhp.exp4_fhp_average_policy_audit.aggregate import csv_file, summary
from experiments.fhp.exp13_fhp_policy_capacity.aggregate import collapse_replicates, difference_summary
from experiments.fhp.exp4_fhp_average_policy_audit.fitting import write_json
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES
from .config import contract, arm_name, fit_directory
from .evaluation import expected_metrics, policy_names


def aggregate(root, *, smoke=False, include_lbr=False):
    from .run import identity
    root, config = Path(root), contract(smoke, include_lbr)
    rows, sources, curves, details, boundaries = [], [], [], {}, []

    def add(seed, arm, metric, value):
        base, rep = arm.rsplit("_rep", 1) if arm != "archived" else (arm, None)
        rows.append({"seed": seed, "arm": base, "replicate": None if rep is None else int(rep),
                     "metric": metric, "value": value})

    for seed in config["seeds"]:
        worker = root / "workers" / f"seed_{seed}"
        read = lambda name: json.loads((worker / name).read_text())
        manifest, success, replay = read("manifest.json"), read("SUCCESS.json"), read("replay_diagnostics.json")
        if (manifest["config"] != config or manifest["provenance"]["seed"] != seed
                or success != {"seed": seed, "status": "complete", "smoke": smoke,
                               "manifest_sha256": identity(manifest)}):
            raise ValueError("Incomplete worker or incompatible audit manifest")
        if sources and any(manifest[k] != sources[0][k] for k in
                           ("audit_commit", "audit_source_sha256", "python", "torch", "numpy")):
            raise ValueError("Source workers used different audit code/runtime")
        sources.append(manifest)
        fits, policy_index = read("fit_metrics.json"), read("policy_index.json")
        if set(policy_index) != set(policy_names(config)):
            raise ValueError("Incomplete playable-policy index")
        if policy_index["archived"]["sha256"] != manifest["provenance"]["source_policy_sha256"]:
            raise ValueError("Archived policy identity changed")
        expected = {(phase, recipe, rep, u) for phase in ("diagnostic", "deployment")
                    for recipe in config["recipes"] for rep in config["replicates"] for u in config["updates"]}
        if len(fits) != len(expected) or {(r["phase"], r["recipe"], r["replicate"], r["updates"]) for r in fits} != expected:
            raise ValueError("Incomplete fitting results")
        matching = {}
        for phase in ("diagnostic", "deployment"):
            for recipe in config["recipes"]:
                for rep in config["replicates"]:
                    path = fit_directory(worker, phase, recipe, rep)
                    fc = json.loads((path / "fit_contract.json").read_text())
                    path_rows = json.loads((path / "metrics.json").read_text())
                    if (fc["config"] != config or fc["seed"] != seed or fc["phase"] != phase
                            or fc["recipe"] != recipe or fc["replicate"] != rep
                            or fc["architecture"] != "standard" or fc["updates"] != config["updates"]
                            or path_rows != [r for r in fits if r["phase"] == phase and r["recipe"] == recipe and r["replicate"] == rep]):
                        raise ValueError("Fitting path identity mismatch")
        for fit in fits:
            phase, recipe, rep, update = (fit[k] for k in ("phase", "recipe", "replicate", "updates"))
            split = "training" if phase == "diagnostic" else "full"
            validation_hash = replay["splits"]["heldout"]["data_sha256"] if phase == "diagnostic" else None
            expected_examples = update * min(config["batch_size"], replay["splits"][split]["groups"])
            if (fit["seed"] != seed or fit["architecture"] != "standard"
                    or fit["data_sha256"] != replay["splits"][split]["data_sha256"]
                    or fit["validation_data_sha256"] != validation_hash
                    or fit["processed_examples"] != expected_examples):
                raise ValueError("Fitting data/budget identity mismatch")
            key = phase, rep, update
            match = tuple(fit[k] for k in ("initial_state_sha256", "sample_sequence_sha256", "processed_examples"))
            if key in matching and matching[key] != match:
                raise ValueError("Learning-rate arms used unmatched initialisations or samples")
            matching[key] = match
            name = arm_name(recipe, rep)
            curve = {k: fit[k] for k in ("seed", "phase", "recipe", "replicate", "updates", "fitting_seconds", "processed_examples")}
            for split_name in ("training", "validation"):
                if fit[split_name] is not None:
                    for metric in ("weighted_ce", "weighted_kl", "weighted_l1"):
                        value = fit[split_name]["all"][metric]
                        curve[f"{split_name}_{metric}"] = value
                        if update == config["updates"][-1]:
                            add(seed, name, f"{phase}_{split_name}_{metric}", value)
            curves.append(curve)
            if update == config["updates"][-1]:
                add(seed, name, f"{phase}_fitting_seconds", fit["fitting_seconds"])
                if phase == "deployment" and fit["policy_sha256"] != policy_index[name]["sha256"]:
                    raise ValueError("Gameplay policy does not match deployed final fit")
        evaluations = read("evaluation_summary.json")
        if len(evaluations) != len(expected_metrics(config)) or {(r["arm"], r["metric"]) for r in evaluations} != expected_metrics(config):
            raise ValueError("Incomplete gameplay evaluation")
        for row in evaluations:
            if row["metric"] != "lbr_mbb_per_hand":
                if row["policy_a_sha256"] != policy_index[row["arm"]]["sha256"] or row["source_seed"] != seed:
                    raise ValueError("Evaluation source/policy identity mismatch")
                expected_deals = config["rule_deals"] if row["kind"] == "rule" else config["direct_deals"]
                if row["num_deal_pairs"] != expected_deals or row["num_games"] != 2 * expected_deals:
                    raise ValueError("Evaluation budget mismatch")
                if row["kind"] == "direct_crossplay" and row["policy_b_sha256"] != policy_index[row["policy_b_name"]]["sha256"]:
                    raise ValueError("Direct-play opponent changed")
            elif row["num_deal_pairs"] != config["lbr_deals"]:
                raise ValueError("Incomplete LBR shards")
            add(seed, row["arm"], row["metric"], row["mean_mbb_per_hand"])
        for name in policy_names(config):
            rule_values = [r["mean_mbb_per_hand"] for r in evaluations if r["arm"] == name and r["metric"].startswith("rule_")]
            if len(rule_values) != len(PUBLISHED_AGENT_NAMES):
                raise ValueError("Incomplete rule suite")
            add(seed, name, "rule_suite_mean", float(np.mean(rule_values)))
        details[str(seed)] = {"replay": replay, "fits": fits, "evaluation": evaluations,
                              "resources": read("audit_resources.json")}
    per_seed = collapse_replicates(rows, config)
    values = defaultdict(dict)
    for row in per_seed:
        values[(row["metric"], row["arm"])][row["seed"]] = row["value"]
    if any(set(v) != set(config["seeds"]) for v in values.values()):
        raise ValueError("Missing source seed")
    summaries = [{"metric": metric, "arm": arm, **summary(list(v.values()))} for (metric, arm), v in sorted(values.items())]
    contrasts = []
    for metric in sorted({m for m, _ in values}):
        for candidate, control in (("adam_0003", "adam_003"), ("adam_0003", "archived"), ("adam_003", "archived")):
            if (metric, candidate) in values and (metric, control) in values:
                diffs = np.asarray([values[(metric, candidate)][s] - values[(metric, control)][s] for s in config["seeds"]])
                contrasts.append({"metric": metric, "contrast": f"{candidate}_minus_{control}", **difference_summary(diffs)})
        # A direct matchup is itself the paired contrast, not a difference of two win rates.
        if metric.startswith("direct_crossplay_"):
            for (m, arm), v in values.items():
                if m == metric:
                    contrasts.append({"metric": metric, "contrast": arm + "_versus_zero", **difference_summary(np.asarray(list(v.values())))})
    for recipe in config["recipes"]:
        def endpoint_mean(update):
            return float(np.mean([r["validation_weighted_ce"] for r in curves if r["phase"] == "diagnostic"
                                  and r["recipe"] == recipe and r["updates"] == update]))
        change = endpoint_mean(config["updates"][-1]) - endpoint_mean(config["updates"][-2])
        boundaries.append({"recipe": recipe, "last_interval_heldout_ce_change": change,
                           "still_improving_at_fixed_budget": change < 0})
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    interpretation = (
        "Matched reset 0.0003 versus 0.003 is primary; archived 0.003 is an additional historical control. "
        "Two fitting replicates are averaged within each of three reused source training seeds. "
        "No checkpoint or rate is selected using held-out or gameplay outcomes. Diagnostic fits use "
        "group-disjoint 90/10 replay; deployed fits use all replay. Held-out fidelity is not exploitability. "
        "All intervals and sign-flip tests are exploratory; three sources give minimum two-sided p=0.25. "
        "A fixed-budget null is not equivalence or evidence that the lower rate cannot help with more updates. "
        "Restricted LBR is not exact exploitability; omitted LBR is not reported as zero."
    )
    write_json(analysis / "summary.json", {"config": config, "status": "complete", "summaries": summaries,
        "paired_contrasts": contrasts, "budget_warnings": boundaries, "interpretation": interpretation})
    write_json(analysis / "source_manifests.json", sources)
    write_json(analysis / "detailed_diagnostics.json", details)
    for name, data in (("per_replicate_metrics", rows), ("per_seed_metrics", per_seed), ("aggregate_metrics", summaries),
                       ("paired_contrasts", contrasts), ("learning_curves", curves), ("budget_warnings", boundaries)):
        csv_file(analysis / f"{name}.csv", data)
    plot(analysis, config, curves, summaries)
    (analysis / "README.md").write_text("# Experiment 15: frozen policy learning-rate audit\n\n" + interpretation
        + "\n\nOnly playable policies, metadata, diagnostics and evaluation results are retained.\n"
        + ("\nSMOKE ONLY: tiny/synthetic results are not scientific evidence.\n" if smoke else ""))
    return analysis


def plot(directory, config, curves, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("training_weighted_kl", "validation_weighted_kl")):
        for recipe in config["recipes"]:
            y, err = [], []
            for u in config["updates"]:
                seed_values = [np.mean([r[metric] for r in curves if r["phase"] == "diagnostic"
                               and r["recipe"] == recipe and r["updates"] == u and r["seed"] == s]) for s in config["seeds"]]
                stats = summary(seed_values)
                y.append(stats["mean"]); err.append(stats["se"] or 0)
            axis.errorbar(config["updates"], y, yerr=err, marker="o", label=recipe, capsize=3)
        axis.set(xlabel="Updates", ylabel=metric, title="Group-disjoint diagnostic fits")
        axis.legend(); axis.grid(alpha=.2)
    fig.suptitle("SMOKE ONLY" if config["smoke"] else "Experiment 15: source-seed mean ± SE")
    fig.savefig(directory / "learning_rate_fidelity.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("rule_suite_mean", "direct_crossplay_archived", "direct_crossplay_matched_003")):
        selected = [r for r in summaries if r["metric"] == metric]
        for i, row in enumerate(selected):
            axis.errorbar(i, row["mean"], yerr=row["se"] or 0, fmt="o", capsize=4)
        axis.axhline(0, color="grey", lw=.7)
        axis.set_xticks(range(len(selected)), [r["arm"] for r in selected], rotation=20)
        axis.set(title=metric, ylabel="mbb/hand; higher is better")
    fig.suptitle("SMOKE ONLY" if config["smoke"] else "Full-replay fixed-budget policies: mean ± source-seed SE")
    fig.savefig(directory / "learning_rate_policy_quality.png", dpi=180); plt.close(fig)
