"""Average fitting replicas within source; never count replicas or hands as seeds."""

from collections import defaultdict
import itertools
import json
from pathlib import Path
import numpy as np

from experiments.fhp.exp4_fhp_average_policy_audit.aggregate import summary, csv_file
from .config import contract, arm_specs
from .evaluation import comparisons, expected_metrics
from .fitting import write_json
from .selection import fit_directory, read_selection, select


def collapse_replicates(rows, config):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["seed"], row["arm"], row["metric"])
        if row["replicate"] in grouped[key] or not np.isfinite(row["value"]):
            raise ValueError("Duplicate/nonfinite replicate measurement")
        grouped[key][row["replicate"]] = row["value"]
    result = []
    for (seed, arm, metric), values in sorted(grouped.items()):
        expected = {None} if arm == "archived" else set(config["replicates"])
        if set(values) != expected:
            raise ValueError("Missing optimizer replicate")
        result.append({"seed": seed, "arm": arm, "metric": metric,
                       "n_fitting_replicates": len(values), "value": float(np.mean(list(values.values())))})
    return result


def paired_contrasts(values, config):
    result = []
    low, high = config["control_updates"]
    for metric in sorted({metric for metric, _ in values}):
        for candidate, control in comparisons(config):
            for budget in (low, high, "tuned"):
                a, b = values.get((metric, f"{candidate}_{budget}")), values.get((metric, f"{control}_{budget}"))
                if a is None or b is None:
                    continue
                diffs = np.asarray([a[s] - b[s] for s in config["seeds"]])
                result.append({"metric": metric, "contrast": f"{candidate}_minus_{control}_{budget}",
                               **difference_summary(diffs)})
            arms = [f"{control}_{low}", f"{control}_{high}", f"{candidate}_{low}", f"{candidate}_{high}"]
            if all((metric, arm) in values for arm in arms):
                diffs = np.asarray([[values[(metric, arm)][s] for arm in arms] for s in config["seeds"]]) @ np.array([1, -1, -1, 1])
                name = ("capacity_by_budget_interaction" if config["experiment_id"] == 13 else
                        f"{candidate}_minus_{control}_by_budget_interaction")
                result.append({"metric": metric, "contrast": name, **difference_summary(diffs)})
    return result


def difference_summary(diffs):
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(diffs))))
    p = float(np.mean(np.abs((signs * diffs).mean(1)) >= abs(diffs.mean()) - 1e-12))
    return {**summary(diffs), "negative_source_differences": int((diffs < 0).sum()),
            "exact_two_sided_sign_flip_p": p}


def aggregate(root, smoke=False, *, config=None, plot_fn=None):
    root, config = Path(root), contract(smoke) if config is None else config
    # Recompute from validation-only screen metadata to verify the locked decision.
    selected = select(root, config)
    read_selection(root / "selection.json", config)
    specs = arm_specs(config, selected)
    rows, sources, details, curve_rows = [], [], {}, []

    def add(seed, arm, metric, value):
        base, replicate = (arm.rsplit("_rep", 1) if arm != "archived" else (arm, None))
        rows.append({"seed": seed, "arm": base, "replicate": None if replicate is None else int(replicate),
                     "metric": metric, "value": value})

    for seed in config["seeds"]:
        worker = root / "workers" / f"seed_{seed}"
        read = lambda name: json.loads((worker / name).read_text())
        manifest, marker, lock = read("manifest.json"), read("SUCCESS.json"), read("locked_selection.json")
        read_selection(worker / "locked_selection.json", config, manifest, seed)
        if (marker != {"seed": seed, "smoke": smoke, "status": "complete", "selection_sha256": selected["selection_sha256"]}
                or lock != selected):
            raise ValueError("Incomplete worker or different selected recipe")
        sources.append(manifest)
        replay = read("replay_diagnostics.json")
        fits, tests, evaluations = read("deployment_metrics.json"), read("diagnostic_test.json"), read("evaluation_summary.json")
        if len(fits) != len(specs) or {r["arm"] for r in fits} != set(specs):
            raise ValueError("Incomplete deployment fits")
        sample_hashes = {}
        for fit in fits:
            if (fit["seed"] != seed or any(fit[k] != v for k, v in specs[fit["arm"]].items())
                    or fit["phase"] != "deployment" or fit["data_sha256"] != replay["data_sha256"]
                    or fit["validation_data_sha256"] is not None):
                raise ValueError("Wrong deployment fit identity")
            key = (fit["replicate"], fit["updates"])
            if key in sample_hashes and sample_hashes[key] != fit["sample_sequence_sha256"]:
                raise ValueError("Deployment capacity arms used different minibatch sequences")
            sample_hashes[key] = fit["sample_sequence_sha256"]
            for metric in ("fitting_seconds", "processed_examples"):
                add(seed, fit["arm"], metric, fit[metric])
            for metric in ("weighted_ce", "weighted_kl", "weighted_l1"):
                add(seed, fit["arm"], f"deployment_training_{metric}", fit["training"]["all"][metric])
            for metric, value in fit["inference"].items():
                add(seed, fit["arm"], f"inference_{metric}", value)
        if (tests["selection_sha256"] != selected["selection_sha256"] or tests["seed"] != seed
                or tests["test_data_sha256"] != replay["splits"]["test"]["data_sha256"]
                or len(tests["rows"]) != len(specs) or {r["arm"] for r in tests["rows"]} != set(specs)):
            raise ValueError("Incomplete or unlocked test results")
        for row in tests["rows"]:
            if any(row[k] != v for k, v in specs[row["arm"]].items()):
                raise ValueError("Wrong diagnostic test identity")
            screen_path = fit_directory(worker, "screen", row["architecture"], row["recipe"], row["replicate"])
            screen = json.loads((screen_path / "metrics.json").read_text())
            expected_hash = next(r["policy_sha256"] for r in screen if r["updates"] == row["updates"])
            if row["policy_sha256"] != expected_hash:
                raise ValueError("Test scores refer to a different diagnostic policy")
            for metric in ("weighted_ce", "weighted_kl", "weighted_l1"):
                add(seed, row["arm"], f"diagnostic_test_{metric}", row["test"]["all"][metric])
        expected = expected_metrics(config)
        if len(evaluations) != len(expected) or {(r["arm"], r["metric"]) for r in evaluations} != expected:
            raise ValueError("Incomplete gameplay evaluation")
        for row in evaluations:
            add(seed, row["arm"], row["metric"], row["mean_mbb_per_hand"])
        screens = []
        for architecture in config["architectures"]:
            for recipe in config["recipes"]:
                for replicate in config["replicates"]:
                    fits_path = fit_directory(worker, "screen", architecture, recipe, replicate) / "metrics.json"
                    for fit in json.loads(fits_path.read_text()):
                        screens.append(fit)
                        curve_rows.append({k: fit[k] for k in ("seed", "architecture", "recipe", "replicate", "updates", "fitting_seconds")}
                                          | {"training_kl": fit["training"]["all"]["weighted_kl"],
                                             "validation_kl": fit["validation"]["all"]["weighted_kl"]})
        details[str(seed)] = {"replay": replay, "screen": screens,
                              "deployment": fits, "test": tests, "evaluation": evaluations,
                              "resources": {phase: read(f"{phase}_resources.json") for phase in ("screen", "deployment")}}
    per_seed = collapse_replicates(rows, config)
    values = defaultdict(dict)
    for row in per_seed:
        values[(row["metric"], row["arm"])][row["seed"]] = row["value"]
    if any(set(v) != set(config["seeds"]) for v in values.values()):
        raise ValueError("Missing source-seed measurement")
    summaries = [{"metric": metric, "arm": arm, **summary(list(data.values()))}
                 for (metric, arm), data in sorted(values.items())]
    contrasts = paired_contrasts(values, config)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_json(analysis / "summary.json", {"config": config, "summaries": summaries,
        "paired_contrasts": contrasts, "selection": selected, "status": "complete",
        "interpretation": config.get("interpretation", "Three reused source trajectories, not six seeds. Fixed 20k capacity contrast is primary; tuned contrasts and all intervals are exploratory. LBR is not exact exploitability.")})
    write_json(analysis / "source_manifests.json", sources)
    write_json(analysis / "detailed_diagnostics.json", details)
    for filename, data in (("per_replicate_metrics", rows), ("per_seed_metrics", per_seed),
                           ("aggregate_metrics", summaries), ("paired_contrasts", contrasts), ("learning_curves", curve_rows)):
        csv_file(analysis / f"{filename}.csv", data)
    (plot if plot_fn is None else plot_fn)(analysis, config, curve_rows, summaries, values)
    (analysis / "README.md").write_text(
        "# " + config.get("analysis_title", "Frozen FHP policy-capacity audit") + "\n\n"
        + config.get("analysis_intro", "Only output-policy fitting changes. Standard/wide arms reuse identical Experiment 2 replay. ") +
        "Two fitting replicas are averaged within each of three source seeds. "
        "Diagnostic fits use group-disjoint 80/10/10 partitions; validation selects one recipe/budget "
        "per architecture globally before held-out test access. Deployment fits use all replay. "
        "Fixed-budget comparisons isolate the network change; tuned comparisons also change optimisation. "
        "CE/KL fidelity is not exploitability. Direct-play wins and restricted LBR are complementary diagnostics. "
        "No exact exploitability or Nash guarantee is claimed. A selected maximum-budget endpoint "
        "or a recipe whose validation CE is still falling at the boundary flags potentially unresolved "
        "optimisation. An uncertain null result is not evidence of equivalence. "
        "Intervals and tests are exploratory; the minimum two-sided source sign-flip p with three seeds is 0.25. "
        "Learning curves and per-replica raw results are retained to diagnose overfitting. "
        "Playable policies are under workers; replay/optimizer states are not duplicated.\n"
        + ("\nSMOKE ONLY: synthetic/tiny results are not scientific evidence.\n" if smoke else ""))
    return analysis


def plot(analysis, config, curves, summaries, values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title = "SMOKE ONLY" if config["smoke"] else "Frozen FHP policy-capacity audit (3 source seeds)"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, axis_key in enumerate(("updates", "fitting_seconds")):
        for row, split in enumerate(("training_kl", "validation_kl")):
            ax = axes[row, column]
            for architecture in config["architectures"]:
                selected = [r for r in curves if r["architecture"] == architecture and r["recipe"] == config["control_recipe"]]
                x, y, error = [], [], []
                for update in config["screen_updates"]:
                    block = [r for r in selected if r["updates"] == update]
                    seed_values = [np.mean([r[split] for r in block if r["seed"] == s]) for s in config["seeds"]]
                    x.append(np.mean([r[axis_key] for r in block]))
                    y.append(np.mean(seed_values))
                    error.append(summary(seed_values)["se"] or 0)
                ax.errorbar(x, y, yerr=error, label=architecture, marker="o", capsize=3)
            ax.set(xlabel=axis_key, ylabel=split, title="Matched Adam 0.003; bars: source-seed SE")
            ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.savefig(analysis / "capacity_learning_curves.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, architecture in zip(axes, config["architectures"]):
        for recipe in config["recipes"]:
            y = [np.mean([r["validation_kl"] for r in curves if r["architecture"] == architecture
                          and r["recipe"] == recipe and r["updates"] == u]) for u in config["screen_updates"]]
            ax.plot(config["screen_updates"], y, label=recipe, marker=".")
        ax.set(title=architecture, xlabel="Updates", ylabel="Validation weighted KL")
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(title + ": optimisation screen")
    fig.savefig(analysis / "optimisation_screen.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("diagnostic_test_weighted_kl", "lbr_mbb_per_hand",
                                     "direct_crossplay_archived", "direct_crossplay_matched_standard")):
        selected = [r for r in summaries if r["metric"] == metric]
        for i, row in enumerate(selected):
            data = list(values[(metric, row["arm"])].values())
            ax.scatter([i]*len(data), data, alpha=.5)
            ax.errorbar(i, row["mean"], yerr=row["se"] or 0, fmt="ko", capsize=4)
        ax.set_xticks(range(len(selected)), [r["arm"] for r in selected], rotation=30, ha="right")
        ax.set_title(metric); ax.grid(axis="y", alpha=.2)
    fig.suptitle(title + ": one source-seed SE; direct-play higher is better, KL/LBR lower is better")
    fig.savefig(analysis / "capacity_policy_quality.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric in zip(axes, ("fitting_seconds", "inference_batch_1_seconds_per_state")):
        for row in summaries:
            if row["metric"] != "lbr_mbb_per_hand" or row["arm"] == "archived":
                continue
            x = next(r["mean"] for r in summaries if r["metric"] == metric and r["arm"] == row["arm"])
            ax.errorbar(x, row["mean"], yerr=row["se"] or 0, fmt="o")
            ax.annotate(row["arm"], (x, row["mean"]), fontsize=8)
        ax.set(xlabel=metric, ylabel="Restricted LBR gain (mbb/hand)"); ax.grid(alpha=.2)
    fig.suptitle(title + ": fitting and inference cost")
    fig.savefig(analysis / "capacity_cost_quality.png", dpi=180); plt.close(fig)
