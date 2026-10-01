"""Source-seed summaries and paired contrasts, not pseudoreplicated hand tests."""

from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from .config import SEEDS, arm_names, contract
from .fitting import write_json


def summary(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("Expected finite source-seed measurements")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    margin = float(t.ppf(.975, len(values)-1) * se) if se is not None else None
    return {"n_source_seeds": len(values), "mean": mean, "se": se,
            "ci95_low": None if margin is None else mean - margin,
            "ci95_high": None if margin is None else mean + margin}


def csv_file(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root, smoke=False):
    root = Path(root)
    config = contract(smoke)
    seeds = (0,) if smoke else SEEDS
    rows, sources = [], []
    for seed in seeds:
        worker = root / "workers" / f"seed_{seed}"
        success = json.loads((worker / "SUCCESS.json").read_text())
        manifest = json.loads((worker / "manifest.json").read_text())
        if success["seed"] != seed or manifest["config"] != config:
            raise ValueError("Worker does not satisfy the audit contract")
        sources.append(manifest)
        for phase in ("deployment", "diagnostic"):
            fits = []
            for sampler in config["samplers"]:
                fits.extend(json.loads((worker / phase / sampler / "metrics.json").read_text()))
            if sorted(r["arm"] for r in fits) != sorted(arm_names(config)):
                raise ValueError("Incomplete fitting endpoints")
            for fit in fits:
                if fit["seed"] != seed:
                    raise ValueError("Fit belongs to a different source seed")
                for metric in ("fitting_seconds", "diagnostic_seconds", "processed_examples"):
                    rows.append({"seed": seed, "arm": fit["arm"], "metric": f"{phase}_{metric}", "value": fit[metric]})
                for split in ("training", "validation"):
                    if fit[split] is not None:
                        for metric in ("weighted_ce", "weighted_kl", "weighted_l1"):
                            rows.append({"seed": seed, "arm": fit["arm"],
                                         "metric": f"{phase}_{split}_{metric}", "value": fit[split]["all"][metric]})
        evaluations = json.loads((worker / "evaluation_summary.json").read_text())
        if {r["arm"] for r in evaluations if r["metric"] == "lbr_mbb_per_hand"} != set(arm_names(config) + ["archived"]):
            raise ValueError("Incomplete LBR evaluation")
        for ev in evaluations:
            rows.append({"seed": seed, "arm": ev["arm"], "metric": ev["metric"], "value": ev["mean_mbb_per_hand"]})
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["metric"], row["arm"])
        if row["seed"] in grouped[key]:
            raise ValueError("Duplicate source-seed measurement")
        grouped[key][row["seed"]] = row["value"]
    summaries = [{"metric": metric, "arm": arm, **summary(list(values.values()))}
                 for (metric, arm), values in sorted(grouped.items())]
    low, high = config["updates"]
    contrasts = [(f"uniform_{high}", f"uniform_{low}"),
                 (f"mass_{low}", f"uniform_{low}"),
                 (f"mass_{high}", f"uniform_{high}"),
                 (f"mass_{high}", f"mass_{low}"),
                 (f"mass_{high}", f"uniform_{low}")]
    paired = []
    for metric in sorted({r["metric"] for r in rows}):
        for candidate, reference in contrasts:
            a, b = grouped.get((metric, candidate)), grouped.get((metric, reference))
            if a is None or b is None:
                continue
            if set(a) != set(seeds) or set(b) != set(seeds):
                raise ValueError("Unpaired source-seed contrast")
            diffs = np.array([a[s] - b[s] for s in seeds])
            signs = np.asarray(list(itertools.product((-1, 1), repeat=len(seeds))))
            p = float(np.mean(np.abs((signs * diffs).mean(1)) >= abs(diffs.mean()) - 1e-12))
            paired.append({"metric": metric, "candidate": candidate, "reference": reference,
                           **summary(diffs), "negative_differences": int((diffs < 0).sum()),
                           "exact_two_sided_sign_flip_p": p})
        # Preserve the factorial question, including a possible interaction.
        arms = [f"uniform_{low}", f"uniform_{high}", f"mass_{low}", f"mass_{high}"]
        if all((metric, arm) in grouped for arm in arms):
            matrix = np.array([[grouped[(metric, arm)][seed] for arm in arms] for seed in seeds])
            for estimand, coefficients in {
                "sampler_main_effect_mass_minus_uniform": [-.5, -.5, .5, .5],
                "update_main_effect_high_minus_low": [-.5, .5, -.5, .5],
                "interaction_mass_increment_minus_uniform_increment": [1, -1, -1, 1],
            }.items():
                diffs = matrix @ np.asarray(coefficients)
                signs = np.asarray(list(itertools.product((-1, 1), repeat=len(seeds))))
                p = float(np.mean(np.abs((signs * diffs).mean(1)) >= abs(diffs.mean()) - 1e-12))
                paired.append({"metric": metric, "candidate": estimand, "reference": "zero",
                               **summary(diffs), "negative_differences": int((diffs < 0).sum()),
                               "exact_two_sided_sign_flip_p": p})
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_json(analysis / "source_manifests.json", sources)
    write_json(analysis / "summary.json", {"config": config, "summaries": summaries,
                                           "paired_contrasts": paired, "status": "complete"})
    csv_file(analysis / "per_seed_metrics.csv", rows)
    csv_file(analysis / "aggregate_metrics.csv", summaries)
    csv_file(analysis / "paired_contrasts.csv", paired)
    # Preserve all action/frequency/round diagnostics in the small analysis download.
    diagnostics = {}
    for seed in seeds:
        worker = root / "workers" / f"seed_{seed}"
        diagnostics[str(seed)] = {"replay": json.loads((worker / "replay_diagnostics.json").read_text()),
                                  "resources": json.loads((worker / "resources.json").read_text()),
                                  "fits": {phase: {sampler: json.loads((worker / phase / sampler / "metrics.json").read_text())
                                                    for sampler in config["samplers"]}
                                           for phase in ("deployment", "diagnostic")},
                                  "evaluation": json.loads((worker / "evaluation_summary.json").read_text())}
    write_json(analysis / "detailed_diagnostics.json", diagnostics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("deployment_training_weighted_kl", "Replay-weighted KL (lower is better)"),
               ("diagnostic_validation_weighted_ce", "Unseen-group weighted CE (lower is better)"),
               ("lbr_mbb_per_hand", "LBR gain, mbb/hand (lower is better)"),
               ("direct_crossplay_archived", "Value vs archived policy, mbb/hand (higher is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        selected = [r for r in summaries if r["metric"] == metric]
        for index, row in enumerate(selected):
            values = grouped[(metric, row["arm"])]
            ax.scatter([index] * len(values), list(values.values()), alpha=.45)
            ax.errorbar(index, row["mean"], yerr=row["se"], fmt="ko", capsize=4)
        ax.set_xticks(range(len(selected)), [r["arm"] for r in selected], rotation=25, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("SMOKE TEST ONLY — not scientific results" if smoke else
                 "Frozen FHP average-policy audit; error bars: one source-seed SE")
    fig.savefig(analysis / "policy_fitting_audit.png", dpi=180)
    plt.close(fig)
    (analysis / "README.md").write_text(
        "# FHP policy-fitting audit\n\n"
        "Four refits of identical Experiment 2 replay, not new regret-learning trajectories. "
        "Diagnostic fits withhold canonical information-set groups; deployment fits use all replay. "
        "Lower CE/KL means better replay fidelity, not necessarily stronger play. "
        "LBR is a restricted exploiter, not exact exploitability or a convergence certificate. "
        "Direct play is duplicate-deal, seat-swapped sampled evaluation. "
        "Intervals in aggregate tables use the three source training seeds, not hands or checkpoints. "
        "Within-policy Monte Carlo uncertainty is retained in detailed_diagnostics.json. "
        "All intervals/tests are exploratory and unadjusted for multiple comparisons; "
        "three seeds cannot support a decisive significance claim. No winner is automatically promoted.\n")
    return analysis
