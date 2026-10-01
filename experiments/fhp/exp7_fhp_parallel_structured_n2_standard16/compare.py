"""Compare completed Experiments 6/7 using small analysis folders only."""
import csv
import json
from pathlib import Path
import numpy as np


def compare_runs(baseline_root: Path, parallel_root: Path, output: Path):
    cohorts = []
    for root, expected_id in ((baseline_root, 6), (parallel_root, 7)):
        directory = Path(root) / "analysis"
        if not (directory / "SUCCESS.json").is_file():
            raise ValueError(f"Analysis is incomplete: {directory}")
        manifest = json.loads((directory / "experiment_manifest.json").read_text())
        if manifest["experiment_id"] != expected_id:
            raise ValueError("Expected Experiment 6 baseline and Experiment 7 parallel run")
        with (directory / "seed_summaries.csv").open() as stream:
            summaries = list(csv.DictReader(stream))
        with (directory / "checkpoint_index.csv").open() as stream:
            checkpoints = list(csv.DictReader(stream))
        if any(row["smoke"].lower() != "false" for row in summaries):
            raise ValueError("Cannot compare smoke runs as production evidence")
        cohorts.append((manifest, summaries, checkpoints))
    for key in ("training_config", "reference_vm", "learner_intraop_threads"):
        if cohorts[0][0]["contract"][key] != cohorts[1][0]["contract"][key]:
            raise ValueError(f"Comparison differs in {key}")
    seeds = [{int(row["seed"]) for row in cohort[1]} for cohort in cohorts]
    if seeds[0] != seeds[1] or seeds[0] != {0, 1, 2}:
        raise ValueError("Both runs must contain all three benchmark seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in sorted(seeds[0]):
        values = [next(row for row in c[1] if int(row["seed"]) == seed) for c in cohorts]
        node_rates = [float(v["final_nodes_touched"]) / float(v["final_training_elapsed_seconds"])
                      for v in values]
        iteration_rates = [float(v["final_iteration"]) / float(v["final_training_elapsed_seconds"])
                           for v in values]
        per_seed.append({"seed": seed, "serial_nodes_per_second": node_rates[0],
                         "parallel_nodes_per_second": node_rates[1],
                         "node_throughput_ratio": node_rates[1] / node_rates[0],
                         "iteration_throughput_ratio": iteration_rates[1] / iteration_rates[0]})
    result = {"seeds": per_seed,
              "mean_node_throughput_ratio": float(np.mean([r["node_throughput_ratio"] for r in per_seed])),
              "mean_iteration_throughput_ratio": float(np.mean([r["iteration_throughput_ratio"] for r in per_seed])),
              "caveat": "Descriptive throughput, not policy-quality evidence. Check source/runtime manifests for same-commit comparison; actor streams differ. Active time excludes output-policy fitting and checkpoint overhead."}
    (output / "sequential_vs_parallel.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for cohort, colour, label in zip(cohorts, ("tab:blue", "tab:orange"),
                                      ("Sequential (Exp. 6)", "8 workers (Exp. 7)")):
        curves = [sorted([r for r in cohort[2] if int(r["seed"]) == seed],
                         key=lambda r: float(r["checkpoint_target_hours"])) for seed in sorted(seeds[0])]
        if any([float(row["checkpoint_target_hours"]) for row in curve] != [6, 12, 18, 24]
               for curve in curves):
            raise ValueError("Unexpected checkpoint schedule")
        x = np.array([[float(r["actual_training_elapsed_seconds"]) / 3600 for r in c] for c in curves])
        y = np.array([[float(r["nodes_touched"]) / 1e6 for r in c] for c in curves])
        for xs, ys in zip(x, y):
            ax.plot(xs, ys, color=colour, alpha=.25)
        ax.errorbar(x.mean(0), y.mean(0), yerr=y.std(0, ddof=1) / np.sqrt(3),
                    color=colour, marker="o", capsize=4, label=label)
    ax.set(xlabel="Active training time (hours)", ylabel="Training nodes (millions)",
           title="FHP: sequential versus parallel collection on n2-standard-16")
    ax.legend()
    fig.savefig(output / "sequential_vs_parallel_nodes.png", dpi=220)
    plt.close(fig)
    return result
