"""Validate and aggregate the three Experiment 7 seed workers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import numpy as np

from experiments.fhp.exp2_fhp_lossless_structured_ucv.aggregate import (
    aggregate_workers as _aggregate_structured_workers,
    task_name as _structured_task_name,
)

from .config import (
    ALGORITHM_ID,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PRODUCTION_SEEDS,
    REFERENCE_VM,
    contract_manifest,
)


def task_name(index: int, seed: int) -> str:
    return _structured_task_name(index, seed, algorithm_id=ALGORITHM_ID)


def aggregate_workers(
    output_root: Path,
    *,
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> Path:
    output_root = Path(output_root)
    summaries, curves, commits = [], [], []
    for index, seed in enumerate(seeds):
        folder = output_root / "workers" / task_name(index, seed)
        manifest = json.loads((folder / "run_manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        if (manifest["experiment_id"] != EXPERIMENT_ID
                or manifest["experiment_name"] != EXPERIMENT_NAME
                or manifest["algorithm_id"] != ALGORITHM_ID
                or manifest["reference_vm"] != REFERENCE_VM
                or manifest["execution_backend"] != "ray_parallel"
                or summary["experiment_id"] != EXPERIMENT_ID):
            raise ValueError(f"Not an Experiment 7 worker: {folder}")
        if summary["execution_diagnostics"]["parallel_num_workers"] != 8:
            raise ValueError("Experiment 7 requires eight traversal actors")
        summaries.append(summary)
        commits.append(manifest["repository_commit"])
        curves.append(json.loads((folder / "checkpoint_manifest.json").read_text()))
    if len(set(commits)) != 1:
        raise ValueError("Experiment 7 workers used different source commits")
    _performance_outputs(output_root / "analysis", summaries, curves, commits[0])
    return _aggregate_structured_workers(
        output_root,
        seeds=seeds,
        experiment_id=EXPERIMENT_ID,
        experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID,
        contract_manifest_fn=contract_manifest,
        experiment_label="Experiment 7",
    )


def _performance_outputs(directory, summaries, curves, commit):
    directory.mkdir(parents=True, exist_ok=True)
    count = len(summaries)
    nodes = np.asarray([[row["nodes_touched"] for row in curve] for curve in curves], dtype=float)
    times = np.asarray([[row["actual_training_elapsed_seconds"] / 3600 for row in curve]
                        for curve in curves], dtype=float)
    rate = np.asarray([s["final_nodes_touched"] / s["final_training_elapsed_seconds"]
                       for s in summaries])
    node_se = nodes.std(axis=0, ddof=1) / np.sqrt(count) if count > 1 else np.zeros(4)
    smoke = all(s["smoke"] for s in summaries)
    result = {
        "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
        "smoke": smoke, "num_seeds": count, "seeds": [s["seed"] for s in summaries],
        "reference_vm": REFERENCE_VM,
        "repository_commit": commit,
        "mean_final_nodes": float(nodes[:, -1].mean()),
        "se_final_nodes": float(node_se[-1]),
        "mean_nodes_per_active_second": float(rate.mean()),
        "se_nodes_per_active_second": float(rate.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.,
        "mean_final_iterations": float(np.mean([s["final_iteration"] for s in summaries])),
        "max_peak_rss_mib": max(s["peak_rss_mib"] for s in summaries),
        "seed_metrics": [
            {"seed": s["seed"], "nodes": s["final_nodes_touched"],
             "iterations": s["final_iteration"],
             "active_seconds": s["final_training_elapsed_seconds"],
             "diagnostics": s["execution_diagnostics"]} for s in summaries],
        "checkpoint_means": [
            {"target_hours": curves[0][j]["checkpoint_target_hours"],
             "mean_actual_training_hours": float(times[:, j].mean()),
             "mean_nodes": float(nodes[:, j].mean()), "se_nodes": float(node_se[j])}
            for j in range(4)],
        "interpretation": "Throughput only; active clock excludes policy-checkpoint fitting, saving and uploads.",
    }
    (directory / "throughput_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for x, y in zip(times, nodes):
        ax.plot(x, y / 1e6, color="tab:blue", alpha=.3, linewidth=1)
    ax.errorbar(times.mean(axis=0), nodes.mean(axis=0) / 1e6, yerr=node_se / 1e6,
                color="tab:blue", marker="o", capsize=4, label="Mean ± one SE")
    ax.set(xlabel="Active training time (hours)", ylabel="Training nodes touched (millions)",
           title="SMOKE ONLY" if smoke else "FHP Experiment 7: eight-worker UCV-ESCHER on n2-standard-16")
    ax.legend()
    fig.savefig(directory / "nodes_by_training_time.png", dpi=220)
    plt.close(fig)
    # Non-overlapping DRIVER intervals. Merge is already in collection; actor
    # times are summed CPU-process wall times, not an additional driver phase.
    names = ("collection", "sync", "regret", "holdout", "learner")
    timings = [[s["execution_diagnostics"][f"cumulative_parallel_{name}_seconds"] / 3600
                for name in names] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bottom = np.zeros(count)
    for i, label in enumerate(("Collection incl. wait/merge", "Snapshot construction",
                               "Regret fitting", "Predictor diagnostic", "Critic + calibration fitting")):
        values = np.asarray(timings)[:, i]
        ax.bar([str(s["seed"]) for s in summaries], values, bottom=bottom, label=label)
        bottom += values
    ax.set(xlabel="Training seed", ylabel="Instrumented driver time (hours)",
           title="SMOKE ONLY" if smoke else "Parallel training cost breakdown")
    ax.legend(fontsize="small")
    fig.savefig(directory / "training_time_breakdown.png", dpi=220)
    plt.close(fig)


__all__ = ["aggregate_workers", "task_name"]
