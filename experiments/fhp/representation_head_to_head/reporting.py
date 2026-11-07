"""Direct-play reporting with separate training-seed and deal uncertainty."""
from collections import defaultdict
import itertools
import math
from pathlib import Path

import numpy as np

from .sources import PINS


def report(results, output, records, costs, engine):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not results:
        raise ValueError("Incomplete evaluation")
    implementation = results[0]["task"]["implementation"]
    expected = {t["task_id"]: t for t in engine.build_tasks(records, "production", implementation)}
    seen, grouped = set(), defaultdict(list)
    for row in results:
        task_id = row["task"]["task_id"]
        if task_id not in expected or task_id in seen:
            raise ValueError("Unexpected or duplicate match shard")
        engine.validate_result(row, expected[task_id])
        seen.add(task_id)
        grouped[row["task"]["cell_id"]].append(row)
    if seen != set(expected):
        raise ValueError("Incomplete production shards; refusing aggregate")
    cells = []
    for cell, rows in sorted(grouped.items()):
        task = rows[0]["task"]
        cells.append(dict(cell_id=cell, comparison_id=task["comparison_id"],
            algorithm=task["a"]["algorithm"], cohort=task["a"]["cohort"], opponent_cohort=task["b"]["cohort"],
            seed=task["a"]["seed"], opponent_seed=task["b"]["seed"],
            nodes_touched=task["a"]["nodes_touched"], opponent_nodes=task["b"]["nodes_touched"],
            **engine.pool_shards(rows)))
    summaries, matrices = [], {}
    for index, (name, left, right) in enumerate(engine.COMPARISONS):
        selected = [c for c in cells if c["comparison_id"] == name]
        if (sorted((c["seed"], c["opponent_seed"]) for c in selected)
                != list(itertools.product(engine.SEEDS, repeat=2))
                or any(c["duplicate_pairs"] != 100_000 for c in selected)):
            raise ValueError("Incomplete cross-seed matrix or deal budget")
        matrix = np.empty((3, 3))
        for cell in selected:
            matrix[cell["seed"], cell["opponent_seed"]] = cell["mean_mbb_per_hand"]
        matrices[name] = matrix
        lo, hi = engine.cluster_interval(matrix, seed=891_223+index)
        flo, fhi = engine.cluster_interval(matrix, confidence=engine.FAMILY_CONFIDENCE, seed=901_223+index)
        mean = float(matrix.mean())
        se = math.sqrt(sum(c["mc_se_mbb_per_hand"]**2 for c in selected))/9
        summaries.append(dict(comparison_id=name, algorithm=engine.REPRESENTATION_ALGORITHM, left_cohort=left, right_cohort=right,
            positive_favours=left, mean_mbb_per_hand=mean, cluster_ci95_low=lo, cluster_ci95_high=hi,
            family_ci97_5_low=flo, family_ci97_5_high=fhi, conditional_mc_se_mbb=se,
            conditional_mc_ci95_low=mean-1.96*se, conditional_mc_ci95_high=mean+1.96*se,
            positive_cells=int((matrix > 0).sum()), cross_seed_cells=9,
            duplicate_pairs=sum(c["duplicate_pairs"] for c in selected),
            left_nodes_mean=float(np.mean([c["nodes_touched"] for c in selected])),
            right_nodes_mean=float(np.mean([c["opponent_nodes"] for c in selected]))))
    algorithm = engine.REPRESENTATION_ALGORITHM
    caveats = [
        "Head-to-head value against the original policy is not exploitability or universal playing strength.",
        "Training configs differ only by encoder selection; historical cohorts also differ in code revision and Python/platform patch versions. This is not a perfectly isolated causal rerun.",
        "Three training seeds per cohort: the independent row/column bootstrap is exploratory; nine cells are not nine independent training replications. Equal seed numbers do not imply paired training randomness.",
        "Training-seed and conditional deal intervals are separate, not a joint confidence region.",
        "The two representation experiments form one prespecified family; 97.5% marginal seed intervals use a Bonferroni adjustment for two comparisons, subject to bootstrap limitations.",
        "Within each algorithm the native active-training clock is matched at 48h. Feature computation changes throughput; total compute and node counts are not forced equal.",
        "All source checkpoints are fixed historical policies. No retraining, checkpoint selection, tuning, rule opponents or LBR is performed."]
    summary = dict(experiment_id=engine.EXPERIMENT_ID, protocol=engine.PROTOCOL, status="complete", smoke=False,
        algorithm=algorithm, training_checkpoint_hours=48, original_player_features=183, rich_player_features=231,
        selected_source_runs={c: p["run_id"] for c,p in PINS[algorithm].items()},
        exact_exploitability=False, lbr_evaluated=False, rule_agents_evaluated=False,
        direct_comparisons=summaries, units="mbb/hand", family_confidence=engine.FAMILY_CONFIDENCE,
        inference="10,000 independent row/column training-seed bootstrap draws; pointwise 95% and two-comparison 97.5% intervals. Conditional Monte Carlo uncertainty reported separately.",
        policy_representation=("full uniform historical trajectory mixture; one draw per player per hand; no thinning"
            if algorithm == "sd" else "native saved average-policy network"), caveats=caveats)
    engine.write_csv(output/"matchups.csv", cells)
    engine.write_csv(output/"direct_comparison_summary.csv", summaries)
    engine.write_json(output/"summary.json", summary)
    engine.write_json(output/"checkpoint_index.json", engine.portable(records))
    engine.write_csv(output/"checkpoint_index.csv", engine.portable(records))
    engine.write_json(output/"training_cost_context.json", dict(records=costs,
        interpretation="Native timing definitions retained; not equal-total-compute evidence."))
    plot_matrix(next(iter(matrices.values())), output, algorithm)
    lines = [f"# Experiment {engine.EXPERIMENT_ID}: {algorithm.upper()} rich vs original, fixed 48-hour policies", "",
        "Positive values favour the rich representation. Units: mbb/hand.", "",
        "| Comparison | Mean | 95% seed interval | 97.5% family interval | 95% conditional deal interval |",
        "|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['left_cohort']} vs {r['right_cohort']} | {r['mean_mbb_per_hand']:.3f} | "
            f"[{r['cluster_ci95_low']:.3f}, {r['cluster_ci95_high']:.3f}] | "
            f"[{r['family_ci97_5_low']:.3f}, {r['family_ci97_5_high']:.3f}] | "
            f"[{r['conditional_mc_ci95_low']:.3f}, {r['conditional_mc_ci95_high']:.3f}] |")
    lines += ["", "## Frozen sources", ""] + [f"- {c}: {p['run_id']}." for c,p in PINS[algorithm].items()]
    lines += ["", "## Interpretation", ""] + ["- "+v for v in caveats]
    (output/"analysis_summary.md").write_text("\n".join(lines)+"\n")
    return summary


def plot_matrix(matrix, output, algorithm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    bound = max(float(np.abs(matrix).max()), .001)
    heat = axis.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound)
    for (i,j), value in np.ndenumerate(matrix):
        axis.text(j,i,f"{value:.2f}",ha="center",va="center",bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
    axis.set(xticks=range(3),yticks=range(3),xlabel="Original policy seed",ylabel="Rich policy seed",
        title=f"{algorithm.upper()}: rich versus original (48h)")
    fig.colorbar(heat, ax=axis, label="mbb/hand; positive favours rich")
    fig.tight_layout()
    fig.savefig(output/"final_48h_head_to_head.png",dpi=180)
    plt.close(fig)
