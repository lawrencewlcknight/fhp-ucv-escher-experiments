"""Direct-play reporting with separate training-seed and deal uncertainty."""
from collections import defaultdict
import itertools
import math
from pathlib import Path

import numpy as np

from . import engine as native


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
            algorithm=task["a"]["algorithm"], opponent=task["b"]["algorithm"],
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
        summaries.append(dict(comparison_id=name, left_algorithm=left, right_algorithm=right,
            positive_favours=left, mean_mbb_per_hand=mean, cluster_ci95_low=lo, cluster_ci95_high=hi,
            family_ci98_33_low=flo, family_ci98_33_high=fhi, conditional_mc_se_mbb=se,
            conditional_mc_ci95_low=mean-1.96*se, conditional_mc_ci95_high=mean+1.96*se,
            positive_cells=int((matrix > 0).sum()), cross_seed_cells=9,
            duplicate_pairs=sum(c["duplicate_pairs"] for c in selected),
            left_nodes_mean=float(np.mean([c["nodes_touched"] for c in selected])),
            right_nodes_mean=float(np.mean([c["opponent_nodes"] for c in selected]))))
    caveats = [
        "Head-to-head value is not exploitability or proof of convergence.",
        "Shared rich AE inputs remove exclusive access to those features as an explanation; architectures, optimisation and tuning histories remain different.",
        "UCV is a selected historical reference; fresh SD/VR runs and fresh deals do not independently replicate UCV training.",
        "Only three training seeds per algorithm: row/column bootstrap intervals are exploratory; nine cells are not nine independent training replications.",
        "Training-seed and conditional deal intervals are separate, not a joint confidence region.",
        "Nominal active training time is 48 hours; native clock definitions differ, so this is not equal-total-compute evidence.",
        "Node counts are descriptive and are not used to normalise match outcomes."]
    summary = dict(experiment_id=41, protocol=native.PROTOCOL, status="complete", smoke=False,
        training_checkpoint_hours=48, matched_representation=True, feature_encoder_id="fhp_hand_board_strategic_ae_v1",
        player_features=231, selected_source_runs=native.RUN_IDS, policy_labels=native.LABELS,
        exact_exploitability=False, lbr_evaluated=False, rule_agents_evaluated=False,
        direct_comparisons=summaries, units="mbb/hand", family_confidence=engine.FAMILY_CONFIDENCE,
        inference="10,000 independent row/column training-seed bootstrap draws; pointwise 95% and Bonferroni-compatible 98.33% intervals for three co-primary comparisons. Conditional Monte Carlo uncertainty reported separately.",
        policy_representations=dict(sd="full uniform historical trajectory mixture; one draw per hand; no thinning",
            ucv="native saved average-policy network", vr="native saved average-policy network"), caveats=caveats)
    engine.write_csv(output/"matchups.csv", cells)
    engine.write_csv(output/"direct_comparison_summary.csv", summaries)
    engine.write_json(output/"summary.json", summary)
    engine.write_json(output/"checkpoint_index.json", engine.portable(records))
    engine.write_csv(output/"checkpoint_index.csv", engine.portable(records))
    engine.write_json(output/"training_cost_context.json", dict(records=costs,
        interpretation="Native timing definitions retained; not equal-total-compute evidence."))
    engine.plots(matrices, summaries, output)
    lines = ["# Experiment 41: matched-representation 48-hour FHP comparison", "",
        "Positive values favour the left algorithm. Units: mbb/hand.", "",
        "| Comparison | Mean | 95% seed interval | 98.33% family interval | 95% conditional deal interval |",
        "|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['left_algorithm']} vs {r['right_algorithm']} | {r['mean_mbb_per_hand']:.3f} | "
            f"[{r['cluster_ci95_low']:.3f}, {r['cluster_ci95_high']:.3f}] | "
            f"[{r['family_ci98_33_low']:.3f}, {r['family_ci98_33_high']:.3f}] | "
            f"[{r['conditional_mc_ci95_low']:.3f}, {r['conditional_mc_ci95_high']:.3f}] |")
    lines += ["", "## Frozen sources", ""] + [f"- {k}: {v} ({native.RUN_IDS[k]})." for k,v in native.LABELS.items()]
    lines += ["", "## Interpretation", ""] + ["- "+v for v in caveats]
    (output/"analysis_summary.md").write_text("\n".join(lines)+"\n")
    return summary
