"""Crossed-model inference with paired continuation-trajectory resampling.

Only three independent trajectories per cohort: intervals are exploratory.
The 36/38 model indices share weights; 16 is independently resampled. The
all-nine-pairing endpoint and same-trajectory endpoint are different estimands.
"""
import numpy as np
from scipy.stats import t
from experiments.fhp.exp37_fhp_critic_budget_evaluation.statistics import collapse_direct, pool_shards

COHORTS = ("exp16", "exp36", "exp38")
PAIRS = (("exp38", "exp16"), ("exp38", "exp36"))


def _intervals(samples, smoke):
    keys = ("bootstrap_ci95_low", "bootstrap_ci95_high", "bootstrap_family_ci_low", "bootstrap_family_ci_high")
    values = (None,)*4 if smoke else map(float, np.quantile(samples, [.025, .975, .0125, .9875]))
    return dict(zip(keys, values))


def comparisons(cells, *, smoke=False, bootstrap_seed=2026102139, replicates=30000):
    seeds = (0,) if smoke else (0, 1, 2)
    index = {(r["cohort_a"], r["cohort_b"], r["seed_a"], r["seed_b"]): r for r in cells}
    expected = {(a, b, i, j) for a, b in PAIRS for i in seeds for j in seeds}
    if len(index) != len(cells) or set(index) != expected:
        raise ValueError("Incomplete or duplicate crossed seed-pair matrix")
    rng = np.random.default_rng(bootstrap_seed)
    n = len(seeds)
    paired = rng.multinomial(n, [1/n]*n, size=replicates)/n
    independent = rng.multinomial(n, [1/n]*n, size=replicates)/n
    weights = {"exp38": paired, "exp36": paired, "exp16": independent}
    primary, matched = [], None
    for a, b in PAIRS:
        means = np.array([[index[(a, b, i, j)]["mean_mbb_per_hand"] for j in seeds] for i in seeds])
        ses = np.array([[index[(a, b, i, j)]["se_mbb_per_hand"] for j in seeds] for i in seeds])
        point, mc_se = float(means.mean()), float(np.sqrt(np.sum(ses**2))/means.size)
        perturbed = means + rng.normal(size=(replicates, n, n))*ses
        samples = np.einsum("bi,bij,bj->b", weights[a], perturbed, weights[b])
        row = dict(comparison=f"{a}_vs_{b}", cohort_a=a, cohort_b=b,
            models_per_configuration=n, model_pairings=means.size,
            paired_training_trajectories=b == "exp36",
            num_deal_pairs=sum(index[(a, b, i, j)]["num_deal_pairs"] for i in seeds for j in seeds),
            mean_mbb_per_hand=point, conditional_deal_se_mbb_per_hand=mc_se,
            conditional_deal_ci95_low=point-1.96*mc_se, conditional_deal_ci95_high=point+1.96*mc_se,
            **_intervals(samples, smoke))
        primary.append(row)
        if b == "exp36":
            # Report the diagonal independently, not as nine independent runs.
            diagonal = np.diagonal(perturbed, axis1=1, axis2=2)
            matched_se = float(np.sqrt(np.sum(np.diag(ses)**2))/n)
            matched_point = float(np.diag(means).mean())
            matched = dict(comparison="exp38_vs_own_exp36_predecessor", trajectories=n,
                num_deal_pairs=sum(index[(a, b, i, i)]["num_deal_pairs"] for i in seeds),
                mean_mbb_per_hand=matched_point, conditional_deal_se_mbb_per_hand=matched_se,
                conditional_deal_ci95_low=matched_point-1.96*matched_se,
                conditional_deal_ci95_high=matched_point+1.96*matched_se,
                **_intervals(np.sum(paired*diagonal, axis=1), smoke))
            # Not a third primary endpoint; retain only exploratory 95% interval.
            matched.pop("bootstrap_family_ci_low")
            matched.pop("bootstrap_family_ci_high")
    return primary, matched


def decision_screen(rows):
    results = {}
    for row in rows:
        low, high = row["bootstrap_family_ci_low"], row["bootstrap_family_ci_high"]
        results[row["comparison"]] = ("smoke_only" if low is None else
            "favours_exp38" if low > 0 else "favours_comparator" if high < 0 else "inconclusive")
    return dict(primary_signals=results, automatic_promotion=False,
        qualification="Exploratory: three trajectories; no full league ranking, convergence certificate or Nash guarantee. Check common opponents, LBR, rare states and total cost.")


def paired_changes(rule_means, lbr):
    """Within-trajectory common-panel changes; seed-level t intervals only."""
    rows, summaries = [], []
    for metric, data, direction in (("rule_panel", rule_means, "higher"), ("lbr_payoff", lbr, "lower")):
        index = {(r["experiment"], r["training_seed"]): r for r in data}
        seeds = sorted(s for c, s in index if c == "exp38")
        if not seeds:
            continue
        differences = [index[("exp38", s)]["mean_mbb_per_hand"] - index[("exp36", s)]["mean_mbb_per_hand"] for s in seeds]
        rows.extend(dict(metric=metric, training_seed=s, change_48h_minus_24h=v) for s, v in zip(seeds, differences))
        mean = float(np.mean(differences))
        half = float(t.ppf(.975, len(seeds)-1)*np.std(differences, ddof=1)/np.sqrt(len(seeds))) if len(seeds)>1 else None
        summaries.append(dict(metric=metric, trajectories=len(seeds), mean_change_mbb_per_hand=mean,
            ci95_low=None if half is None else mean-half, ci95_high=None if half is None else mean+half,
            better_direction=direction, uncertainty="Exploratory paired training-seed t interval; no separate deal-level interval"))
    return rows, summaries
