"""Crossed training-seed uncertainty and exact pooling of deal-pair moments.

Deal streams are independent BETWEEN match shards; seats share randomness
WITHIN each duplicated pair. Crossed model resampling is exploratory with only
three trained models per configuration. It is not a nine-independent-seed CI.
"""
from collections import defaultdict

import numpy as np

COHORTS = ("exp10", "exp20", "exp36")
PAIRS = (("exp20", "exp10"), ("exp36", "exp10"), ("exp36", "exp20"))


def pool_shards(rows, scale):
    """Pool sample moments, including between-shard variation (not mean SEs)."""
    n = sum(r["num_deal_pairs"] for r in rows)
    mean = sum(r["num_deal_pairs"] * r["mean_chips_per_hand"] for r in rows) / n
    m2 = sum(((r["num_deal_pairs"] - 1) * r["std_chips_per_pair"] ** 2 if r["num_deal_pairs"] > 1 else 0.)
             + r["num_deal_pairs"] * (r["mean_chips_per_hand"] - mean) ** 2 for r in rows)
    if n < 2:
        raise ValueError("At least two duplicated pairs required for uncertainty")
    std = float(np.sqrt(m2 / (n - 1)))
    se = std / np.sqrt(n)
    return dict(num_deal_pairs=n, num_games=2*n, mean_chips_per_hand=mean,
        std_chips_per_pair=std, se_chips_per_hand=float(se),
        mean_mbb_per_hand=mean*scale, se_mbb_per_hand=float(se*scale),
        ci95_low_mbb_per_hand=float((mean-1.96*se)*scale),
        ci95_high_mbb_per_hand=float((mean+1.96*se)*scale),
        policy_a_player0_mean_chips=sum(r["num_deal_pairs"]*r["policy_a_player0_mean_chips"] for r in rows)/n,
        policy_a_player1_mean_chips=sum(r["num_deal_pairs"]*r["policy_a_player1_mean_chips"] for r in rows)/n)


def collapse_direct(results, scale):
    groups = defaultdict(list)
    for row in results:
        if row["kind"] == "direct_crossplay":
            groups[(row["cohort_a"], row["cohort_b"], row["seed_a"], row["seed_b"])].append(row)
    return [dict(cohort_a=a, cohort_b=b, seed_a=i, seed_b=j,
                 comparison=f"{a}_vs_{b}", **pool_shards(rows, scale))
            for (a, b, i, j), rows in sorted(groups.items())]


def crossed_summary(cells, *, smoke=False, bootstrap_seed=2026102037, replicates=30000):
    """Resample each cohort's models; reuse its weights in all comparisons.

    An independent Gaussian perturbation with each cell's deal-level SE also
    propagates finite-match uncertainty. This is a conservative sensitivity
    addition: empirical model effects themselves already contain deal noise.
    Independent resampling across configurations does NOT assume equal seed
    labels produced paired training trajectories.
    """
    seeds = (0,) if smoke else (0, 1, 2)
    index = {(r["cohort_a"], r["cohort_b"], r["seed_a"], r["seed_b"]): r for r in cells}
    expected = {(a, b, i, j) for a, b in PAIRS for i in seeds for j in seeds}
    if len(index) != len(cells) or set(index) != expected:
        raise ValueError("Incomplete or duplicate crossed seed-pair matrix")
    rng = np.random.default_rng(bootstrap_seed)
    weights = {c: rng.multinomial(len(seeds), [1/len(seeds)]*len(seeds), size=replicates)/len(seeds)
               for c in COHORTS}
    summaries = []
    for a, b in PAIRS:
        means = np.array([[index[(a, b, i, j)]["mean_mbb_per_hand"] for j in seeds] for i in seeds])
        ses = np.array([[index[(a, b, i, j)]["se_mbb_per_hand"] for j in seeds] for i in seeds])
        point = float(means.mean())
        mc_se = float(np.sqrt(np.sum(ses**2))/means.size)
        row = dict(comparison=f"{a}_vs_{b}", cohort_a=a, cohort_b=b,
            models_per_configuration=len(seeds), model_pairings=means.size,
            num_deal_pairs=sum(index[(a, b, i, j)]["num_deal_pairs"] for i in seeds for j in seeds),
            mean_mbb_per_hand=point, conditional_deal_se_mbb_per_hand=mc_se,
            conditional_deal_ci95_low=point-1.96*mc_se, conditional_deal_ci95_high=point+1.96*mc_se,
            bootstrap_ci95_low=None, bootstrap_ci95_high=None,
            bootstrap_family_ci_low=None, bootstrap_family_ci_high=None)
        if not smoke:
            perturbed = means + rng.normal(size=(replicates, *means.shape))*ses
            samples = np.einsum("bi,bij,bj->b", weights[a], perturbed, weights[b])
            lo, hi = np.quantile(samples, [.025, .975])
            # Bonferroni across the THREE primary comparisons, not nine cells.
            flo, fhi = np.quantile(samples, [.05/(2*len(PAIRS)), 1-.05/(2*len(PAIRS))])
            row.update(bootstrap_ci95_low=float(lo), bootstrap_ci95_high=float(hi),
                       bootstrap_family_ci_low=float(flo), bootstrap_family_ci_high=float(fhi))
        summaries.append(row)
    return summaries


def leader_screen(rows):
    """Only an exploratory H2H screen, never an automatic algorithm promotion."""
    wins = defaultdict(set)
    for r in rows:
        if r["bootstrap_family_ci_low"] is None:
            continue
        if r["bootstrap_family_ci_low"] > 0:
            wins[r["cohort_a"]].add(r["cohort_b"])
        if r["bootstrap_family_ci_high"] < 0:
            wins[r["cohort_b"]].add(r["cohort_a"])
    leaders = [c for c in COHORTS if wins[c] == set(COHORTS)-{c}]
    return dict(provisional_head_to_head_leader=leaders[0] if leaders else None,
        status="exploratory_head_to_head_signal" if leaders else "no_clear_head_to_head_leader",
        automatic_promotion=False,
        qualification="Only three training seeds per configuration. Check common opponents, LBR, rare-state failures and cost before choosing; no Nash or exploitability guarantee.")
