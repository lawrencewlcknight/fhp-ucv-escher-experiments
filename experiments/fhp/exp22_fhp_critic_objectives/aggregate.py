"""Selection-locked contrasts and seed-level uncertainty; no significance gate."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from experiments.critic_target_cache_benchmark import write_json
from fhp_escher.checkpointing import sha256_file
from .config import contract
from .run import choose_candidates


def paired(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    se = float(values.std(ddof=1)/np.sqrt(n)) if n > 1 else None
    mean = float(values.mean())
    return dict(mean_difference=mean, source_seeds=n,
                paired_seed_t95=None if se is None else
                [mean-float(t.ppf(.975, n-1))*se, mean+float(t.ppf(.975, n-1))*se])


def aggregate(root, pilot=False, smoke=False):
    root = Path(root)
    seeds = (0,) if pilot or smoke else (0, 1, 2)
    per_seed, manifests = [], []
    for seed in seeds:
        directory = root / "workers" / f"seed_{seed}"
        success = json.loads((directory / "SUCCESS.json").read_text())
        if (success.get("status") != "complete" or success.get("seed") != seed
                or success.get("smoke") != smoke):
            raise ValueError("Incomplete or mixed audit worker")
        records = {}
        for name in ("manifest", "summary", "selection_lock", "fitting", "assessment", "panels", "source_identity", "unchanged_assessment"):
            path = directory / f"{name}.json"
            if sha256_file(path) != success["artifacts"].get(path.name):
                raise ValueError("Audit artifact checksum mismatch")
            records[name] = json.loads(path.read_text())
        manifest = records["manifest"]
        if manifest["config"] != contract(smoke) or manifest["seed"] != seed:
            raise ValueError("Mixed source seed or scientific contract")
        if not smoke and manifest["provenance"].get("synthetic"):
            raise ValueError("Synthetic sources cannot enter production analysis")
        manifests.append(manifest)
        selection = records["selection_lock"]
        if (selection["fitting_sha256"] != success["artifacts"]["fitting.json"] or
                selection["candidates"] != choose_candidates(records["fitting"])):
            raise ValueError("Selection lock differs from validation-only rule")
        expected = {(objective, step, fold) for objective in contract(smoke)["objectives"]
                    for step in contract(smoke)["updates"] for fold in (0, 1)}
        rows = records["assessment"]
        if len(rows) != len(expected) or {(r["objective"], r["updates"], r["member"]) for r in rows} != expected:
            raise ValueError("Incomplete or duplicated assessment candidates")
        selected = {}
        for name, decision in selection["candidates"].items():
            candidates = [r for r in rows if r["objective"] == decision["objective"] and r["updates"] == decision["updates"]]
            selected[name] = dict(**decision, assessment={rep: {
                metric: float(np.mean([r["assessment"][rep]["summary"][metric] for r in candidates]))
                for metric in candidates[0]["assessment"][rep]["summary"]}
                for rep in ("online", "deployed")})
        per_seed.append(dict(seed=seed, selected=selected, worker_summary=records["summary"],
                             all_candidate_assessment=rows, fitting=records["fitting"],
                             unchanged_assessment=records["unchanged_assessment"]))
    if len({(m["code_sha256"], m["commit"], json.dumps(m["runtime"], sort_keys=True)) for m in manifests}) != 1:
        raise ValueError("Mixed code/runtime: pilot reuse requires identical protocol and code")
    contrasts = {}
    for objective in ("weighted_td", "regret_variance"):
        for reference in ("matched_td", "best_td_control"):
            key = objective + "_minus_" + reference
            contrasts[key] = paired([s["selected"][objective]["assessment"]["deployed"]["mean_variance"]
                                    - s["selected"][reference]["assessment"]["deployed"]["mean_variance"]
                                    for s in per_seed])
    summary = dict(experiment_id=22, smoke=smoke, pilot=pilot, seeds=list(seeds), comparisons=contrasts,
                   interpretation="Negative differences favour candidate. Late-stage mechanism screen only.",
                   limitations=["No exploitability, policy quality, early-stage or long-run conclusion.",
                                "One fit path per source/objective; finite probe panels; no equivalence claim.",
                                "Three source seeds maximum; paired t intervals are exploratory and pointwise.",
                                "Pilot seed 0 is reused once, not an additional independent replicate.",
                                "Mean of fold-specific tail quantiles is not a pooled quantile."])
    analysis = root / "analysis"
    write_json(analysis / "summary.json", summary)
    write_json(analysis / "per_seed.json", per_seed)
    write_json(analysis / "source_manifests.json", manifests)
    lines = ["# Experiment 22 — late-state critic objective audit", "",
             "Execution-only smoke; not scientific evidence." if smoke else
             "One-source-seed pilot; no seed-level uncertainty estimate." if pilot else
             "Three source seeds; candidate budgets were selected without final-assessment access.", "",
             "Negative differences favour the candidate. All contrasts are exploratory.", "",
             "| Selection-locked contrast | Deployed variance difference | Seed-level 95% interval |",
             "|---|---:|---|"]
    for name, result in contrasts.items():
        lines.append(f"| {name} | {result['mean_difference']:.6g} | {result['paired_seed_t95']} |")
    lines += ["", "## Selection-locked candidates", "",
              "Variances/TD error and p95 are means across the two critic members. The p95 column is",
              "not a pooled quantile. Cost sums both members and charges each method its own data cost.", "",
              "| Seed | Objective | Updates | Online variance | Deployed variance | Deployed p95 | Deployed TD MSE | Data + fit seconds |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed in per_seed:
        for objective in contract(smoke)["objectives"]:
            selected = seed["selected"][objective]
            online, deployed = (selected["assessment"][rep] for rep in ("online", "deployed"))
            cost = sum(r["standalone_data_plus_fit_seconds"] for r in seed["fitting"]
                       if r["objective"] == objective and r["updates"] == selected["updates"])
            lines.append(f"| {seed['seed']} | {objective} | {selected['updates']} | "
                         f"{online['mean_variance']:.6g} | {deployed['mean_variance']:.6g} | "
                         f"{deployed['p95_variance']:.6g} | {deployed['mean_td_mse']:.6g} | {cost:.2f} |")
    lines += ["", "## Untouched source and complete audit cost", "",
              "| Seed | Original deployed variance | Audit seconds | CPU seconds | Peak RSS MiB |",
              "|---:|---:|---:|---:|---:|"]
    for seed in per_seed:
        original = np.mean([r["assessment"]["deployed"]["summary"]["mean_variance"]
                            for r in seed["unchanged_assessment"]])
        cost = seed["worker_summary"]
        lines.append(f"| {seed['seed']} | {original:.6g} | {cost['elapsed_audit_seconds']:.2f} | "
                     f"{cost['observed_cpu_seconds']:.2f} | {cost['peak_rss_mib']:.2f} |")
    lines += ["", "The audit total includes all four fitting paths, probes, snapshots and intermediate syncing.",
              "It excludes source loading/download and cloud setup/final upload; see source manifests,",
              "source-fetch diagnostics and Batch job elapsed time for those additional costs."]
    lines += ["", "Review per-seed tail diagnostics, TD error and standalone data-plus-fit costs together.",
              "No automatic promotion to full training. A negative screen does not rule out early-stage benefit.",
              "Same-update comparisons are not compute-matched. Source loading, panel generation, fitting,",
              "selection, persistence, final assessment and cloud job costs must be counted separately."]
    (analysis / "report.md").write_text("\n".join(lines)+"\n")
    return analysis
