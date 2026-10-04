"""Small, seed-level summaries; folds, iterations and rollouts are not replicates."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from experiments.critic_target_cache_benchmark import write_json
from fhp_escher.checkpointing import sha256_file
from .config import contract


def paired_summary(short, full):
    short, full = np.asarray(short, float), np.asarray(full, float)
    difference = short-full
    n = len(short)
    se = float(difference.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    interval = ([float(difference.mean()-t.ppf(.975, n-1)*se),
                 float(difference.mean()+t.ppf(.975, n-1)*se)] if n > 1 else None)
    return dict(short_mean=float(short.mean()), full_mean=float(full.mean()),
                short_minus_full=float(difference.mean()), paired_difference_se=se,
                paired_difference_95_t_interval=interval, independent_source_seeds=n,
                ratio_of_seed_means=float(short.mean()/full.mean()) if full.mean() > 0 else None,
                favourable_seed_count=int(np.sum(difference < 0)))


def aggregate(root, smoke=False):
    root = Path(root)
    config = contract(smoke)
    short, full = map(str, config["updates"])
    per_seed, detailed, manifests = [], [], []
    for seed in config["seeds"]:
        folder = root / "workers" / f"seed_{seed}"
        success = json.loads((folder / "SUCCESS.json").read_text())
        if success.get("status") != "complete" or success.get("seed") != seed or success.get("smoke") != smoke:
            raise ValueError("Incomplete or mixed worker")
        required = ["manifest.json", "summary.json"] + [f"boundary_{b}.json" for b in range(1, config["boundaries"]+1)]
        for name in required:
            if sha256_file(folder / name) != success["artifacts"].get(name):
                raise ValueError("Audit JSON checksum mismatch")
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["config"] != config or manifest["provenance"]["seed"] != seed:
            raise ValueError("Worker contract/source seed mismatch")
        manifests.append(manifest)
        worker_summary = json.loads((folder / "summary.json").read_text())
        if (worker_summary["seed"] != seed or worker_summary["smoke"] != smoke
                or worker_summary["boundaries"] != config["boundaries"]
                or worker_summary["final_iteration"] - worker_summary["source_iteration"] != config["boundaries"]):
            raise ValueError("Incomplete continuation audit")
        boundaries = [json.loads((folder / f"boundary_{b}.json").read_text())
                      for b in range(1, config["boundaries"]+1)]
        for b, record in enumerate(boundaries, 1):
            if (record["boundary"] != b or record["seed"] != seed or not record["diagnostic_state_and_rng_preserved"]
                    or record["iteration"] != worker_summary["source_iteration"] + b
                    or len(record["critics"]) != 2 or len(record["probes"]) != 4*config["probes_per_player_fold"]
                    or success["artifacts"].get(record["weights_path"]) != record["weights_sha256"]):
                raise ValueError("Incomplete or incompatible diagnostic boundary")
        rows = [p for b in boundaries for p in b["probes"]]
        critics = [c for b in boundaries for c in b["critics"]]
        metrics = {}
        for budget in (short, full):
            metrics[budget] = {key: float(np.mean([p["arms"][budget][key] for p in rows])) for key in (
                "mean_legal_action_advantage_variance", "online_heldout_td_mse", "deployed_heldout_td_mse")}
            metrics[budget].update({key: float(np.mean([c["arms"][budget][key] for c in critics])) for key in (
                "online_training_probe_td_mse", "deployed_training_probe_td_mse", "estimated_native_fit_seconds")})
        per_seed.append(dict(seed=seed, arms=metrics,
                             mean_phase_seconds={key: float(np.mean([b["phase_seconds"][key] for b in boundaries]))
                                                 for key in ("collection", "regret", "critics", "calibration")}))
        detailed.extend(boundaries)
    if len({(m["audit_commit"], m["code_sha256"]) for m in manifests}) != 1:
        raise ValueError("Mixed worker code revisions")
    if len({json.dumps(m["runtime"], sort_keys=True) for m in manifests}) != 1:
        raise ValueError("Mixed worker runtimes")
    if not smoke and len({m["provenance"]["state_sha256"] for m in manifests}) != len(config["seeds"]):
        raise ValueError("Source states unexpectedly duplicated across seeds")
    comparisons = {key: paired_summary([s["arms"][short][key] for s in per_seed],
                                      [s["arms"][full][key] for s in per_seed]) for key in per_seed[0]["arms"][short]}
    analysis = root / "analysis"
    write_json(analysis / "summary.json", dict(config=config, comparisons=comparisons,
        mean_phase_seconds_per_iteration={key: float(np.mean([s["mean_phase_seconds"][key] for s in per_seed]))
                                         for key in per_seed[0]["mean_phase_seconds"]},
        conclusion="Diagnostic screen only. Review variance, held-out error and time jointly; no automatic promotion.",
        limitations=["Three source seeds; adjacent boundaries are correlated within seed.",
                     "Conditional full-history variance on a sampled probe distribution, not exact exploitability or global regret.",
                     "Fresh-rollout TD error uses the original frozen bootstrap target, not oracle action values.",
                     "Only one shortened fit at each control boundary; not a persistent 5000-update training trajectory.",
                     "Short-fit time is a native-prefix estimate including cache and an estimated finalisation cost."]))
    write_json(analysis / "per_seed.json", per_seed)
    write_json(analysis / "detailed_results.json", detailed)
    write_json(analysis / "source_manifests.json", manifests)
    lines = ["# Experiment 17: frozen-data critic-budget audit", "",
             "Lower variance and error are better. Uncertainty is across source training seeds, not rollout samples.", "",
             "| Metric | Short fit | Full fit | Short minus full (95% paired t interval) |",
             "|---|---:|---:|---|"]
    for key, row in comparisons.items():
        interval = row["paired_difference_95_t_interval"]
        uncertainty = "smoke only" if interval is None else f"[{interval[0]:.5g}, {interval[1]:.5g}]"
        lines.append(f"| {key} | {row['short_mean']:.5g} | {row['full_mean']:.5g} | {row['short_minus_full']:.5g} {uncertainty} |")
    lines += ["", "This is a mechanism screen, not evidence of stronger play or long-run convergence.",
              "The full-fit branch alone advances to the next boundary. No large training states are produced."]
    (analysis / "report.md").write_text("\n".join(lines) + "\n")
    return analysis
