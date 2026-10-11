"""Integrity-checked, separate primary/secondary reports; no test-set reselection."""
import json
from pathlib import Path

import numpy as np

from experiments.critic_target_cache_benchmark import write_json
from experiments.fhp.exp22_fhp_critic_objectives.run import choose_candidates
from fhp_escher.checkpointing import sha256_file
from .config import contract, STAGES
from .states import commit, PHASE


def read_stage(root, stage, smoke):
    directory = root/"audits"/stage
    success = json.loads((directory/"SUCCESS.json").read_text())
    if (success.get("status") != "complete" or success.get("seed") != 0 or success.get("smoke") != smoke):
        raise ValueError("Incomplete/mixed stage worker")
    result = {}
    for name in ("manifest", "summary", "selection_lock", "fitting", "assessment", "panels", "source_identity", "unchanged_assessment"):
        path = directory/(name+".json")
        if sha256_file(path) != success["artifacts"].get(path.name):
            raise ValueError("Audit artifact checksum mismatch")
        result[name] = json.loads(path.read_text())
    config = contract(smoke, stage)
    manifest = result["manifest"]
    if (manifest["config"] != config or manifest["stage"] != stage or manifest["provenance"]["phase"] != PHASE
            or manifest["provenance"]["smoke"] != smoke or not result["summary"]["source_state_preserved"]):
        raise ValueError("Invalid phase or scientific contract")
    expected = {(o, u, m) for o in config["objectives"] for u in config["updates"] for m in (0, 1)}
    for key in ("fitting", "assessment"):
        rows = result[key]
        if len(rows) != len(expected) or {(r["objective"], r["updates"], r["member"]) for r in rows} != expected:
            raise ValueError("Incomplete or duplicated budget curve")
    if (result["selection_lock"]["candidates"] != choose_candidates(result["fitting"])
            or result["selection_lock"]["fitting_sha256"] != success["artifacts"]["fitting.json"]):
        raise ValueError("Selection lock was altered")
    for fold in (0, 1):
        fitted = [r for r in result["fitting"] if r["member"] == fold]
        for key in ("initial_online_sha256", "initial_optimizer_sha256"):
            if len({r[key] for r in fitted}) != 1:
                raise ValueError("Unmatched starting conditions")
        if len({r["fixed_fit_panel"]["fixed_replay_indices_sha256"] for r in fitted}) != 1:
            raise ValueError("Training diagnostic panel changed")
        for step in config["updates"]:
            matched = [r for r in fitted if r["updates"] == step and r["objective"] != "replay_td"]
            if len({r["minibatch_stream_sha256"] for r in matched}) != 1:
                raise ValueError("Fresh-data minibatch streams are not matched")
        for r in fitted:
            if r["updates"] == 0 and r["deployed_sha256"] != result["source_identity"]["critics"][fold]["target"]:
                raise ValueError("Zero-update control refreshed the target")
    sets = [{r["history_sha256"] for r in p["roots"]} for p in result["panels"].values()]
    if sum(map(len, sets)) != len(set.union(*sets)):
        raise ValueError("Probe split overlap")
    return result


def curves(stage):
    result = []
    for objective in stage["manifest"]["config"]["objectives"]:
        for updates in stage["manifest"]["config"]["updates"]:
            fitted = [r for r in stage["fitting"] if (r["objective"], r["updates"]) == (objective, updates)]
            assessed = [r for r in stage["assessment"] if (r["objective"], r["updates"]) == (objective, updates)]
            item = dict(objective=objective, updates=updates,
                selected=stage["selection_lock"]["candidates"][objective]["updates"] == updates,
                standalone_data_plus_fit_seconds=sum(r["standalone_data_plus_fit_seconds"] for r in fitted),
                snapshot_and_diagnostic_seconds=sum(r["snapshot_and_selection_seconds"] for r in fitted))
            for representation in ("online", "deployed"):
                item[representation] = dict(
                    assessment={key: float(np.mean([r["assessment"][representation]["summary"][key] for r in assessed]))
                                for key in assessed[0]["assessment"][representation]["summary"]},
                    selection_variance=float(np.mean([r["selection"][representation]["summary"]["mean_variance"] for r in fitted])),
                    fit_variance=float(np.mean([r["fixed_fit_panel"][representation]["corrected_regret"]["mean_variance"] for r in fitted])),
                    fit_objectives={key: float(np.mean([r["fixed_fit_panel"][representation]["objectives"][key] for r in fitted]))
                                    for key in fitted[0]["fixed_fit_panel"][representation]["objectives"]})
            result.append(item)
    return result


def aggregate(root, *, smoke=False, local=False):
    if local and not smoke:
        raise ValueError("Only local execution smoke may omit historical late state")
    root = Path(root)
    stages = STAGES[:2] if local else STAGES
    verify = json.loads((root/"checkpoint_verification.json").read_text())
    if verify.get("status") != "complete" or verify["smoke"] != smoke:
        raise ValueError("Missing full checkpoint round-trip")
    references = {}
    for name in (("reference",) if local else ("reference", "late_reference")):
        value = json.loads((root/name/"SUCCESS.json").read_text())
        if value["status"] != "complete" or value["smoke"] != smoke:
            raise ValueError("Reference did not complete")
        if name == "late_reference" and value["final_iteration"] != value["source_iteration"]+1:
            raise ValueError("Late observation must follow exactly one normal iteration")
        references[name] = value
    details = {stage: read_stage(root, stage, smoke) for stage in stages}
    if len({(d["manifest"]["commit"], d["manifest"]["code_sha256"]) for d in details.values()}) != 1:
        raise ValueError("Mixed code across stage audits")
    all_curves = {stage: curves(details[stage]) for stage in stages}
    limitations = [
        "One seed: no seed-level confidence interval, replication, equivalence, or controller-superiority claim.",
        "Early stages share a new trajectory; late is a historical 24h state plus one iteration, not a coherent three-stage trajectory.",
        "Conditional within-history probe variance is not exploitability, global CFR variance, or poker performance.",
        "Fixed-fitting-panel scores are in-sample diagnostics. Only independent assessment measures generalisation.",
        "Quantiles are means of within-member quantiles, not pooled quantiles; folds/roots are not seed replicates.",
        "Equal update counts are not equal compute/data; count data, fitting, probes, snapshots and cloud overhead.",
        "One fit per method/state; adaptive allocation would need separate end-to-end tests against fixed budgets and schedules.",
        "No automatic promotion of any method; final assessment never reselects a candidate.",
        "Root distribution is frozen-behaviour sampled, not information-set-balanced; replay/infoset overlap remains possible.",
    ]
    for experiment_id in (24, 25):
        directory = root/"analysis"/f"exp{experiment_id}"
        selected_curves = {stage: [r for r in rows if experiment_id == 25 or r["objective"] == "replay_td"]
                           for stage, rows in all_curves.items()}
        summary = dict(experiment_id=experiment_id, smoke=smoke, source_seed=0,
                       stages=list(stages), curves=selected_curves, limitations=limitations,
                       seed_level_confidence_intervals=None)
        write_json(directory/"summary.json", summary)
        title = "Pre-fit TD update budgets" if experiment_id == 24 else "Critic objective generalisation"
        lines = [f"# Experiment {experiment_id} — {title}", "",
                 "Execution smoke only; not scientific evidence." if smoke else "One-seed development evidence.", "",
                 "Budgets are selected on independent selection panels before constructing final-assessment panels.",
                 "The zero-update candidate retains the existing deployed target without a history refresh.", ""]
        for stage, rows in selected_curves.items():
            identity = details[stage]["manifest"]["provenance"]
            lines += [f"## {stage}", "", f"Iteration {identity['iteration']}; nodes {identity['nodes']}; "
                      f"active time {identity['elapsed_training_seconds']:.2f} s. Phase: {identity['phase']}.", "",
                      "| Objective | Updates | Selected | Online fit variance | Online test variance | Deployed selection | Deployed test | Deployed test p95 | Data + fit s | Diagnostic/snapshot s |",
                      "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                online, deployed = r["online"], r["deployed"]
                lines.append(f"| {r['objective']} | {r['updates']} | {r['selected']} | {online['fit_variance']:.6g} | "
                    f"{online['assessment']['mean_variance']:.6g} | {deployed['selection_variance']:.6g} | "
                    f"{deployed['assessment']['mean_variance']:.6g} | {deployed['assessment']['p95_variance']:.6g} | "
                    f"{r['standalone_data_plus_fit_seconds']:.2f} | {r['snapshot_and_diagnostic_seconds']:.2f} |")
        lines += ["", "## Interpretation and costs", "", *[f"- {text}" for text in limitations], "",
                  "Fit-panel losses for every objective and representation, all tails and all candidate curves are in summary.json.",
                  "Full state-level details, selection locks, source identities, and audit costs are retained in ../per_stage.json.",
                  "Reference/training, loading, checkpoint/hash/upload and workflow costs are in ../summary.json and Batch diagnostics.",
                  "Per-candidate cost excludes diagnostic overhead; the adjacent column exposes it. Shared reference costs are not claimed twice."]
        (directory/"report.md").write_text("\n".join(lines)+"\n")
    summary = dict(experiments=[24, 25], smoke=smoke, local_smoke=local, source_seed=0,
        commit=commit(), stages=list(stages), references=references, checkpoint_verification=verify,
        setup_costs={name: json.loads((root/name).read_text())
                     for name in ("reference_initialisation.json", "late_source_load.json", "diagnostics/source_fetch.json")
                     if (root/name).is_file()},
        audits={s: dict(**details[s]["summary"], source_load_seconds=details[s]["manifest"]["source_load_seconds"]) for s in stages},
        limitations=limitations)
    write_json(root/"analysis/summary.json", summary)
    write_json(root/"analysis/per_stage.json", details)
    paths = ["analysis/summary.json", "analysis/per_stage.json", "analysis/exp24/summary.json", "analysis/exp25/summary.json"]
    write_json(root/"SUCCESS.json", dict(status="complete", smoke=smoke, local_smoke=local,
        commit=commit(), stages=list(stages), experiments=[24, 25],
        artifacts={p: sha256_file(root/p) for p in paths}))
    return summary
