"""Integrity-checked, seed-level evidence; absence from a reservoir is not non-visitation."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import math
import statistics

from .common import write_json, verify, complete, file_hash, digest, revision
from .config import contract, SOURCES


def read(path):
    return json.loads(Path(path).read_text())


def uncertainty(values):
    """Independent training seeds, not board positions or fitting replicas."""
    mean = statistics.mean(values)
    # Prespecified three-source experiment: t interval is descriptive and low-power.
    interval = None
    if len(values) == 3:
        half = 4.30265273 * statistics.stdev(values) / math.sqrt(3)
        interval = [mean-half, mean+half]
    return dict(n_seeds=len(values), mean=mean, per_seed=values,
                descriptive_95pct_t_interval=interval)


def case_evidence(predictions):
    rows = [r for r in predictions["rows"] if r["strict_winner"]]
    supported = [r for r in rows if r["retained_target"] is not None]
    adequate = [r for r in supported if r["retained_target"]["count"] >= 2
                and r["retained_target"]["effective_rows"] >= 1.5]
    # Prespecified descriptive flags, NOT automatic causal conclusions.
    fit = [r for r in adequate if r["retained_target"]["target"][0] <= .05 and r["deployed"][0] > .5]
    old = [r for r in adequate if r["retained_target"]["target"][0] > .5]
    return dict(guaranteed_winner_positions=len(rows), exact_retained_positions=len(supported),
        adequately_supported_positions=len(adequate),
        sensible_replay_but_large_deployed_fold=len(fit), empirical_replay_folds_over_half=len(old),
        current_regret_folds_over_half=sum(r["current_regret_policy"][0] > .5 for r in rows),
        fitting_mismatch_case_ids=[r["id"] for r in fit],
        note="Flags describe retained labels/current strategy; they do not identify the exact historical average.")


def continuation_evidence(path):
    summary = read(path/"summary.json")
    visits, generated, accepted, sampled = (Counter() for _ in range(4))
    trace_count = truncations = 0
    for file in sorted(path.glob("iteration_*.json")):
        data = read(file)
        for row in data["merges"]:
            visits.update(row["visit_counts"])
            trace_count += len(row["terminal_traces"])
            truncations += row.get("truncated_traces", 0)
            for name in ("regret", "average_policy"):
                if name in row:
                    generated.update({name+"/"+k:v for k,v in row[name]["generated_categories"].items()})
                    accepted.update({name+"/"+k:v for k,v in row[name]["accepted_categories"].items()})
        for fit in data["regret_fits"]:
            sampled.update(fit["sampled_categories"])
    return dict(**summary, visit_counts=dict(visits), generated_categories=dict(generated),
        arrival_accepted_categories=dict(accepted), regret_fit_sampled_categories=dict(sampled),
        recorded_terminal_traces=trace_count, truncated_terminal_traces=truncations,
        inference="A zero count in this short continuation is inconclusive about historical coverage.")


def aggregate(directory, *, smoke=False, local=False, pilot=False):
    root = Path(directory)
    protocol = read(root/"protocol.json")
    cfg = contract(smoke)
    if protocol["config"] != json.loads(json.dumps(cfg)) or protocol["synthetic"] != local:
        raise ValueError("Protocol or synthetic/real provenance differs")
    if protocol["commit"] != revision():
        raise ValueError("Cannot aggregate artifacts produced by another commit")
    panel = read(root/"panel.json")
    if panel["protocol"] != protocol["config"]:
        raise ValueError("Panel protocol differs")
    seeds = [0] if pilot else cfg["seeds"]
    summaries, effects = [], defaultdict(dict)
    for seed in seeds:
        worker = root/f"workers/seed_{seed}"
        stages = {}
        for hours in (24, 48):
            path = worker/f"offline_{hours}h"
            success = verify(path)
            manifest = read(path/"manifest.json")
            if (success["smoke"] != smoke or success["synthetic"] != local
                    or manifest["seed"] != seed or manifest["hours"] != hours
                    or manifest["config"] != protocol["config"]
                    or manifest["panel_sha256"] != digest(panel)
                    or manifest["commit"] != protocol["commit"]):
                raise ValueError("Offline stage provenance differs")
            summary = read(path/"summary.json")
            if not local and summary["source"]["run"] != SOURCES[hours]:
                raise ValueError("Unexpected source run")
            if not summary["estimator_expectation_verified"]:
                raise ValueError("Estimator identity not checked")
            stages[str(hours)] = dict(summary=summary,
                evidence=case_evidence(read(path/"case_predictions.json")))
        success = verify(worker/"continuation")
        cont = continuation_evidence(worker/"continuation")
        if (success["smoke"] != smoke or success["synthetic"] != local
                or cont["complete_iterations"] != cfg["continuation_iterations"]
                or cont["diagnostic_samples_added"] != 0 or (not local and cont["ray_actors"] != 8)):
            raise ValueError("Incomplete continuation or changed training process")
        game = read(worker/"gameplay.json") if not local else None
        if not local and (not game["complete"] or not game["rows"]):
            raise ValueError("Missing gameplay guardrail")
        refits = worker/"offline_48h/refits"
        candidates = read(refits/"candidates.json")
        lock = read(refits/"fitting_lock.json")
        assessment = read(refits/"assessment.json")
        if (assessment["lock_sha256"] != file_hash(refits/"fitting_lock.json")
                or not lock["no_selection"] or not stages["48"]["summary"]["refitting"]["source_preserved"]):
            raise ValueError("Assessment was not locked or source modified")
        if lock["candidates"] != [{k:c[k] for k in ("name", "policy_sha256")} for c in candidates]:
            raise ValueError("Candidates changed after locking")
        for c in candidates:
            if file_hash(refits/(c["name"]+".pkl")) != c["policy_sha256"]:
                raise ValueError("Candidate checksum mismatch")
        for rep in range(cfg["fit_replicates"]):
            for budget in cfg["updates"]:
                match = [c for c in candidates if c["replicate"] == rep and c["updates"] == budget]
                if (not match or len({c["minibatch_stream_sha256"] for c in match}) != 1
                        or len({c["initial_model_sha256"] for c in match}) != 1
                        or len({c["initial_optimizer_sha256"] for c in match}) != 1):
                    raise ValueError("Refit arms used unmatched starting states/minibatches")
        scores = {r["name"]:r["metrics"] for r in assessment["scores"]}
        by_arm = defaultdict(list)
        for c in candidates:
            by_arm[(c["arm"], c["updates"])].append(c)
        seed_metrics = {}
        for (arm, budget), group in by_arm.items():
            for domain in ("assessment/all", "assessment/quads", "assessment/straight_flush"):
                if domain not in scores["unchanged"]:
                    continue
                for field in ("mean_gap_bb", "mean_fold_probability", "p95_gap_bb"):
                    metric = statistics.mean(scores[c["name"]][domain][field] for c in group)
                    ordinary = statistics.mean(scores[c["name"]][domain][field]
                        for c in by_arm[("ordinary", budget)])
                    key = f"{arm}/{budget}/{domain}/{field}"
                    seed_metrics[key] = dict(value=metric,
                        change_vs_unchanged=metric-scores["unchanged"][domain][field],
                        change_vs_ordinary=metric-ordinary)
                    effects[key][seed] = seed_metrics[key]
        summaries.append(dict(seed=seed, historical=stages, continuation=cont, gameplay=game,
            refit_effects=seed_metrics, eligibility=read(refits/"eligibility.json")))
    grouped = {key:{field:uncertainty([by_seed[s][field] for s in sorted(by_seed)])
                    for field in ("value", "change_vs_unchanged", "change_vs_ordinary")}
               for key,by_seed in effects.items()}
    result = dict(experiment=28, seeds=seeds, smoke=smoke, local_smoke=local, pilot=pilot,
        commit=protocol["commit"], per_seed=summaries, refit_effects=grouped,
        endpoints="Lower conditional guaranteed-winner fold/gap is better. Not whole-game exploitability.",
        inference="Fitting replicas are averaged within source seed. Board positions are correlated, not independent replicates.",
        limitations=["Retained replay is an empirical teacher, not the true CFR average.",
            "Current regret strategy cannot be substituted for the historical average teacher.",
            "Holdout board classes are absent from this refit and prior audit, not necessarily historical training.",
            "Targeted weighting uses only existing targets; probes never enter replay.",
            "A short diagnostic continuation may encounter no exact rare state.",
            "Three source seeds and a modest match panel cannot establish a global exploitability reduction."])
    output = root/("pilot_analysis" if pilot else "analysis")
    write_json(output/"summary.json", result)
    write_json(output/"per_seed.json", summaries)
    lines = ["# Experiment 28 — rare-state coverage and learning-path audit", "",
        f"{'NON-SCIENTIFIC SMOKE' if smoke else 'Development experiment'}; source seeds {seeds}.", "",
        "## Historical coverage and observed mismatch", "",
        "Counts concern guaranteed-winner development positions, not independent hands or a natural play frequency.", "",
        "| Seed | Hours | Exact replay coverage | Adequate support | Sensible replay / deployed fold > 0.5 | Current regret folds > 0.5 |",
        "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        for hours,h in s["historical"].items():
            e=h["evidence"]
            lines.append(f"| {s['seed']} | {hours} | {e['exact_retained_positions']}/{e['guaranteed_winner_positions']} | {e['adequately_supported_positions']} | {e['sensible_replay_but_large_deployed_fold']} | {e['current_regret_folds_over_half']} |")
    lines += ["", "## Interpretation", "",
        "A well-supported, low-fold replay target paired with a high-fold deployed policy is evidence of an empirical fitting mismatch. It is not proof that the exact historical average was correct. High-fold replay labels shift attention upstream, but alone cannot distinguish historical averaging from regret approximation or noisy targets. The passive continuation records actual estimator, insertion and fitting events to investigate that distinction.", "",
        "## Fixed refit endpoints on new board classes", "",
        "Changes below are conditional payoff-gap differences in BB; negative is better. Average fitting replicas within each source seed before comparing seeds. All candidates were locked before assessment.", "",
        "| Arm | Extra updates | Source seeds | Gap change vs unchanged | Gap change vs ordinary fit |",
        "|---|---:|---:|---:|---:|"]
    for key,g in sorted(grouped.items()):
        if key.endswith("/assessment/all/mean_gap_bb"):
            arm,budget,*_=key.split("/")
            lines.append(f"| {arm} | {budget} | {g['value']['n_seeds']} | {g['change_vs_unchanged']['mean']:.6f} | {g['change_vs_ordinary']['mean']:.6f} |")
    lines += ["", "Targeted/random arms are omitted when no sufficiently supported sensible rare labels exist; this is an informative identification limit, not an experiment failure.", "",
        "## Continuation and costs", ""]
    for s in summaries:
        c=s["continuation"]
        cost=sum(sum(h["summary"]["costs"].values()) for h in s["historical"].values())+c["wall_seconds"]
        if s["gameplay"]: cost+=s["gameplay"]["seconds"]
        lines.append(f"- Seed {s['seed']}: {c['complete_iterations']} complete continuation iterations, {c['recorded_terminal_traces']} terminal rare-state estimator traces; measured audit/refit/continuation/gameplay subprocess work {cost/60:.1f} minutes. Source transfer and bootstrap are additional; see resource diagnostics and source manifests.")
    lines += ["", "## What to do next", "",
        "1. A supported replay–policy mismatch plus improvement over both ordinary and random refits supports a policy-fitting repair. Check the gameplay guardrail before any larger claim.",
        "2. Correct estimator expectation but poor current regret predictions suggests inspecting actual sampled targets and regret-fitting traces; no observed natural trace leaves this unresolved.",
        "3. Sparse/absent replay with no supported repair targets motivates a later sampling/retention intervention. Do not manufacture oracle labels in this audit.",
        "4. Replicate promising mechanisms before funding a full training modification; a negative late-stage audit does not exclude earlier benefits.", "",
        "## Limitations", ""]+["- "+s for s in result["limitations"]]
    output.mkdir(parents=True,exist_ok=True)
    (output/"report.md").write_text("\n".join(lines)+"\n")
    if pilot:
        complete(output,experiment=28,pilot=True,seeds=seeds,commit=protocol["commit"])
    else:
        complete(root,experiment=28,seeds=seeds,smoke=smoke,local_smoke=local,commit=protocol["commit"])
    return result
