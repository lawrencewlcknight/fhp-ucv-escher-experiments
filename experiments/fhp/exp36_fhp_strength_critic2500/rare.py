"""Batched deployed-policy diagnostics, not reach-weighted exploitability."""
import json
from pathlib import Path
from time import perf_counter

from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file
from fhp_escher.game import load_fhp_game
from experiments.fhp.exp28_fhp_rare_state_audit import diagnostics, panels
from experiments.fhp.exp35_fhp_feature_rare_audit import panel as previous_panel
from experiments.fhp.exp35_fhp_feature_rare_audit.run import interval
from experiments.fhp.retrospective_exp7_exp8_evaluation.run import shared, protocol

INPUTS = Path(previous_panel.__file__).with_name("inputs.json")


def implementation_hashes():
    return {name: sha256_file(path) for name, path in (
        ("inputs", INPUTS), ("panel", previous_panel.__file__),
        ("states", panels.__file__), ("diagnostics", diagnostics.__file__),
        ("interval", Path(previous_panel.__file__).with_name("run.py")))}


def build_panel(smoke=False):
    inputs = json.loads(INPUTS.read_text())
    # Do not call the previously examined Exp35 positions a new holdout.
    used = previous_panel.build_panel(inputs, smoke=False)["assessment_board_classes"]
    excluded = sorted(set(map(tuple, inputs["prior_board_classes"])) | set(map(tuple, used)))
    panel = previous_panel.build_panel(dict(inputs, prior_board_classes=excluded), smoke=smoke)
    panel.update(excluded_board_classes=excluded,
        scope="Terminal call/fold decisions only; new assessment board classes relative to the recorded strategic audits and Experiment 35. Not guaranteed absent from training. Known examples remain labelled development. No reach-weighted whole-game interpretation.")
    return panel


def assess(record, panel):
    started = perf_counter()
    policy = LoadedFHPPolicy(load_fhp_game(), record["checkpoint_path"])
    cases, _, features, masks = diagnostics.panel_arrays(panel, policy.feature_encoder)
    probabilities = diagnostics.probabilities(policy.model, features, masks)
    return dict(experiment=record["experiment"], seed=record["seed"],
        checkpoint_sha256=record["checkpoint_sha256"],
        metrics=diagnostics.metrics(cases, probabilities),
        positions=[dict(id=c["id"], probabilities=p.tolist()) for c, p in zip(cases, probabilities)],
        seconds=perf_counter() - started)


def run(output, records, fingerprint, *, smoke, controls=("exp20", "exp33", "exp10")):
    panel_path = output / "rare_panel.json"
    receipt_path = output / "rare_panel_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != fingerprint or receipt["sha256"] != sha256_file(panel_path):
            raise ValueError("Rare panel resume checksum mismatch")
        panel = json.loads(panel_path.read_text())
    else:
        started = perf_counter()
        panel = build_panel(smoke)
        panel["construction_seconds"] = perf_counter() - started
        shared._write_json(panel_path, panel)
        shared._write_json(receipt_path, dict(fingerprint=fingerprint, sha256=sha256_file(panel_path)))
    panel_sha = sha256_file(panel_path)
    results = []
    for record in records:
        if record["training_hours"] != 24 or (smoke and record["seed"] != 0):
            continue
        path = output / "rare_results" / f"{record['experiment']}_seed{record['seed']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            body = saved["result"]
            if (saved["fingerprint"] != fingerprint or saved["panel_sha256"] != panel_sha
                    or saved["result_sha256"] != shared._digest(body)
                    or body["checkpoint_sha256"] != record["checkpoint_sha256"]
                    or body["experiment"] != record["experiment"] or body["seed"] != record["seed"]):
                raise ValueError("Rare-result resume mismatch")
        else:
            body = assess(record, panel)
            shared._write_json(path, dict(fingerprint=fingerprint, panel_sha256=panel_sha,
                                         result=body, result_sha256=shared._digest(body)))
        results.append(body)
    rows = [dict(experiment=r["experiment"], training_seed=r["seed"], group=g, **values)
            for r in results for g, values in r["metrics"].items()]
    protocol._write_csv(output / "rare_by_seed.csv", rows)
    index = {(r["experiment"], r["training_seed"], r["group"]): r for r in rows}
    differences = []
    for control in controls:
        for group in sorted({r["group"] for r in rows}):
            for metric in ("mean_fold_probability", "mean_gap_bb", "p95_gap_bb", "max_gap_bb", "fraction_fold_over_half"):
                values = [r[metric] - index[(control, r["training_seed"], group)][metric]
                          for r in rows if r["experiment"] == "exp36" and r["group"] == group]
                differences.append(dict(comparator=control, group=group, metric=metric, **interval(values)))
    summary = dict(panel_sha256=panel_sha, policy_count=len(results),
        by_seed=rows,
        paired_seed_differences=differences, lower_is_better=True,
        scope=panel["scope"], panel_construction_seconds=panel["construction_seconds"],
        inference_seconds=sum(r["seconds"] for r in results),
        uncertainty="Pointwise training-seed intervals, not position-level intervals. Seed labels do not guarantee common training randomness. Diagnostic, not a proof of whole-game improvement.")
    shared._write_json(output / "rare_summary.json", summary)
    return summary
