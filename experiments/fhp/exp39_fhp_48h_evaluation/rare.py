"""Reuse the exact published Exp37 panel, never relabel it a fresh holdout."""
import json
from experiments.fhp.exp36_fhp_strength_critic2500 import rare as base
from .statistics import paired_changes

PANEL_RUN_ID = "exp37-critic-compare-20261019-222256"
PANEL_SHA256 = "092dcd1c88d40037df03afe0441cbc03d297cca9d4ed06e18bec6d7af642f2f4"
SCOPE = ("Exact frozen Experiment 37 diagnostic panel; all positions are now previously examined, "
    "including those with historical 'assessment' labels. Not fresh holdout evidence. "
    "Conditional terminal-call losses, not reach-weighted whole-game exploitability.")
implementation_hashes = base.implementation_hashes


def load_panel(path):
    if base.sha256_file(path) != PANEL_SHA256:
        raise ValueError("Require the exact frozen Experiment 37 rare-panel checksum")
    return json.loads(path.read_text())


def run(output, records, fingerprint, *, panel_path, smoke):
    panel = load_panel(panel_path)
    # Save the exact bytes: provenance checksum must survive copying.
    (output / "rare_panel.json").write_bytes(panel_path.read_bytes())
    base.shared._write_json(output / "rare_panel_receipt.json", dict(
        fingerprint=fingerprint, source_run=PANEL_RUN_ID, sha256=PANEL_SHA256, scope=SCOPE))
    results = []
    for record in records:
        if smoke and record["seed"] != 0:
            continue
        path = output / "rare_results" / f"{record['experiment']}_seed{record['seed']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            body = saved["result"]
            if (saved["fingerprint"] != fingerprint or saved["panel_sha256"] != PANEL_SHA256
                    or saved["result_sha256"] != base.shared._digest(body)
                    or body["checkpoint_sha256"] != record["checkpoint_sha256"]
                    or body["experiment"] != record["experiment"] or body["seed"] != record["seed"]):
                raise ValueError("Rare-result resume mismatch")
        else:
            body = base.assess(record, panel)
            base.shared._write_json(path, dict(fingerprint=fingerprint, panel_sha256=PANEL_SHA256,
                result=body, result_sha256=base.shared._digest(body)))
        results.append(body)
    rows = [dict(experiment=r["experiment"], training_seed=r["seed"], group=g, **values)
            for r in results for g, values in r["metrics"].items()]
    base.protocol._write_csv(output / "rare_by_seed.csv", rows)
    changes = []
    for group in sorted({r["group"] for r in rows}):
        for metric in ("mean_fold_probability", "mean_gap_bb", "p95_gap_bb", "max_gap_bb", "fraction_fold_over_half"):
            values = [dict(r, mean_mbb_per_hand=r[metric]) for r in rows if r["group"] == group]
            _, summaries = paired_changes(values, [])
            for summary in summaries:
                changes.append(dict(group=group, metric=metric, trajectories=summary["trajectories"],
                    mean_change=summary["mean_change_mbb_per_hand"], ci95_low=summary["ci95_low"],
                    ci95_high=summary["ci95_high"], better_direction="lower"))
    base.protocol._write_csv(output / "rare_paired_changes.csv", changes)
    result = dict(panel_sha256=PANEL_SHA256, policy_count=len(results), by_seed=rows,
        paired_48h_minus_24h_changes=changes, scope=SCOPE,
        inference_seconds=sum(r["seconds"] for r in results),
        uncertainty="Paired three-trajectory intervals for 38 minus 36 only; rare positions are not independent training replicates. 16 versus 38 is descriptive, not seed-paired.")
    base.shared._write_json(output / "rare_summary.json", result)
    return result
