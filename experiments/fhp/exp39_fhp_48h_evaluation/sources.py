"""Validate native policy payloads and both 24-to-48h continuation lineages."""
import json

from experiments.fhp.exp10_fhp_hand_board_features import config as original
from experiments.fhp.exp36_fhp_strength_critic2500 import config as richer
from experiments.fhp.exp16_fhp_hand_board_48h import contract as old_contract
from experiments.fhp.exp38_fhp_strength_critic2500_48h import contract as new_contract
from experiments.fhp.exp36_fhp_strength_critic2500 import evaluate as existing

native = existing.sources
CONFIGS = {"exp16": original, "exp36": richer, "exp38": richer}
HOURS = {"exp16": 48, "exp36": 24, "exp38": 48}
LABELS = {"exp16": "Original features / 10,000 critic updates / 48h",
          "exp36": "Richer AE features / 2,500 critic updates / 24h",
          "exp38": "Richer AE features / 2,500 critic updates / 48h"}


def metadata(root, contract, seed, hours):
    worker = root / "workers" / contract.worker_name(seed)
    return contract.validate_metadata(
        {name: (worker / name).read_bytes() for name in contract.METADATA_FILES}, seed=seed, hours=hours)


def validate_lineage(source_root, destination_root, source_records, continued, contract):
    index = {(r["seed"], r["training_hours"]): r for r in source_records}
    for seed in contract.SEEDS:
        data = metadata(source_root, contract, seed, 24)
        metadata(destination_root, contract, seed, 48)
        worker = destination_root / "workers" / contract.worker_name(seed)
        lineage = json.loads((worker / "continuation_source.json").read_text())
        final = data["checkpoint_manifest.json"][-1]
        expected = dict(seed=seed, source_total_hours=24, total_hours=48,
            source_commit=contract.TRAINING_REF, source_state_path=final["training_state_path"],
            source_state_sha256=final["training_state_sha256"],
            source_summary_sha256=data["SUCCESS.json"]["summary_sha256"])
        if (any(lineage.get(k) != v for k, v in expected.items()) or not
                lineage.get("source_worker", "").endswith(f"/{contract.SOURCE_RUN_ID}/workers/{contract.worker_name(seed)}")):
            raise ValueError(f"Invalid continuation lineage: {contract.EXPERIMENT_NAME}, seed {seed}")
        rows = [r for r in continued if r["seed"] == seed]
        for record in rows:
            hour = record["training_hours"]
            if hour <= 24 and any(record[k] != index[(seed, hour)][k] for k in
                    ("checkpoint_sha256", "nodes_touched", "outer_iteration", "training_elapsed_seconds")):
                raise ValueError("Imported 6–24h policy history differs from its source")
        last = next(r for r in rows if r["training_hours"] == 48)
        if any(last[k] <= index[(seed, 24)][k] for k in ("nodes_touched", "outer_iteration")):
            raise ValueError("No additional training occurred after restore")


def discover(roots, source10):
    early_old = existing.discover(source10, "exp10")
    early_new = existing.discover(roots["exp36"], "exp36")
    records = {"exp36": early_new}
    for cohort, parent, root, contract in (
            ("exp16", early_old, source10, old_contract),
            ("exp38", early_new, roots["exp36"], new_contract)):
        records[cohort] = native.discover_checkpoints(roots[cohort], cohort,
            config=CONFIGS[cohort], hours=contract.HOURS, label=LABELS[cohort],
            cache=True, allow_continuation=True)
        validate_lineage(root, roots[cohort], parent, records[cohort], contract)
    selected = []
    for cohort in HOURS:
        for record in records[cohort]:
            if record["training_hours"] == HOURS[cohort]:
                selected.append(dict(record, selected_for_evaluation=True, label=LABELS[cohort]))
    return selected
