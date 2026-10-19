"""Verify a tiny saved 48h endpoint can continue again; no production training.

Embedded by the new workflow into the original Experiment 36 learner checkout.
Its native smoke first creates short 24h/48h schedule analogues and verifies
bitwise restoration. This adds a 48h -> 72h analogue to prove the final state
is a training continuation, not just a playable policy export.
"""
import json
from pathlib import Path
import sys


def verify_final_restart(output_root):
    from experiments.fhp.exp36_fhp_strength_critic2500.config import EXPERIMENT
    from experiments.fhp.strategic_feature_suite.run import _run_task
    from experiments.fhp.strategic_feature_suite.aggregate import task_name, aggregate_workers
    from fhp_escher.checkpointing import sha256_file
    root = Path(output_root)
    source = root/"extension_smoke/workers"/task_name(0, 0, experiment=EXPERIMENT)
    before = json.loads((source/"summary.json").read_text())
    old_rows = json.loads((source/"checkpoint_manifest.json").read_text())
    state = source/old_rows[-1]["training_state_path"]
    source_hash = sha256_file(state)
    result_root = root/"final_restart_smoke"
    result = _run_task(task_index=0, output_root=result_root, smoke=True, resume=True,
        total_hours=72, source_worker=source, experiment=EXPERIMENT)
    worker = result_root/"workers"/task_name(0, 0, experiment=EXPERIMENT)
    rows = json.loads((worker/"checkpoint_manifest.json").read_text())
    if (before["checkpoint_count"] != 8 or result["checkpoint_count"] != 12
            or not result["resumed_from_training_state"] or result["final_iteration"] <= before["final_iteration"]
            or result["final_nodes_touched"] <= before["final_nodes_touched"]):
        raise RuntimeError("Final-endpoint restart did not advance training")
    if any(old["sha256"] != new["sha256"] or sha256_file(worker/new["path"]) != old["sha256"]
           for old, new in zip(old_rows, rows[:8])):
        raise RuntimeError("Restart modified historical policies")
    if (sha256_file(state) != source_hash or any("training_state_path" in r for r in rows[:-1])
            or len(list((worker/"training_states").glob("*.pt"))) != 1
            or sha256_file(worker/rows[-1]["training_state_path"]) != rows[-1]["training_state_sha256"]
            or (worker/"continuation_inputs").exists()):
        raise RuntimeError("Final restart-state integrity/retention check failed")
    aggregate_workers(result_root, seeds=(0,), total_hours=72, experiment=EXPERIMENT)
    report = dict(status="passed", smoke=True, resumed_from_full_state=True,
        source_state_unchanged=True, imported_policies_unchanged=True, checkpoint_count=12,
        new_full_states=1, scope="Tiny native 48-to-72 schedule analogue; not a production-state restore")
    (root/"final_restart_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    print(json.dumps(verify_final_restart(sys.argv[1]), indent=2))
