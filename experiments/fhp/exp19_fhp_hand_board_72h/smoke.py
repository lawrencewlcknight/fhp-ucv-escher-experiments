"""Small second-continuation check, also embedded into the pinned learner VM.

Only imports modules present at the original Experiment 10 learner commit.
Run after its native smoke has produced the 24 -> 48 checkpoint-prefix test.
The smoke thresholds are milliseconds, not production training hours.
"""
import json
from pathlib import Path
import sys


def verify_second_continuation(output_root):
    from experiments.fhp.exp10_fhp_hand_board_features.run import _run_task
    from experiments.fhp.exp10_fhp_hand_board_features.aggregate import aggregate_workers, task_name
    from fhp_escher.checkpointing import sha256_file

    output_root = Path(output_root)
    source = output_root / "extension_smoke" / "workers" / task_name(0, 0)
    before = json.loads((source / "summary.json").read_text())
    old_rows = json.loads((source / "checkpoint_manifest.json").read_text())
    state = source / old_rows[-1]["training_state_path"]
    state_hash = sha256_file(state)
    extended_root = output_root / "second_extension_smoke"
    result = _run_task(task_index=0, output_root=extended_root, smoke=True,
                       resume=True, total_hours=72, source_worker=source)
    destination = extended_root / "workers" / task_name(0, 0)
    rows = json.loads((destination / "checkpoint_manifest.json").read_text())
    if (before["checkpoint_count"] != 8 or result["checkpoint_count"] != 12
            or not result["resumed_from_training_state"]
            or result["final_iteration"] <= before["final_iteration"]
            or result["final_nodes_touched"] <= before["final_nodes_touched"]):
        raise RuntimeError("Second completed-endpoint continuation did not advance")
    for old, new in zip(old_rows, rows[:8]):
        if (old["sha256"] != new["sha256"]
                or sha256_file(destination / new["path"]) != old["sha256"]):
            raise RuntimeError("Second continuation changed imported policies")
    if (sha256_file(state) != state_hash
            or any("training_state_path" in row for row in rows[:-1])
            or len(list((destination / "training_states").glob("*.pt"))) != 1
            or sha256_file(destination / rows[-1]["training_state_path"]) != rows[-1]["training_state_sha256"]
            or (destination / "continuation_inputs").exists()):
        raise RuntimeError("Second continuation violated final-state-only retention")
    aggregate_workers(extended_root, seeds=(0,), total_hours=72)
    report = {"status": "passed", "smoke": True,
              "source_checkpoint_count": 8, "final_checkpoint_count": 12,
              "resumed_from_full_state": True, "source_state_unchanged": True,
              "imported_policies_unchanged": True, "new_full_states": 1,
              "scope": "small 48-to-72 checkpoint-prefix test; not a production-state restore"}
    (output_root / "second_continuation_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(verify_second_continuation(Path(sys.argv[1])), indent=2))
