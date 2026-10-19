"""Add workflow identity and restart inventory after native aggregation.

This standard-library script is embedded into the pinned learner job; it changes
reporting only. Native manifests and full states retain Experiment 36 identity.
"""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def finish(root, workflow_ref, run_uri, *, cloud=None):
    root = Path(root)
    if cloud is None:
        def cloud(*args):
            return subprocess.check_output(["gcloud", "storage", *args])
    source = json.loads((root/"source_contract.json").read_text())
    analysis = root/"analysis"
    # Never advertise workflow completion before the restart objects are checked.
    (analysis/"SUCCESS.json").unlink(missing_ok=True)
    if (source["experiment_id"] != 38 or source["total_active_hours"] != 48
            or source["source_active_hours"] != 24 or source["seeds"] != [0, 1, 2]):
        raise ValueError("Unexpected Experiment 38 source receipt")
    extensions, restarts = [], []
    for row in source["source_states"]:
        seed = row["seed"]
        worker_name = row["source_worker"].rsplit("/", 1)[1]
        worker = root/"workers"/worker_name
        manifest = json.loads((worker/"run_manifest.json").read_text())
        summary = json.loads((worker/"summary.json").read_text())
        success = json.loads((worker/"SUCCESS.json").read_text())
        lineage = json.loads((worker/"continuation_source.json").read_text())
        checkpoints = json.loads((worker/"checkpoint_manifest.json").read_text())
        if (manifest["repository_commit"] != source["training_ref"] or manifest["experiment_id"] != 36
                or manifest["training_config_sha256"] != source["config_sha256"]
                or summary["status"] != "complete" or success["status"] != "complete"
                or summary["seed"] != seed or not summary["resumed_from_training_state"]
                or summary["checkpoint_count"] != 8 or summary["final_training_elapsed_seconds"] < 48*3600
                or summary["execution_diagnostics"]["critic_train_steps_per_fit"] != [2500, 2500]
                or hashlib.sha256((worker/"summary.json").read_bytes()).hexdigest() != success["summary_sha256"]
                or lineage["source_state_sha256"] != row["state_sha256"]
                or lineage["source_worker"] != row["source_worker"]
                or lineage["source_commit"] != source["training_ref"]
                or lineage["source_summary_sha256"] != row["metadata_sha256"]["summary.json"]
                or lineage["source_total_hours"] != 24 or lineage["total_hours"] != 48
                or [r["checkpoint_id"] for r in checkpoints] != [f"time_{h:02d}h" for h in range(6, 49, 6)]
                or any("training_state_path" in r for r in checkpoints[:-1])):
            raise ValueError(f"Incomplete or incompatible continuation output for seed {seed}")
        if [{k: r[k] for k in ("checkpoint_id", "path", "sha256")} for r in checkpoints[:4]] != row["source_checkpoints"]:
            raise ValueError("Original 24h checkpoint prefix changed during continuation")
        final = checkpoints[-1]
        state_path = PurePosixPath(final["training_state_path"])
        if state_path.is_absolute() or len(state_path.parts) != 2 or state_path.parts[0] != "training_states" or ".." in state_path.parts:
            raise ValueError("Invalid final state path")
        state_uri = f"{run_uri}/workers/{worker_name}/{state_path}"
        obj = json.loads(cloud("objects", "describe", state_uri, "--format=json"))
        if int(obj.get("size", 0)) <= 0 or int(obj["size"]) != final["training_state_size_bytes"]:
            raise ValueError(f"Missing/incomplete uploaded 48h state for seed {seed}")
        restarts.append(dict(seed=seed, source_worker=f"{run_uri}/workers/{worker_name}",
            training_state_uri=state_uri, training_state_sha256=final["training_state_sha256"],
            training_state_size_bytes=int(obj["size"]), object_generation=obj.get("generation"),
            cumulative_active_seconds=summary["final_training_elapsed_seconds"], training_ref=source["training_ref"],
            full_state=True, reload_for_training=True))
        extensions.append(dict(seed=seed, source_active_seconds=row["source_active_seconds"],
            final_active_seconds=summary["final_training_elapsed_seconds"],
            added_active_seconds=summary["final_training_elapsed_seconds"]-row["source_active_seconds"],
            source_nodes=row["source_nodes"], final_nodes=summary["final_nodes_touched"],
            added_nodes=summary["final_nodes_touched"]-row["source_nodes"],
            source_iterations=row["source_iterations"], final_iterations=summary["final_iteration"],
            added_iterations=summary["final_iteration"]-row["source_iterations"]))
    for file in ("experiment_manifest.json", "throughput_summary.json"):
        target = analysis/file
        data = json.loads(target.read_text())
        native = analysis/("learner_"+file)
        if data["experiment_id"] == 36:
            write_json(native, data)
        elif data["experiment_id"] != 38 or not native.is_file():
            raise ValueError("Unexpected native aggregation identity")
        data.update(experiment_id=38, experiment_name=source["experiment_name"], learner_experiment_id=36,
            workflow_ref=workflow_ref, source_run_id=source["source_run_id"], training_ref=source["training_ref"],
            total_active_hours=48, source_active_hours=24, automatic_evaluation=False)
        if file == "experiment_manifest.json":
            data["contract"] = source
        write_json(target, data)
    write_json(analysis/"experiment38_contract.json", dict(source, workflow_ref=workflow_ref))
    write_json(analysis/"restart_inventory.json", dict(experiment_id=38, total_active_hours=48, seeds=restarts,
        instructions="Use the native Experiment 36 stage-source and worker commands at training_ref in a NEW run directory, with a horizon greater than 48h. Do not load only a .pkl policy. No further training is started automatically."))
    with (analysis/"continuation_progress.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(extensions[0]))
        writer.writeheader()
        writer.writerows(extensions)
    lines = ["# Experiment 38 — Experiment 36 continued to 48 active hours", "",
        "All three original seeds continued from complete 24-hour states. Algorithm, richer features, critic budget (2,500), eight Ray actors and learner code are unchanged. No evaluation was run.",
        "", "| Seed | Additional active hours | Added million nodes | Added iterations |", "|---|---:|---:|---:|"]
    for row in extensions:
        lines.append(f"| {row['seed']} | {row['added_active_seconds']/3600:.3f} | {row['added_nodes']/1e6:.3f} | {row['added_iterations']} |")
    lines += ["", "The original 6/12/18/24h policies are retained; new policies are at 30/36/42/48h. The full 48h state for every seed is listed in restart_inventory.json. Native worker and checkpoint identifiers deliberately remain Experiment 36, while this workflow and its analysis are Experiment 38.",
        "", "Active time is cumulative and stops at the first completed iteration after 48h. Policy extraction, saving, uploads and VM setup are additional wall time. Throughput is not policy strength or convergence evidence."]
    (analysis/"analysis_summary.md").write_text("\n".join(lines)+"\n")
    write_json(analysis/"SUCCESS.json", dict(status="complete", experiment_id=38, full_restart_states=3, automatic_evaluation=False))
    return restarts


if __name__ == "__main__":
    finish(sys.argv[1], sys.argv[2], sys.argv[3])
