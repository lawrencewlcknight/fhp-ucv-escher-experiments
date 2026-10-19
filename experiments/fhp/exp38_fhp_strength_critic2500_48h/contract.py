"""Dependency-free continuation identity and cheap cloud source validation.

Experiment 38 identifies the workflow. Learners retain the native Experiment 36
identity and source code revision: no renaming/migration of saved model states.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

EXPERIMENT_ID = 38
EXPERIMENT_NAME = "exp38_fhp_strength_critic2500_48h"
SOURCE_RUN_ID = "exp36-rich2500-20261018-205125"
TRAINING_REF = "6ab85bbcd03627281e411243419d64ce9e9236d5"
CONFIG_SHA256 = "c9dbd0be75986bda4e34c060da47b937f4feb657d9888ccdee6a506cfc1854ed"
ALGORITHM_ID = "strategic_ae_critic2500_ucv_escher"
LEARNER_NAME = "exp36_fhp_strength_critic2500"
SEEDS = (0, 1, 2)
SOURCE_HOURS, TOTAL_HOURS = 24, 48
HOURS = tuple(range(6, 49, 6))
NEW_HOURS = (30, 36, 42, 48)
RUNTIME = {"python_version": "3.11.16", "torch_version": "2.7.0+cpu", "numpy_version": "1.26.4",
           "ray_version": "2.51.2", "torch_intraop_threads": 8, "torch_interop_threads": 8,
           "traversal_execution": "ray_parallel", "frozen_critic_target_cache": True}
PARALLEL_SETTINGS = {"parallel_num_workers": 8, "parallel_collection_chunk_size": 1200,
    "parallel_ray_object_store_memory": 4294967296, "parallelize_independent_learners": False,
    "parallel_cache_actor_snapshots": True}
METADATA_FILES = ("run_manifest.json", "runtime_manifest.json", "checkpoint_manifest.json", "summary.json", "SUCCESS.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def worker_name(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected a production seed 0, 1 or 2")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"


def safe_path(value, directory):
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != directory or ".." in path.parts:
        raise ValueError(f"Unsafe {directory} artifact path")
    return str(path)


def validate_metadata(raw, *, seed, hours=SOURCE_HOURS):
    if hours not in (SOURCE_HOURS, TOTAL_HOURS):
        raise ValueError("Only original 24h sources and completed 48h outputs are valid")
    worker_name(seed)
    data = {k: json.loads(v) for k, v in raw.items()}
    manifest, runtime = data["run_manifest.json"], data["runtime_manifest.json"]
    summary, success, rows = data["summary.json"], data["SUCCESS.json"], data["checkpoint_manifest.json"]
    expected = dict(experiment_id=36, experiment_name=LEARNER_NAME, algorithm_id=ALGORITHM_ID,
        seed=seed, smoke=False, repository_commit=TRAINING_REF, execution_backend="ray_parallel",
        training_duration_seconds=hours*3600, training_config_sha256=CONFIG_SHA256, training_state_retention="final")
    if any(manifest.get(k) != v for k, v in expected.items()) or digest(manifest["training_config"]) != CONFIG_SHA256:
        raise ValueError(f"Incompatible pinned Experiment 36 learning contract, seed {seed}")
    if any(runtime.get(k) != v for k, v in RUNTIME.items()):
        raise ValueError(f"Pinned runtime mismatch, seed {seed}")
    if runtime.get("parallel_settings") != PARALLEL_SETTINGS:
        raise ValueError("Pinned parallel settings mismatch")
    for record in (manifest, runtime):
        if record.get("reference_vm", {}).get("machine_type") != "n2-standard-16":
            raise ValueError("Require original n2-standard-16 machine class")
        if record.get("feature_encoder", {}).get("id") != "fhp_hand_board_strategic_ae_v1":
            raise ValueError("Wrong richer input encoder")
    elapsed = summary.get("final_training_elapsed_seconds", float("nan"))
    if (summary.get("status") != "complete" or success.get("status") != "complete"
            or summary.get("seed") != seed or summary.get("checkpoint_count") != hours//6
            or not math.isfinite(elapsed) or elapsed < hours*3600
            or (hours == SOURCE_HOURS and elapsed >= TOTAL_HOURS*3600)):
        raise ValueError("Incomplete source training or incompatible training clock")
    if summary.get("execution_diagnostics", {}).get("critic_train_steps_per_fit") != [2500, 2500]:
        raise ValueError("Actual critic budgets must both remain 2,500")
    if any(type(summary.get(k)) is not int or summary[k] <= 0 for k in ("final_nodes_touched", "final_iteration")):
        raise ValueError("Source progress counters are missing or invalid")
    if hashlib.sha256(raw["summary.json"]).hexdigest() != success.get("summary_sha256"):
        raise ValueError("Summary checksum mismatch")
    targets = list(range(6, hours+1, 6))
    expected_ids = [f"time_{h:02d}h" for h in targets]
    if ([r["checkpoint_id"] for r in rows] != expected_ids
            or [r["checkpoint_id"] for r in manifest["checkpoint_schedule"]] != expected_ids
            or [r["target_training_seconds"] for r in manifest["checkpoint_schedule"]] != [h*3600 for h in targets]):
        raise ValueError("Incomplete or reordered six-hour checkpoint schedule")
    for h, row in zip(targets, rows):
        safe_path(row["path"], "checkpoints")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or row["checkpoint_target_seconds"] != h*3600:
            raise ValueError("Invalid policy checksum or checkpoint target")
        if h != hours and any(k.startswith("training_state_") for k in row):
            raise ValueError("Only final checkpoint may advertise a full state")
    safe_path(rows[-1]["training_state_path"], "training_states")
    if not re.fullmatch(r"[0-9a-f]{64}", rows[-1]["training_state_sha256"]):
        raise ValueError("Missing full-state checksum")
    if hours == TOTAL_HOURS and summary.get("resumed_from_training_state") is not True:
        raise ValueError("The 48h result must continue an existing full state")
    return data


def cloud_preflight(bucket, *, cloud=None):
    """Read JSON and object metadata, not multi-gigabyte training payloads."""
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", bucket):
        raise ValueError("Expected a gs:// bucket without a path")
    if cloud is None:
        def cloud(*args):
            return subprocess.check_output(["gcloud", "storage", *args])
    sources = []
    for seed in SEEDS:
        uri = f"{bucket}/{SOURCE_RUN_ID}/workers/{worker_name(seed)}"
        raw = {name: cloud("cat", uri+"/"+name) for name in METADATA_FILES}
        data = validate_metadata(raw, seed=seed)
        final = data["checkpoint_manifest.json"][-1]
        obj = json.loads(cloud("objects", "describe", uri+"/"+final["training_state_path"], "--format=json"))
        if int(obj.get("size", 0)) <= 0:
            raise ValueError("Missing/empty full restart state")
        if int(obj["size"]) != final.get("training_state_size_bytes"):
            raise ValueError("Restart object size differs from the saved checkpoint manifest")
        sources.append(dict(seed=seed, source_worker=uri,
            metadata_sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
            state_path=final["training_state_path"], state_sha256=final["training_state_sha256"],
            source_checkpoints=[{k: r[k] for k in ("checkpoint_id", "path", "sha256")} for r in data["checkpoint_manifest.json"]],
            state_size_bytes=int(obj["size"]), state_generation=obj.get("generation"),
            source_nodes=data["summary.json"].get("final_nodes_touched"),
            source_iterations=data["summary.json"].get("final_iteration"),
            source_active_seconds=data["summary.json"]["final_training_elapsed_seconds"],
            remaining_active_seconds=TOTAL_HOURS*3600-data["summary.json"]["final_training_elapsed_seconds"]))
    return dict(experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        learner_experiment_id=36, training_ref=TRAINING_REF, config_sha256=CONFIG_SHA256,
        source_run_id=SOURCE_RUN_ID, source_active_hours=SOURCE_HOURS, total_active_hours=TOTAL_HOURS,
        additional_active_hours_nominal=24, checkpoint_hours=HOURS, new_checkpoint_hours=NEW_HOURS,
        seeds=SEEDS, source_states=sources, runtime=RUNTIME, parallel_settings=PARALLEL_SETTINGS,
        automatic_evaluation=False, full_state_retention="final_only",
        state_integrity="Payload hashes and native restore validation run on the training VM before learning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = cloud_preflight(args.bucket.rstrip("/"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Verified all three 24h restart-state objects; source receipt: {args.output}")


if __name__ == "__main__":
    main()
