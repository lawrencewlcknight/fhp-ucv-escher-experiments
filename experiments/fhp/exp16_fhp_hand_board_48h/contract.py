"""Dependency-free continuation contract and cloud metadata preflight.

The learner keeps Experiment 10's identity and exact source commit. Experiment
16 identifies the continuation workflow, not a different algorithm or state format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

EXPERIMENT_ID = 16
EXPERIMENT_NAME = "exp16_fhp_hand_board_48h"
SOURCE_RUN_ID = "exp10-features-20261001-161740"
EXP9_RUN_ID = "exp9-cache24-20261001-132550"
EXP8_RUN_ID = "exp8-par48-20261001-005208"
TRAINING_REF = "e66d4da515eb212e5026a965ac5c39c86144c901"
CONFIG_SHA256 = "065734572e18f4597dd8f494ea39230c1f193b9d68aa775138c4a8a3e39eddad"
ALGORITHM_ID = "hand_board_cached_parallel_ucv_escher"
SEEDS = (0, 1, 2)
TOTAL_HOURS = 48
HOURS = tuple(range(6, 49, 6))
EVALUATED_HOURS = (24, 30, 36, 42, 48)
RUNTIME = {"python_version": "3.11.16", "torch_version": "2.7.0+cpu",
           "numpy_version": "1.26.4", "ray_version": "2.51.2",
           "torch_intraop_threads": 8, "torch_interop_threads": 8,
           "traversal_execution": "ray_parallel", "frozen_critic_target_cache": True}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def worker_name(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Expected a production seed 0, 1 or 2")
    return f"task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"


def safe_path(value, directory):
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != directory or ".." in path.parts:
        raise ValueError(f"Unsafe {directory} artifact path: {value}")
    return str(path)


def validate_metadata(metadata, *, seed, hours=24):
    """Validate JSON bytes without reading large training-state payloads."""
    data = {name: json.loads(raw) for name, raw in metadata.items()}
    manifest, runtime = data["run_manifest.json"], data["runtime_manifest.json"]
    summary, success, rows = data["summary.json"], data["SUCCESS.json"], data["checkpoint_manifest.json"]
    expected = {"experiment_id": 10, "experiment_name": "exp10_fhp_hand_board_features",
                "algorithm_id": ALGORITHM_ID, "seed": seed, "smoke": False,
                "repository_commit": TRAINING_REF, "execution_backend": "ray_parallel",
                "training_duration_seconds": hours * 3600, "training_config_sha256": CONFIG_SHA256}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Incompatible Experiment 10 manifest for seed {seed}")
    if digest(manifest["training_config"]) != CONFIG_SHA256:
        raise ValueError("Learning configuration differs from the completed Experiment 10 run")
    if any(runtime.get(k) != v for k, v in RUNTIME.items()):
        raise ValueError(f"Pinned source runtime mismatch for seed {seed}")
    if runtime.get("reference_vm", {}).get("machine_type") != "n2-standard-16":
        raise ValueError("Continuation requires the original n2-standard-16 machine class")
    elapsed = summary.get("final_training_elapsed_seconds", float("nan"))
    if (summary.get("status") != "complete" or success.get("status") != "complete"
            or summary.get("seed") != seed or summary.get("checkpoint_count") != hours // 6
            or not math.isfinite(elapsed) or elapsed < hours * 3600
            or (hours == 24 and elapsed >= TOTAL_HOURS * 3600)):
        raise ValueError("Incomplete source training or incompatible elapsed training budget")
    if hashlib.sha256(metadata["summary.json"]).hexdigest() != success.get("summary_sha256"):
        raise ValueError("Summary checksum mismatch")
    expected_ids = [f"time_{h:02d}h" for h in range(6, hours + 1, 6)]
    if ([r["checkpoint_id"] for r in rows] != expected_ids
            or [r["checkpoint_id"] for r in manifest["checkpoint_schedule"]] != expected_ids
            or [r["target_training_seconds"] for r in manifest["checkpoint_schedule"]]
               != [h * 3600 for h in range(6, hours + 1, 6)]):
        raise ValueError("Incomplete or reordered six-hour checkpoint schedule")
    for index, row in enumerate(rows):
        safe_path(row["path"], "checkpoints")
        if not re.fullmatch("[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Missing policy checksum")
        if row.get("checkpoint_target_seconds") != (index + 1) * 6 * 3600:
            raise ValueError("Invalid checkpoint target")
        if index < len(rows) - 1 and any(k.startswith("training_state_") for k in row):
            raise ValueError("Only the final checkpoint may advertise a full training state")
    safe_path(rows[-1]["training_state_path"], "training_states")
    if not re.fullmatch("[0-9a-f]{64}", rows[-1]["training_state_sha256"]):
        raise ValueError("Missing final full-state checksum")
    return data


METADATA_FILES = ("run_manifest.json", "runtime_manifest.json", "checkpoint_manifest.json",
                  "summary.json", "SUCCESS.json")


def cloud_preflight(bucket, *, cloud=None):
    """Read small manifests and object metadata; never download .pt states."""
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", bucket):
        raise ValueError("Expected a gs:// bucket without a path")
    if cloud is None:
        def cloud(*args):
            return subprocess.check_output(["gcloud", "storage", *args])
    sources = []
    for seed in SEEDS:
        uri = f"{bucket}/{SOURCE_RUN_ID}/workers/{worker_name(seed)}"
        raw = {name: cloud("cat", uri + "/" + name) for name in METADATA_FILES}
        data = validate_metadata(raw, seed=seed)
        final = data["checkpoint_manifest.json"][-1]
        state = json.loads(cloud("objects", "describe", uri + "/" + final["training_state_path"], "--format=json"))
        if int(state.get("size", 0)) <= 0:
            raise ValueError(f"Missing/empty resumable state: {uri}")
        sources.append({"seed": seed, "source_worker": uri,
                        "metadata_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
                        "state_path": final["training_state_path"], "state_sha256": final["training_state_sha256"],
                        "state_size_bytes": int(state["size"]), "state_generation": state.get("generation"),
                        "final_training_elapsed_seconds": data["summary.json"]["final_training_elapsed_seconds"]})
    # Also fail before training if the prespecified learned opponent panel is absent.
    panel = []
    for run, algorithm, horizon in (
        (EXP9_RUN_ID, "cached_parallel_structured_ucv_escher", 24),
        (EXP8_RUN_ID, "parallel_structured_ucv_escher_48h", 48),
    ):
        # Worker directory names are discovered from manifests, not guessed.
        paths = cloud("ls", f"{bucket}/{run}/workers/*/run_manifest.json").decode().splitlines()
        seen = set()
        for path in paths:
            manifest = json.loads(cloud("cat", path))
            if manifest.get("algorithm_id") != algorithm or manifest.get("training_duration_seconds") != horizon * 3600:
                raise ValueError(f"Wrong fixed-panel algorithm or horizon: {run}")
            seed = manifest.get("seed")
            if type(seed) is not int or seed not in SEEDS or seed in seen or manifest.get("smoke") is not False:
                raise ValueError(f"Invalid fixed-panel seed: {run}")
            seen.add(seed)
            worker = path.rsplit("/", 1)[0]
            success = json.loads(cloud("cat", worker + "/SUCCESS.json"))
            rows = json.loads(cloud("cat", worker + "/checkpoint_manifest.json"))
            if success.get("status") != "complete" or rows[-1]["checkpoint_target_seconds"] != horizon * 3600:
                raise ValueError(f"Incomplete fixed opponent panel: {run}")
            cloud("objects", "describe", worker + "/" + safe_path(rows[-1]["path"], "checkpoints"), "--format=json")
            panel.append({"run_id": run, "seed": seed, "hours": horizon,
                          "policy_path": rows[-1]["path"], "policy_sha256": rows[-1]["sha256"]})
        if seen != set(SEEDS):
            raise ValueError(f"Incomplete fixed-panel seeds: {run}")
    return {"experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
            "training_ref": TRAINING_REF, "runtime": RUNTIME, "source_run_id": SOURCE_RUN_ID,
            "total_active_hours": TOTAL_HOURS, "checkpoint_hours": HOURS,
            "source_states": sources, "full_state_retention": "final_only",
            "fixed_panel_runs": {"exp9": EXP9_RUN_ID, "exp8": EXP8_RUN_ID},
            "fixed_panel_policies": panel,
            "payload_hash_verification": "performed on training VM before restore"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = cloud_preflight(args.bucket.rstrip("/"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified three source state objects and fixed opponent panel; contract: {args.output}")


if __name__ == "__main__":
    main()
