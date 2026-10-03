"""Read-only, checksummed Experiment 9 input staging; no learner is restored."""
import gc
import json
from pathlib import Path
import subprocess

import torch

from experiments.fhp.exp4_fhp_average_policy_audit.data import relative_path
from fhp_escher.checkpointing import load_checkpoint_payload, sha256_file
from fhp_escher.features import ENCODER_ID, encoder_from_metadata
from .config import SOURCE_ALGORITHM, SOURCE_EXPERIMENT, SOURCE_STATE_TYPE, source_worker_name


def source_row(worker):
    worker = Path(worker)
    manifest = json.loads((worker / "run_manifest.json").read_text())
    if (manifest.get("experiment_id") != 9 or manifest.get("experiment_name") != SOURCE_EXPERIMENT
            or manifest.get("algorithm_id") != SOURCE_ALGORITHM
            or manifest.get("training_duration_seconds") != 86400):
        raise ValueError("Requires the completed original 24-hour Experiment 9 source")
    if not (worker / "SUCCESS.json").is_file():
        raise ValueError("Source worker is incomplete")
    rows = json.loads((worker / "checkpoint_manifest.json").read_text())
    selected = [r for r in rows if r["checkpoint_id"] == "time_24h"]
    if (len(selected) != 1 or selected[0]["checkpoint_target_seconds"] != 86400
            or not selected[0].get("training_state_path")
            or not selected[0].get("training_state_sha256")):
        raise ValueError("Requires one 24-hour endpoint with retained full replay state")
    return manifest, selected[0]


def load_source(worker, seed, *, smoke=False):
    manifest, row = source_row(worker)
    if manifest["seed"] != seed:
        raise ValueError("Source seed mismatch")
    policy_path = relative_path(worker, row["path"])
    state_path = relative_path(worker, row["training_state_path"])
    if not state_path.is_file() or state_path.suffix != ".pt":
        raise ValueError("Expected Experiment 9's monolithic final .pt state")
    if sha256_file(policy_path) != row["sha256"] or sha256_file(state_path) != row["training_state_sha256"]:
        raise ValueError("Source checksum mismatch")
    template = load_checkpoint_payload(policy_path)
    synthetic = bool(template.get("audit_synthetic_fixture"))
    if synthetic and not smoke:
        raise ValueError("Synthetic source is permitted only for smoke")
    if manifest.get("smoke") and not synthetic:
        raise ValueError("Training smoke is not a production replay source")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (state.get("type") != SOURCE_STATE_TYPE or state.get("schema_version") != 1
            or state.get("seed") != seed or state.get("checkpoint_id") != "time_24h"):
        raise ValueError("Unsupported Experiment 9 state schema/seed/endpoint")
    if (template["experiment_id"] != 9 or template["algorithm_id"] != SOURCE_ALGORITHM
            or template["seed"] != seed or template["outer_iteration"] != row["outer_iteration"]
            or state["solver"]["num_iteration"] != row["outer_iteration"]):
        raise ValueError("Policy/replay provenance disagreement")
    encoder_from_metadata(template["feature_encoder"])
    model = template["policy_model"]
    if (template["feature_encoder"]["id"] != ENCODER_ID
            or model["type"] != "structured_fhp_mlp_v1"
            or model["hidden_layers"] != [192, 192] or model["branch_width"] != 64):
        raise ValueError("Unexpected source policy architecture or encoder")
    expected = {"gamma": 2.0, "average_policy_learning_rate": .003,
                "average_policy_train_steps": 20000, "average_policy_reset_each_fit": True,
                "ave_policy_batch_size": 2048, "structured_branch_width": 64,
                "ave_policy_buffer_size": 1_000_000,
                "average_policy_loss": "grouped_soft_target_cross_entropy",
                "cache_frozen_critic_targets": True}
    if not synthetic:
        for key, value in expected.items():
            if state["config"].get(key) != value or manifest["training_config"].get(key) != value:
                raise ValueError(f"Source fitting contract differs: {key}")
        if any(list(cfg.get("average_policy_network_layers", [])) != [192, 192]
               for cfg in (state["config"], manifest["training_config"])):
            raise ValueError("Source fitting contract differs: policy network layers")
        if (state["solver"]["training_elapsed_seconds"] < 86400
                or state.get("repository_commit") != manifest.get("repository_commit")):
            raise ValueError("Source training duration/commit mismatch")
    saved_model = state["average_policy_trainer"]["model"]
    if (set(saved_model) != set(template["policy_state_dict"])
            or any(not torch.equal(value, saved_model[name])
                   for name, value in template["policy_state_dict"].items())):
        raise ValueError("Saved policy and replay-state model disagree")
    buffer = state["average_policy_trainer"]["buffer"]
    if not synthetic and int(buffer["size"]) != 1_000_000:
        raise ValueError("Requires 1,000,000 replay rows per source")
    iteration = int(state["solver"]["num_iteration"])
    del state, saved_model
    gc.collect()  # Release all critic/calibration buffers before grouping policy replay.
    return buffer, iteration, template, {
        "seed": seed, "source_policy_sha256": row["sha256"],
        "source_state_sha256": row["training_state_sha256"],
        "source_commit": manifest.get("repository_commit"),
        "source_experiment": SOURCE_EXPERIMENT, "checkpoint_id": "time_24h",
        "outer_iteration": iteration, "nodes_touched": row["nodes_touched"],
        "synthetic": synthetic,
    }


def fetch_source(bucket, run_id, seed, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    remote = f"{bucket.rstrip('/')}/{run_id}/workers/{source_worker_name(seed)}"
    for name in ("run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"):
        subprocess.run(["gcloud", "storage", "cp", f"{remote}/{name}", str(directory / name)], check=True)
    manifest, row = source_row(directory)
    if manifest["seed"] != seed:
        raise ValueError("Source seed mismatch before replay download")
    for key in ("path", "training_state_path"):
        path = relative_path(directory, row[key])
        if key == "training_state_path" and path.suffix != ".pt":
            raise ValueError("Unexpected source state layout")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", f"{remote}/{row[key]}", str(path)], check=True)
    return directory
