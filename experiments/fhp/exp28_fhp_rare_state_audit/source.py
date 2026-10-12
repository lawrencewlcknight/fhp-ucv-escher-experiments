"""Reuse strict Exp10 metadata validation for its 24h and 48h continuation."""
import json
from pathlib import Path
import subprocess
import time

from experiments.fhp.exp16_fhp_hand_board_48h.contract import METADATA_FILES, validate_metadata, safe_path
from experiments.fhp.exp17_fhp_critic_budget.source import checked_bucket, validate_runtime
from .config import SOURCES, SEEDS, PRIOR_AUDIT, TRAINING_REF, worker_name
from .common import file_hash, write_json

OPPONENT = "exp9-cache24-20261001-132550/workers/task_000_cached_parallel_structured_ucv_escher_seed_0/checkpoints/cached_parallel_structured_ucv_escher_seed_0_time_24h.pkl"
OPPONENT_SHA256 = "9328bb5e0e59ecb4aa43dbbf5211098d0a3efe1371e4bfb0dc7684449648e449"


def fetch_opponent(bucket, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", f"{checked_bucket(bucket)}/{OPPONENT}", str(path)], check=True)
    if file_hash(path) != OPPONENT_SHA256:
        raise ValueError("Fixed opponent checksum differs")


def metadata(bucket, seed, hours):
    if seed not in SEEDS or hours not in SOURCES:
        raise ValueError("Invalid source seed/horizon")
    uri = f"{checked_bucket(bucket)}/{SOURCES[hours]}/workers/{worker_name(seed)}"
    raw = {n: subprocess.check_output(["gcloud", "storage", "cat", f"{uri}/{n}"]) for n in METADATA_FILES}
    return uri, raw, validate_metadata(raw, seed=seed, hours=hours)


def fetch(bucket, seed, hours, directory):
    start = time.perf_counter()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    uri, raw, data = metadata(bucket, seed, hours)
    for name, content in raw.items():
        (root / name).write_bytes(content)
    final = data["checkpoint_manifest.json"][-1]
    for field, folder in (("path", "checkpoints"), ("training_state_path", "training_states")):
        path = safe_path(final[field], folder)
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", f"{uri}/{path}", str(root / path)], check=True)
    result = validate(root, seed, hours)[2]
    write_json(root / "download.json", dict(seconds=time.perf_counter()-start, **result))
    return result


def validate(root, seed, hours):
    root = Path(root)
    data = validate_metadata({n:(root/n).read_bytes() for n in METADATA_FILES}, seed=seed, hours=hours)
    final = data["checkpoint_manifest.json"][-1]
    state = root / safe_path(final["training_state_path"], "training_states")
    policy = root / safe_path(final["path"], "checkpoints")
    if file_hash(state) != final["training_state_sha256"] or file_hash(policy) != final["sha256"]:
        raise ValueError("Source checksum mismatch before deserialisation")
    provenance = dict(seed=seed, hours=hours, run=SOURCES[hours], commit=TRAINING_REF,
        state_sha256=final["training_state_sha256"], policy_sha256=final["sha256"],
        state_bytes=state.stat().st_size, metadata_sha256={n:file_hash(root/n) for n in METADATA_FILES})
    if (root/"download.json").exists():
        provenance["source_transfer_and_verification_seconds"] = json.loads((root/"download.json").read_text())["seconds"]
    return state, policy, provenance


def load(root, seed, hours):
    import torch
    from fhp_escher.checkpointing import load_checkpoint_payload
    from experiments.fhp.exp10_fhp_hand_board_features.config import EXPERIMENT_CONFIG
    from experiments.fhp.exp10_fhp_hand_board_features.training_state import STATE_TYPE
    state_path, policy_path, provenance = validate(root, seed, hours)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    policy = load_checkpoint_payload(policy_path)
    if (state["type"] != STATE_TYPE or state["seed"] != seed
            or state["repository_commit"] != TRAINING_REF or state["config"] != EXPERIMENT_CONFIG
            or state["checkpoint_id"] != f"time_{hours:02d}h"
            or policy["seed"] != seed or policy["outer_iteration"] != state["solver"]["num_iteration"]
            or state["feature_encoder"] != policy["feature_encoder"]):
        raise ValueError("State/policy/source contract differs")
    if any(not torch.equal(t, state["average_policy_trainer"]["model"][n])
           for n,t in policy["policy_state_dict"].items()):
        raise ValueError("Playable policy differs from source training-state policy")
    return state, policy, provenance


def fetch_library(bucket, path):
    raw = subprocess.check_output(["gcloud", "storage", "cat",
        f"{checked_bucket(bucket)}/{PRIOR_AUDIT}/library.json"])
    data = json.loads(raw)
    import hashlib
    if hashlib.sha256(raw).hexdigest() != "fcbf4cd00ff2cfb24552c9dc423b4bb4673069b34d7a90ee9f01dfd227b3da5d":
        raise ValueError("Previously inspected strategic library changed")
    if data.get("seed") != 20261006 or len(data.get("boards", [])) != 96:
        raise ValueError("Unexpected previous strategic library")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)
    return data
