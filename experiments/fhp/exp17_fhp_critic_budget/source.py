"""Read-only source acquisition, pinned state checks and runtime validation."""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

from experiments.fhp.exp16_fhp_hand_board_48h.contract import (
    METADATA_FILES, validate_metadata, safe_path,
)
from .config import SOURCE_RUN_ID, TRAINING_REF, SEEDS, RUNTIME, worker_name


def checked_bucket(bucket):
    bucket = bucket.rstrip("/")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", bucket):
        raise ValueError("Expected a gs:// bucket, without a path")
    return bucket


def preflight(bucket):
    bucket = checked_bucket(bucket)
    records = []
    for seed in SEEDS:
        uri = f"{bucket}/{SOURCE_RUN_ID}/workers/{worker_name(seed)}"
        raw = {name: subprocess.check_output(["gcloud", "storage", "cat", f"{uri}/{name}"])
               for name in METADATA_FILES}
        data = validate_metadata(raw, seed=seed)
        final = data["checkpoint_manifest.json"][-1]
        state = json.loads(subprocess.check_output([
            "gcloud", "storage", "objects", "describe",
            uri + "/" + safe_path(final["training_state_path"], "training_states"), "--format=json"]))
        if int(state.get("size", 0)) <= 0:
            raise ValueError("Missing/empty source training state")
        records.append(dict(seed=seed, uri=uri, state_size=int(state["size"]),
                            state_sha256=final["training_state_sha256"]))
    return records


def fetch_source(bucket, seed, directory):
    if seed not in SEEDS:
        raise ValueError("Unexpected seed")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    uri = f"{checked_bucket(bucket)}/{SOURCE_RUN_ID}/workers/{worker_name(seed)}"
    for name in METADATA_FILES:
        subprocess.run(["gcloud", "storage", "cp", f"{uri}/{name}", str(directory / name)], check=True)
    data = validate_metadata({p: (directory / p).read_bytes() for p in METADATA_FILES}, seed=seed)
    final = data["checkpoint_manifest.json"][-1]
    relative = safe_path(final["training_state_path"], "training_states")
    (directory / relative).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", f"{uri}/{relative}", str(directory / relative)], check=True)
    # Verify immediately, and again before unpickling in the worker.
    return validate_local(directory, seed)


def validate_local(directory, seed):
    from fhp_escher.checkpointing import sha256_file
    directory = Path(directory).resolve()
    data = validate_metadata({p: (directory / p).read_bytes() for p in METADATA_FILES}, seed=seed)
    final = data["checkpoint_manifest.json"][-1]
    state = (directory / safe_path(final["training_state_path"], "training_states")).resolve()
    if not state.is_relative_to(directory) or sha256_file(state) != final["training_state_sha256"]:
        raise ValueError("Corrupt or unsafe source state")
    return state, dict(seed=seed, source_run_id=SOURCE_RUN_ID, source_commit=TRAINING_REF,
                       state_sha256=final["training_state_sha256"],
                       metadata_sha256={p: hashlib.sha256((directory / p).read_bytes()).hexdigest()
                                        for p in METADATA_FILES})


def validate_runtime():
    import numpy as np
    import ray
    import torch
    actual = dict(python_version=platform.python_version(), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, ray_version=ray.__version__)
    if any(actual[k] != RUNTIME[k] for k in actual):
        raise ValueError(f"Production audit requires pinned source runtime {RUNTIME}; got {actual}")
    return actual


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Metadata-only preflight; no local ML dependencies required")
    parser.add_argument("--bucket", required=True)
    print(json.dumps(preflight(parser.parse_args().bucket), indent=2))
