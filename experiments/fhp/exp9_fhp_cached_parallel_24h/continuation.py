"""Import a completed Experiment 9 endpoint into a separate extension run."""
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import json
import shutil
import subprocess

from fhp_escher.checkpointing import sha256_file
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _repository_commit, _config_sha256
from .config import (
    EXPERIMENT_ID, EXPERIMENT_NAME, ALGORITHM_ID, EXPERIMENT_CONFIG,
    REFERENCE_VM, checkpoint_schedule, smoke_config, validate_total_hours,
)


def _safe_relative(path):
    part = PurePosixPath(path)
    if part.is_absolute() or ".." in part.parts or len(part.parts) != 2:
        raise ValueError("Unsafe source checkpoint path")
    if part.parts[0] not in ("checkpoints", "training_states"):
        raise ValueError("Unexpected source checkpoint directory")
    return str(part)


def stage_source(source: str, destination: Path, *, total_hours: int, seed: int,
                 smoke=False, source_commit=None):
    """Copy only small policy artifacts and the final full state, never train."""
    total_hours = validate_total_hours(total_hours)
    destination = Path(destination).resolve()
    source = source.rstrip("/")
    if not source.startswith("gs://"):
        source = str(Path(source).resolve())
        if Path(source) == destination or Path(source) in destination.parents:
            raise ValueError("Extension must use a separate output directory")
    identity = {"source_worker": source, "total_hours": total_hours, "seed": seed}
    marker = destination / "continuation_source.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if any(prior.get(k) != v for k, v in identity.items()):
            raise ValueError("Destination already belongs to another continuation")
        return prior
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Extension destination is not empty")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def fetch(relative, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.startswith("gs://"):
            subprocess.run(["gcloud", "storage", "cp", source + "/" + relative, str(target)], check=True)
        else:
            shutil.copy2(Path(source) / relative, target)

    # Do not expose a partially imported run if a transfer fails.
    with TemporaryDirectory(prefix="exp9-import-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "worker"
        stage.mkdir()
        metadata = {}
        for name in ("SUCCESS.json", "summary.json", "run_manifest.json",
                     "checkpoint_manifest.json", "runtime_manifest.json"):
            fetch(name, stage / name)
            metadata[name] = json.loads((stage / name).read_text())
        summary, manifest = metadata["summary.json"], metadata["run_manifest.json"]
        rows = metadata["checkpoint_manifest.json"]
        success = metadata["SUCCESS.json"]
        config = smoke_config() if smoke else EXPERIMENT_CONFIG
        if (manifest["experiment_id"] != EXPERIMENT_ID
                or manifest["experiment_name"] != EXPERIMENT_NAME
                or manifest["algorithm_id"] != ALGORITHM_ID
                or manifest["execution_backend"] != "ray_parallel"
                or manifest["reference_vm"] != REFERENCE_VM
                or manifest["training_config_sha256"] != _config_sha256(config)
                or manifest["repository_commit"] != (source_commit or _repository_commit())
                or manifest["smoke"] != smoke
                or summary["seed"] != seed or manifest["seed"] != seed
                or summary["status"] != "complete" or success["status"] != "complete"
                or sha256_file(stage / "summary.json") != success["summary_sha256"]
                or summary["checkpoint_count"] != len(rows)):
            raise ValueError("Source is not a compatible completed Experiment 9 run")
        # Each completed six-hour scheduled checkpoint is retained on extension.
        source_hours = len(rows) * 6
        old_schedule = checkpoint_schedule(smoke=smoke, total_hours=source_hours)
        if manifest["checkpoint_schedule"] != list(old_schedule):
            raise ValueError("Source schedule is not the complete Experiment 9 prefix")
        if not source_hours < total_hours <= source_hours + 48:
            raise ValueError("An extension must add 6 to 48 active hours")
        if not smoke and summary["final_training_elapsed_seconds"] >= total_hours * 3600:
            raise ValueError("New horizon must exceed actual source training time")
        if [r["checkpoint_id"] for r in rows] != [r["checkpoint_id"] for r in old_schedule]:
            raise ValueError("Source checkpoint manifest is incomplete or reordered")
        for row in rows:
            relative = _safe_relative(row["path"])
            fetch(relative, stage / relative)
            if sha256_file(stage / relative) != row["sha256"]:
                raise ValueError("Source policy checksum mismatch")
        final = rows[-1]
        relative = _safe_relative(final["training_state_path"])
        # An imported state is a temporary input. Never duplicate it into the
        # extension's training_states outputs or upload it at early checkpoints.
        local_state = "continuation_inputs/source_state.pt"
        fetch(relative, stage / local_state)
        if sha256_file(stage / local_state) != final["training_state_sha256"]:
            raise ValueError("Source full-training-state checksum mismatch")
        lineage = dict(identity, source_total_hours=source_hours,
                       source_commit=manifest["repository_commit"],
                       source_state_sha256=final["training_state_sha256"],
                       source_state_path=relative,
                       local_state_path=local_state,
                       source_summary_sha256=success["summary_sha256"])
        for row in rows:
            for key in tuple(row):
                if key.startswith("training_state_"):
                    row.pop(key)
        (stage / "checkpoint_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
        (stage / "continuation_source.json").write_text(json.dumps(lineage, indent=2) + "\n")
        # These describe completion of the source, NOT of the new extension.
        # The retained source checkpoint manifest/runtime enable exact restore.
        for name in ("SUCCESS.json", "summary.json", "run_manifest.json"):
            (stage / name).unlink()
        if destination.exists():
            destination.rmdir()  # validated empty destination only
        stage.rename(destination)
    return lineage
