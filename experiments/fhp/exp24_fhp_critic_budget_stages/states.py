"""Phase-tagged pre-fit snapshots and complete end-of-iteration checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

from experiments.critic_target_cache_benchmark import write_json
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline import training_state as storage
from experiments.fhp.exp2_fhp_lossless_structured_ucv import training_state as serial_state
from experiments.fhp.exp10_fhp_hand_board_features import config as baseline, training_state as parallel_state
from experiments.fhp.exp22_fhp_critic_objectives.source import extract_verified_snapshot
from .config import STAGES

SNAPSHOT_TYPE = "exp24_pre_fit_critic_snapshot_not_resumable"
PHASE = "before_ensemble_fit_after_collection_regret_and_calibration"


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3], text=True).strip()


def save_state(solver, directory, *, stage, smoke, elapsed, pre_fit, lineage):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Refusing to overwrite captured state")
    started = time.perf_counter()
    config = baseline.smoke_config() if smoke and lineage.get("synthetic") else baseline.EXPERIMENT_CONFIG
    builder = serial_state.build_training_state if pre_fit or not hasattr(solver, "_workers") else parallel_state.build_training_state
    # Never let a previous policy extraction's RNG snapshot override this phase.
    old_checkpoint_rng = solver._checkpoint_resume_rng_state
    solver._checkpoint_resume_rng_state = None
    try:
        payload = builder(solver, seed=0, checkpoint_id=stage,
            repository_commit=commit(), config=config, captured_checkpoints=[])
    finally:
        solver._checkpoint_resume_rng_state = old_checkpoint_rng
    payload["solver"]["training_elapsed_seconds"] = elapsed
    metadata = dict(stage=stage, phase=PHASE if pre_fit else "completed_outer_iteration",
        seed=0, smoke=smoke, elapsed_training_seconds=elapsed,
        iteration=solver.num_iteration, nodes=solver.nodes_touched, episodes=solver.episode,
        critic_versions=[m.target_version for m in solver.q_value_trainer.members],
        lineage=lineage, commit=commit(), pre_fit=pre_fit,
        resumable=not pre_fit, ray_actors=len(getattr(solver, "_workers", [])),
        policy_extraction="not performed: this is a training state, not a fitted playable average policy")
    if pre_fit:
        # Do not duplicate the massive average-policy replay or claim that a
        # mid-iteration diagnostic can be resumed with a normal iteration call.
        payload = {key: payload[key] for key in ("config", "seed", "solver", "regret_trainers",
            "q_ensemble", "calibration", "replay_rng", "feature_encoder", "repository_commit")}
        payload["type"] = SNAPSHOT_TYPE
        payload["calibration"] = {k: v for k, v in payload["calibration"].items() if k != "buffer"}
    payload["stage_audit"] = metadata
    if not pre_fit:
        # Native parallel states already retain these streams; serial local
        # smoke also needs them for an exact next-iteration round-trip.
        payload["stage_regret_replay_rng"] = [deepcopy(t.buffer.rng.bit_generator.state)
                                              for t in solver.regret_trainers]
    info = storage.save_sharded_training_state(directory, payload)
    metadata.update(storage=info, capture_seconds=time.perf_counter()-started)
    write_json(directory.parent / (directory.name+".json"), metadata)
    return metadata


def read_state(directory, *, pre_fit):
    directory = Path(directory)
    metadata = json.loads((directory.parent / (directory.name+".json")).read_text())
    info = storage.training_state_storage_info(directory, verify=True)
    if info != metadata["storage"]:
        raise ValueError("Captured state storage identity changed")
    payload = storage.read_sharded_training_state(directory, verify=False)
    embedded = payload["stage_audit"]
    if any(metadata[k] != v for k, v in embedded.items()):
        raise ValueError("Phase manifest differs from saved state")
    if (metadata["pre_fit"] != pre_fit or metadata["seed"] != 0 or metadata["stage"] not in STAGES
            or metadata["phase"] != (PHASE if pre_fit else "completed_outer_iteration")):
        raise ValueError("Wrong checkpoint phase / identity")
    expected = baseline.smoke_config() if metadata["smoke"] and metadata["lineage"].get("synthetic") else baseline.EXPERIMENT_CONFIG
    if payload["config"] != expected or payload["seed"] != 0:
        raise ValueError("Reference learning configuration differs from Experiment 10")
    if pre_fit and payload["type"] != SNAPSHOT_TYPE:
        raise ValueError("Not a pre-fit audit snapshot")
    if not metadata["smoke"] and metadata["lineage"].get("synthetic"):
        raise ValueError("Synthetic state cannot be production evidence")
    return payload, metadata


def load_audit_solver(directory):
    payload, metadata = read_state(directory, pre_fit=True)
    return extract_verified_snapshot(payload), metadata


def restore_reference(directory, solver):
    """Restore full driver/replay/Adam/history/RNG/actor state, never a pre-fit snapshot."""
    payload, metadata = read_state(directory, pre_fit=False)
    native = parallel_state if metadata["ray_actors"] else serial_state
    native.restore_training_state(solver, payload, seed=0,
        repository_commit=metadata["commit"], config=payload["config"])
    for trainer, rng in zip(solver.regret_trainers, payload["stage_regret_replay_rng"], strict=True):
        trainer.buffer.rng.bit_generator.state = deepcopy(rng)
    return metadata
