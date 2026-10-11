"""Verified source extraction without starting Ray actors or resuming training."""
from copy import deepcopy
import gc

from experiments.fhp.exp17_fhp_critic_budget.source import fetch_source, preflight, validate_local, validate_runtime
from experiments.fhp.exp17_fhp_critic_budget.config import TRAINING_REF


def extract_solver(payload):
    from experiments.fhp.exp10_fhp_hand_board_features import config as baseline
    from experiments.fhp.exp10_fhp_hand_board_features.training_state import STATE_TYPE
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
    from unbiased_escher.fhp_structured_solver import StructuredFHPGroupedWideUCVEscher
    from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import install_cache
    if (payload.get("type") != STATE_TYPE or payload.get("repository_commit") != TRAINING_REF
            or payload.get("checkpoint_id") != "time_24h" or payload.get("seed") not in (0, 1, 2)
            or payload.get("config") != baseline.EXPERIMENT_CONFIG):
        raise ValueError("Expected exact frozen Experiment 10 final-state identity/configuration")
    if len(payload["q_ensemble"]["members"]) != 2:
        raise ValueError("Expected two cross-fitted critics")
    small = deepcopy(payload["config"])
    for key in ("advantage_buffer_size", "ave_policy_buffer_size", "calibration_buffer_size"):
        small[key] = 1
    small["baseline_buffer_size"] = 2
    kwargs = _solver_kwargs(payload["seed"], small)
    kwargs.pop("cache_frozen_critic_targets")
    solver = StructuredFHPGroupedWideUCVEscher(**kwargs)
    install_cache(solver, enabled=True)
    if payload["feature_encoder"] != solver.feature_encoder.metadata():
        raise ValueError("Source feature encoder mismatch")
    for trainer, saved in zip(solver.regret_trainers, payload["regret_trainers"], strict=True):
        trainer.model.load_state_dict(saved["model"])
        if trainer.predictor_enabled:
            raise ValueError("Predictive regret core unsupported")
    for member, saved, rng in zip(solver.q_value_trainer.members, payload["q_ensemble"]["members"],
                                  payload["replay_rng"]["q_members"], strict=True):
        member.model.load_state_dict(saved["model"])
        member.target_model.load_state_dict(saved["target_model"])
        member.optimizer.load_state_dict(saved["optimizer"])
        member.target_history.clear()
        member.target_history.extend(deepcopy(saved["target_history"]))
        member.target_version = saved["target_version"]
        replay = saved["buffer"]
        for name in ("history", "next_history", "next_state", "reward", "action",
                     "next_legal_actions_mask", "next_player", "done"):
            setattr(member.buffer, name+"_buf", replay[name])
        member.buffer.size, member.buffer.cur_id = replay["size"], replay["cur_id"]
        member.buffer.buffer_size = len(replay["history"])
        member.buffer.rng.bit_generator.state = deepcopy(rng)
    calibration = payload["calibration"]
    for name in ("model", "target_model"):
        getattr(solver.calibration_trainer, name).load_state_dict(calibration[name])
    solver.calibration_trainer.target_version = calibration["target_version"]
    solver.num_iteration = payload["solver"]["num_iteration"]
    solver.nodes_touched = payload["solver"]["nodes_touched"]
    solver.episode = payload["solver"]["episode"]
    return solver


def load_solver(source, seed):
    from experiments.fhp.exp10_fhp_hand_board_features.training_state import read_training_state
    path, provenance = validate_local(source, seed)
    payload = read_training_state(path)
    if payload.get("seed") != seed:
        raise ValueError("Source payload seed mismatch")
    solver = extract_solver(payload)
    del payload
    gc.collect()
    return solver, provenance
