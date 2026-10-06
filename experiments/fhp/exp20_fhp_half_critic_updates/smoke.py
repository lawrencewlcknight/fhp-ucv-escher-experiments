"""Fixed-iteration, disk-backed restart check using the real eight Ray actors."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import torch

from .config import smoke_config
from .worker import _make_solver
from .training_state import build_training_state, restore_training_state, save_training_state, read_training_state
from experiments.fhp.exp8_fhp_parallel_48h.training_state import (
    build_training_state as parallel_state,
)
from experiments.critic_target_cache_benchmark import target_audit

LEARNING_FIELDS = ("regret_trainers", "average_policy_trainer", "q_ensemble",
                   "calibration", "gate_controller", "rng", "replay_rng")


def learning_state(solver):
    payload = parallel_state(solver, seed=0, config={}, repository_commit="cache-check",
                             checkpoint_id="fixed-iteration", captured_checkpoints=[])
    return {**{key: payload[key] for key in LEARNING_FIELDS},
            "actors": payload["parallel"]["actors"],
            "regret_replay_rng": payload["parallel"]["regret_replay_rng"],
            "counters": {key: payload["solver"][key]
                         for key in ("num_iteration", "episode", "nodes_touched")}}


def verify_cache_equivalence(directory):
    """Same fixed workload and actor streams, differing only in target caching."""
    outputs = []
    initial_state = None
    for cached in (False, True):
        torch.set_num_threads(1)
        config = dict(smoke_config(), cache_frozen_critic_targets=cached)
        solver = _make_solver(0, config, smoke=True)
        try:
            current_initial_state = learning_state(solver)
            if initial_state is None:
                initial_state = current_initial_state
            else:
                assert_identical(initial_state, current_initial_state,
                                 "cached-vs-baseline.initial")
                print("Cached/uncached initial learner, actor and RNG states match exactly.",
                      flush=True)
            for _ in range(3):
                solver.iteration()
                # Also exercise the output fitter and its RNG consumption.
                solver.ave_policy_trainer.train_model(solver.num_iteration)
            if cached and solver._cumulative_cached_critic_fit_calls != 6:
                raise RuntimeError("Integration smoke did not cache both critics each iteration")
            integrated = learning_state(solver)
            # In addition, test the production 2,048-row minibatch, eight
            # fitting threads, and a 3-row cache tail, on real collected rows.
            torch.set_num_threads(8)
            for member in solver.q_value_trainer.members:
                size = len(member.buffer)
                for name in ("history", "next_history", "next_state", "reward", "action",
                             "next_legal_actions_mask", "next_player", "done"):
                    source = getattr(member.buffer, name + "_buf")[:size]
                    setattr(member.buffer, name + "_buf", source[np.arange(2051) % size].copy())
                member.buffer.size = member.buffer.buffer_size = 2051
                member.buffer.cur_id = 0
                member.batch_size, member.train_steps = 2048, 3
                audit = target_audit(member, solver.num_iteration, 7919)
                if not (audit["allclose"] and audit["construction_preserved_rng"]):
                    raise RuntimeError("Production-shaped cache target audit failed")
                member.train_model(solver.num_iteration)
            outputs.append((integrated, learning_state(solver)))
        finally:
            solver.close()
    assert_identical(outputs[0], outputs[1], "cached-vs-baseline")
    result = {"status": "passed", "learning_state_bitwise_identical": True,
              "initial_learning_state_bitwise_identical": True,
              "iterations_per_arm": 3, "traversal_actors": 8,
              "additional_critic_batch_size": 2048, "additional_critic_replay_rows": 2051,
              "additional_critic_fitting_threads": 8,
              "scope": "small fixed-workload integration, not a long-run performance guarantee"}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cache_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.set_num_threads(1)
    return result


def assert_identical(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        equal = torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        equal = np.array_equal(a, b, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_identical(a[key], b[key], f"{path}.{key}")
        return
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_identical(x, y, f"{path}[{i}]")
        return
    else:
        equal = a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))
    assert equal, f"Learning states differ at {path}"


def _verify_continuation(directory: Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = smoke_config()
    kwargs = dict(seed=0, config=config, repository_commit="smoke-restart-check")
    def snapshot(solver):
        return build_training_state(solver, **kwargs, checkpoint_id="iteration_1",
                                    captured_checkpoints=[])
    solver = _make_solver(0, config, smoke=True)
    try:
        solver.iteration()
        save_training_state(directory / "resume.pt", snapshot(solver))
        solver.iteration()
        expected = snapshot(solver)
        assert solver.episode == 2 * 2 * config["num_traversals"]
        assert solver._effective_parallel_learner_threads == 1
    finally:
        solver.close()
    solver = _make_solver(0, config, smoke=True)
    try:
        restore_training_state(solver, read_training_state(directory / "resume.pt"), **kwargs)
        solver.iteration()
        actual = snapshot(solver)
        for field in LEARNING_FIELDS:
            assert_identical(expected[field], actual[field], field)
        assert_identical(expected["parallel"]["actors"], actual["parallel"]["actors"], "actors")
        assert_identical(expected["parallel"]["regret_replay_rng"],
                         actual["parallel"]["regret_replay_rng"], "regret sampler RNG")
        for field in ("_cumulative_cached_critic_fit_calls", "_cumulative_cached_critic_rows"):
            assert_identical(expected["critic_cache"]["counters"][field],
                             actual["critic_cache"]["counters"][field], field)
        for key in ("num_iteration", "episode", "nodes_touched"):
            assert_identical(expected["solver"][key], actual["solver"][key], key)
        result = {"status": "passed", "actor_count": len(solver._workers),
                  "iterations": solver.num_iteration, "trajectories": solver.episode,
                  "nodes": solver.nodes_touched, "learning_state_bitwise_identical": True}
    finally:
        solver.close()
    (directory / "restart_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_continuation(directory: Path) -> dict:
    with TemporaryDirectory(prefix="fhp-restart-test-") as temporary:
        result = _verify_continuation(Path(temporary))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "restart_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
