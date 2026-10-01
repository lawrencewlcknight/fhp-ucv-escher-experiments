"""Fixed-iteration, disk-backed restart check using the real eight Ray actors."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import torch

from .config import smoke_config
from .worker import _make_solver
from .training_state import build_training_state, restore_training_state, save_training_state, read_training_state


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
    assert equal, f"Restart changed {path}"


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
        for field in ("regret_trainers", "average_policy_trainer", "q_ensemble",
                      "calibration", "gate_controller", "rng", "replay_rng"):
            assert_identical(expected[field], actual[field], field)
        assert_identical(expected["parallel"]["actors"], actual["parallel"]["actors"], "actors")
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
