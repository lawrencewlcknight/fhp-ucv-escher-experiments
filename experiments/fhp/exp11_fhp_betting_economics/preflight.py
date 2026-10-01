"""Cloud-only production-capacity checkpoint preflight (no learning)."""
from copy import deepcopy
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import EXPERIMENT_CONFIG
from .worker import _make_solver
from .diagnostics import execution_diagnostics
from .training_state import build_training_state, save_training_state, read_training_state, restore_training_state


def capacity_preflight(output_root: Path, *, config=None, smoke=False):
    """Synthetic state stays in a private temporary directory, never uploaded."""
    config = deepcopy(EXPERIMENT_CONFIG if config is None else config)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    solver = _make_solver(0, config, smoke=smoke)
    try:
        average = solver.ave_policy_trainer.buffer
        average.cur_id = average.buffer_size
        for member in solver.q_value_trainer.members:
            member.buffer.size = member.buffer.buffer_size
            member.buffer.cur_id = 0
        calibration = solver.calibration_trainer.buffer
        calibration.size = calibration.capacity
        calibration.cursor = 0
        # Touch every allocated array page, including transient regret replay.
        buffers = [average, calibration, *(t.buffer for t in solver.regret_trainers),
                   *(m.buffer for m in solver.q_value_trainer.members)]
        import numpy as np
        for buffer in buffers:
            for value in vars(buffer).values():
                if isinstance(value, np.ndarray):
                    value.fill(1)
        kwargs = dict(seed=0, config=config, repository_commit="capacity-preflight")
        with TemporaryDirectory(prefix="exp11-capacity-") as temporary:
            path = Path(temporary) / "synthetic.pt"
            payload = build_training_state(solver, **kwargs, checkpoint_id="capacity",
                                           captured_checkpoints=[])
            save_training_state(path, payload)
            del payload
            loaded = read_training_state(path)
            restore_training_state(solver, loaded, **kwargs)
            if (loaded["average_policy_trainer"]["buffer"]["size"] != average.buffer_size
                    or len(loaded["parallel"]["actors"]) != 8):
                raise RuntimeError("Production-capacity checkpoint did not round-trip")
            result = {"status": "passed", "synthetic_no_training": True,
                      "serialized_bytes": path.stat().st_size,
                      "diagnostics": execution_diagnostics(solver)}
            del loaded
            gc.collect()
        (output_root / "capacity_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        solver.close()
