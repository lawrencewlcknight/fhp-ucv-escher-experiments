"""Infrastructure startup must not consume the central learner's RNG stream."""

from importlib import import_module
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unbiased_escher.parallel_solver import worker_seed


class FakeRay:
    """Exercise startup, reused runtimes and cleanup without real processes."""

    def __init__(self, *, noisy=False, initialized=False, failure=None):
        self.noisy = noisy
        self.initialized = initialized
        self.failure = failure
        self.events = []
        self.seeds = []

    def event(self, name):
        self.events.append(name)
        if self.noisy:
            for _ in range(len(self.events)):
                random.random()
            np.random.random(len(self.events))
            torch.rand(len(self.events))
        if self.failure == name:
            raise RuntimeError(f"Injected {name} failure")

    def is_initialized(self):
        self.event("is_initialized")
        return self.initialized

    def init(self, **kwargs):
        self.initialized = True
        self.event("init")

    def remote(self, **kwargs):
        self.event("remote")

        def make_actor(*args):
            self.event("actor")
            self.seeds.append(args[2])
            return SimpleNamespace(ping=SimpleNamespace(remote=lambda: self.event("ping")))

        return lambda cls: SimpleNamespace(remote=make_actor)

    def get(self, refs):
        self.event("get")
        return refs

    def kill(self, worker, **kwargs):
        self.event("kill")

    def shutdown(self):
        self.event("shutdown")
        self.initialized = False


@pytest.mark.parametrize("experiment", [
    "exp10_fhp_hand_board_features", "exp11_fhp_betting_economics",
])
@pytest.mark.parametrize("initialized,failure", [
    (False, None), (True, None), (False, "init"),
    (False, "get"), (True, "get"),
])
def test_startup_preserves_post_model_rng(monkeypatch, experiment, initialized, failure):
    worker = import_module(f"experiments.fhp.{experiment}.worker")
    config = import_module(f"experiments.fhp.{experiment}.config").smoke_config()
    smoke = import_module(f"experiments.fhp.{experiment}.smoke")
    # A no-draw runtime supplies the expected post-model-initialisation state.
    baseline_ray = FakeRay(initialized=initialized)
    monkeypatch.setitem(sys.modules, "ray", baseline_ray)
    baseline = worker._make_solver(0, config, smoke=True)
    try:
        expected = baseline._capture_rng_state()
        expected_weights = [p.detach().clone() for t in baseline.regret_trainers
                            for p in t.model.parameters()]
        expected_seeds = list(baseline_ray.seeds)
    finally:
        baseline.close()

    noisy_ray = FakeRay(noisy=True, initialized=initialized, failure=failure)
    monkeypatch.setitem(sys.modules, "ray", noisy_ray)
    if failure:
        with pytest.raises(RuntimeError, match=f"Injected {failure} failure"):
            worker._make_solver(0, config, smoke=True)
        # Capture the process state even though construction did not return.
        smoke.assert_identical(expected, baseline._capture_rng_state(), "failed-startup.rng")
        assert ("shutdown" in noisy_ray.events) == (not initialized)
        if failure == "get":
            assert noisy_ray.events.count("kill") == 8
    else:
        solver = worker._make_solver(0, config, smoke=True)
        try:
            smoke.assert_identical(expected, solver._capture_rng_state(), "startup.rng")
            actual_weights = [p.detach().clone() for t in solver.regret_trainers
                              for p in t.model.parameters()]
            smoke.assert_identical(expected_weights, actual_weights, "startup.models")
            assert noisy_ray.seeds == expected_seeds == [worker_seed(0, i) for i in range(8)]
            assert ("init" in noisy_ray.events) == (not initialized)
        finally:
            solver.close()
        assert ("shutdown" in noisy_ray.events) == (not initialized)

    # Restoring the original seed would incorrectly repeat model-init draws.
    assert not torch.equal(expected["torch"], torch.Generator().manual_seed(0).get_state())
