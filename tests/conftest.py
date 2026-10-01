"""Shared integration-test fixtures."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture
def noisy_ray_startup(monkeypatch):
    """Force unequal infrastructure RNG consumption across real Ray startups."""
    ray = pytest.importorskip("ray")
    original_init = ray.init
    calls = []

    def init_with_rng_noise(*args, **kwargs):
        result = original_init(*args, **kwargs)
        draws = 1 + 17 * len(calls)
        for _ in range(draws):
            random.random()
        np.random.random(draws)
        torch.rand(draws)
        calls.append(draws)
        return result

    monkeypatch.setattr(ray, "init", init_with_rng_noise)
    return calls
