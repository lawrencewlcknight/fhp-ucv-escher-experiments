"""Small, real-Ray checks of selected-architecture transfer and cache safety."""

import random

import numpy as np
import pytest
import torch

from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.config import smoke_config
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.worker import _solver_kwargs
from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import smoke_config as structured
from experiments.fhp.exp3_fhp_wider_lossless_structured_ucv.config import smoke_config as wider
from unbiased_escher.grouped_parallel_solver import (
    ParallelGroupedWideUCVEscher, ParallelStructuredGroupedUCVEscher,
)
from vr_deep_cfr.logger import Logger


def make_kwargs(config):
    kwargs = _solver_kwargs(7, config)
    kwargs["logger"] = Logger(verbose=False)
    # Force multiple chunks to exercise cache reuse within a traverser phase.
    kwargs["num_traversals"] = 6
    return dict(
        **kwargs, parallel_num_workers=2, parallel_run_seed=7,
        parallel_collection_chunk_size=4,
        parallel_ray_object_store_memory=128 * 1024**2,
    )


def state_of(solver):
    models = [
        *(t.model for t in solver.regret_trainers),
        solver.ave_policy_trainer.model,
        *(m.model for m in solver.q_value_trainer.members),
        *(m.target_model for m in solver.q_value_trainer.members),
        solver.calibration_trainer.model,
    ]
    return [x.detach().clone() for model in models for x in model.parameters()]


@pytest.mark.ray
@pytest.mark.parametrize("config_fn,cls", [
    (smoke_config, ParallelGroupedWideUCVEscher),
    (structured, ParallelStructuredGroupedUCVEscher),
    (wider, ParallelStructuredGroupedUCVEscher),
])
def test_real_ray_cached_and_uncached_selected_solvers_match(config_fn, cls):
    pytest.importorskip("ray")
    config = config_fn()
    results = []
    for cached in (False, True):
        solver = cls(**make_kwargs(config), parallel_cache_actor_snapshots=cached)
        try:
            assert solver.q_value_trainer.ensemble_size == 2
            assert solver.critic_target_average_window == 4
            assert solver.average_policy_network_layers == config["average_policy_network_layers"]
            assert solver.fixed_control_variate_beta == 1
            assert not solver.use_instantaneous_predictor
            checkpoint_iterations = []
            solver._maybe_run_training_time_checkpoint = lambda: checkpoint_iterations.append(
                (solver.num_iteration, [len(m.target_history) for m in solver.q_value_trainer.members])
            )
            random.seed(91)
            np.random.seed(91)
            torch.manual_seed(91)
            for _ in range(2):
                solver.iteration()
            assert checkpoint_iterations == [(1, [1, 1]), (2, [2, 2])]
            assert solver.episode == 24  # same total traversal budget, not per actor
            assert solver._effective_parallel_learner_threads == 1  # shared calibration RNG
            solver.ave_policy_trainer.reset()
            loss = solver.ave_policy_trainer.train_model(solver.num_iteration)
            results.append((
                solver.nodes_touched, loss, state_of(solver),
                solver._parallel_worker_snapshot_reload_count,
                solver._parallel_dispatch_count,
            ))
        finally:
            solver.close()
    uncached, cached = results
    assert uncached[:2] == cached[:2]
    assert all(torch.equal(a, b) for a, b in zip(uncached[2], cached[2]))
    assert uncached[3] == 16 and cached[3] == 8
    assert uncached[4] == cached[4] == 8
