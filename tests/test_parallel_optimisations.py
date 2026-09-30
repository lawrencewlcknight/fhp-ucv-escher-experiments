"""Exact replay/preprocessing equivalence; no expensive production training."""

from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.float32_solver import (
    Float32LegacyReservoirBuffer,
)
from unbiased_escher.efficient_replay import CompactCircularBuffer
from unbiased_escher.grouped_wide_solver import GroupedSoftTargetCrossEntropyAvePolicyTrainer
from unbiased_escher.parallel_solver import (
    ParallelUnbiasedControlVariateEscher, _append_circular, _append_calibration,
)
from unbiased_escher.solver import CalibrationBuffer, ResidualCalibrationTrainer
from vr_deep_cfr.logger import Logger
from vr_deep_cfr.solver import ReservoirBuffer


def reservoir_payload(count, offset=0):
    x = np.arange(offset, offset + count, dtype=np.float64)
    return dict(
        infostates=np.column_stack((x, x / 3, -x)),
        values=np.column_stack((x, -x)),
        legal_masks=np.ones((count, 2)),
        iterations=x[:, None] + 1,
    )


@pytest.mark.parametrize("cls", [ReservoirBuffer, Float32LegacyReservoirBuffer])
@pytest.mark.parametrize("capacity", [1, 7, 23])
def test_legacy_reservoir_batch_preserves_rows_and_rng(cls, capacity):
    ref, fast = cls(capacity, 3, 2), cls(capacity, 3, 2)
    np.random.seed(761)
    state = np.random.get_state()
    offset = 0
    for count in (0, 3, 4, 97, 1, 58):
        payload = reservoir_payload(count, offset)
        np.random.set_state(state)
        for row in zip(*(payload[k] for k in ("infostates", "values", "legal_masks", "iterations"))):
            ref.add(*row)
        after = np.random.get_state()
        np.random.set_state(state)
        fast.add_batch(payload)
        got = np.random.get_state()
        assert got[0] == after[0] and got[2:] == after[2:]
        np.testing.assert_array_equal(got[1], after[1])
        assert ref.cur_id == fast.cur_id
        for field in ("infostate_buf", "q_value_buf", "q_value_mask_buf", "iteration_buf"):
            np.testing.assert_array_equal(getattr(ref, field), getattr(fast, field))
        state = after
        offset += count
    random.seed(5)
    a = ref.sample(1)
    random.seed(5)
    b = fast.sample(1)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("capacity", [1, 3, 13])
def test_contiguous_ring_merges_equal_scalar_adds(capacity):
    ref = CompactCircularBuffer(capacity, 4, 3, 2)
    fast = deepcopy(ref)
    cref, cfast = CalibrationBuffer(capacity, 4), CalibrationBuffer(capacity, 4)
    for count in (0, 2, 17, 1, 36):
        p = {
            "histories": np.arange(count * 4).reshape(count, 4),
            "actions": np.arange(count) % 2,
            "next_histories": np.arange(count * 4).reshape(count, 4) + 1,
            "next_states": np.arange(count * 3).reshape(count, 3),
            "next_legal_masks": np.ones((count, 2)),
            "next_players": np.arange(count) % 2,
            "dones": np.arange(count) % 2,
            "rewards": np.arange(count) / 3,
        }
        for row in zip(*p.values()):
            ref.add(*row)
        _append_circular(fast, p)
        for field, value in vars(ref).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(fast, field))
        assert (ref.cur_id, ref.size) == (fast.cur_id, fast.size)
        c = {"features": p["histories"], "targets": p["rewards"]}
        for row in zip(c["features"], c["targets"]):
            cref.add(*row)
        _append_calibration(cfast, c)
        np.testing.assert_array_equal(cref.features, cfast.features)
        np.testing.assert_array_equal(cref.targets, cfast.targets)
        assert (cref.cursor, cref.size) == (cfast.cursor, cfast.size)


def reference_grouping(trainer, iteration):
    """Pre-optimisation implementation, including stable first-mask selection."""
    b = trainer.buffer
    size = min(b.cur_id, b.buffer_size)
    features = np.asarray(b.infostate_buf[:size], dtype=np.float32)
    policies = np.asarray(b.q_value_buf[:size], dtype=np.float64)
    masks = np.asarray(b.q_value_mask_buf[:size], dtype=np.float32)
    iterations = np.asarray(b.iteration_buf[:size], dtype=np.float64).reshape(-1)
    w = np.power(np.maximum(iterations, 0) / float(iteration) * 2, trainer.gamma,
                 dtype=np.float64)
    unique, inverse, counts = np.unique(
        features, axis=0, return_inverse=True, return_counts=True,
    )
    masses = np.zeros(len(unique), dtype=np.float64)
    numerators = np.zeros((len(unique), policies.shape[1]), dtype=np.float64)
    np.add.at(masses, inverse, w)
    np.add.at(numerators, inverse, policies * w[:, None])
    order = np.argsort(inverse, kind="stable")
    first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    grouped_masks = masks[order[first]]
    for i, group in enumerate(inverse):
        if not np.array_equal(masks[i], grouped_masks[group]):
            raise ValueError("Legal-action masks differ")
    return tuple(torch.as_tensor(x, dtype=torch.float32) for x in (
        unique, numerators / masses[:, None], grouped_masks,
        masses * (float(len(unique)) / float(size)),
    ))


def make_trainer(batch_size):
    trainer = GroupedSoftTargetCrossEntropyAvePolicyTrainer(
        3, 2, [8, 8], 0.003, 31, batch_size, 4, Logger(verbose=False), "cpu", 2,
    )
    for i in range(31):
        trainer.add_data([i % 7, (i % 7) / 3, 1], [i / 31, 1-i / 31], [1, 1], i+1)
    return trainer


def reference_fit(trainer, iteration):
    features, targets, masks, weights = reference_grouping(trainer, iteration)
    size = len(features)
    for _ in range(trainer.train_steps):
        if trainer.batch_size == -1 or trainer.batch_size >= size:
            indices = torch.arange(size)
        else:
            indices = torch.as_tensor(random.sample(range(size), trainer.batch_size))
        logits = trainer.model(features.index_select(0, indices))
        logits = logits.masked_fill(masks.index_select(0, indices) != 1, -1e20)
        loss = torch.mean(
            -(targets.index_select(0, indices) * torch.log_softmax(logits, dim=-1)).sum(1)
            * weights.index_select(0, indices)
        )
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
    return loss.item()


@pytest.mark.parametrize("batch_size", [-1, 7, 2048, 3])
def test_grouped_targets_loss_gradients_parameters_and_rng_are_unchanged(batch_size):
    torch.set_num_threads(1)
    ref = make_trainer(batch_size)
    fast = deepcopy(ref)
    expected = reference_grouping(ref, 32)
    actual = fast._grouped_training_data(32)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    random.seed(71)
    expected_loss = reference_fit(ref, 32)
    expected_rng = random.getstate()
    random.seed(71)
    assert fast.train_model(32) == expected_loss
    assert random.getstate() == expected_rng
    for x, y in zip(ref.model.parameters(), fast.model.parameters()):
        assert torch.equal(x, y)
        assert torch.equal(x.grad, y.grad)
    for x, y in zip(ref.optimizer.state.values(), fast.optimizer.state.values()):
        assert all(torch.equal(x[k], y[k]) for k in x)


def test_grouped_mask_validation_still_rejects_inconsistent_repeated_states():
    trainer = make_trainer(-1)
    trainer.buffer.q_value_mask_buf[7, 0] = 0
    with pytest.raises(ValueError, match="Legal-action masks"):
        trainer._grouped_training_data(32)


def test_grouped_validation_across_chunk_boundary():
    trainer = make_trainer(-1)
    b = trainer.buffer
    b.infostate_buf = np.tile(b.infostate_buf[:1], (65_537, 1))
    b.q_value_buf = np.tile(b.q_value_buf[:1], (65_537, 1))
    b.q_value_mask_buf = np.ones((65_537, 2))
    b.iteration_buf = np.ones((65_537, 1))
    b.cur_id = b.buffer_size = 65_537
    assert all(torch.equal(x, y) for x, y in zip(
        reference_grouping(trainer, 2), trainer._grouped_training_data(2),
    ))
    b.q_value_mask_buf[-1, 1] = 0
    with pytest.raises(ValueError, match="Legal-action masks"):
        trainer._grouped_training_data(2)


@pytest.mark.parametrize("iteration", [0, 1, 100, 1_000_000])
def test_broadcast_calibration_features_and_predictions_are_exact(iteration):
    trainer = ResidualCalibrationTrainer(
        infostate_size=190, action_size=3, hidden_layers=[8],
        learning_rate=.001, buffer_size=10, batch_size=2, train_steps=1,
        device="cpu", minimum_variance=1e-5,
    )
    info = np.arange(190) / 7
    disagreement = [-1.0, 1e-9, 1e6]
    expected = np.stack([
        trainer.feature(info, a, iteration, disagreement[a], 1) for a in range(3)
    ])
    means, variances, features = trainer.predict_all(info, iteration, disagreement, 1)
    np.testing.assert_array_equal(features, expected)
    with torch.no_grad():
        outputs = trainer.target_model(torch.as_tensor(expected))
        expected_var = torch.exp(torch.clamp(outputs[:, 1], -8, 6)) + 1e-5
    np.testing.assert_array_equal(means, outputs[:, 0].numpy().astype(np.float64))
    np.testing.assert_array_equal(variances, expected_var.numpy().astype(np.float64))


@pytest.mark.parametrize("shared_generator", [False, True])
def test_shared_rng_learners_fall_back_to_sequential_update_order(shared_generator):
    calls = []
    def train(name):
        calls.append((name, random.random()))
        return 1.0
    shared = np.random.default_rng(91) if shared_generator else None
    members = [SimpleNamespace(buffer=SimpleNamespace(rng=shared)) for _ in range(2)]
    driver = SimpleNamespace(
        calibration_trainer=SimpleNamespace(
            buffer=SimpleNamespace(rng=shared), train_model=lambda: train("calibration"),
        ),
        q_value_trainer=SimpleNamespace(
            members=members, train_model=lambda t: train("critics"),
        ),
        _parallelize_independent_learners=True, num_iteration=1,
    )
    random.seed(6)
    expected = [random.random(), random.random()]
    random.seed(6)
    assert ParallelUnbiasedControlVariateEscher._train_independent_control_learners(
        driver,
    ) == (1., 1.)
    assert calls == list(zip(("calibration", "critics"), expected))
    assert driver._effective_parallel_learner_threads == 1


def test_private_rng_learners_still_run_concurrently_and_restore_torch_threads():
    from threading import Barrier

    barrier = Barrier(3)
    def learner(seed):
        buffer = SimpleNamespace(rng=np.random.default_rng(seed))
        def train(*_):
            barrier.wait(timeout=10)  # fails if the independent branch is serialised
            return float(buffer.rng.random(5).mean())
        return SimpleNamespace(buffer=buffer, train_model=train)
    members = [learner(1), learner(2)]
    calibration = learner(3)
    driver = SimpleNamespace(
        calibration_trainer=calibration,
        q_value_trainer=SimpleNamespace(members=members),
        _parallelize_independent_learners=True, num_iteration=7,
        _parallel_learner_threads=3, _parallel_learner_intraop_threads=1,
    )
    original_threads = torch.get_num_threads()
    expected = [float(np.random.default_rng(seed).random(5).mean()) for seed in (1, 2, 3)]
    assert ParallelUnbiasedControlVariateEscher._train_independent_control_learners(
        driver,
    ) == (expected[2], float(np.mean(expected[:2])))
    assert driver._effective_parallel_learner_threads == 3
    assert torch.get_num_threads() == original_threads
