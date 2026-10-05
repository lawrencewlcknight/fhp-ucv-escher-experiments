"""Storage parity, not merely approximate policy similarity."""
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from unbiased_escher.lossless_replay import (
    CODEBOOK, STORAGE_ID, EncodedFeatures, encode, feature_state, restore_features,
)
from unbiased_escher.efficient_replay import CompactReservoirBuffer, CompactCircularBuffer
from unbiased_escher.fhp_structured_solver import EncodedCalibrationBuffer
from unbiased_escher.solver import CalibrationBuffer
from fhp_escher.game import load_fhp_game
from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
from experiments.fhp.exp10_fhp_hand_board_features.config import smoke_config
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
from unbiased_escher.fhp_structured_solver import StructuredFHPGroupedWideUCVEscher
from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import install_cache
from experiments.fhp.exp10_fhp_hand_board_features.smoke import assert_identical
from experiments.fhp.exp2_fhp_lossless_structured_ucv import training_state as states
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.training_state import (
    save_sharded_training_state, read_sharded_training_state,
)


def make_solver(storage="dense"):
    torch.set_num_threads(1)
    config = dict(smoke_config(), replay_storage=storage)
    kwargs = _solver_kwargs(0, config)
    kwargs.pop("cache_frozen_critic_targets")
    solver = StructuredFHPGroupedWideUCVEscher(**kwargs)
    install_cache(solver, enabled=True)
    return solver


def normalise(value):
    if isinstance(value, dict):
        if value.get("encoding") == STORAGE_ID:
            return np.concatenate((CODEBOOK[value["codes"]], value["tail"]), axis=1)
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(normalise(v) for v in value)
    return value


def test_encoder_round_trip():
    game, encoder = load_fhp_game(), FHPHandBoardFeatureEncoder()
    rng = np.random.default_rng(221)
    for _ in range(100):
        state = game.new_initial_state()
        while True:
            values = [] if state.is_chance_node() else [encoder.full_state(state)]
            if not state.is_chance_node() and not state.is_terminal():
                values += [encoder.information_state(state)]
            for vector in values:
                assert np.array_equal(CODEBOOK[encode(vector)].view(np.uint32), vector.view(np.uint32))
            if state.is_terminal():
                break
            state.apply_action(int(rng.choice(state.legal_actions())))


@pytest.mark.parametrize("invalid", [0.123456, -0.0, np.nan, np.inf, -1, 2])
def test_fail_closed(invalid):
    with pytest.raises(ValueError, match="exactly representable"):
        encode(np.array([invalid], dtype=np.float32))


def test_reservoir_sampling_and_admission():
    buffers = [CompactReservoirBuffer(13, 7, 3, seed=43, feature_array_factory=f)
               for f in (None, EncodedFeatures)]
    rng = np.random.default_rng(23)
    for count in (7, 15, 31):
        batch = dict(infostates=rng.choice(CODEBOOK, (count, 7)),
                     values=rng.normal(size=(count, 3)).astype(np.float32),
                     legal_masks=np.ones((count, 3), np.float32),
                     iterations=np.full((count, 1), 2, np.float32))
        for b in buffers:
            b.add_batch(batch)
        assert_identical(buffers[0].sample(9), buffers[1].sample(9))
        assert_identical(buffers[0].rng.bit_generator.state, buffers[1].rng.bit_generator.state)
    assert buffers[1].nbytes() < buffers[0].nbytes()
    with pytest.raises(TypeError):
        np.asarray(buffers[1].infostate_buf)


def test_circular_and_calibration():
    circular = [CompactCircularBuffer(7, 5, 3, 3, seed=21, feature_array_factory=f)
                for f in (None, EncodedFeatures)]
    calibration = [CalibrationBuffer(7, 9), EncodedCalibrationBuffer(7, 9, 3)]
    for i in range(21):
        for b in circular:
            b.add(np.full(5, 1/3, np.float32), 1, np.zeros(5), np.ones(3), [1,1,0], 1, 0, 1.234)
        for b in calibration:
            b.add(np.array([0, .5, 1, 1, 0, 0, .123, i / 7, 1], np.float32), .341)
    assert_identical(circular[0].sample(4), circular[1].sample(4))
    random.seed(31)
    expected = calibration[0].sample(4, "cpu")
    random.seed(31)
    assert_identical(expected, calibration[1].sample(4, "cpu"))


def test_checkpoint_migration(tmp_path):
    dense = np.tile(CODEBOOK, (101, 1))
    coded = EncodedFeatures(dense.shape)
    restore_features(coded, dense, len(dense))
    destination = tmp_path / "state"
    save_sharded_training_state(destination, {"features": feature_state(coded, len(dense))})
    saved = read_sharded_training_state(destination)
    assert isinstance(saved["features"]["codes"], np.memmap)
    restored = np.empty_like(dense)
    restore_features(restored, saved["features"], len(dense))
    assert_identical(dense, restored)
    saved["features"]["encoding"] = "unknown"
    with pytest.raises(ValueError):
        restore_features(coded, saved["features"], len(dense))


def test_end_to_end_and_streamed_restart(tmp_path):
    results = []
    kwargs = dict(seed=0, config={}, repository_commit="test", checkpoint_id="audit", captured_checkpoints=[])
    for storage in ("dense", STORAGE_ID):
        solver = make_solver(storage)
        for _ in range(2):
            solver.iteration()
            solver.ave_policy_trainer.train_model(solver.num_iteration)
        checkpoint = tmp_path / storage
        regret_rng = [deepcopy(t.buffer.rng.bit_generator.state) for t in solver.regret_trainers]
        save_sharded_training_state(checkpoint, states.build_training_state(solver, **kwargs))
        solver.iteration()
        solver.ave_policy_trainer.train_model(solver.num_iteration)
        expected = states.build_training_state(solver, **kwargs)
        resumed = make_solver(storage)
        states.restore_training_state(resumed, read_sharded_training_state(checkpoint),
                                     seed=0, config={}, repository_commit="test")
        # Exp8 adds these to the production parallel continuation schema;
        # this unit test uses Exp2's sequential schema instead.
        for trainer, rng in zip(resumed.regret_trainers, regret_rng):
            trainer.buffer.rng.bit_generator.state = rng
        resumed.iteration()
        resumed.ave_policy_trainer.train_model(resumed.num_iteration)
        actual = states.build_training_state(resumed, **kwargs)
        fields = ("regret_trainers", "average_policy_trainer", "q_ensemble", "calibration", "gate_controller", "rng", "replay_rng")
        for field in fields:
            assert_identical(expected[field], actual[field], "restart." + field)
        results.append(normalise({k: expected[k] for k in fields}))
    assert_identical(*results, "dense-vs-coded")


@pytest.mark.parametrize("batch_size", [-1, 7])
def test_grouping_duplicates_and_parameter_updates(batch_size):
    states_out = []
    for storage in ("dense", STORAGE_ID):
        solver = make_solver(storage)
        trainer = solver.ave_policy_trainer
        trainer.batch_size = batch_size
        rng = np.random.default_rng(19)
        # More than one grouping chunk; both duplicates and unequal masses.
        trainer.buffer = CompactReservoirBuffer(40_000, 213, 3, seed=17,
                            feature_array_factory=EncodedFeatures if storage != "dense" else None)
        keys = rng.choice(CODEBOOK, (71, 213))
        payload = dict(infostates=keys[rng.integers(71, size=40_000)],
                       values=rng.dirichlet([1, 2, 3], size=40_000).astype(np.float32),
                       legal_masks=np.ones((40_000, 3), np.float32),
                       iterations=rng.integers(1, 101, size=(40_000, 1)).astype(np.float32))
        trainer.buffer.add_batch(payload)
        random.seed(56)
        loss = trainer.train_model(100)
        states_out.append((deepcopy(trainer.model.state_dict()), deepcopy(trainer.optimizer.state_dict()),
                           loss, random.getstate(), trainer.grouped_num_information_sets))
    assert_identical(*states_out)


def test_cached_targets_and_critic_gradients_identical():
    from adaptive_escher.frozen_target_cache import build_target_cache
    results = []
    for storage in ("dense", STORAGE_ID):
        solver = make_solver(storage)
        solver.iteration()
        values = []
        for member in solver.q_value_trainer.members:
            targets = build_target_cache(member, solver.num_iteration)
            rows = slice(0, min(len(member.buffer), member.batch_size))
            features = torch.as_tensor(member.buffer.history_buf[rows])
            actions = torch.as_tensor(member.buffer.action_buf[rows], dtype=torch.long)
            predictions = member.model(features).gather(1, actions[:, None]).squeeze(1)
            member.optimizer.zero_grad()
            loss = member.loss_fn(predictions, targets[rows])
            loss.backward()
            values.append((targets, predictions.detach(),
                           [p.grad.clone() for p in member.model.parameters()]))
        results.append(values)
    assert_identical(*results)


def test_encoded_checkpoint_rejects_invalid_code():
    features = EncodedFeatures((1, 3))
    features[:] = np.zeros((1, 3))
    state = feature_state(features, 1)
    state["codes"][0, 0] = 255
    with pytest.raises(ValueError, match="Invalid replay feature code"):
        restore_features(EncodedFeatures((1, 3)), state, 1)
