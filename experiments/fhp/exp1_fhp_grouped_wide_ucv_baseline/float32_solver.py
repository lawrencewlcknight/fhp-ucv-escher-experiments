"""Experiment-1 solver with legacy sampling and float32 replay storage.

This module deliberately changes storage precision only.  In particular, it
does not use the compact replay implementation from FHP Experiments 2 and 3,
whose private random generators and reduced-width integer arrays would alter
more than the failed baseline's memory representation.
"""

from __future__ import annotations

import numpy as np

from unbiased_escher.grouped_wide_solver import (
    GroupedSoftTargetCrossEntropyAvePolicyTrainer,
    GroupedWideUnbiasedControlVariateEscher,
    TemporallyAveragedCrossFittedQEnsemble,
    TemporallyAveragedCrossFittedQMember,
)
from vr_deep_cfr.solver import CircularBuffer, ReservoirBuffer
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer


REPLAY_FLOAT_DTYPE = np.dtype(np.float32)
REPLAY_INTEGER_DTYPE = np.dtype(int)


class Float32LegacyReservoirBuffer(ReservoirBuffer):
    """Legacy Algorithm-R reservoir with float32-valued arrays.

    ``add`` and ``sample`` are inherited so replacement and minibatch draws
    continue to consume the global NumPy and Python RNG streams exactly as in
    the original Experiment 1 implementation.
    """

    def reset(self):
        if hasattr(self, "cur_id"):
            self.cur_id = 0
            return

        self.infostate_buf = np.ones(
            (self.buffer_size, self.infostate_size), dtype=REPLAY_FLOAT_DTYPE
        )
        self.q_value_buf = np.ones(
            (self.buffer_size, self.action_size), dtype=REPLAY_FLOAT_DTYPE
        )
        self.q_value_mask_buf = np.ones(
            (self.buffer_size, self.action_size), dtype=REPLAY_FLOAT_DTYPE
        )
        self.iteration_buf = np.ones(
            (self.buffer_size, 1), dtype=REPLAY_FLOAT_DTYPE
        )
        self.cur_id = 0


class Float32LegacyCircularBuffer(CircularBuffer):
    """Legacy critic replay with float32 values and unchanged integer arrays."""

    def reset(self):
        self.history_buf = np.ones(
            (self.buffer_size, self.history_size), dtype=REPLAY_FLOAT_DTYPE
        )
        self.action_buf = np.ones(self.buffer_size, dtype=REPLAY_INTEGER_DTYPE)
        self.next_history_buf = np.ones(
            (self.buffer_size, self.history_size), dtype=REPLAY_FLOAT_DTYPE
        )
        self.next_state_buf = np.ones(
            (self.buffer_size, self.state_size), dtype=REPLAY_FLOAT_DTYPE
        )
        self.next_legal_actions_mask_buf = np.ones(
            (self.buffer_size, self.action_size), dtype=REPLAY_INTEGER_DTYPE
        )
        self.next_player_buf = np.ones(self.buffer_size, dtype=REPLAY_INTEGER_DTYPE)
        self.done_buf = np.ones(self.buffer_size, dtype=REPLAY_INTEGER_DTYPE)
        self.reward_buf = np.ones(self.buffer_size, dtype=REPLAY_FLOAT_DTYPE)
        self.cur_id = 0
        self.size = 0


class _Float32ReservoirMixin:
    def init_buffer(self):
        return Float32LegacyReservoirBuffer(
            self.buffer_size,
            self.input_size,
            self.output_size,
            device=self.device,
        )


class Float32DCFRPlusRegretTrainer(_Float32ReservoirMixin, VRDCFRPlusRegretTrainer):
    pass


class Float32GroupedAvePolicyTrainer(
    _Float32ReservoirMixin,
    GroupedSoftTargetCrossEntropyAvePolicyTrainer,
):
    pass


class Float32TemporallyAveragedQMember(TemporallyAveragedCrossFittedQMember):
    def init_buffer(self):
        return Float32LegacyCircularBuffer(
            self.buffer_size,
            self.input_size,
            self.state_size,
            self.output_size,
            device=self.device,
        )


class Float32TemporallyAveragedQEnsemble(TemporallyAveragedCrossFittedQEnsemble):
    """The selected two-fold critic, changing only member replay storage."""

    member_class = Float32TemporallyAveragedQMember


class Float32GroupedWideUnbiasedControlVariateEscher(
    GroupedWideUnbiasedControlVariateEscher
):
    """Raw FHP Experiment 1 with training-precision replay values."""

    replay_storage_contract = {
        "continuous_dtype": REPLAY_FLOAT_DTYPE.name,
        "integer_dtype": REPLAY_INTEGER_DTYPE.name,
        "sampling": "legacy_global_rng",
        "representation": "raw_openspiel_information_state",
    }
    nonpredictive_regret_trainer_class = Float32DCFRPlusRegretTrainer
    average_policy_trainer_class = Float32GroupedAvePolicyTrainer
    q_ensemble_class = Float32TemporallyAveragedQEnsemble


def validate_replay_storage(solver) -> int:
    """Validate the storage-only contract and return allocated replay bytes."""
    arrays = []
    reservoirs = [
        *(trainer.buffer for trainer in solver.regret_trainers),
        solver.ave_policy_trainer.buffer,
    ]
    for buffer in reservoirs:
        continuous = (
            buffer.infostate_buf,
            buffer.q_value_buf,
            buffer.q_value_mask_buf,
            buffer.iteration_buf,
        )
        if any(array.dtype != REPLAY_FLOAT_DTYPE for array in continuous):
            raise TypeError("Experiment 1 reservoir values must use float32")
        if hasattr(buffer, "rng"):
            raise TypeError("Experiment 1 must retain legacy global-RNG sampling")
        arrays.extend(continuous)

    for member in solver.q_value_trainer.members:
        buffer = member.buffer
        continuous = (
            buffer.history_buf,
            buffer.next_history_buf,
            buffer.next_state_buf,
            buffer.reward_buf,
        )
        integer = (
            buffer.action_buf,
            buffer.next_legal_actions_mask_buf,
            buffer.next_player_buf,
            buffer.done_buf,
        )
        if any(array.dtype != REPLAY_FLOAT_DTYPE for array in continuous):
            raise TypeError("Experiment 1 critic values must use float32")
        if any(array.dtype != REPLAY_INTEGER_DTYPE for array in integer):
            raise TypeError("Experiment 1 critic integers must remain native-width")
        if hasattr(buffer, "rng"):
            raise TypeError("Experiment 1 must retain legacy global-RNG sampling")
        arrays.extend((*continuous, *integer))

    calibration = solver.calibration_trainer
    if calibration is not None:
        if (
            calibration.buffer.features.dtype != REPLAY_FLOAT_DTYPE
            or calibration.buffer.targets.dtype != REPLAY_FLOAT_DTYPE
        ):
            raise TypeError("Experiment 1 calibration values must use float32")
        arrays.extend((calibration.buffer.features, calibration.buffer.targets))
    return sum(int(array.nbytes) for array in arrays)


__all__ = [
    "Float32GroupedWideUnbiasedControlVariateEscher",
    "Float32LegacyCircularBuffer",
    "Float32LegacyReservoirBuffer",
    "REPLAY_FLOAT_DTYPE",
    "REPLAY_INTEGER_DTYPE",
    "validate_replay_storage",
]
