"""Compact, vectorised CPU replay for large-poker UCV-ESCHER runs.

The buffers retain the existing replay contracts and without-replacement SGD
sampling. They store tensors in their eventual training precision, use an
independent NumPy generator per learner, and provide an exact batched reservoir
update so Ray results do not have to be merged one Python sample at a time.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


class CompactReservoirBuffer:
    """Algorithm-R reservoir with float32 storage and vectorised ingestion."""

    def __init__(
        self,
        buffer_size: int,
        infostate_size: int,
        action_size: int,
        *,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.buffer_size = int(buffer_size)
        self.infostate_size = int(infostate_size)
        self.action_size = int(action_size)
        self.device = str(device)
        self.rng = np.random.default_rng(int(seed))
        self.infostate_buf = np.empty(
            (self.buffer_size, self.infostate_size), dtype=np.float32
        )
        self.q_value_buf = np.empty(
            (self.buffer_size, self.action_size), dtype=np.float32
        )
        self.q_value_mask_buf = np.empty(
            (self.buffer_size, self.action_size), dtype=np.float32
        )
        self.iteration_buf = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.cur_id = 0

    def reset(self) -> None:
        self.cur_id = 0

    def add(self, infostate, q_value, q_value_mask, iteration) -> None:
        stream_index = int(self.cur_id)
        if stream_index < self.buffer_size:
            target = stream_index
        else:
            target = int(self.rng.integers(0, stream_index + 1))
        if target < self.buffer_size:
            self.infostate_buf[target] = infostate
            self.q_value_buf[target] = q_value
            self.q_value_mask_buf[target] = q_value_mask
            self.iteration_buf[target] = iteration
        self.cur_id += 1

    def add_batch(self, payload: Mapping[str, np.ndarray]) -> None:
        """Apply exact sequential Algorithm-R choices with vectorised writes."""
        count = int(len(payload["iterations"]))
        if count == 0:
            return

        source_offset = 0
        direct_count = min(count, max(0, self.buffer_size - int(self.cur_id)))
        if direct_count:
            target = slice(int(self.cur_id), int(self.cur_id) + direct_count)
            source = slice(0, direct_count)
            self.infostate_buf[target] = payload["infostates"][source]
            self.q_value_buf[target] = payload["values"][source]
            self.q_value_mask_buf[target] = payload["legal_masks"][source]
            self.iteration_buf[target] = payload["iterations"][source]
            self.cur_id += direct_count
            source_offset = direct_count

        remaining = count - source_offset
        if remaining:
            positions = np.arange(
                int(self.cur_id) + 1,
                int(self.cur_id) + remaining + 1,
                dtype=np.int64,
            )
            candidate_targets = self.rng.integers(
                low=0,
                high=positions,
                dtype=np.int64,
            )
            accepted = np.flatnonzero(candidate_targets < self.buffer_size)
            if accepted.size:
                accepted_targets = candidate_targets[accepted]
                # If several stream items select one slot, only the final item
                # survives after the equivalent sequential Algorithm-R updates.
                _, reversed_indices = np.unique(
                    accepted_targets[::-1], return_index=True
                )
                last = accepted.size - 1 - reversed_indices
                source_indices = source_offset + accepted[last]
                target_indices = accepted_targets[last]
                self.infostate_buf[target_indices] = payload["infostates"][
                    source_indices
                ]
                self.q_value_buf[target_indices] = payload["values"][source_indices]
                self.q_value_mask_buf[target_indices] = payload["legal_masks"][
                    source_indices
                ]
                self.iteration_buf[target_indices] = payload["iterations"][
                    source_indices
                ]
            self.cur_id += remaining

    def sample(self, num_samples: int = -1):
        length = min(int(self.cur_id), self.buffer_size)
        if num_samples < 0 or int(num_samples) >= length:
            indices = slice(0, length)
        else:
            indices = self.rng.choice(length, size=int(num_samples), replace=False)
        arrays = (
            self.infostate_buf[indices],
            self.q_value_buf[indices],
            self.q_value_mask_buf[indices],
            self.iteration_buf[indices],
        )
        return tuple(
            torch.as_tensor(array, device=self.device)
            for array in arrays
        )

    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.infostate_buf,
                self.q_value_buf,
                self.q_value_mask_buf,
                self.iteration_buf,
            )
        )

    def __len__(self) -> int:
        return int(self.cur_id)


class CompactCircularBuffer:
    """Typed circular transition replay with vectorised minibatch selection."""

    def __init__(
        self,
        buffer_size: int,
        history_size: int,
        state_size: int,
        action_size: int,
        *,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.buffer_size = int(buffer_size)
        self.history_size = int(history_size)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = str(device)
        self.rng = np.random.default_rng(int(seed))
        self.history_buf = np.empty(
            (self.buffer_size, self.history_size), dtype=np.float32
        )
        self.action_buf = np.empty(self.buffer_size, dtype=np.int16)
        self.next_history_buf = np.empty(
            (self.buffer_size, self.history_size), dtype=np.float32
        )
        self.next_state_buf = np.empty(
            (self.buffer_size, self.state_size), dtype=np.float32
        )
        self.next_legal_actions_mask_buf = np.empty(
            (self.buffer_size, self.action_size), dtype=np.int8
        )
        self.next_player_buf = np.empty(self.buffer_size, dtype=np.int8)
        self.done_buf = np.empty(self.buffer_size, dtype=np.int8)
        self.reward_buf = np.empty(self.buffer_size, dtype=np.float32)
        self.cur_id = 0
        self.size = 0

    def reset(self) -> None:
        self.cur_id = 0
        self.size = 0

    def add(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ) -> None:
        index = int(self.cur_id)
        self.history_buf[index] = history
        self.action_buf[index] = action
        self.next_history_buf[index] = next_history
        self.next_state_buf[index] = next_state
        self.next_legal_actions_mask_buf[index] = next_legal_actions_mask
        self.next_player_buf[index] = next_player
        self.done_buf[index] = done
        self.reward_buf[index] = reward
        self.cur_id = (index + 1) % self.buffer_size
        self.size = min(int(self.size) + 1, self.buffer_size)

    def sample(self, num_samples: int = -1):
        length = int(self.size)
        if num_samples < 0 or int(num_samples) >= length:
            indices = slice(0, length)
        else:
            indices = self.rng.choice(length, size=int(num_samples), replace=False)
        float_arrays = (
            self.history_buf[indices],
            self.next_history_buf[indices],
            self.next_state_buf[indices],
            self.reward_buf[indices],
        )
        int_arrays = (
            self.next_legal_actions_mask_buf[indices],
            self.next_player_buf[indices],
            self.action_buf[indices],
            self.done_buf[indices],
        )
        return (
            *(torch.as_tensor(array, device=self.device) for array in float_arrays),
            *(
                torch.as_tensor(array, dtype=torch.int64, device=self.device)
                for array in int_arrays
            ),
        )

    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.history_buf,
                self.action_buf,
                self.next_history_buf,
                self.next_state_buf,
                self.next_legal_actions_mask_buf,
                self.next_player_buf,
                self.done_buf,
                self.reward_buf,
            )
        )

    def __len__(self) -> int:
        return int(self.size)


class CompactCalibrationBuffer:
    """Float32 residual replay with its own vectorised sampler."""

    def __init__(self, capacity: int, feature_size: int, *, seed: int = 0):
        self.capacity = int(capacity)
        self.features = np.empty((capacity, feature_size), dtype=np.float32)
        self.targets = np.empty(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(int(seed))
        self.cursor = 0
        self.size = 0

    def add(self, features, target: float) -> None:
        self.features[self.cursor] = features
        self.targets[self.cursor] = target
        self.cursor = (int(self.cursor) + 1) % self.capacity
        self.size = min(int(self.size) + 1, self.capacity)

    def sample(self, count: int, device: str):
        count = min(int(count), int(self.size))
        indices = self.rng.choice(int(self.size), size=count, replace=False)
        return (
            torch.as_tensor(self.features[indices], device=device),
            torch.as_tensor(self.targets[indices], device=device),
        )

    def nbytes(self) -> int:
        return int(self.features.nbytes + self.targets.nbytes)

    def __len__(self) -> int:
        return int(self.size)
