"""Lossless, symmetry-aware neural features for fixed-limit flop hold'em."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear


ENCODER_ID = "fhp_lossless_suit_canonical_v1"
ENCODER_VERSION = 1
NUM_RANKS = 13
NUM_SUITS = 4
CARDS_PER_DECK = NUM_RANKS * NUM_SUITS
MAX_ACTIONS_PER_ROUND = 5
ACTION_TOKENS = {"c": 0, "r": 1, "f": 2}
_SEQUENCES_RE = re.compile(r"\[Sequences: (.*)\]$")


@dataclass(frozen=True)
class FeatureLayout:
    """Contiguous card and context sections consumed by a structured MLP."""

    name: str
    total_size: int
    card_size: int

    @property
    def context_size(self) -> int:
        return self.total_size - self.card_size

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_size": self.total_size,
            "card_size": self.card_size,
        }


POLICY_LAYOUT = FeatureLayout("player_information_state", 183, 104)
FULL_STATE_LAYOUT = FeatureLayout("critic_full_state", 263, 156)


def _raw_components(state, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = np.asarray(state.information_state_tensor(int(player)), dtype=np.float32)
    if tensor.shape != (190,):
        raise ValueError(f"Expected OpenSpiel FHP tensor of size 190, got {tensor.shape}")
    hole = tensor[2:54].reshape(NUM_RANKS, NUM_SUITS).T.copy()
    board = tensor[54:106].reshape(NUM_RANKS, NUM_SUITS).T.copy()
    return hole, board, tensor


def _raw_cards(state, player: int) -> tuple[np.ndarray, np.ndarray]:
    hole, board, _ = _raw_components(state, player)
    return hole, board


def _canonicalise_suits(*channels: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sort equal-status suits by their rank signatures.

    Poker payoffs are invariant to a global permutation of suit names. Sorting
    the complete per-suit signatures therefore removes only redundant naming,
    not strategically relevant information. Tied signatures are identical, so
    their relative ordering cannot change the returned tensors.
    """

    if not channels:
        raise ValueError("At least one card channel is required")
    if any(channel.shape != (NUM_SUITS, NUM_RANKS) for channel in channels):
        raise ValueError("Card channels must have shape (4, 13)")
    signatures = [
        tuple(float(value) for channel in channels for value in channel[suit])
        for suit in range(NUM_SUITS)
    ]
    order = sorted(range(NUM_SUITS), key=lambda suit: signatures[suit], reverse=True)
    return tuple(channel[order].copy() for channel in channels)


def _betting_sequence(state, player: int) -> str:
    match = _SEQUENCES_RE.search(state.information_state_string(int(player)))
    if match is None:
        raise ValueError("OpenSpiel FHP information-state string has no betting sequence")
    return match.group(1)


def _betting_features(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    rounds = sequence.split("|")
    if len(rounds) > 2:
        raise ValueError(f"FHP has at most two betting rounds: {sequence!r}")
    round_index = len(rounds) - 1
    round_one_hot = np.zeros(2, dtype=np.float32)
    round_one_hot[round_index] = 1.0
    actions = np.zeros((2, MAX_ACTIONS_PER_ROUND, 3), dtype=np.float32)
    for round_id, action_string in enumerate(rounds):
        if len(action_string) > MAX_ACTIONS_PER_ROUND:
            raise ValueError(f"Unexpectedly long FHP betting round: {sequence!r}")
        for index, token in enumerate(action_string):
            if token not in ACTION_TOKENS:
                raise ValueError(f"Unsupported FHP action token {token!r}")
            actions[round_id, index, ACTION_TOKENS[token]] = 1.0
    return round_one_hot, actions.reshape(-1)


def _betting_features_from_tensor(
    tensor: np.ndarray, *, postflop: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Decode reachable decision history without formatting a state string.

    Deal actions and unused padding are both zero in OpenSpiel's generic tensor,
    but the three flop deals create a gap in the otherwise contiguous non-zero
    betting actions. Terminal folds are intentionally omitted: this encoder is
    used for decision states, while terminal critic transitions carry a
    separate `done` flag and never bootstrap from their next-history value.
    """

    pairs = np.asarray(tensor[106:162], dtype=np.float32).reshape(28, 2)
    active = np.flatnonzero(np.any(pairs != 0.0, axis=1))
    split = len(active)
    if postflop and len(active) > 1:
        gaps = np.flatnonzero(np.diff(active) > 1)
        if len(gaps):
            split = int(gaps[0]) + 1
    per_round = (active[:split], active[split:]) if postflop else (active, ())
    actions = np.zeros((2, MAX_ACTIONS_PER_ROUND, 3), dtype=np.float32)
    for round_id, indices in enumerate(per_round):
        if len(indices) > MAX_ACTIONS_PER_ROUND:
            raise ValueError("OpenSpiel FHP betting round exceeds five actions")
        for slot, index in enumerate(indices):
            pair = tuple(int(value) for value in pairs[int(index)])
            token = {(1, 0): 0, (0, 1): 1, (1, 1): 1}.get(pair)
            if token is None:
                raise ValueError(f"Unsupported OpenSpiel FHP action bits: {pair}")
            actions[round_id, slot, token] = 1.0
    round_one_hot = np.asarray(
        [0.0, 1.0] if postflop else [1.0, 0.0], dtype=np.float32
    )
    return round_one_hot, actions.reshape(-1)


def _rank_counts(cards: np.ndarray, divisor: float) -> np.ndarray:
    return cards.sum(axis=0, dtype=np.float32) / float(divisor)


def _suit_counts(cards: np.ndarray, divisor: float) -> np.ndarray:
    return cards.sum(axis=1, dtype=np.float32) / float(divisor)


def _hole_flags(hole: np.ndarray) -> np.ndarray:
    rank_counts = hole.sum(axis=0)
    suit_counts = hole.sum(axis=1)
    return np.asarray(
        [float(np.max(rank_counts) == 2), float(np.max(suit_counts) == 2)],
        dtype=np.float32,
    )


def _five_card_category(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Return an exact one-hot five-card hand category after the flop."""

    result = np.zeros(9, dtype=np.float32)
    cards = hole + board
    if int(cards.sum()) != 5:
        return result
    rank_counts = cards.sum(axis=0).astype(np.int8)
    suit_counts = cards.sum(axis=1).astype(np.int8)
    present = rank_counts > 0
    straight = any(np.all(present[start : start + 5]) for start in range(9))
    straight = bool(straight or np.all(present[[12, 0, 1, 2, 3]]))
    flush = bool(np.max(suit_counts) == 5)
    multiplicities = sorted((int(value) for value in rank_counts if value), reverse=True)
    if straight and flush:
        category = 8
    elif multiplicities[0] == 4:
        category = 7
    elif multiplicities[:2] == [3, 2]:
        category = 6
    elif flush:
        category = 5
    elif straight:
        category = 4
    elif multiplicities[0] == 3:
        category = 3
    elif multiplicities[:2] == [2, 2]:
        category = 2
    elif multiplicities[0] == 2:
        category = 1
    else:
        category = 0
    result[category] = 1.0
    return result


class FHPFeatureEncoder:
    """Exact FHP features modulo the payoff-irrelevant names of the suits."""

    encoder_id = ENCODER_ID
    version = ENCODER_VERSION
    policy_layout = POLICY_LAYOUT
    full_state_layout = FULL_STATE_LAYOUT

    def __init__(self, *, cache_entries: int = 4096):
        self.cache_entries = int(cache_entries)
        if self.cache_entries < 0:
            raise ValueError("cache_entries cannot be negative")
        self._information_cache = OrderedDict()
        self._full_state_cache = OrderedDict()

    def _cached(self, cache, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def _remember(self, cache, key, value):
        if self.cache_entries == 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cache_entries:
            cache.popitem(last=False)
        return value

    @property
    def policy_size(self) -> int:
        return self.policy_layout.total_size

    @property
    def full_state_size(self) -> int:
        return self.full_state_layout.total_size

    def metadata(self) -> dict[str, object]:
        return {
            "id": self.encoder_id,
            "version": self.version,
            "lossless_modulo_suit_symmetry": True,
            "policy_layout": self.policy_layout.to_dict(),
            "full_state_layout": self.full_state_layout.to_dict(),
            "card_order": "channel_suit_rank",
            "betting_encoding": "two_rounds_five_slots_fold_call_raise_one_hot",
        }

    def information_state(self, state, player: int | None = None) -> np.ndarray:
        if player is None:
            player = int(state.current_player())
        player = int(player)
        if player not in (0, 1):
            raise ValueError("A policy information state requires player 0 or 1")
        key = (tuple(state.history()), player)
        cached = self._cached(self._information_cache, key)
        if cached is not None:
            return cached
        raw_hole, raw_board, raw_tensor = _raw_components(state, player)
        hole, board = _canonicalise_suits(raw_hole, raw_board)
        round_one_hot, betting = _betting_features_from_tensor(
            raw_tensor, postflop=bool(raw_board.sum())
        )
        player_one_hot = np.zeros(2, dtype=np.float32)
        player_one_hot[player] = 1.0
        features = np.concatenate(
            [
                hole.reshape(-1),
                board.reshape(-1),
                player_one_hot,
                round_one_hot,
                betting,
                _rank_counts(hole, 2.0),
                _rank_counts(board, 3.0),
                _suit_counts(hole, 2.0),
                _suit_counts(board, 3.0),
                _hole_flags(hole),
                _five_card_category(hole, board),
            ]
        ).astype(np.float32, copy=False)
        if features.shape != (self.policy_size,):
            raise AssertionError(f"Policy feature shape changed: {features.shape}")
        return self._remember(self._information_cache, key, features)

    def full_state(self, state) -> np.ndarray:
        key = tuple(state.history())
        cached = self._cached(self._full_state_cache, key)
        if cached is not None:
            return cached
        hole0, board0, tensor0 = _raw_components(state, 0)
        hole1, board1, _ = _raw_components(state, 1)
        if not np.array_equal(board0, board1):
            raise ValueError("Players disagree about the public board")
        hole0, hole1, board = _canonicalise_suits(hole0, hole1, board0)
        round_one_hot, betting = _betting_features_from_tensor(
            tensor0, postflop=bool(board0.sum())
        )
        acting = np.zeros(2, dtype=np.float32)
        if int(state.current_player()) in (0, 1):
            acting[int(state.current_player())] = 1.0
        features = np.concatenate(
            [
                hole0.reshape(-1),
                hole1.reshape(-1),
                board.reshape(-1),
                acting,
                round_one_hot,
                betting,
                _rank_counts(hole0, 2.0),
                _rank_counts(hole1, 2.0),
                _rank_counts(board, 3.0),
                _suit_counts(hole0, 2.0),
                _suit_counts(hole1, 2.0),
                _suit_counts(board, 3.0),
                _hole_flags(hole0),
                _hole_flags(hole1),
                _five_card_category(hole0, board),
                _five_card_category(hole1, board),
            ]
        ).astype(np.float32, copy=False)
        if features.shape != (self.full_state_size,):
            raise AssertionError(f"Full-state feature shape changed: {features.shape}")
        return self._remember(self._full_state_cache, key, features)


class StructuredFHPMLP(nn.Module):
    """Small CPU-friendly two-branch MLP for cards and betting context."""

    checkpoint_model_type = "structured_fhp_mlp_v1"

    def __init__(
        self,
        layout: FeatureLayout,
        hidden_layers,
        output_size: int,
        *,
        branch_width: int,
    ):
        super().__init__()
        hidden_layers = [int(width) for width in hidden_layers]
        if not hidden_layers or int(branch_width) <= 0:
            raise ValueError("Structured FHP MLP requires positive hidden widths")
        self.layout = layout
        self.input_size = int(layout.total_size)
        self.output_size = int(output_size)
        self.hidden_layers = hidden_layers
        self.branch_width = int(branch_width)
        self.card_encoder = SonnetLinear(
            layout.card_size, self.branch_width, activation=True
        )
        self.context_encoder = SonnetLinear(
            layout.context_size, self.branch_width, activation=True
        )
        widths = [2 * self.branch_width, *hidden_layers]
        self.trunk = nn.Sequential(
            *[
                SonnetLinear(input_width, output_width, activation=True)
                for input_width, output_width in zip(widths[:-1], widths[1:])
            ]
        )
        self.output = ZeroInitLinear(hidden_layers[-1], self.output_size, False)

    def reset_parameters(self) -> None:
        self.card_encoder.reset_parameters()
        self.context_encoder.reset_parameters()
        for layer in self.trunk:
            layer.reset_parameters()
        self.output.reset_parameters()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "type": self.checkpoint_model_type,
            "layout": self.layout.to_dict(),
            "hidden_layers": list(self.hidden_layers),
            "branch_width": self.branch_width,
            "output_size": self.output_size,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cards = inputs[..., : self.layout.card_size]
        context = inputs[..., self.layout.card_size :]
        joined = torch.cat(
            [self.card_encoder(cards), self.context_encoder(context)], dim=-1
        )
        return self.output(self.trunk(joined))


def encoder_from_metadata(metadata: Mapping[str, object] | None):
    if metadata is None:
        return None
    if metadata.get("id") != ENCODER_ID or int(metadata.get("version", -1)) != 1:
        raise ValueError(f"Unsupported FHP feature encoder: {metadata!r}")
    encoder = FHPFeatureEncoder()
    if dict(metadata) != encoder.metadata():
        raise ValueError("Checkpoint FHP encoder metadata differs from the v1 contract")
    return encoder


__all__ = [
    "ENCODER_ID",
    "FHPFeatureEncoder",
    "FULL_STATE_LAYOUT",
    "FeatureLayout",
    "POLICY_LAYOUT",
    "StructuredFHPMLP",
    "encoder_from_metadata",
]
