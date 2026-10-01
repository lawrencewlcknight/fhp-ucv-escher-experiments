"""Experiment 12: privileged showdown comparison for full-state critics only.

FHP ends on the flop, so two hole cards plus the completed three-card board
determine each player's showdown hand. This is strength conditional on reaching
showdown, NOT the realised winner after folding or an action-value target.
No future cards are sampled and no game tree or best response is evaluated.
"""
from __future__ import annotations

import numpy as np

from .features import (
    FHPFeatureEncoder, FeatureLayout, FULL_STATE_LAYOUT, POLICY_LAYOUT,
    _five_card_category,
)
from .hand_board_features import _showdown_tiebreaks


ENCODER_ID = "fhp_lossless_critic_showdown_v1"
FEATURE_NAMES = ("showdown_player0_wins", "showdown_tie", "showdown_player1_wins")
SHOWDOWN_FULL_STATE_LAYOUT = FeatureLayout(
    "critic_full_state_showdown_v1", FULL_STATE_LAYOUT.total_size + len(FEATURE_NAMES),
    FULL_STATE_LAYOUT.card_size,
)


def _hand_rank(hole, board):
    counts = (hole + board).sum(axis=0).astype(np.int8)
    # Reuse only the exact rank calculation from the hand-feature experiment;
    # none of that experiment's policy features are enabled here.
    return (int(np.argmax(_five_card_category(hole, board))),
            *_showdown_tiebreaks(counts))


def critic_showdown_features(hole0, hole1, board) -> np.ndarray:
    """Three float32 indicators in fixed player order; zero before a full flop."""
    if any(channel.shape != (4, 13) for channel in (hole0, hole1, board)):
        raise ValueError("Expected suit-by-rank card channels")
    result = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if int(board.sum()) != 3:
        return result
    if int(hole0.sum()) != 2 or int(hole1.sum()) != 2:
        raise ValueError("Showdown comparison requires two private cards per player")
    rank0, rank1 = _hand_rank(hole0, board), _hand_rank(hole1, board)
    index = 0 if rank0 > rank1 else 2 if rank0 < rank1 else 1
    result[index] = 1
    return result


class FHPCriticShowdownFeatureEncoder(FHPFeatureEncoder):
    """Unchanged policy encoding, with three extra full-state context inputs."""

    encoder_id = ENCODER_ID
    version = 1
    policy_layout = POLICY_LAYOUT
    full_state_layout = SHOWDOWN_FULL_STATE_LAYOUT

    # Inherit information_state and all policy hooks unchanged. In particular,
    # do not derive a policy feature by calling full_state or reading player 1-p.
    def _extra_full_state_features(self, hole0, hole1, board):
        return (critic_showdown_features(hole0, hole1, board),)

    def metadata(self):
        return {
            **super().metadata(),
            "base_encoder_id": FHPFeatureEncoder.encoder_id,
            "additive_features": list(FEATURE_NAMES),
            "feature_scope": "critic_only",
            "policy_inputs_unchanged": True,
            "base_policy_prefix_size": POLICY_LAYOUT.total_size,
            "base_full_state_prefix_size": FULL_STATE_LAYOUT.total_size,
            "full_state_extra_order": "player0_win_tie_player1_win",
            "availability": "all_zero_until_three_public_board_cards",
            "meaning": "five_card_showdown_comparison_not_fold_outcome_or_payoff",
        }
