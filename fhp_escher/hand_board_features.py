"""Additive exact-card features for Experiment 10; no new state merging.

FHP ends after its three-card flop. Straight features describe current rank
structure, never the probability of receiving a turn or river. All policy
features use only the acting player's cards and the public board.
"""
from __future__ import annotations

import numpy as np

from .features import FHPFeatureEncoder, FeatureLayout, POLICY_LAYOUT, FULL_STATE_LAYOUT

ENCODER_ID = "fhp_lossless_hand_board_v2"
FEATURE_NAMES = (
    *(f"showdown_tiebreak_rank_{i}" for i in range(1, 6)),
    "hole_high_rank", "hole_low_rank", "hole_rank_gap",
    "board_high_rank", "board_middle_rank", "board_low_rank",
    "board_unpaired", "board_one_pair", "board_trips",
    "board_rainbow", "board_two_tone", "board_monotone",
    "pocket_overpair", "pocket_underpair", "pocket_between_board_ranks",
    "pocket_rank_on_board",
    "hole_matches_highest_board_rank", "hole_matches_middle_distinct_board_rank",
    "hole_matches_lowest_board_rank", "hole_overcard_fraction",
    "hole_board_matching_fraction", "combined_max_suit_fraction",
    "board_best_straight_window_fraction", "combined_best_straight_window_fraction",
    "combined_longest_rank_run_fraction",
)
NUM_EXTRA_FEATURES = len(FEATURE_NAMES)
HAND_POLICY_LAYOUT = FeatureLayout("player_information_state_hand_board_v2",
                                  POLICY_LAYOUT.total_size + NUM_EXTRA_FEATURES,
                                  POLICY_LAYOUT.card_size)
HAND_FULL_STATE_LAYOUT = FeatureLayout("critic_full_state_hand_board_v2",
                                      FULL_STATE_LAYOUT.total_size + 2 * NUM_EXTRA_FEATURES,
                                      FULL_STATE_LAYOUT.card_size)
# Rank indices 0..12 mean 2..A. Include A2345 explicitly; no wraparound otherwise.
STRAIGHT_WINDOWS = tuple(tuple(range(start, start + 5)) for start in range(9)) + (
    (12, 0, 1, 2, 3),)


def _straight_high(present):
    for start in range(8, -1, -1):
        if all(present[r] for r in range(start, start + 5)):
            return start + 4
    return 3 if all(present[r] for r in (12, 0, 1, 2, 3)) else None


def _showdown_tiebreaks(counts):
    """Five-card ranking tuple *within* the existing hand category.

Flush/high card: five descending ranks; pair: pair then three kickers;
two pair: two pair ranks then kicker; trips: trips then two kickers;
straight: high card (five for a wheel); full house: trips then pair;
quads: quads then kicker. Padding is encoded as zero, below every real rank.
    """
    present = counts > 0
    high = _straight_high(present)
    if high is not None:
        return [high]
    multiplicities = sorted((int(counts[r]), r) for r in range(13) if counts[r])
    multiplicities.reverse()
    # Ordering (multiplicity, rank) gives the correct tuple for every paired hand.
    return [r for _, r in multiplicities]


def hand_board_features(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Thirty bounded float32 features; deterministic, suit-invariant, RNG-free."""
    if hole.shape != (4, 13) or board.shape != (4, 13):
        raise ValueError("Expected suit-by-rank card channels")
    result = np.zeros(NUM_EXTRA_FEATURES, dtype=np.float32)
    hole_counts = hole.sum(axis=0).astype(np.int8)
    hole_ranks = np.repeat(np.arange(13), hole_counts)
    if len(hole_ranks) != 2:
        raise ValueError("Hand features require exactly two private cards")
    low, high = (int(r) for r in hole_ranks)
    result[5:8] = ((high + 2) / 14, (low + 2) / 14, (high - low) / 12)
    # Partial-board chance transitions may be encoded for bookkeeping. No
    # made-hand/board assertion is supplied until all three public cards exist.
    if int(board.sum()) != 3:
        return result
    board_counts = board.sum(axis=0).astype(np.int8)
    counts = hole_counts + board_counts
    tie = _showdown_tiebreaks(counts)
    result[:len(tie)] = [(r + 2) / 14 for r in tie]
    board_ranks = np.repeat(np.arange(13), board_counts)[::-1]
    result[8:11] = (board_ranks + 2) / 14
    distinct = np.flatnonzero(board_counts)
    result[11 + (3 - len(distinct))] = 1  # unpaired / paired / trips
    suit_max = int(board.sum(axis=1).max())
    result[14 + suit_max - 1] = 1  # rainbow / two-tone / monotone
    if low == high:
        relation = (0 if high > distinct[-1] else 1 if high < distinct[0]
                    else 3 if board_counts[high] else 2)
        result[17 + relation] = 1
    result[21] = hole_counts[distinct[-1]] > 0
    result[22] = len(distinct) == 3 and hole_counts[distinct[1]] > 0
    result[23] = hole_counts[distinct[0]] > 0
    result[24] = np.count_nonzero(hole_ranks > distinct[-1]) / 2
    result[25] = np.count_nonzero(board_counts[hole_ranks]) / 2
    result[26] = (hole + board).sum(axis=1).max() / 5
    board_present, present = board_counts > 0, counts > 0
    result[27] = max(sum(board_present[r] for r in w) for w in STRAIGHT_WINDOWS) / 3
    result[28] = max(sum(present[r] for r in w) for w in STRAIGHT_WINDOWS) / 5
    longest = current = 0
    # Ace can be low OR high, never part of a wrapping QKA23 sequence.
    for exists in [present[12], *present]:
        current = current + 1 if exists else 0
        longest = max(longest, current)
    result[29] = longest / 5
    return result


class FHPHandBoardFeatureEncoder(FHPFeatureEncoder):
    encoder_id = ENCODER_ID
    version = 2
    policy_layout = HAND_POLICY_LAYOUT
    full_state_layout = HAND_FULL_STATE_LAYOUT

    def _extra_policy_features(self, hole, board):
        return (hand_board_features(hole, board),)

    def _extra_full_state_features(self, hole0, hole1, board):
        return (hand_board_features(hole0, board), hand_board_features(hole1, board))

    def metadata(self):
        return {**super().metadata(), "base_encoder_id": FHPFeatureEncoder.encoder_id,
                "additive_features": list(FEATURE_NAMES),
                "base_policy_prefix_size": POLICY_LAYOUT.total_size,
                "base_full_state_prefix_size": FULL_STATE_LAYOUT.total_size,
                "full_state_extra_order": "player0_then_player1",
                "rank_normalisation": "rank_value_2_through_14_divided_by_14; missing_zero",
                "postflop_features": "zero_until_three_board_cards; no_future_streets"}
