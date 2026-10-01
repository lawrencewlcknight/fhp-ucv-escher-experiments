"""Experiment 11: additive public betting economics, not card abstractions.

Derived solely from the existing encoded betting history. The bounded cache
shares values across all deals with the same public betting sequence. No
state-string parsing, game-tree search or floating-point chip accumulation is
performed in the per-state path.
"""
from functools import lru_cache

import numpy as np

from .features import FHPFeatureEncoder, FeatureLayout, POLICY_LAYOUT, FULL_STATE_LAYOUT
from .game import FHP_GAME_PARAMETERS

ENCODER_ID = "fhp_lossless_betting_economics_v1"
FEATURE_NAMES = (
    "pot_bb", "call_cost_bb", "immediate_pot_odds",
    "own_round_contribution_bb", "opponent_round_contribution_bb",
    "raises_remaining",
    "is_button_small_blind", "acts_last_this_round", "is_next_to_act",
    "facing_call", "last_action_was_raise",
    "last_aggressor_self_this_round", "last_aggressor_opponent_this_round",
    "last_aggressor_self_previous_round", "last_aggressor_opponent_previous_round",
    "last_action_was_opponent_check",
)
NUM_EXTRA_FEATURES = len(FEATURE_NAMES)
BETTING_POLICY_LAYOUT = FeatureLayout("player_information_state_betting_economics_v1",
    POLICY_LAYOUT.total_size + NUM_EXTRA_FEATURES, POLICY_LAYOUT.card_size)
BETTING_FULL_STATE_LAYOUT = FeatureLayout("critic_full_state_betting_economics_v1",
    FULL_STATE_LAYOUT.total_size + 2 * NUM_EXTRA_FEATURES, FULL_STATE_LAYOUT.card_size)
BLINDS = tuple(map(int, FHP_GAME_PARAMETERS["blind"].split()))
BIG_BLIND = max(BLINDS)
RAISE_SIZES = tuple(map(int, FHP_GAME_PARAMETERS["raiseSize"].split()))
FIRST_PLAYERS = tuple(int(x) - 1 for x in FHP_GAME_PARAMETERS["firstPlayer"].split())
MAX_RAISES = tuple(map(int, FHP_GAME_PARAMETERS["maxRaises"].split()))


@lru_cache(maxsize=512)
def _economics_by_history(round_id: int, betting_bytes: bytes):
    """Both seat perspectives; inputs are already represented by the v1 prefix.

    Betting slots encode check/call or raise. As in v1, terminal folds are omitted;
    decision-state features are exact and terminal critic targets are masked by
    the existing `done` flag. Preflop round contributions include posted blinds.
    """
    if round_id not in (0, 1) or len(betting_bytes) != 30 * 4:
        raise ValueError("Invalid fixed-FHP round/history encoding")
    history = np.frombuffer(betting_bytes, dtype=np.float32).reshape(2, 5, 3)
    spent = list(BLINDS)
    previous_aggressor = None
    for street in range(round_id + 1):
        start_spent = [0, 0] if street == 0 else spent.copy()
        actor = FIRST_PLAYERS[street]
        raises = 0
        aggressor = last_actor = None
        last_raise = last_check = False
        found_padding = False
        for slot in history[street]:
            if not np.any(slot):
                found_padding = True
                continue
            if found_padding or not (np.array_equal(slot, (1, 0, 0))
                                     or np.array_equal(slot, (0, 1, 0))):
                raise ValueError("Expected contiguous check/call or raise slots")
            cost = max(spent) - spent[actor]
            last_actor = actor
            last_raise = bool(slot[1])
            last_check = not last_raise and cost == 0
            if last_raise:
                spent[actor] = max(spent) + RAISE_SIZES[street]
                raises += 1
                aggressor = actor
            else:
                spent[actor] += cost
            actor = 1 - actor
        if raises > MAX_RAISES[street]:
            raise ValueError("History exceeds fixed-FHP raise cap")
        if street < round_id:
            previous_aggressor = aggressor
    pot = sum(spent)
    features = np.zeros((2, NUM_EXTRA_FEATURES), dtype=np.float32)
    for player in (0, 1):
        opponent = 1 - player
        call = max(0, spent[opponent] - spent[player])
        features[player] = (
            pot / BIG_BLIND, call / BIG_BLIND, call / (pot + call),
            (spent[player] - start_spent[player]) / BIG_BLIND,
            (spent[opponent] - start_spent[opponent]) / BIG_BLIND,
            MAX_RAISES[round_id] - raises,
            player == 0, player != FIRST_PLAYERS[round_id], player == actor,
            call > 0, last_raise,
            aggressor == player, aggressor == opponent,
            previous_aggressor == player, previous_aggressor == opponent,
            last_actor == opponent and last_check,
        )
    # Callers may share this cached array: make accidental mutation fail loudly.
    features.flags.writeable = False
    return features


def betting_economics_features(player, round_one_hot, betting):
    if player not in (0, 1):
        raise ValueError("Betting features require player 0 or 1")
    round_one_hot = np.asarray(round_one_hot)
    if not (np.array_equal(round_one_hot, (1, 0)) or np.array_equal(round_one_hot, (0, 1))):
        raise ValueError("Expected one-hot preflop/flop round")
    encoded = np.asarray(betting, dtype=np.float32)
    if encoded.shape != (30,):
        raise ValueError("Expected thirty exact betting-history features")
    return _economics_by_history(int(round_one_hot[1]), encoded.tobytes())[player]


class FHPBettingEconomicsFeatureEncoder(FHPFeatureEncoder):
    encoder_id = ENCODER_ID
    version = 1
    policy_layout = BETTING_POLICY_LAYOUT
    full_state_layout = BETTING_FULL_STATE_LAYOUT

    def _extra_policy_context(self, player, round_one_hot, betting):
        return (betting_economics_features(player, round_one_hot, betting),)

    def _extra_full_state_context(self, round_one_hot, betting):
        return tuple(betting_economics_features(p, round_one_hot, betting) for p in (0, 1))

    def metadata(self):
        return {**super().metadata(), "base_encoder_id": FHPFeatureEncoder.encoder_id,
                "additive_features": list(FEATURE_NAMES),
                "base_policy_prefix_size": POLICY_LAYOUT.total_size,
                "base_full_state_prefix_size": FULL_STATE_LAYOUT.total_size,
                "full_state_extra_order": "player0_then_player1",
                "big_blind_chips": BIG_BLIND,
                "pot_definition": "sum_of_committed_chips_before_call",
                "pot_odds_definition": "call_cost_divided_by_pot_plus_call_cost",
                "round_contributions": "include_blinds_preflop; reset_to_zero_on_flop",
                "raises_remaining": "raw_count; posted_blind_does_not_consume_raise",
                "aggression_window": "current_round_and_previous_round_separately",
                "position": "seat0_button_small_blind; acts_first_preflop_last_flop",
                "card_features": "unchanged_v1; no_experiment10_descriptors"}
