"""Additive Experiment 29--33 encoders, preserving the full Experiment 10 prefix."""
import numpy as np

from .features import FHPFeatureEncoder, FeatureLayout
from .hand_board_features import FHPHandBoardFeatureEncoder, HAND_POLICY_LAYOUT, HAND_FULL_STATE_LAYOUT
from .betting_economics_features import betting_economics_features
from .relative_strength import (OVERALL_NAMES, CLASS_NAMES, CLASS_FEATURE_NAMES, BLOCKER_NAMES,
                                relative_features, card_ids)
from .preflop_strength import FEATURE_NAMES as PREFLOP_NAMES, load_table, starting_indices

INTERACTION_NAMES = ("pot_bb", "call_cost_bb", "pot_odds", "raises_remaining_fraction",
                     "acts_last", "next_to_act", "call_closes_final_round",
                     "equity_minus_pot_odds", "equity_times_pot_odds",
                     "equity_times_closes", "equity_times_raises", "equity_times_position")
PACKAGES = {"A": OVERALL_NAMES, "AB": OVERALL_NAMES + CLASS_FEATURE_NAMES,
            "AC": OVERALL_NAMES + BLOCKER_NAMES, "D": PREFLOP_NAMES,
            "AE": OVERALL_NAMES + INTERACTION_NAMES}
ENCODER_IDS = {kind: f"fhp_hand_board_strategic_{kind.lower()}_v1" for kind in PACKAGES}


class StrategicFeatureEncoder(FHPFeatureEncoder):
    version = 1

    def __init__(self, kind, *, cache_entries=4096):
        if kind not in PACKAGES:
            raise ValueError(f"Unknown strategic feature package {kind}")
        super().__init__(cache_entries=cache_entries)
        self.kind, self.encoder_id = kind, ENCODER_IDS[kind]
        self.policy_layout = FeatureLayout(f"strategic_{kind}_player_v1",
            HAND_POLICY_LAYOUT.total_size + len(PACKAGES[kind]), HAND_POLICY_LAYOUT.card_size)
        self.full_state_layout = FeatureLayout(f"strategic_{kind}_critic_v1",
            HAND_FULL_STATE_LAYOUT.total_size + 2 * len(PACKAGES[kind]), HAND_FULL_STATE_LAYOUT.card_size)
        self._base = FHPHandBoardFeatureEncoder(cache_entries=0)
        if kind == "D":
            load_table()  # fail before allocating replay if the versioned asset is missing

    def _additional(self, hole, board, player, round_one_hot, betting):
        if self.kind == "D":
            if np.any(board):
                return np.zeros(len(PREFLOP_NAMES), np.float32)
            return load_table()[0][int(starting_indices(np.asarray([card_ids(hole)]))[0])]
        overall, classes, blockers = relative_features(hole, board)
        if self.kind == "A":
            return overall
        if self.kind == "AB":
            return np.concatenate((overall, classes))
        if self.kind == "AC":
            return np.concatenate((overall, blockers))
        interaction = np.zeros(len(INTERACTION_NAMES), np.float32)
        if overall[0]:
            e = betting_economics_features(player, round_one_hot, betting)
            # On the last round, a check/call closes betting iff a prior flop
            # action exists and this is the next actor. Preflop blind option is
            # deliberately excluded; off-turn critic perspective is masked.
            closes = bool(e[8] and np.any(betting.reshape(2, 5, 3)[1]))
            equity = overall[4]
            interaction[:] = (e[0], e[1], e[2], e[5] / 3, e[7], e[8], closes,
                equity - e[2], equity * e[2], equity * closes, equity * e[5] / 3,
                equity * e[7])
        return np.concatenate((overall, interaction))

    def information_state(self, state, player=None):
        player = int(state.current_player()) if player is None else int(player)
        key = (tuple(state.history()), player)
        cached = self._cached(self._information_cache, key)
        if cached is not None:
            return cached
        base = self._base.information_state(state, player)
        result = np.concatenate((base, self._additional(
            base[:52].reshape(4, 13), base[52:104].reshape(4, 13), player,
            base[106:108], base[108:138]))).astype(np.float32, copy=False)
        return self._remember(self._information_cache, key, result)

    def full_state(self, state):
        key = tuple(state.history())
        cached = self._cached(self._full_state_cache, key)
        if cached is not None:
            return cached
        base = self._base.full_state(state)
        board = base[104:156].reshape(4, 13)
        extras = [self._additional(base[p*52:(p+1)*52].reshape(4, 13), board, p,
                                  base[158:160], base[160:190]) for p in (0, 1)]
        result = np.concatenate((base, *extras)).astype(np.float32, copy=False)
        return self._remember(self._full_state_cache, key, result)

    def metadata(self):
        result = {**super().metadata(), "base_encoder_id": FHPHandBoardFeatureEncoder.encoder_id,
            "base_policy_prefix_size": HAND_POLICY_LAYOUT.total_size,
            "base_full_state_prefix_size": HAND_FULL_STATE_LAYOUT.total_size,
            "feature_package": self.kind, "additive_features": list(PACKAGES[self.kind]),
            "full_state_extra_order": "player0_then_player1; own-plus-public-only per perspective",
            "classes": list(CLASS_NAMES) if self.kind in ("AB", "AC") else [],
            "opponent_prior": "uniform legal holdings, NOT action-conditioned or actual hidden cards",
            "board_table_cache_entries_per_process": 64,
            "unavailable_features": "zero; availability indicator distinguishes missing from zero strength",
            "blocker_reference": "1176 board-compatible holdings before own-card removal; 1081 afterwards",
            "grouping": "unchanged exact canonical information sets; no lossy buckets"}
        if self.kind == "D":
            result["preflop_table_sha256"] = load_table()[1]
            result["preflop_table_method"] = "exact full flop enumeration, suit-orbit weighted; no Monte Carlo"
        return result
