"""Exact showdown semantics and the privileged-information boundary."""
from itertools import permutations
import random

import numpy as np
import pytest
import torch

from fhp_escher.features import FHPFeatureEncoder, encoder_from_metadata, _raw_cards
from fhp_escher.critic_showdown_features import (
    ENCODER_ID, FEATURE_NAMES, FHPCriticShowdownFeatureEncoder, critic_showdown_features,
)
from fhp_escher.game import load_fhp_game


def card_ids(text):
    return ['23456789TJQKA'.index(c[0]) * 4 + 'cdhs'.index(c[1]) for c in text.split()]


def cards(text):
    values = np.zeros((4, 13), dtype=np.float32)
    for c in card_ids(text):
        values[c % 4, c // 4] = 1
    return values


def flop(hole0, hole1, board):
    state = load_fhp_game().new_initial_state()
    for c in card_ids(hole0 + ' ' + hole1):
        state.apply_action(c)
    state.apply_action(1)
    state.apply_action(1)
    for c in card_ids(board):
        state.apply_action(c)
    assert not state.is_terminal() and not state.is_chance_node()
    return state


def showdown(state):
    state = state.clone()
    while not state.is_terminal():
        assert not state.is_chance_node()  # never draw hypothetical future cards
        state.apply_action(1)
    return state


@pytest.mark.parametrize('hole0,hole1,board,winner', [
    ('Ac Jd', 'Kc Td', '9h 5s 2c', 0),  # high card
    ('8c 8d', '7c 7d', 'Kh 9s 2c', 0),  # pair rank
    ('Ac Kd', 'Ah Qd', 'As Ks 2c', 0),  # two pairs versus one pair
    ('Ac Ad', 'Kc Kd', 'Ah Ks 2c', 0),  # trips rank
    ('Ac 2d', 'Kc Kh', '3h 4s 5c', 0),  # wheel
    ('Ac Kc', 'Ad Ah', 'Qc 9c 2c', 0),  # flush
    ('Ac Ad', 'Kh Qd', 'Ah Ks Kc', 0),  # full house
    ('9c 9d', 'Ac Ad', '9h 9s Kc', 0),  # quads
    ('Ac 2c', 'Kd Kh', '3c 4c 5c', 0),  # straight flush
    ('Ac Jd', 'Ah Td', 'As 9h 2c', 0),  # same pair, different kickers
    ('Ac Kd', 'Ad Kh', 'Qh 9s 2c', 1),  # tie
    ('Ac 2d', '2c 6d', '3h 4s 5c', 2),  # wheel loses to six-high straight
])
def test_showdown_indicator_matches_independent_game_returns(hole0, hole1, board, winner):
    values = critic_showdown_features(cards(hole0), cards(hole1), cards(board))
    np.testing.assert_array_equal(values, np.eye(3, dtype=np.float32)[winner])
    terminal = showdown(flop(hole0, hole1, board))
    assert np.sign(terminal.returns()[0]) == (1, 0, -1)[winner]


def test_random_complete_deals_match_openspiel_and_keep_original_inputs():
    game = load_fhp_game()
    rng = np.random.default_rng(12012)
    old, new = FHPFeatureEncoder(), FHPCriticShowdownFeatureEncoder()
    for _ in range(500):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(int(rng.choice(state.legal_actions())))
            else:
                for player in (0, 1):
                    np.testing.assert_array_equal(new.information_state(state, player),
                                                  old.information_state(state, player))
                full = new.full_state(state)
                assert full.shape == (266,) and full.dtype == np.float32
                np.testing.assert_array_equal(full[:263], old.full_state(state))
                _, board = _raw_cards(state, 0)
                if int(board.sum()) != 3:
                    np.testing.assert_array_equal(full[263:], [0, 0, 0])
                else:
                    payoff = showdown(state).returns()[0]
                    expected = [float(payoff > 0), float(payoff == 0), float(payoff < 0)]
                    np.testing.assert_array_equal(full[263:], expected)
                state.apply_action(1)


@pytest.mark.parametrize('board', ['', '3h', '3h 4s'])
def test_unavailable_is_not_encoded_as_a_tie(board):
    values = critic_showdown_features(cards('Ac 2d'), cards('Kc Kh'), cards(board))
    np.testing.assert_array_equal(values, np.zeros(3))


def test_features_describe_showdown_not_fold_winner():
    state = flop('Kc Kd', 'Ac Ad', '9h 5s 2c')
    encoder = FHPCriticShowdownFeatureEncoder()
    state.apply_action(1)  # Player 1 checks.
    state.apply_action(2)  # Player 0 bets.
    state.apply_action(0)  # Player 1 folds the stronger hand.
    assert state.is_terminal() and state.returns()[0] > 0
    np.testing.assert_array_equal(encoder.full_state(state)[263:], [0, 0, 1])


def test_suit_invariance_player_swap_and_rng_preservation():
    a, b, board = cards('Ac 2c'), cards('Kd Kh'), cards('3c 4c 5c')
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    for permutation in permutations(range(4)):
        order = list(permutation)
        np.testing.assert_array_equal(critic_showdown_features(a[order], b[order], board[order]),
                                      [1, 0, 0])
        np.testing.assert_array_equal(critic_showdown_features(b[order], a[order], board[order]),
                                      [0, 0, 1])
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.random.get_rng_state(), before[2])


class OwnInformationOnly:
    """Policy inference fails if any hidden opponent tensor is requested."""
    def __init__(self, state, player):
        self.state, self.player = state, player

    def current_player(self):
        return self.state.current_player()

    def history(self):
        return self.state.history()

    def information_state_tensor(self, player):
        assert player == self.player, "Opponent's actual cards leaked to policy inputs"
        return self.state.information_state_tensor(player)

    def legal_actions(self, player=None):
        return self.state.legal_actions(self.player if player is None else player)

    def legal_actions_mask(self, player=None):
        return self.state.legal_actions_mask(self.player if player is None else player)


@pytest.mark.parametrize('player', [0, 1])
def test_opponent_cards_cannot_change_information_set_network_inputs(player, tmp_path):
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _make_solver
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import smoke_config
    from fhp_escher.checkpointing import save_policy_checkpoint, LoadedFHPPolicy

    old_config = smoke_config()
    config = dict(old_config, feature_encoder_id=ENCODER_ID)
    old, new = _make_solver(0, old_config), _make_solver(0, config)
    assert new.infostate_size == old.infostate_size == 183
    assert new.ave_policy_trainer.model.layout == old.ave_policy_trainer.model.layout
    assert new.calibration_trainer.model.input_size == old.calibration_trainer.model.input_size == 189
    assert all(t.model.input_size == 183 for t in new.regret_trainers)
    assert all(m.input_size == 266 and m.state_size == 183 for m in new.q_value_trainer.members)
    assert new.q_value_trainer.ensemble_size == 2

    def decision(opponent):
        holes = ['Kc Kd', opponent] if player == 0 else [opponent, 'Kc Kd']
        state = flop(*holes, '9h 5s 2c')
        if player == 0:
            state.apply_action(1)
        assert state.current_player() == player
        return state

    a, b = decision('Qc Qd'), decision('Ac Ad')
    assert a.information_state_string(player) == b.information_state_string(player)
    assert not np.array_equal(new.get_history_tensor(a)[263:], new.get_history_tensor(b)[263:])

    # Deliberately make the critic depend on the privileged comparison. With
    # two folds exactly ONE held-out critic supplies the control, so its
    # ensemble variance is zero; calibration cannot learn the comparison via
    # a critic-disagreement input either. Residual TARGETS still change as
    # intended; they are not information supplied to the calibration model.
    for member in new.q_value_trainer.members:
        member.get_baseline = lambda s, p: np.repeat(
            np.dot(new.get_history_tensor(s)[263:], [1, 0, -1]), 3)
    assert not np.array_equal(new.q_value_trainer.get_baseline(a, player),
                              new.q_value_trainer.get_baseline(b, player))
    for fold in (0, 1):
        new.q_value_trainer.active_fold = fold
        features = []
        for state in (a, b):
            view = OwnInformationOnly(state, player)
            expected = old.get_infostate_tensor(view, player)
            np.testing.assert_array_equal(new.get_infostate_tensor(view, player), expected)
            np.testing.assert_array_equal(new.regret_trainers[player].get_infostate_tensor(view), expected)
            np.testing.assert_array_equal(new.ave_policy_trainer.get_infostate_tensor(view), expected)
            _, disagreement = new.q_value_trainer.get_baseline_and_disagreement(state, player)
            np.testing.assert_array_equal(disagreement, np.zeros(3))
            _, _, inputs = new.calibration_trainer.predict_all(expected, 7, disagreement, player)
            features.append(inputs)
        np.testing.assert_array_equal(features[0], features[1])

    # Grouping and replay objectives remain identical, not merely their sizes.
    for i, state in enumerate((a, b, a, b)):
        mask = np.asarray(state.legal_actions_mask(), np.float32)
        for solver in (old, new):
            trainer = solver.ave_policy_trainer
            trainer.buffer.add(trainer.get_infostate_tensor(state), mask / mask.sum(), mask, i + 1)
    for x, y in zip(old.ave_policy_trainer._grouped_training_data(4),
                    new.ave_policy_trainer._grouped_training_data(4)):
        assert torch.equal(x, y)

    path = save_policy_checkpoint(new, tmp_path / 'policy.pkl', seed=0, config=config,
                                 checkpoint_row={})
    loaded = LoadedFHPPolicy(load_fhp_game(), path)
    assert loaded.model.input_size == 183
    # Clear caches so the guarded call really exercises encoding, not a cache hit.
    loaded.feature_encoder._information_cache.clear()
    probs_a = loaded.action_probabilities(OwnInformationOnly(a, player))
    probs_b = loaded.action_probabilities(OwnInformationOnly(b, player))
    assert probs_a == probs_b


def test_versioned_metadata_fails_closed():
    for encoder in (FHPFeatureEncoder(), FHPCriticShowdownFeatureEncoder()):
        assert encoder_from_metadata(encoder.metadata()).metadata() == encoder.metadata()
    wrong = FHPCriticShowdownFeatureEncoder().metadata()
    wrong['additive_features'] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match='contract'):
        encoder_from_metadata(wrong)
