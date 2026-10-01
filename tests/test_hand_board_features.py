"""Information preservation, exact poker semantics and old-checkpoint compatibility."""
from itertools import permutations

import numpy as np
import pytest
import torch

from fhp_escher.features import FHPFeatureEncoder, encoder_from_metadata, _raw_cards, _five_card_category
from fhp_escher.hand_board_features import (
    ENCODER_ID, FEATURE_NAMES, FHPHandBoardFeatureEncoder, hand_board_features,
)
from fhp_escher.game import load_fhp_game


def cards(text):
    result = np.zeros((4, 13), np.float32)
    for token in text.split():
        result['cdhs'.index(token[1]), '23456789TJQKA'.index(token[0])] = 1
    return result


@pytest.mark.parametrize('hole,board,tie', [
    ('Ac Kd', 'Qh 9s 2c', [14, 13, 12, 9, 2]),  # high card
    ('Ac Ad', 'Kh 9s 2c', [14, 13, 9, 2]),
    ('Ac Kd', 'Ah Ks 2c', [14, 13, 2]),
    ('Ac Ad', 'Ah Ks 2c', [14, 13, 2]),
    ('Ac 2d', '3h 4s 5c', [5]),  # wheel
    ('Tc Jd', 'Qh Ks Ac', [14]),
    ('Ac Kc', 'Qc 9c 2c', [14, 13, 12, 9, 2]),  # flush
    ('Ac Ad', 'Ah Ks Kc', [14, 13]),
    ('Ac Ad', 'Ah As Kc', [14, 13]),
    ('Ac 2c', '3c 4c 5c', [5]),
])
def test_exact_showdown_tiebreaks(hole, board, tie):
    values = hand_board_features(cards(hole), cards(board))
    expected = np.zeros(5, np.float32)
    expected[:len(tie)] = np.asarray(tie) / 14
    np.testing.assert_array_equal(values[:5], expected)
    assert len(values) == len(FEATURE_NAMES) == 30
    assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))


@pytest.mark.parametrize('hole,board,names', [
    ('Ac Ad', 'Kh 9s 2c', ['pocket_overpair', 'board_unpaired', 'board_rainbow']),
    ('2c 2d', 'Kh 9h 3h', ['pocket_underpair', 'board_monotone']),
    ('8c 8d', 'Kh 9s 2c', ['pocket_between_board_ranks']),
    ('Kc Kd', 'Kh 9s 2c', ['pocket_rank_on_board', 'hole_matches_highest_board_rank']),
    ('9c Ad', 'Kh 9s 2c', ['hole_matches_middle_distinct_board_rank']),
    ('2h Ad', 'Kh 9s 2c', ['hole_matches_lowest_board_rank']),
    ('Ac Jd', 'Kh Ks 2h', ['board_one_pair', 'board_two_tone']),
    ('Ac Jd', 'Kh Ks Kc', ['board_trips']),
])
def test_board_relationships(hole, board, names):
    values = dict(zip(FEATURE_NAMES, hand_board_features(cards(hole), cards(board))))
    for name in names:
        assert values[name] == 1, name


def test_preflop_missing_board_and_no_wrapping_straight():
    values = hand_board_features(cards('Ac 2d'), cards(''))
    assert np.count_nonzero(values[:5]) == 0 and np.count_nonzero(values[8:]) == 0
    np.testing.assert_allclose(values[5:8], [1, 2/14, 1])
    wrapping = hand_board_features(cards('Ac 2d'), cards('3h Ks Qc'))
    # QKA23 is high card, not a straight. Wheel rank-run is A23, not a wrap of five.
    np.testing.assert_allclose(wrapping[:5], np.asarray([14,13,12,3,2])/14)
    assert wrapping[-1] == np.float32(3/5)


def decision_states(count=100):
    game = load_fhp_game()
    rng = np.random.default_rng(492)
    for _ in range(count):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probabilities = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probabilities)))
            else:
                yield state.clone()
                state.apply_action(1)  # check/call to reach showdown


def test_existing_information_is_a_bitwise_unchanged_prefix():
    old, new = FHPFeatureEncoder(), FHPHandBoardFeatureEncoder()
    for state in decision_states(50):
        for player in (0, 1):
            features = new.information_state(state, player)
            assert features.shape == (213,)
            np.testing.assert_array_equal(features[:183], old.information_state(state, player))
            hole, board = _raw_cards(state, player)
            np.testing.assert_array_equal(features[183:], hand_board_features(hole, board))
        full = new.full_state(state)
        assert full.shape == (323,)
        np.testing.assert_array_equal(full[:263], old.full_state(state))
        for player in (0,1):
            hole, board = _raw_cards(state, player)
            np.testing.assert_array_equal(full[263+30*player:293+30*player], hand_board_features(hole, board))


def test_all_suit_permutations_and_card_input_order_are_invariant():
    rng = np.random.default_rng(21)
    for _ in range(100):
        ids = rng.choice(52, 5, replace=False)
        hole, board = np.zeros((4,13), np.float32), np.zeros((4,13), np.float32)
        for i,c in enumerate(ids):
            (hole if i<2 else board)[c%4,c//4] = 1
        expected = hand_board_features(hole, board)
        for order in permutations(range(4)):
            np.testing.assert_array_equal(expected, hand_board_features(hole[list(order)], board[list(order)]))


def test_hand_order_agrees_with_openspiel_showdown():
    # Independent payoff oracle: random full deals, no folds/raises, equal stakes.
    rng = np.random.default_rng(794)
    game = load_fhp_game()
    for _ in range(250):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probabilities = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probabilities)))
            else:
                state.apply_action(1)
        ranks = []
        for player in (0,1):
            hole, board = _raw_cards(state, player)
            ranks.append((int(np.argmax(_five_card_category(hole, board))),
                          *hand_board_features(hole, board)[:5]))
        expected = int(ranks[0] > ranks[1]) - int(ranks[0] < ranks[1])
        assert expected == np.sign(state.returns()[0])


def test_versioned_encoder_roundtrip_fails_closed():
    for encoder in (FHPFeatureEncoder(), FHPHandBoardFeatureEncoder()):
        assert encoder_from_metadata(encoder.metadata()).metadata() == encoder.metadata()
    wrong = FHPHandBoardFeatureEncoder().metadata()
    wrong['additive_features'] = list(reversed(wrong['additive_features']))
    with pytest.raises(ValueError, match='contract'):
        encoder_from_metadata(wrong)
    with pytest.raises(ValueError, match='Unsupported'):
        encoder_from_metadata({'id': 'unknown'})


def test_grouping_targets_and_old_network_contract_are_unchanged():
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _make_solver
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import smoke_config
    old = _make_solver(0, smoke_config())
    new = _make_solver(0, dict(smoke_config(), feature_encoder_id=ENCODER_ID))
    assert old.infostate_size == 183 and new.infostate_size == 213
    assert new.q_value_trainer.members[0].input_size == 323
    assert new.calibration_trainer.model.layout.card_size == 104
    assert new.calibration_trainer.model.input_size == old.calibration_trainer.model.input_size + 30
    states = list(decision_states(4))
    for i,state in enumerate(states + states):
        mask = np.asarray(state.legal_actions_mask(), np.float32)
        target = mask / mask.sum()
        for solver in (old, new):
            trainer = solver.ave_policy_trainer
            trainer.buffer.add(trainer.get_infostate_tensor(state), target, mask, i%3+1)
    a = old.ave_policy_trainer._grouped_training_data(3)
    b = new.ave_policy_trainer._grouped_training_data(3)
    assert torch.equal(a[0], b[0][:,:183])
    for left,right in zip(a[1:],b[1:]):
        assert torch.equal(left,right)
