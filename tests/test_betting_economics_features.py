"""Check every public betting sequence against independent OpenSpiel accounting."""
import re

import numpy as np
import pytest
import torch

from fhp_escher.features import FHPFeatureEncoder, encoder_from_metadata
from fhp_escher.betting_economics_features import (
    ENCODER_ID, FEATURE_NAMES, FHPBettingEconomicsFeatureEncoder,
    betting_economics_features, _economics_by_history,
)
from fhp_escher.game import load_fhp_game


def spent(state):
    # Independent simulator oracle, used only in tests, never training features.
    match = re.search(r"Spent: \[P0: (\d+)\s+P1: (\d+)\s*\]", str(state))
    assert match
    return np.asarray([int(x) for x in match.groups()])


def deal(state):
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    return state


def initial():
    return deal(load_fhp_game().new_initial_state())


def features(state, player):
    values = FHPBettingEconomicsFeatureEncoder().information_state(state, player)[183:]
    return dict(zip(FEATURE_NAMES, values))


def test_initial_blinds_pot_and_odds_are_not_the_displayed_pot():
    s = initial()
    f = features(s, 0)
    assert '[Pot: 200]' in s.information_state_string(0)
    assert f['pot_bb'] == 1.5  # 150 committed, NOT 200 including the call.
    assert f['call_cost_bb'] == .5 and f['immediate_pot_odds'] == .25
    assert f['own_round_contribution_bb'] == .5
    assert f['opponent_round_contribution_bb'] == 1
    assert f['raises_remaining'] == 3
    assert f['is_button_small_blind'] == f['is_next_to_act'] == 1
    assert f['acts_last_this_round'] == 0
    other = features(s, 1)
    assert other['call_cost_bb'] == other['immediate_pot_odds'] == 0
    assert other['is_next_to_act'] == 0


def test_check_call_and_previous_round_aggression_are_distinct():
    s = initial()
    s.apply_action(1)  # small blind completes; this is NOT a check.
    assert features(s, 1)['last_action_was_opponent_check'] == 0
    s.apply_action(2)  # big blind raises.
    f = features(s, 0)
    assert f['last_action_was_raise'] == f['last_aggressor_opponent_this_round'] == 1
    assert f['call_cost_bb'] == 1 and f['raises_remaining'] == 2
    s.apply_action(1)
    deal(s)
    f = features(s, 1)
    assert f['own_round_contribution_bb'] == f['opponent_round_contribution_bb'] == 0
    assert f['last_aggressor_self_previous_round'] == 1
    assert f['last_action_was_raise'] == f['last_aggressor_self_this_round'] == 0
    assert f['raises_remaining'] == 3
    assert f['is_next_to_act'] == 1 and f['acts_last_this_round'] == 0
    s.apply_action(1)  # postflop check by big blind.
    f = features(s, 0)
    assert f['last_action_was_opponent_check'] == f['acts_last_this_round'] == 1


def test_exhaustive_betting_tree_matches_simulator_and_preserves_v1_prefix():
    old, new = FHPFeatureEncoder(), FHPBettingEconomicsFeatureEncoder()
    seen = set()

    def visit(state, round_start):
        deal(state)
        if state.is_terminal():
            # Full-state next-history tensors must remain finite even at folds.
            assert np.isfinite(new.full_state(state)).all()
            return
        actor = state.current_player()
        seq = state.information_state_string(actor).split('[Sequences: ')[1].removesuffix(']')
        seen.add(seq)
        rounds = seq.split('|')
        street = len(rounds) - 1
        actual = spent(state)
        for player in (0, 1):
            values = new.information_state(state, player)
            np.testing.assert_array_equal(values[:183], old.information_state(state, player))
            f = dict(zip(FEATURE_NAMES, values[183:]))
            assert values.shape == (199,) and values.dtype == np.float32
            assert np.isfinite(values).all()
            assert f['pot_bb'] == actual.sum() / 100
            call = max(0, actual[1-player] - actual[player])
            assert f['call_cost_bb'] == call / 100
            assert f['immediate_pot_odds'] == np.float32(call / (actual.sum() + call))
            assert f['own_round_contribution_bb'] == (actual[player] - round_start[player]) / 100
            assert f['opponent_round_contribution_bb'] == (actual[1-player] - round_start[1-player]) / 100
            assert f['raises_remaining'] == 3 - rounds[-1].count('r')
            assert f['is_next_to_act'] == (player == actor)
            assert f['acts_last_this_round'] == (player != (0,1)[street])
            assert f['facing_call'] == (call > 0)
        af = features(state, actor)
        called = state.child(1)
        assert spent(called)[actor] - actual[actor] == af['call_cost_bb'] * 100
        assert (af['raises_remaining'] > 0) == (2 in state.legal_actions())
        full = new.full_state(state)
        assert full.shape == (295,)
        np.testing.assert_array_equal(full[:263], old.full_state(state))
        for player in (0,1):
            np.testing.assert_array_equal(full[263+16*player:279+16*player],
                                          new.information_state(state, player)[183:])
        for action in state.legal_actions():
            child = state.child(action)
            next_start = spent(child) if child.is_chance_node() else round_start
            visit(child, next_start)

    visit(initial(), np.zeros(2, dtype=int))
    assert len(seen) > 50


def test_private_cards_cannot_affect_betting_descriptors():
    game = load_fhp_game()
    rng = np.random.default_rng(567)
    new = FHPBettingEconomicsFeatureEncoder()
    reference = {}
    for _ in range(30):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(int(rng.choice(state.legal_actions())))
            else:
                for p in (0,1):
                    key = (p, state.information_state_string(p).split('[Sequences:')[1])
                    values = new.information_state(state, p)[183:]
                    np.testing.assert_array_equal(reference.setdefault(key, values), values)
                state.apply_action(1)


def test_cache_is_bounded_readonly_and_rng_free():
    _economics_by_history.cache_clear()
    state = np.random.get_state()
    history = np.zeros(30, np.float32)
    first = betting_economics_features(0, [1,0], history)
    second = betting_economics_features(0, [1,0], history)
    np.testing.assert_array_equal(first, second)
    assert _economics_by_history.cache_info().hits == 1
    assert _economics_by_history.cache_info().maxsize == 512
    with pytest.raises(ValueError):
        first[0] = 999
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


def test_versions_are_separate_and_fail_closed():
    from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
    for encoder in (FHPFeatureEncoder(), FHPHandBoardFeatureEncoder(), FHPBettingEconomicsFeatureEncoder()):
        assert encoder_from_metadata(encoder.metadata()).metadata() == encoder.metadata()
    wrong = FHPBettingEconomicsFeatureEncoder().metadata()
    wrong['pot_definition'] = 'pot_including_call'
    with pytest.raises(ValueError, match='contract'):
        encoder_from_metadata(wrong)
    assert 'showdown_tiebreak_rank_1' not in FEATURE_NAMES


def test_grouping_and_all_networks_use_only_the_new_context_features():
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _make_solver
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import smoke_config
    old = _make_solver(0, smoke_config())
    new = _make_solver(0, dict(smoke_config(), feature_encoder_id=ENCODER_ID))
    assert old.infostate_size == 183 and new.infostate_size == 199
    assert new.q_value_trainer.members[0].input_size == 295
    assert new.regret_trainers[0].input_size == new.ave_policy_trainer.input_size == 199
    assert new.calibration_trainer.model.input_size == old.calibration_trainer.model.input_size + 16
    for i in range(12):
        s = initial()
        for _ in range(i % 3):
            s.apply_action(1)
            deal(s)
        mask = np.asarray(s.legal_actions_mask(), np.float32)
        for solver in (old, new):
            t = solver.ave_policy_trainer
            t.buffer.add(t.get_infostate_tensor(s), mask / mask.sum(), mask, i % 3 + 1)
    a, b = (s.ave_policy_trainer._grouped_training_data(3) for s in (old,new))
    assert torch.equal(a[0], b[0][:,:183])
    assert all(torch.equal(x,y) for x,y in zip(a[1:], b[1:]))
