"""Model-independent terminal decisions, disjoint from recorded prior board classes."""
from itertools import combinations
import numpy as np

from experiments.fhp.exp28_fhp_rare_state_audit.panels import (
    canonical_board, make_state, histories, hand_fact, feature_hash,
)
from experiments.fhp.exp28_fhp_rare_state_audit.common import digest
from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
from fhp_evaluation.cards import five_card_rank, card_name


def build_panel(inputs, smoke=False):
    previous = set(map(tuple, inputs['prior_board_classes']))
    rng = np.random.default_rng(2026101335)
    windows = [tuple(range(s, s+5)) for s in range(9)] + [(12, 0, 1, 2, 3)]
    sf = {canonical_board(tuple(4*r for r in ranks))
          for w in windows for ranks in combinations(w, 3)} - previous
    # Trips-board ranks have all been inspected. Paired boards plus a pocket
    # pair provide genuine new quads positions rather than relabelled holdouts.
    quads = {canonical_board((4*r, 4*r+1, 4*k)) for r in range(13)
             for k in range(13) if r != k} - previous
    count = 1 if smoke else 8
    selected = []
    for pool in (sf, quads):
        pool = sorted(pool)
        if len(pool) < count:
            raise ValueError('Insufficient fresh board classes; never reuse a holdout silently')
        selected.extend(pool[i] for i in rng.choice(len(pool), count, replace=False))
    encoder = FHPHandBoardFeatureEncoder()
    rows = {}
    for partition, boards in [('development', [(6, 10, 50), (24, 26, 27)]),
                              ('assessment', selected)]:
        for board in boards:
            hands = list(combinations([c for c in range(52) if c not in board], 2))
            strong = [h for h in hands if five_card_rank((*board, *h))[0] in (7, 8)]
            ordinary = [h for h in hands if five_card_rank((*board, *h))[0] < 7]
            controls = [ordinary[i] for i in rng.choice(len(ordinary),
                        min(len(ordinary), 1 if smoke else 4), replace=False)]
            # Keep all side-card ranks on known trips boards, including A/5.
            for hand in sorted(set(strong + controls)):
                fact = hand_fact(tuple(board), tuple(hand))
                for pre, post, seat, fold, win in histories():
                    state = make_state(board, hand, pre, post, seat)
                    assert state.legal_actions() == [0, 1] and state.child(1).is_terminal()
                    gap = win-fold if fact['strict_winner'] else None
                    if gap is not None and not np.isclose(
                            (state.child(1).returns()[seat]-state.child(0).returns()[seat])/100, gap):
                        raise ValueError('OpenSpiel terminal payoff differs from certificate')
                    row = dict(board=list(board), hand=list(hand), preflop=list(pre),
                        flop=list(post), seat=seat, partition=partition,
                        board_class=list(canonical_board(tuple(board))), **fact,
                        guaranteed_call_gain_bb=gap, fold_value_bb=fold,
                        hand_names=[card_name(c) for c in hand],
                        board_names=[card_name(c) for c in board],
                        feature_hash=feature_hash(encoder.information_state(state, seat)))
                    row['id'] = digest(dict(board_class=row['board_class'], feature_hash=row['feature_hash']))
                    # First representative retains the explicitly reported club
                    # side-card examples; do not weight suit-equivalent copies.
                    rows.setdefault(row['id'], row)
    result = list(rows.values())
    if any(tuple(r['board_class']) in previous for r in result if r['partition']=='assessment'):
        raise ValueError('Assessment board leakage')
    for cat in (7, 8):
        if not any(r['partition']=='assessment' and r['category']==cat and r['strict_winner'] for r in result):
            raise ValueError('Fresh panel lacks a guaranteed-winner category')
    return dict(version=1, smoke=smoke, rows=result, assessment_board_classes=sorted(selected),
        scope='Terminal call/fold only. Fresh relative to the recorded strategic library and Exp28; '
              'not guaranteed absent from training. Not a reach-weighted whole-game metric.',
        controls='Ordinary hands are descriptive: no optimal-action labels from uniform equity.')
