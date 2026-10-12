"""Frozen suit-class holdouts and exact OpenSpiel terminal-call diagnostics."""
from functools import lru_cache
from itertools import combinations, permutations
import numpy as np

from fhp_escher.game import load_fhp_game
from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
from fhp_evaluation.cards import five_card_rank, card_name
from .common import digest


@lru_cache(maxsize=24000)
def canonical_board(board):
    return min(tuple(sorted(4*(c//4)+p[c%4] for c in board)) for p in permutations(range(4)))


def make_state(board, hand, pre, post, player, opponent=None):
    game = load_fhp_game()
    other = tuple(c for c in range(52) if c not in (*board, *hand))[:2] if opponent is None else tuple(opponent)
    if len(set((*hand, *other, *board))) != 7:
        raise ValueError("Overlapping cards")
    s = game.new_initial_state()
    holes = (*hand, *other) if player == 0 else (*other, *hand)
    for action in (*holes, *pre, *board, *post):
        if action not in s.legal_actions():
            raise ValueError("Invalid audit history")
        s.apply_action(int(action))
    if s.current_player() != player:
        raise ValueError("Wrong acting seat")
    return s


def from_case(case, opponent=None):
    return make_state(case["board"], case["hand"], case["preflop"], case["flop"], case["seat"], opponent)


@lru_cache(maxsize=1)
def histories():
    """Enumerate legal entries/terminal call points using OpenSpiel, not guessed histories."""
    game = load_fhp_game()
    s = game.new_initial_state()
    for c in (0, 1, 2, 3):
        s.apply_action(c)
    entries, result = [], []
    def walk_pre(st, path):
        if st.is_terminal():
            return
        if st.is_chance_node():
            entries.append((st, path))
        else:
            for a in st.legal_actions():
                walk_pre(st.child(a), path+(a,))
    walk_pre(s, ())
    def walk_post(st, pre, path):
        if st.is_terminal():
            return
        if st.legal_actions() == [0, 1] and st.child(1).is_terminal():
            p = st.current_player()
            # Both players have contributed equally after the terminal call.
            fold = st.child(0).returns()[p] / 100.
            # fold is negative; the winner receives the OTHER player's contribution.
            win = -fold + 1.
            result.append((pre, path, p, fold, win))
        for a in st.legal_actions():
            walk_post(st.child(a), pre, path+(a,))
    for state, pre in entries:
        for c in (4, 5, 6):
            state.apply_action(c)
        walk_post(state, pre, ())
    return result


@lru_cache(maxsize=32768)
def hand_fact(board, hand):
    rank = five_card_rank((*board, *hand))
    opponents = list(combinations([c for c in range(52) if c not in (*board, *hand)], 2))
    comparison = [five_card_rank((*board, *h)) for h in opponents]
    wins = sum(rank > r for r in comparison)
    ties = sum(rank == r for r in comparison)
    return dict(category=rank[0], unbeatable=(wins+ties == len(opponents)),
                strict_winner=(wins == len(opponents)), uniform_equity=(wins+.5*ties)/len(opponents))


def build_panel(prior_library, config):
    """Choose board classes before reading any model predictions.

    Earlier development AND reference boards are excluded from assessment.
    The previous audit's reserved assessment set stays reserved and is excluded
    too: this experiment creates its own fresh holdout, with no source dependency.
    """
    previous = {canonical_board(tuple(r["board"])) for r in prior_library.get("boards", [])}
    previous |= {canonical_board(tuple(r["cards"][4:])) for r in prior_library.get("reference_deals", [])}
    previous |= {canonical_board(tuple(b)) for b in prior_library.get("assessment_board_classes", [])}
    known = [(6, 10, 50), (24, 26, 27)]
    previous |= {canonical_board(b) for b in known}
    # All monotone boards supporting a straight flush, and all trips boards.
    runs = [tuple(range(start, start+5)) for start in range(9)] + [(12, 0, 1, 2, 3)]
    straight_boards = {tuple(sorted(4*r for r in ranks)) for window in runs for ranks in combinations(window, 3)}
    trips = {(4*r, 4*r+1, 4*r+2) for r in range(13)}
    pools = [sorted(straight_boards-previous), sorted(trips-previous)]
    rng = np.random.default_rng(config["random_seed"])
    dev, assessment = list(known), []
    # Scarce trips classes: reserve assessment FIRST. There may be few left
    # after previous audits. Never fill a holdout with an already inspected class.
    for pool in pools:
        order = [pool[i] for i in rng.permutation(len(pool))]
        ntest = min(len(order), max(1, config["assessment_boards"]//2))
        assessment.extend(order[:ntest])
        dev.extend(order[ntest:ntest+config["additional_dev_boards"]//2])
    if not assessment:
        raise ValueError("No fresh assessment board classes")
    rows, encoder = [], FHPHandBoardFeatureEncoder()
    for split, boards in (("development", dev), ("assessment", assessment)):
        for board in boards:
            hands = list(combinations([c for c in range(52) if c not in board], 2))
            ranked = [(h, five_card_rank((*board, *h))[0]) for h in hands]
            strong = [h for h, cat in ranked if cat in (7, 8)]
            # Keep all side-card ranks for trips; small smoke remains bounded.
            limit = config["hands_per_category"] if config["smoke"] else 52
            strong = strong[:limit]
            ordinary = [h for h, cat in ranked if cat in (0, 1, 2, 3, 4, 5, 6)]
            control = [ordinary[i] for i in rng.choice(len(ordinary),
                min(config["controls_per_board"], len(ordinary)), replace=False)]
            for hand in sorted(set(strong+control)):
                fact = hand_fact(tuple(board), tuple(hand))
                for pre, post, seat, fold, win in histories():
                    st = make_state(board, hand, pre, post, seat)
                    features = encoder.information_state(st, seat)
                    values = [fold, (2*fact["uniform_equity"]-1)*win, None]
                    row = dict(board=list(board), hand=list(hand), preflop=list(pre), flop=list(post), seat=seat,
                        partition=split, board_class=list(canonical_board(tuple(board))), **fact,
                        fold_value_bb=fold, call_value_uniform_bb=values[1],
                        guaranteed_call_gain_bb=(-fold if not fact["strict_winner"] else win-fold)
                            if fact["unbeatable"] else None,
                        feature_hash=feature_hash(features), hand_names=[card_name(c) for c in hand],
                        board_names=[card_name(c) for c in board])
                    row["id"] = digest({k:row[k] for k in ("board_class", "feature_hash")})
                    rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        # Equal-status suit permutations can produce the same information set.
        rows = list({r["id"]: r for r in rows}.values())
    test = {tuple(r["board_class"]) for r in rows if r["partition"] == "assessment"}
    development = {tuple(r["board_class"]) for r in rows if r["partition"] == "development"}
    if test & (previous | development):
        raise ValueError("Assessment leakage")
    return dict(protocol=config, rows=rows, previous_board_classes=[list(b) for b in sorted(previous)],
        holdout_board_classes=[list(b) for b in sorted(test)],
        note="uniform equity is descriptive only; primary dominance does not assume an opponent range")


def feature_hash(features):
    import hashlib
    return hashlib.sha256(np.asarray(features, dtype="<f4").tobytes()).hexdigest()


def board_classes(features):
    """Canonical board identity from lossless card channels; preflop gets None."""
    result = []
    for row in features:
        cards = np.flatnonzero(row[52:104] > .5)
        result.append(canonical_board(tuple(sorted(4*int(c%13)+int(c//13) for c in cards)))
                      if len(cards) == 3 else None)
    return result


def categories(features):
    a = np.asarray(features)
    return np.where(a[:, 107] > .5, np.argmax(a[:, 174:183], axis=1), -1)
