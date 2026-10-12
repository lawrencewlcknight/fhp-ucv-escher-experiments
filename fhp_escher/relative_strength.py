"""Exact FHP card statistics, with bounded, board-shared vectorized tables.

No actual opponent cards, opponent policy, random draws or future streets are
used. Card ids are rank*4+suit. Tables contain all 1,176 hands on a public board;
blocker subtraction leaves exactly 1,081 opponents for each queried own hand.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np

CLASS_NAMES = (
    "unpaired", "pair_below_board_top", "board_top_pair_kicker_below_jack",
    "board_top_pair_kicker_jack_plus", "overpair", "two_pair", "trips",
    "straight", "flush", "full_house_or_better",
)
OVERALL_NAMES = ("available", "win_probability", "tie_probability",
                 "loss_probability", "equity", "unbeatable")
CLASS_FEATURE_NAMES = tuple(f"{c}_{s}" for c in CLASS_NAMES
                            for s in ("equity", "legal_mass", "supported"))
BLOCKER_NAMES = tuple(f"{c}_{s}" for c in CLASS_NAMES
                     for s in ("removed_fraction", "board_supported")) + (
                         "nuts_removed_fraction", "nuts_legal_mass")
_PAIRS = np.asarray(list(combinations(range(52), 2)), dtype=np.int16)
_POWERS = np.asarray([15**i for i in range(4, -1, -1)], dtype=np.int32)
_STRAIGHT = np.full(8192, -1, dtype=np.int8)
for _mask in range(8192):
    if _mask & 0x100F == 0x100F:
        _STRAIGHT[_mask] = 3  # wheel: rank index of five
    for _start in range(9):
        if _mask & (31 << _start) == 31 << _start:
            _STRAIGHT[_mask] = _start + 4


def rank_five(cards):
    """Return comparable integer ranks, categories and ordered tiebreaks in batches."""
    cards = np.asarray(cards, dtype=np.int16)
    if cards.ndim != 2 or cards.shape[1] != 5:
        raise ValueError("Expected N by 5 cards")
    ranks = cards // 4
    counts = np.eye(13, dtype=np.int8)[ranks].sum(axis=1)
    mask = ((counts > 0) * (1 << np.arange(13))).sum(axis=1)
    straight = _STRAIGHT[mask]
    flush = np.all(cards % 4 == cards[:, :1] % 4, axis=1)
    pairs = (counts == 2).sum(axis=1)
    trips, quads = (counts == 3).any(axis=1), (counts == 4).any(axis=1)
    category = np.zeros(len(cards), dtype=np.int8)
    category[pairs == 1] = 1
    category[pairs == 2] = 2
    category[trips] = 3
    category[straight >= 0] = 4
    category[flush] = 5
    category[trips & (pairs == 1)] = 6
    category[quads] = 7
    category[flush & (straight >= 0)] = 8
    order = np.argsort(-(counts * 16 + np.arange(13)), axis=1)[:, :5]
    tie = np.where(np.take_along_axis(counts, order, axis=1) > 0, order + 2, 0)
    is_straight = (category == 4) | (category == 8)
    tie[is_straight] = 0
    tie[is_straight, 0] = straight[is_straight] + 2
    return category.astype(np.int32) * 15**5 + tie @ _POWERS, category, tie


def canonical_board(board):
    """Return public-board suit canonicalisation and original->canonical suit map."""
    board = tuple(sorted(map(int, board)))
    signatures = [sum(1 << (c // 4) for c in board if c % 4 == s) for s in range(4)]
    order = sorted(range(4), key=lambda s: (-signatures[s], s))
    mapping = tuple(order.index(s) for s in range(4))
    return tuple(sorted(c // 4 * 4 + mapping[c % 4] for c in board)), mapping


def _classes(category, tie, board):
    result = np.zeros(len(category), dtype=np.int8)
    top = max(board) // 4 + 2
    pair = category == 1
    result[pair & (tie[:, 0] < top)] = 1
    result[pair & (tie[:, 0] == top) & (tie[:, 1] < 11)] = 2
    result[pair & (tie[:, 0] == top) & (tie[:, 1] >= 11)] = 3
    result[pair & (tie[:, 0] > top)] = 4
    for cat, label in ((2, 5), (3, 6), (4, 7), (5, 8)):
        result[category == cat] = label
    result[category >= 6] = 9
    return result


class BoardTable:
    """Linear-size rank-prefix sums; deliberately no hand-by-hand payoff matrix."""
    def __init__(self, board):
        board = tuple(board)
        if len(board) != 3 or len(set(board)) != 3 or not all(0 <= c < 52 for c in board):
            raise ValueError("Three distinct board cards required")
        self.hands = _PAIRS[~np.isin(_PAIRS, board).any(axis=1)]
        cards = np.column_stack((self.hands, np.tile(board, (len(self.hands), 1))))
        rank, category, tie = rank_five(cards)
        _, codes = np.unique(rank, return_inverse=True)
        rank_count = int(codes.max()) + 1
        classes = _classes(category, tie, board)
        self.lookup = np.full((52, 52), -1, dtype=np.int16)
        self.lookup[self.hands[:, 0], self.hands[:, 1]] = np.arange(len(self.hands))
        # Each column is a fixed board-relative class, plus one nuts column.
        membership = np.column_stack((np.eye(len(CLASS_NAMES), dtype=np.int32)[classes],
                                      rank == rank.max())).astype(np.int32)
        board_mass = membership.sum(axis=0)
        by_card = np.zeros((52, membership.shape[1]), dtype=np.int32)
        np.add.at(by_card, self.hands[:, 0], membership)
        np.add.at(by_card, self.hands[:, 1], membership)
        mass = board_mass - by_card[self.hands].sum(axis=1) + membership
        wins, ties = [], []
        for j in range(len(CLASS_NAMES)):
            weights = membership[:, j]
            counts = np.bincount(codes, weights=weights, minlength=rank_count).astype(np.int32)
            prefix = np.cumsum(counts) - counts
            cr = np.zeros((52, rank_count), dtype=np.int32)
            np.add.at(cr, (self.hands[:, 0], codes), weights)
            np.add.at(cr, (self.hands[:, 1], codes), weights)
            lesser = np.cumsum(cr, axis=1) - cr
            wins.append(prefix[codes] - lesser[self.hands, codes[:, None]].sum(axis=1))
            ties.append(counts[codes] - cr[self.hands, codes[:, None]].sum(axis=1) + weights)
        wins, ties = np.stack(wins, axis=1), np.stack(ties, axis=1)
        legal_mass = mass[:, :len(CLASS_NAMES)]
        assert np.all(legal_mass.sum(axis=1) == 1081)
        assert np.all((wins >= 0) & (ties >= 0) & (wins + ties <= legal_mass))
        win, tie = wins.sum(axis=1), ties.sum(axis=1)
        loss = 1081 - win - tie
        self.overall = np.column_stack((np.ones(len(win)), win / 1081, tie / 1081,
                                      loss / 1081, (win + .5 * tie) / 1081,
                                      loss == 0)).astype(np.float32)
        equity = np.divide(wins + .5 * ties, legal_mass,
                           out=np.zeros_like(wins, dtype=float), where=legal_mass > 0)
        self.classes = np.stack((equity, legal_mass / 1081, legal_mass > 0), axis=2).reshape(len(win), -1).astype(np.float32)
        removed = np.divide(board_mass - mass, board_mass, out=np.zeros_like(mass, dtype=float),
                            where=board_mass > 0)
        support = np.broadcast_to(board_mass[:len(CLASS_NAMES)] > 0, legal_mass.shape)
        self.blockers = np.column_stack((
            np.stack((removed[:, :len(CLASS_NAMES)], support), axis=2).reshape(len(win), -1),
            removed[:, -1], mass[:, -1] / 1081)).astype(np.float32)
        self.equity_twice_count = 2 * win + tie  # exact integer numerator for table generation
        for array in (self.hands, self.lookup, self.overall, self.classes, self.blockers,
                      self.equity_twice_count):
            array.flags.writeable = False


@lru_cache(maxsize=64)
def board_table(board):
    return BoardTable(board)


def card_ids(channel):
    suit, rank = np.nonzero(channel)
    return tuple(sorted(map(int, rank * 4 + suit)))


def relative_features(hole, board):
    private, public = card_ids(hole), card_ids(board)
    if len(private) != 2 or set(private) & set(public):
        raise ValueError("Two distinct own cards, disjoint from public board, required")
    if len(public) != 3:
        return (np.zeros(len(OVERALL_NAMES), np.float32),
                np.zeros(len(CLASS_FEATURE_NAMES), np.float32),
                np.zeros(len(BLOCKER_NAMES), np.float32))
    canonical, mapping = canonical_board(public)
    hand = sorted(c // 4 * 4 + mapping[c % 4] for c in private)
    table = board_table(canonical)
    index = int(table.lookup[tuple(hand)])
    if index < 0:
        raise ValueError("Illegal private/public card combination")
    return table.overall[index], table.classes[index], table.blockers[index]
