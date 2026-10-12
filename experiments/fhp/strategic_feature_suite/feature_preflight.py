"""Read-only representation cost and correctness gate on sampled legal states."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fhp_escher.game import load_fhp_game
from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
from fhp_escher.relative_strength import board_table, canonical_board, BoardTable


def run_feature_preflight(directory, *, experiment, samples=128):
    # A private generator avoids altering learner or actor random streams.
    rng = np.random.default_rng(290033)
    game = load_fhp_game()
    states = []
    for _ in range(samples):
        cards = rng.choice(52, 7, replace=False)
        state = game.new_initial_state()
        for card in cards[:4]:
            state.apply_action(int(card))
        states.append(state.clone())
        state.apply_action(1)
        state.apply_action(1)
        for card in cards[4:]:
            state.apply_action(int(card))
        states.append(state)
    metrics = {}
    candidate = experiment.encoder()
    candidate.cache_entries = 0
    encoders = (("experiment10", FHPHandBoardFeatureEncoder(cache_entries=0)),
                ("candidate", candidate))
    for name, encoder in encoders:
        board_table.cache_clear()
        start = perf_counter()
        rows = [(encoder.information_state(s, s.current_player()), encoder.full_state(s)) for s in states]
        elapsed = perf_counter() - start
        metrics[name] = {"seconds": elapsed, "microseconds_per_policy_plus_critic": elapsed / len(states) * 1e6}
        if name == "experiment10":
            reference = rows
        else:
            for (policy, critic), (p0, c0) in zip(rows, reference):
                np.testing.assert_array_equal(policy[:len(p0)], p0)
                np.testing.assert_array_equal(critic[:len(c0)], c0)
                if not (np.all(np.isfinite(policy)) and np.all(np.isfinite(critic))):
                    raise RuntimeError("Non-finite strategic features")
            metrics["board_cache"] = board_table.cache_info()._asdict()
    table = BoardTable(canonical_board((0, 5, 10))[0])
    retained_bytes = sum(v.nbytes for v in vars(table).values() if isinstance(v, np.ndarray))
    report = dict(status="passed", sampled_states=len(states), feature_encoder=candidate.metadata(),
        timings=metrics, candidate_to_baseline_encoding_time_ratio=metrics["candidate"]["seconds"]/metrics["experiment10"]["seconds"],
        board_cache_array_bytes_at_64_entries=64 * retained_bytes,
        interpretation="Synthetic cold/diverse-board encoding diagnostic; not end-to-end training speed. Memory excludes Python overhead and temporary construction arrays. Feature costs remain in the active collection clock.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "feature_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    board_table.cache_clear()
    return report
