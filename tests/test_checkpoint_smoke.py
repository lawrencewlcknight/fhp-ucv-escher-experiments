import json
from pathlib import Path

import pytest

from experiments.fhp.ucv_escher_baseline.config import smoke_config
from experiments.fhp.ucv_escher_baseline.run import run_experiment
from fhp_escher.checkpointing import LoadedFHPPolicy, load_checkpoint_payload
from fhp_escher.game import load_fhp_game


@pytest.mark.smoke
def test_smoke_run_saves_reloadable_final_policy(tmp_path: Path):
    run_dir = run_experiment(
        seed=7,
        config=smoke_config(),
        target_nodes=500,
        max_training_seconds=300,
        output_root=tmp_path,
    )
    final_checkpoint = run_dir / "final_policy_checkpoint.pkl"
    payload = load_checkpoint_payload(final_checkpoint)
    assert payload["algorithm"] == "UCV-ESCHER"
    assert payload["seed"] == 7
    assert payload["nodes_touched"] > 0
    assert payload["checkpoint_kind"] == "outer_iteration"
    checkpoint_manifest = json.loads(
        (run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert [row["checkpoint_kind"] for row in checkpoint_manifest] == [
        "outer_iteration"
    ]

    game = load_fhp_game()
    restored = LoadedFHPPolicy(game, final_checkpoint)
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    probabilities = restored.action_probabilities(state)
    assert set(probabilities) == set(state.legal_actions())
    assert sum(probabilities.values()) == pytest.approx(1.0)
