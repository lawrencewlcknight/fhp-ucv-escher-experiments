import json
from pathlib import Path

import pytest

from experiments.fhp.exp1_ucv_escher_baseline.config import smoke_config
from experiments.fhp.exp1_ucv_escher_baseline.run import run_experiment
from fhp_escher.checkpointing import LoadedFHPPolicy, load_checkpoint_payload
from fhp_escher.game import load_fhp_game


@pytest.mark.smoke
def test_smoke_run_saves_reloadable_final_policy(tmp_path: Path):
    run_dir = run_experiment(
        seed=7,
        config=smoke_config(),
        checkpoint_training_seconds=(1e-6, 2e-6),
        output_root=tmp_path,
    )
    final_checkpoint = run_dir / "final_policy_checkpoint.pkl"
    payload = load_checkpoint_payload(final_checkpoint)
    assert run_dir.name.startswith("exp1_fhp_ucv_escher_baseline_")
    assert payload["algorithm"] == "UCV-ESCHER"
    assert payload["algorithm_id"] == "ucv_escher"
    assert payload["execution_backend"] == "sequential"
    assert payload["experiment_id"] == 1
    assert payload["experiment_name"] == "exp1_fhp_ucv_escher_baseline"
    assert payload["seed"] == 7
    assert payload["nodes_touched"] > 0
    assert payload["checkpoint_kind"] == "training_time_checkpoint"
    assert payload["checkpoint_target_seconds"] == 2e-6
    checkpoint_manifest = json.loads(
        (run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert [row["checkpoint_target_seconds"] for row in checkpoint_manifest] == [
        1e-6,
        2e-6,
    ]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "training_time_budget"
    assert summary["checkpoint_count"] == 2
    assert summary["capacity_assessment"] == "checkpoint_schedule_completed"

    game = load_fhp_game()
    restored = LoadedFHPPolicy(game, final_checkpoint)
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    probabilities = restored.action_probabilities(state)
    assert set(probabilities) == set(state.legal_actions())
    assert sum(probabilities.values()) == pytest.approx(1.0)
