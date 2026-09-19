from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import (
    CHECKPOINT_HOURS,
    EXPERIMENT_CONFIG,
    PRODUCTION_SEEDS,
    checkpoint_schedule,
    smoke_config,
    task_schedule,
    validate_contract,
)
from experiments.fhp.exp2_fhp_lossless_structured_ucv.run import _run_task
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _make_solver
from fhp_escher.checkpointing import LoadedFHPPolicy, load_checkpoint_payload
from fhp_escher.features import (
    ENCODER_ID,
    FHPFeatureEncoder,
    StructuredFHPMLP,
    _betting_features,
    _betting_sequence,
    _canonicalise_suits,
)
from fhp_escher.game import load_fhp_game


def _decision_state():
    state = load_fhp_game().new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    return state


def _load_batch_builder():
    path = Path(__file__).parents[1] / "gcp" / "exp2_lossless_structured_batch.py"
    spec = importlib.util.spec_from_file_location("exp2_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_encoder_dimensions_and_exact_betting_history():
    encoder = FHPFeatureEncoder()
    state = _decision_state()
    before = encoder.information_state(state, 0)
    after = encoder.information_state(state.child(1), 1)
    assert before.shape == (183,)
    assert encoder.full_state(state).shape == (263,)
    assert before.dtype == np.float32
    assert not np.array_equal(before, after)
    assert encoder.metadata()["id"] == ENCODER_ID
    assert np.count_nonzero(before[:104]) == 2


def test_suit_canonicalisation_is_invariant_and_lossless_for_rank_patterns():
    hole = np.zeros((4, 13), dtype=np.float32)
    board = np.zeros((4, 13), dtype=np.float32)
    hole[0, 12] = 1
    hole[0, 11] = 1
    board[0, 10] = 1
    board[1, 4] = 1
    board[2, 3] = 1
    expected = _canonicalise_suits(hole, board)
    for permutation in (
        [1, 0, 2, 3],
        [3, 2, 1, 0],
        [2, 0, 3, 1],
    ):
        observed = _canonicalise_suits(hole[permutation], board[permutation])
        assert all(np.array_equal(a, b) for a, b in zip(expected, observed))
    different_rank = hole.copy()
    different_rank[0, 11] = 0
    different_rank[0, 9] = 1
    changed = _canonicalise_suits(different_rank, board)
    assert not np.array_equal(expected[0], changed[0])


def test_compact_betting_encoding_has_no_reachable_history_collisions():
    game = load_fhp_game()
    encodings = {}

    def visit(state):
        if state.is_chance_node():
            visit(state.child(state.chance_outcomes()[0][0]))
            return
        player = 0 if state.is_terminal() else state.current_player()
        sequence = _betting_sequence(state, player)
        encoded = np.concatenate(_betting_features(sequence)).tobytes()
        prior = encodings.setdefault(encoded, sequence)
        assert prior == sequence
        if not state.is_terminal():
            for action in state.legal_actions():
                visit(state.child(action))

    visit(game.new_initial_state())
    assert len(set(encodings.values())) == 162


def test_contract_and_structured_solver_configuration():
    validate_contract(
        seeds=PRODUCTION_SEEDS,
        schedule=checkpoint_schedule(),
        config=EXPERIMENT_CONFIG,
        smoke=False,
    )
    assert CHECKPOINT_HOURS == (6, 12, 18, 24)
    assert task_schedule() == tuple(
        ("lossless_structured_ucv_escher", seed) for seed in PRODUCTION_SEEDS
    )
    solver = _make_solver(0, smoke_config())
    assert solver.raw_open_spiel_infostate_size == 190
    assert solver.infostate_size == 183
    assert solver.q_value_trainer.members[0].input_size == 263
    assert isinstance(solver.ave_policy_trainer.model, StructuredFHPMLP)
    assert isinstance(solver.regret_trainers[0].model, StructuredFHPMLP)
    assert isinstance(solver.q_value_trainer.members[0].model, StructuredFHPMLP)
    assert solver.ave_policy_trainer.buffer.infostate_buf.dtype == np.float32
    assert solver.q_value_trainer.members[0].buffer.history_buf.dtype == np.float32


def test_batch_job_is_three_independent_retryable_vms():
    builder = _load_batch_builder()
    args = SimpleNamespace(
        kind="train",
        repo_url=builder.REPO_URL,
        repo_ref="deadbeef",
        bucket_root="gs://test-bucket",
        run_id="exp2-fhp-test",
        service_account="batch@example.iam.gserviceaccount.com",
        parallelism=3,
        project_id="test-project",
        region="europe-west1",
        controller_action="orchestrate",
    )
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == 3
    assert group["parallelism"] == 3
    assert group["taskSpec"]["maxRetryCount"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "129600s"
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "exp2_fhp_lossless_structured_ucv" in script
    assert "EXP2_REMOTE_TASK_URI" in script
    assert "batch_diagnostics monitor" in script
    assert "exp1_fhp_grouped_wide_ucv_baseline" not in script


def test_structured_checkpoint_round_trip(tmp_path):
    summary = _run_task(
        task_index=0,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    assert summary["status"] == "complete"
    assert summary["checkpoint_count"] == 4
    worker = tmp_path / "workers" / "task_000_lossless_structured_ucv_escher_seed_0"
    checkpoint = worker / "final_policy_checkpoint.pkl"
    payload = load_checkpoint_payload(checkpoint)
    assert payload["feature_encoder"]["id"] == ENCODER_ID
    assert payload["policy_model"]["type"] == "structured_fhp_mlp_v1"
    policy = LoadedFHPPolicy(load_fhp_game(), checkpoint)
    probabilities = policy.action_probabilities(_decision_state())
    assert np.isclose(sum(probabilities.values()), 1.0)
    resumed = _run_task(
        task_index=0,
        output_root=tmp_path,
        smoke=True,
        resume=True,
    )
    assert resumed["resumed_from_training_state"] is True
    assert resumed["final_nodes_touched"] == summary["final_nodes_touched"]
    states = json.loads(
        (worker / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert len(states) == 4
    assert all((worker / row["training_state_path"]).is_file() for row in states)
