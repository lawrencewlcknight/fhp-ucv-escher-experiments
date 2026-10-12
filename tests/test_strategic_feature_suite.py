"""Poker oracles, leakage boundaries, frozen contracts, cloud lifecycle and resume."""
from itertools import combinations, permutations
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fhp_escher.game import load_fhp_game
from fhp_escher.features import encoder_from_metadata, make_feature_encoder, _raw_cards
from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
from fhp_escher.strategic_features import StrategicFeatureEncoder, PACKAGES, INTERACTION_NAMES
from fhp_escher.relative_strength import (BoardTable, rank_five, canonical_board,
    relative_features, board_table, CLASS_NAMES)
from fhp_escher.preflop_strength import load_table, starting_indices, ASSET, generate_table
from fhp_evaluation.cards import five_card_rank
from experiments.fhp.exp10_fhp_hand_board_features import config as baseline
from experiments.fhp.strategic_feature_suite.config import get_experiment, SPECS
from experiments.fhp.strategic_feature_suite import evaluate
from gcp.strategic_features_batch import build_job

ROOT = Path(__file__).resolve().parents[1]


def channel(cards):
    a = np.zeros((4, 13), np.float32)
    for c in cards:
        a[c % 4, c // 4] = 1
    return a


def state_for(own=(48, 45), other=(3, 7), board=(24, 30, 34), actions=()):
    state = load_fhp_game().new_initial_state()
    for c in (*own, *other):
        state.apply_action(c)
    if board:
        state.apply_action(1)
        state.apply_action(1)
        for c in board:
            state.apply_action(c)
    for action in actions:
        state.apply_action(action)
    return state


def test_vector_ranking_matches_independent_scalar_ranker():
    rng = np.random.default_rng(912)
    cards = np.stack([rng.choice(52, 5, replace=False) for _ in range(5000)])
    rank, category, ties = rank_five(cards)
    for i, hand in enumerate(cards):
        expected = five_card_rank(hand)
        assert category[i] == expected[0]
        assert tuple(ties[i, :len(expected)-1]) == expected[1:]
        assert not np.any(ties[i, len(expected)-1:])
        assert rank[i] == expected[0] * 15**5 + sum(x*15**(4-j) for j,x in enumerate(expected[1:]))


@pytest.mark.parametrize("board", [(0, 1, 2), (0, 4, 8), (48, 45, 38), (12, 13, 44)])
def test_board_tables_match_enumeration_and_class_mass(board):
    table = BoardTable(board)
    for i in (0, 133, 679, 1175):
        own = tuple(table.hands[i])
        legal = [h for h in table.hands if not set(own) & set(h)]
        assert len(legal) == 1081
        our_rank = five_card_rank((*own, *board))
        ranks = [five_card_rank((*hand, *board)) for hand in legal]
        win = sum(our_rank > r for r in ranks)
        tie = sum(our_rank == r for r in ranks)
        loss = 1081-win-tie
        np.testing.assert_allclose(table.overall[i], [1, win/1081, tie/1081,
            loss/1081, (win+.5*tie)/1081, loss == 0], rtol=1e-6)
        classes = table.classes[i].reshape(-1, 3)
        assert classes[:, 1].sum() == pytest.approx(1)
        assert (classes[:, 0]*classes[:, 1]).sum() == pytest.approx(table.overall[i, 4])
        assert np.all(classes[classes[:, 2] == 0] == 0)
        assert np.all((table.blockers[i] >= 0) & (table.blockers[i] <= 1))
    assert np.all(table.overall[:, 1:4].sum(axis=1) == pytest.approx(1))


@pytest.mark.parametrize("kind", tuple(PACKAGES))
def test_exact_prefix_privacy_scope_and_metadata(kind):
    encoder, base = StrategicFeatureEncoder(kind), FHPHandBoardFeatureEncoder()
    for board in ((), (24, 30, 34)):
        state, changed = state_for(board=board), state_for(other=(9, 11), board=board)
        a, b = encoder.information_state(state, 0), encoder.information_state(changed, 0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:213], base.information_state(state, 0))
        assert a.shape == (213+len(PACKAGES[kind]),)
        full = encoder.full_state(state)
        np.testing.assert_array_equal(full[:323], base.full_state(state))
        np.testing.assert_array_equal(full[323:323+len(PACKAGES[kind])], a[213:])
        assert full.shape == (323+2*len(PACKAGES[kind]),)
        assert np.isfinite(full).all()
        if not board and kind != "D":
            assert np.count_nonzero(a[213:]) == 0
        if board and kind == "D":
            assert np.count_nonzero(a[213:]) == 0
    assert encoder_from_metadata(encoder.metadata()).metadata() == encoder.metadata()
    wrong = dict(encoder.metadata(), feature_package="tampered")
    with pytest.raises(ValueError, match="contract"):
        encoder_from_metadata(wrong)


def test_suit_invariance_including_empty_suits_and_repeated_board_ranks():
    for private, public in (((49, 45), (0, 4, 8)), ((48, 45), (12, 13, 44)), ((3, 8), (20, 25, 34))):
        expected = relative_features(channel(private), channel(public))
        for order in permutations(range(4)):
            actual = relative_features(channel(private)[list(order)], channel(public)[list(order)])
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
    assert board_table.cache_info().maxsize == 64


def test_preflop_asset_exact_reproducible_and_correct_weighting():
    payload = json.loads(ASSET.read_text())
    regenerated = generate_table()
    for k in payload.keys() - {"generation_seconds"}:
        assert payload[k] == regenerated[k], k
    assert payload["physical_boards"] == 22100
    assert payload["board_orbits"] == 1755
    assert not payload["sampled"]
    table, digest = load_table()
    assert len(digest) == 64 and table.shape == (169, 13)
    np.testing.assert_allclose(table[:, 3:].sum(axis=1), 1, atol=1e-7)
    for hand in ((48,49), (48,44), (48,45)):
        expected = starting_indices(np.asarray([hand]))[0]
        for perm in permutations(range(4)):
            changed = [c//4*4+perm[c%4] for c in hand]
            assert starting_indices(np.asarray([changed]))[0] == expected


def test_betting_interactions_call_closure_matches_game():
    encoder = StrategicFeatureEncoder("AE")
    for actions in ((), (1,), (2,), (2,2), (1,2), (2,2,2)):
        state = state_for(actions=actions)
        actor = state.current_player()
        features = encoder.information_state(state, actor)
        interaction = dict(zip(INTERACTION_NAMES, features[219:]))
        assert interaction["call_closes_final_round"] == state.child(1).is_terminal()
        equity = features[217]
        assert interaction["equity_minus_pot_odds"] == pytest.approx(equity-interaction["pot_odds"])
        hole, board = _raw_cards(state, actor)
        assert equity == relative_features(hole, board)[0][4]


@pytest.mark.parametrize("number", tuple(SPECS))
def test_frozen_contract(number):
    exp = get_experiment(number)
    assert exp.EXPERIMENT_CONFIG == dict(baseline.EXPERIMENT_CONFIG,
        feature_encoder_id=exp.encoder().encoder_id)
    assert exp.PRODUCTION_SEEDS == (0,1,2)
    assert exp.parallel_settings() == baseline.parallel_settings()
    assert exp.checkpoint_schedule() == baseline.checkpoint_schedule()
    assert len(exp.checkpoint_schedule(total_hours=72)) == 12
    exp.validate_contract(seeds=(0,1,2), schedule=exp.checkpoint_schedule(),
                          config=exp.EXPERIMENT_CONFIG, smoke=False)
    with pytest.raises(ValueError):
        exp.validate_contract(seeds=(0,1,2), schedule=exp.checkpoint_schedule(),
            config=dict(exp.EXPERIMENT_CONFIG, baseline_network_train_steps=5000), smoke=False)
    manifest = exp.contract_manifest()
    assert manifest["training_state_retention"] == "final"
    assert not manifest["baseline_retrained"]
    assert manifest["feature_encoder"] == exp.encoder().metadata()


def job_args(number, kind, extension=False):
    return SimpleNamespace(experiment=number, kind=kind, run_id=f"exp{number}-test",
        repo_ref="a"*40, bucket_root="gs://test-bucket", project_id="test-project", region="europe-west1",
        service_account="runner@example.com", repo_url="https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git",
        parallelism=3, max_hours=48, total_hours=48 if extension else 24,
        source_run_id=f"exp{number}-source" if extension else "", resume=False)


@pytest.mark.parametrize("number", tuple(SPECS))
@pytest.mark.parametrize("kind", ("smoke", "train", "aggregate", "controller", "evaluate"))
def test_cloud_job_wiring(number, kind):
    job = build_job(job_args(number, kind))
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    script = spec["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert spec["maxRetryCount"] == 0
    assert "exp20_fhp" not in script and "EXP20" not in script
    assert f"exp{number}_fhp_{SPECS[number][0]}" in script or kind == "controller"
    if kind == "evaluate":
        assert "--candidate-run" in script and "--exp10-run" in script
        assert "exp10-features-20261001-161740" in script
        assert " worker " not in script
        assert '"$BUCKET_ROOT/$RUN_ID/evaluation"' in script
    if kind == "train":
        assert "FEATURE_REMOTE_TASK_URI" in script
        assert '--total-hours "$FEATURE_TOTAL_HOURS"' in script
        assert " stage-source " in script
    if kind == "smoke":
        assert "baseline-preflight" in script
        assert script.index(" baseline-preflight ") < script.index(" capacity-preflight ")
    if kind in ("train", "smoke", "evaluate"):
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"


@pytest.mark.parametrize("number", tuple(SPECS))
def test_cloud_extension(number):
    for kind in ("train", "aggregate", "controller"):
        job = build_job(job_args(number, kind, extension=True))
        env = job["taskGroups"][0]["taskSpec"]["environment"]["variables"]
        assert env["FEATURE_TOTAL_HOURS"] == "48"
        assert env["FEATURE_SOURCE_RUN_ID"] == f"exp{number}-source"
    args = job_args(number, "train", extension=True)
    args.source_run_id = "exp10-features-20261001-161740"
    with pytest.raises(ValueError):
        build_job(args)


def test_automatic_evaluation_and_no_baseline_training():
    source = (ROOT / "gcp/run_strategic_features.sh").read_text()
    assert "ensure_stage evaluate" in source
    assert 'if [[ "$FEATURE_TOTAL_HOURS" == 24 ]]' in source
    assert "exp10_fhp_hand_board_features.run" not in source
    assert "evaluate-resume" in source
    for number, (slug, _, _) in SPECS.items():
        launcher = ROOT / f"gcp/run_exp{number}_{slug}.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        env = dict(os.environ, PROJECT_ID="p", REGION="europe-west1", BUCKET="test-bucket",
            SA_EMAIL="r@example.com", REPO_REF="a"*40, RUN_ID=f"exp{number}-dryrun",
            FEATURE_TOTAL_HOURS="24", FEATURE_SOURCE_RUN_ID="")
        result = subprocess.run(["bash", str(launcher), "dry-run-evaluate"], env=env,
                                text=True, capture_output=True, check=True)
        assert "no cloud actions" in result.stdout


@pytest.mark.parametrize("number", tuple(SPECS))
def test_evaluation_only_matches_historical_exp10_at_24h(number):
    exp = get_experiment(number)
    records = [dict(experiment=e, seed=s, training_hours=h, checkpoint_path=f"/{e}/{s}/{h}",
        checkpoint_sha256="a"*64, nodes_touched=1) for e in (f"exp{number}", "exp10")
        for s in (0,1,2) for h in (6,12,18,24)]
    args = SimpleNamespace(smoke=False, rule_deals=10, lbr_deals=4, lbr_shard_deals=2,
        lbr_rollouts=16, base_seed=20260922, crossplay_deals=50)
    tasks = evaluate.build_tasks(records, args, experiment=exp)
    direct = [t for t in tasks if t["kind"] == "direct_crossplay"]
    assert len(direct) == 3
    assert {t["training_hours"] for t in tasks} == {24}
    assert all(t["policy_b_path"].startswith("/exp10/") for t in direct)
    assert not any(t["kind"] == "temporal_crossplay" for t in tasks)
    args.smoke = True
    assert {t["training_seed"] for t in evaluate.build_tasks(records,args,experiment=exp)} == {0}


@pytest.mark.ray
@pytest.mark.parametrize("number", tuple(SPECS))
def test_native_ray_lifecycle(number, tmp_path):
    if os.environ.get("RUN_STRATEGIC_RAY_TESTS") != "1":
        pytest.skip("Set RUN_STRATEGIC_RAY_TESTS=1 for real eight-actor lifecycle tests")
    from experiments.fhp.strategic_feature_suite.run import main
    main(["smoke", "--output-root", str(tmp_path / "smoke"), "--no-resume"], experiment=get_experiment(number))


@pytest.mark.parametrize("number", tuple(SPECS))
def test_sequential_policy_reload_and_grouping(number, tmp_path):
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
    from unbiased_escher.fhp_structured_solver import StructuredFHPGroupedWideUCVEscher
    from fhp_escher.checkpointing import save_policy_checkpoint, LoadedFHPPolicy
    exp = get_experiment(number)
    def sequential(config):
        kwargs = _solver_kwargs(0, config)
        kwargs.pop("cache_frozen_critic_targets")
        return StructuredFHPGroupedWideUCVEscher(**kwargs)
    solver = sequential(exp.smoke_config())
    old = sequential(baseline.smoke_config())
    assert solver.infostate_size == exp.encoder().policy_size
    states = [state_for(actions=a) for a in ((), (1,), (2,))]
    for i, state in enumerate(states * 2):
        mask = np.asarray(state.legal_actions_mask(), np.float32)
        for learner in (solver, old):
            trainer = learner.ave_policy_trainer
            trainer.buffer.add(trainer.get_infostate_tensor(state), mask/mask.sum(), mask, i%3+1)
    a, b = solver.ave_policy_trainer._grouped_training_data(3), old.ave_policy_trainer._grouped_training_data(3)
    assert torch.equal(a[0][:,:213], b[0])
    for x,y in zip(a[1:], b[1:]):
        assert torch.equal(x,y)
    path = tmp_path / "policy.pkl"
    save_policy_checkpoint(solver, path, seed=0, config=exp.smoke_config(), checkpoint_row={})
    loaded = LoadedFHPPolicy(load_fhp_game(), path)
    for state in states:
        assert sum(loaded.action_probabilities(state).values()) == pytest.approx(1)
    baseline_path = tmp_path / "baseline.pkl"
    save_policy_checkpoint(old, baseline_path, seed=0, config=baseline.smoke_config(), checkpoint_row={})
    # Exercise actual duplicate gameplay and exact-flop LBR paths, not only
    # mocked reporting. These tiny, untrained policies have no scientific value.
    task = dict(task_id="real-smoke", policy_a_path=str(path), policy_b_path=str(baseline_path),
                policy_a_name="candidate", policy_b_name="baseline", evaluation_seed=29033,
                num_deals=2, lbr_rollouts=2, lbr_seed=193, opponent=evaluate.protocol.PUBLISHED_AGENT_NAMES[0])
    for kind in ("direct_crossplay", "rule", "lbr"):
        result = evaluate.protocol._evaluation_worker(dict(task, kind=kind))
        assert np.isfinite(result["mean_mbb_per_hand"])
