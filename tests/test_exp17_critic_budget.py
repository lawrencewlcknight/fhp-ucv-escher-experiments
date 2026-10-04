"""Native-prefix equivalence, recursive estimator fidelity and audit contracts."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess

import numpy as np
import pytest
import torch

from experiments.fhp.exp17_fhp_critic_budget import config as c
from experiments.fhp.exp17_fhp_critic_budget.fitting import audit_member, cpu_state, averaged_target
from experiments.fhp.exp17_fhp_critic_budget.diagnostics import FrozenRollouts, evaluate
from experiments.fhp.exp17_fhp_critic_budget.run import synthetic_solver, audit_boundary, run_worker
from experiments.fhp.exp17_fhp_critic_budget.aggregate import aggregate, paired_summary
from experiments.fhp.exp10_fhp_hand_board_features.smoke import assert_identical
from experiments.critic_target_cache_benchmark import rng_hash, snapshot
from gcp.exp17_critic_budget_batch import build_job


def comparable_snapshot(member):
    state = snapshot(member)
    state["history"] = list(state["history"])
    return state


@pytest.fixture
def solver():
    torch.set_num_threads(1)
    return synthetic_solver(0)


def test_contract():
    config = c.contract()
    assert config["updates"] == [5000, 10000]
    assert config["boundaries"] == 3 and config["seeds"] == [0, 1, 2]
    assert config["probes_per_player_fold"] * 4 == 64
    assert config["repeats_per_probe"] == 128
    assert c.SOURCE_RUN_ID == "exp10-features-20261001-161740"


def test_native_prefix_matches_two_independent_native_fits(solver):
    member = solver.q_value_trainer.members[0]
    member.train_steps = 4
    # Fill the temporal window, so replacing vs incorrectly appending both
    # checkpoints is distinguishable in this test.
    for _ in range(4):
        member.train_model(solver.num_iteration)
    initial = snapshot(member)
    sampler = deepcopy(member.buffer.rng.bit_generator.state)
    global_rng = solver._capture_rng_state()
    metric, states, _ = audit_member(member, solver.num_iteration, updates=[2, 4], probe_rows=16, diagnostic_seed=1)
    full_state, full_rng = comparable_snapshot(member), rng_hash(member)
    for updates in (2, 4):
        member.model.load_state_dict(initial["model"])
        member.target_model.load_state_dict(initial["target"])
        member.optimizer.load_state_dict(deepcopy(initial["optimizer"]))
        member.target_history = deepcopy(initial["history"])
        member.target_version = initial["version"]
        member.buffer.rng.bit_generator.state = deepcopy(sampler)
        solver._restore_rng_state(global_rng)
        member.train_steps = updates
        member.train_model(solver.num_iteration)
        assert_identical(states[updates]["online"], cpu_state(member.model))
        assert_identical(states[updates]["deployed"], cpu_state(member.target_model))
        if updates == 4:
            assert_identical(full_state, comparable_snapshot(member))
            assert full_rng == rng_hash(member)
    assert metric["target_version_after"] == metric["target_version_before"] + 1
    assert metric["arms"]["2"]["prefix_seconds_including_cache"] < metric["arms"]["4"]["prefix_seconds_including_cache"]


def frozen_identity(solver):
    targets = [cpu_state(m.target_model) for m in solver.q_value_trainer.members]
    snapshots = [{u: dict(online=t, deployed=t) for u in (1, 2)} for t in targets]
    return snapshots, targets


@pytest.mark.parametrize("traverser,fold", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_readonly_recursive_estimator_matches_native_dfs(solver, traverser, fold):
    snapshots, targets = frozen_identity(solver)
    audit = FrozenRollouts(solver, snapshots, targets, [1, 2])
    state = audit.select_state(traverser, fold, np.random.default_rng(83))
    original_add = solver.regret_trainers[traverser].add_data
    recorded = []
    def add(features, advantage, *args):
        recorded.append(advantage.copy())
        return original_add(features, advantage, *args)
    solver.regret_trainers[traverser].add_data = add
    solver.q_value_trainer.begin_trajectory(fold)
    for rng_seed in range(5):
        np.random.seed(rng_seed)
        expected = solver.dfs(state.clone(), traverser)
        root_advantage = recorded[-1]
        np.random.seed(rng_seed)
        actual, advantage = audit.rollout(state.clone(), traverser, fold, np.random)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(advantage, np.stack([root_advantage, root_advantage]), rtol=1e-12, atol=1e-12)
    native, _ = solver.q_value_trainer.get_baseline_and_disagreement(state, traverser)
    np.testing.assert_allclose(audit.info(state, traverser, fold)[3][0], native)


def test_same_critics_give_identical_diagnostics_and_preserve_rng(solver):
    snapshots, targets = frozen_identity(solver)
    before = [rng_hash(m) for m in solver.q_value_trainer.members]
    rows = evaluate(solver, snapshots, targets, c.contract(True), seed=0, boundary=1)
    assert before == [rng_hash(m) for m in solver.q_value_trainer.members]
    assert len(rows) == 4
    for row in rows:
        assert row["arms"]["1"] == row["arms"]["2"]
        assert row["paired_advantage_mean_difference"] == [0.] * solver.action_size
        assert row["max_policy_centred_residual"] < 1e-12
        assert row["heldout_critic"] == 1-row["trajectory_fold"]


def test_heldout_td_matches_native_target_formula(solver):
    from adaptive_escher.frozen_target_cache import frozen_targets
    snapshots, targets = frozen_identity(solver)
    audit = FrozenRollouts(solver, snapshots, targets, [1, 2])
    for traverser in (0, 1):
        for fold in (0, 1):
            state = audit.select_state(traverser, fold, np.random.default_rng(2))
            for seed in range(6):
                action, actual = audit.heldout_td(state, traverser, fold, np.random.default_rng(seed))
                rng = np.random.default_rng(seed)
                proposal = audit.info(state, traverser, fold)[2]
                assert action == int(rng.choice(len(proposal), p=proposal))
                child = audit.chance(state.child(action), rng)
                terminal = child.is_terminal()
                next_state = np.zeros(solver.infostate_size) if terminal else solver.get_infostate_tensor(child)
                mask = np.zeros(solver.action_size) if terminal else child.legal_actions_mask()
                batch = tuple(torch.tensor(np.asarray([item]), dtype=dtype) for item, dtype in (
                    (solver.get_history_tensor(state), torch.float32),
                    (solver.get_history_tensor(child), torch.float32), (next_state, torch.float32),
                    (child.returns()[0]/solver.max_utility, torch.float32), (mask, torch.int64),
                    (child.current_player(), torch.int64), (action, torch.int64), (int(terminal), torch.int64)))
                expected = frozen_targets(solver.q_value_trainer.members[1-fold], batch, solver.num_iteration).item()
                assert actual == pytest.approx(expected, abs=1e-6)


def test_observing_entire_iteration_preserves_native_learning(solver, tmp_path):
    # Two identically initialised learners, one with the audit hooks and one
    # with the original iteration; diagnostics must not leak into the next fit.
    reference = synthetic_solver(0)
    rng = solver._capture_rng_state()
    audit_boundary(solver, c.contract(True), seed=0, boundary=1, directory=tmp_path)
    observed_rng = solver._capture_rng_state()
    reference._restore_rng_state(rng)
    reference.iteration()
    assert_identical(observed_rng, reference._capture_rng_state())
    assert solver.nodes_touched == reference.nodes_touched
    for a, b in zip(solver.q_value_trainer.members, reference.q_value_trainer.members):
        assert_identical(comparable_snapshot(a), comparable_snapshot(b))
        assert_identical(a.buffer.rng.bit_generator.state, b.buffer.rng.bit_generator.state)
    for a, b in zip(solver.regret_trainers, reference.regret_trainers):
        assert_identical(cpu_state(a.model), cpu_state(b.model))
    assert_identical(cpu_state(solver.calibration_trainer.model), cpu_state(reference.calibration_trainer.model))


def test_full_smoke_report_reuse_and_corruption(tmp_path):
    root = tmp_path / "audit"
    worker = run_worker(None, root, 0, smoke=True)
    analysis = aggregate(root, smoke=True)
    assert (analysis / "report.md").exists()
    assert not list(root.rglob("training_states"))
    assert len(list(worker.glob("*.pt"))) == 2
    assert json.loads((analysis / "summary.json").read_text())["comparisons"]["mean_legal_action_advantage_variance"]["independent_source_seeds"] == 1
    before = (worker / "summary.json").read_bytes()
    run_worker(None, root, 0, smoke=True)
    assert (worker / "summary.json").read_bytes() == before
    path = worker / "boundary_1.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        aggregate(root, smoke=True)
    with pytest.raises(ValueError, match="corrupt"):
        run_worker(None, root, 0, smoke=True)


def test_pairing_uses_seeds_not_rollouts():
    result = paired_summary([1, 2, 3], [2, 3, 4])
    assert result["independent_source_seeds"] == 3
    assert result["short_minus_full"] == -1
    assert result["paired_difference_95_t_interval"] == [-1., -1.]


@pytest.mark.parametrize("kind", ["controller", "smoke", "train", "aggregate"])
def test_cloud_wiring(kind, tmp_path):
    args = Namespace(kind=kind, repo_url="https://example.org/repo.git", repo_ref="a"*40,
        bucket_root="gs://test-bucket", run_id="exp17-test", project_id="test-project", region="europe-west1",
        service_account="runner@test-project.iam.gserviceaccount.com", parallelism=3, controller_action="orchestrate")
    job = build_job(args)
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    script = spec["runnables"][0]["script"]["text"]
    path = tmp_path / "generated.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert spec["maxRetryCount"] == 0
    if kind in ("train", "smoke"):
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
        assert spec["computeResource"]["cpuMilli"] == 16000
        assert "3.11.16" in script and "fetch-source" in script
        assert 'SOURCE="$WORK_ROOT/source"' in script
        assert group["taskCount"] == (3 if kind == "train" else 1)
    if kind == "controller":
        assert "run_exp17_critic_budget.sh" in script
    if kind == "aggregate":
        assert "--exclude='\\.pt$'" in script


def test_production_rejects_synthetic_and_nested_source(tmp_path):
    with pytest.raises(ValueError, match="Production requires"):
        run_worker(None, tmp_path, 0)
    with pytest.raises(ValueError, match="outside"):
        run_worker(tmp_path / "source", tmp_path, 0)


def test_temporal_average_discards_oldest_not_newest():
    history = [{"x": torch.tensor([float(i)])} for i in (1, 2, 3, 4)]
    target = averaged_target(history, {"x": torch.tensor([9.])}, 4)
    assert target["x"].item() == (2+3+4+9)/4


@pytest.mark.skipif(os.environ.get("RUN_EXP17_RAY_TEST") != "1", reason="Explicit native eight-actor integration")
def test_native_parallel_state_restore_and_audit(tmp_path):
    from experiments.fhp.exp10_fhp_hand_board_features import config as source, training_state as ts
    from experiments.fhp.exp10_fhp_hand_board_features.worker import _make_solver
    from experiments.fhp.exp10_fhp_hand_board_features.smoke import learning_state
    config = source.smoke_config()
    config["baseline_network_train_steps"] = 2
    kwargs = dict(seed=0, config=config, repository_commit="audit-smoke")
    solver = _make_solver(0, config, smoke=True)
    try:
        solver.iteration()
        path = tmp_path / "source.pt"
        ts.save_training_state(path, ts.build_training_state(solver, **kwargs,
            checkpoint_id="time_24h", captured_checkpoints=[]))
        audit_boundary(solver, c.contract(True), seed=0, boundary=1, directory=tmp_path)
        observed = learning_state(solver)
    finally:
        solver.close()
    solver = _make_solver(0, config, smoke=True)
    try:
        ts.restore_training_state(solver, ts.read_training_state(path), **kwargs)
        solver.iteration()
        expected = learning_state(solver)
        assert_identical(observed, expected)
    finally:
        solver.close()
