"""Estimator algebra, independent splits, native deployment and scientific gates."""
from argparse import Namespace
from copy import deepcopy
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from unbiased_escher.estimator import control_variate_advantage
from experiments.fhp.exp17_fhp_critic_budget.run import synthetic_solver
from experiments.fhp.exp17_fhp_critic_budget.fitting import cpu_state, averaged_target
from experiments.fhp.exp22_fhp_critic_objectives.config import contract, OBJECTIVES
from experiments.fhp.exp22_fhp_critic_objectives.estimator import corrected, recursive_estimates, paired_variance_loss, root_td_loss
from experiments.fhp.exp22_fhp_critic_objectives.data import frozen_sampler, record_trace, generate_panel, assess
from experiments.fhp.exp22_fhp_critic_objectives.fitting import fit_objective
from experiments.fhp.exp22_fhp_critic_objectives.run import run_worker, choose_candidates, source_identity
from experiments.fhp.exp22_fhp_critic_objectives.aggregate import aggregate
from gcp.exp22_critic_objectives_batch import build_job
from gcp.exp22_critic_objectives_gate import validate


@pytest.fixture
def solver():
    torch.set_num_threads(1)
    return synthetic_solver(0)


def test_contract():
    c = contract()
    assert c["updates"] == [2500, 5000, 10000]
    assert c["source_run_id"] == "exp10-features-20261001-161740"
    assert c["stage"] == "frozen_24h_only"
    assert len(c["objectives"]) == 4


@pytest.mark.parametrize("action", (0, 1, 2))
def test_torch_correction_matches_production(action):
    q, policy, proposal = np.array([1.3, -2., .4]), np.array([.1, .6, .3]), np.array([.3, .2, .5])
    native = control_variate_advantage(q, beta=np.ones(3), sampled_action=action,
        sample_probability=proposal[action], sampled_return=2.5, policy=policy, legal_actions_mask=np.ones(3))
    value, advantage = corrected(torch.tensor(q[None]), torch.tensor(policy[None]), torch.ones(1, 3),
                                 torch.tensor([action]), torch.tensor([proposal[action]]), torch.tensor([2.5]))
    np.testing.assert_allclose(value, native.policy_value, atol=1e-12)
    np.testing.assert_allclose(advantage[0], native.advantages, atol=1e-12)


def test_exact_conditional_unbiasedness_variance_identity_and_gradient():
    # Enumerate independent action/return pairs, not Monte Carlo assertions.
    policy, proposal = torch.tensor([.2, .3, .5], dtype=torch.double), torch.tensor([.1, .5, .4], dtype=torch.double)
    true_values = torch.tensor([2., -1., 4.], dtype=torch.double)
    baseline = torch.tensor([.7, .1, -2.], dtype=torch.double, requires_grad=True)
    estimates, probabilities = [], []
    for action, noise in itertools.product(range(3), (-.4, .4)):
        _, r = corrected(baseline[None], policy[None], torch.ones(1, 3), torch.tensor([action]),
                         proposal[action:action+1], true_values[action:action+1]+noise)
        estimates.append(r[0]); probabilities.append(proposal[action]/2)
    estimates, probabilities = torch.stack(estimates), torch.stack(probabilities)
    mean = (estimates*probabilities[:, None]).sum(0)
    torch.testing.assert_close(mean, true_values-policy @ true_values)
    variance = ((estimates-mean).square().mean(1)*probabilities).sum()
    pairs = ((estimates[:, None]-estimates[None, :]).square().mean(2)
             * probabilities[:, None]*probabilities[None, :]).sum()/2
    torch.testing.assert_close(variance, pairs)
    pairs.backward()
    assert torch.isfinite(baseline.grad).all() and baseline.grad.abs().sum() > 0
    # Centring does not make zero critics or differing true actions zero-variance.
    assert variance.item() > 0


def test_pair_loss_preserves_state_specific_mean():
    r = torch.tensor([[3., -2.], [1., 0.], [6., 2.], [5., 3.]])
    offsets = torch.tensor([[7., -8.], [7., -8.], [-4., 2.], [-4., 2.]])
    assert paired_variance_loss(r, torch.ones_like(r)) == paired_variance_loss(r+offsets, torch.ones_like(r))


def test_weighted_td_expected_risk_is_uniform_over_legal_actions():
    model = torch.nn.Linear(1, 3, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.], [-1.], [99.]]))
    proposal = (.8, .2)
    expected = 0.
    for action in (0, 1):
        batch = dict(features=torch.ones(1, 1, 1, dtype=torch.double),
                     action=torch.tensor([[action]]), td_target=torch.tensor([.5]),
                     mask=torch.tensor([[[1., 1., 0.]]]),
                     probability=torch.tensor([[proposal[action]]], dtype=torch.double))
        expected += proposal[action]*root_td_loss(model, batch, weighted=True)
    torch.testing.assert_close(expected, torch.tensor(((2-.5)**2+(-1-.5)**2)/2, dtype=torch.double))


def test_two_level_exact_tree_unbiased_even_with_inaccurate_critics():
    policy = torch.tensor([[.3, .7]], dtype=torch.double)
    proposal = torch.tensor([.6, .4], dtype=torch.double)
    leaves = torch.tensor([[1., 4.], [-2., 3.]], dtype=torch.double)
    truth = (leaves*policy).sum(1)
    for oracle in (False, True):
        qroot = truth.clone() if oracle else torch.tensor([8., -4.], dtype=torch.double)
        qchild = leaves.clone() if oracle else torch.zeros_like(leaves)
        qroot.requires_grad_(); qchild.requires_grad_()
        values, probs = [], []
        for a, b in itertools.product(range(2), repeat=2):
            child, _ = corrected(qchild[a:a+1], policy, torch.ones(1, 2), torch.tensor([b]),
                                  proposal[b:b+1], leaves[a, b:b+1])
            _, root = corrected(qroot[None], policy, torch.ones(1, 2), torch.tensor([a]),
                                 proposal[a:a+1], child)
            values.append(root[0]); probs.append(proposal[a]*proposal[b])
        values, probs = torch.stack(values), torch.stack(probs)
        mean = (values*probs[:, None]).sum(0)
        torch.testing.assert_close(mean, truth-(truth*policy[0]).sum())
        grad = torch.autograd.grad(mean.sum(), (qroot, qchild), retain_graph=True)
        assert max(g.abs().max().item() for g in grad) < 1e-12
        variance = ((values-mean).square().mean(1)*probs).sum().item()
        assert variance < 1e-25 if oracle else variance > 1


@pytest.mark.parametrize("member", (0, 1))
def test_recorded_recursive_estimator_matches_native_readonly(solver, member):
    sampler = frozen_sampler(solver)
    panel, _ = generate_panel(sampler, contract(True), seed=0, member=member, split="fit", seen=set())
    model = solver.q_value_trainer.members[member].target_model
    _, actual = recursive_estimates(model, panel.batch(np.arange(panel.trace_count)))
    for index, root in enumerate(panel.roots):
        state = solver.game.new_initial_state()
        for action in root["history"]:
            state.apply_action(action)
        for repeat in range(4):
            rng = np.random.default_rng(np.random.SeedSequence([contract(True)["random_seed"], 0, member, 0, 1, index, repeat]))
            _, expected = sampler.rollout(state, root["traverser"], 1-member, rng)
            np.testing.assert_allclose(actual[index*4+repeat].detach(), expected[0], atol=2e-6, rtol=2e-5)
    measured = assess(model, panel, 64)
    assert measured["summary"]["max_centring_error"] < 1e-10


def test_split_roots_disjoint_and_generation_preserves_training_state(solver):
    seen, sampler = set(), frozen_sampler(solver)
    before = source_identity(solver)
    sets = []
    for fold in (0, 1):
        for split in ("fit", "selection", "assessment"):
            panel, metadata = generate_panel(sampler, contract(True), seed=0, member=fold, split=split, seen=seen)
            sets.append({r["history_sha256"] for r in panel.roots})
            assert len(panel.arrays["terminal"]) == 16
            assert metadata["trace_count"] == 16
    assert len(set.union(*sets)) == sum(map(len, sets))
    assert source_identity(solver) == before


def test_candidate_prefixes_common_initialisation_and_native_average(solver):
    member = solver.q_value_trainer.members[0]
    before = source_identity(solver)
    panel, _ = generate_panel(frozen_sampler(solver), contract(True), seed=0, member=0, split="fit", seen=set())
    initials, optimiser_initials, matched_streams = [], [], []
    for objective in OBJECTIVES:
        saved = []
        rows = fit_objective(member, solver.num_iteration, panel, objective, contract(True), seed=0, fold=0,
                             callback=lambda c, m: saved.append(c))
        initials.extend(r["initial_online_sha256"] for r in rows)
        optimiser_initials.extend(r["initial_optimizer_sha256"] for r in rows)
        if objective != "replay_td":
            matched_streams.append([r["minibatch_stream_sha256"] for r in rows])
        assert all(r["initialization_seconds"] <= r["fit_prefix_seconds"] for r in rows)
        assert all(r["cumulative_online_input_rows"] >= r["cumulative_root_traces"] for r in rows)
        assert [c["updates"] for c in saved] == [1, 2, 3]
        for candidate in saved:
            expected = averaged_target(member.target_history, candidate["online"], member.target_average_window)
            for key in expected:
                torch.testing.assert_close(expected[key], candidate["deployed"][key], rtol=0, atol=0)
        assert saved[0]["optimizer"]["state"]
    assert len(set(initials)) == 1
    assert len(set(optimiser_initials)) == 1
    assert matched_streams[0] == matched_streams[1] == matched_streams[2]
    assert source_identity(solver) == before


def test_replay_td_prefix_matches_native_cached_fit(solver):
    member = solver.q_value_trainer.members[0]
    saved = []
    config = contract(True)
    panel, _ = generate_panel(frozen_sampler(solver), config, seed=0, member=0, split="fit", seen=set())
    fit_objective(member, solver.num_iteration, panel, "replay_td", config, seed=0, fold=0,
                  callback=lambda candidate, metrics: saved.append(candidate))
    for candidate in saved:
        reference = deepcopy(member)
        reference.buffer.rng = np.random.default_rng(np.random.SeedSequence([config["random_seed"], 0, 0, 91]))
        reference.train_steps = candidate["updates"]
        # Bypass the source's instance-level timer wrapper; call the native class fit.
        type(reference).train_model(reference, solver.num_iteration)
        for representation, model in (("online", reference.model), ("deployed", reference.target_model)):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, candidate[representation][key], rtol=0, atol=0)


def test_full_local_smoke_selection_and_corruption(tmp_path):
    directory = tmp_path / "workers" / "seed_0"
    run_worker(None, directory, 0, smoke=True)
    result = aggregate(tmp_path, smoke=True)
    summary = json.loads((result / "summary.json").read_text())
    assert summary["smoke"] and summary["seeds"] == [0]
    report = (result / "report.md").read_text()
    assert "Selection-locked candidates" in report and "Original deployed variance" in report
    assert summary["comparisons"]["regret_variance_minus_best_td_control"]["paired_seed_t95"] is None
    rows = json.loads((directory / "fitting.json").read_text())
    selected = choose_candidates(rows)
    for row in rows:
        row["assessment"] = {"mean_variance": -999999}
    assert choose_candidates(rows) == selected
    with pytest.raises(FileExistsError):
        run_worker(None, directory, 0, smoke=True)
    (directory / "selection_lock.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        aggregate(tmp_path, smoke=True)


def test_source_extractor_uses_native_features_and_does_not_start_ray(solver):
    from experiments.fhp.exp22_fhp_critic_objectives.source import extract_solver
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.training_state import build_training_state
    from experiments.fhp.exp10_fhp_hand_board_features import config as baseline, training_state as ts
    from experiments.fhp.exp17_fhp_critic_budget.config import TRAINING_REF
    # Production identity on a tiny fixture: test extraction logic, not acceptance
    # of this fixture as cloud evidence (the real path is checksum/manifest guarded).
    payload = build_training_state(solver, seed=0, checkpoint_id="time_24h", repository_commit=TRAINING_REF,
                                    config=baseline.EXPERIMENT_CONFIG, captured_checkpoints=[])
    payload["type"] = ts.STATE_TYPE
    extracted = extract_solver(payload)
    assert not hasattr(extracted, "_workers")
    for actual, expected in zip(extracted.q_value_trainer.members, solver.q_value_trainer.members):
        for key in cpu_state(expected.model):
            torch.testing.assert_close(cpu_state(actual.model)[key], cpu_state(expected.model)[key])
    assert extracted.calibration_trainer.target_version == solver.calibration_trainer.target_version


@pytest.mark.parametrize("kind", ("smoke", "run", "pilot", "full", "aggregate-pilot", "aggregate"))
def test_batch_jobs_bash_and_limits(kind, tmp_path):
    args = Namespace(kind=kind, run_id="exp22-critic-test", bucket_root="gs://example-bucket", repo_ref="a"*40,
        service_account="runner@example.iam.gserviceaccount.com", project_id="project", region="europe-west1",
        repo_url="https://example.org/repo.git", pilot_run_id="exp22-critic-pilot", parallelism=2)
    job = build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == (2 if kind == "full" else 1)
    assert group["taskSpec"]["maxRetryCount"] == 0
    if kind == "run":
        assert group["parallelism"] == 1
        assert group["taskSpec"]["maxRunDuration"] == "86400s"
        assert group["taskSpec"]["computeResource"] == {"cpuMilli": 16000, "memoryMib": 64000}
        assert "batch jobs submit" not in group["taskSpec"]["runnables"][0]["script"]["text"]
    for i, runnable in enumerate(group["taskSpec"]["runnables"]):
        path = tmp_path / f"script{i}.sh"
        path.write_text(runnable["script"]["text"])
        subprocess.run(["bash", "-n", str(path)], check=True)
    assert group["taskSpec"]["runnables"][-1]["alwaysRun"]
    assert "uv python install 3.11.16" in group["taskSpec"]["runnables"][0]["script"]["text"]


def test_gate_checks_source_mode_commit_and_hashes():
    import hashlib
    manifest = json.dumps(dict(commit="a"*40, config=dict(experiment_id=22, smoke=False), provenance={})).encode()
    summary = json.dumps(dict(source_state_preserved=True, candidates=24)).encode()
    success = json.dumps(dict(status="complete", seed=0, smoke=False, artifacts={
        "manifest.json": hashlib.sha256(manifest).hexdigest(), "summary.json": hashlib.sha256(summary).hexdigest()})).encode()
    validate(success, manifest, summary, "a"*40, False)
    with pytest.raises(ValueError):
        validate(success, manifest, summary, "b"*40, False)
    with pytest.raises(ValueError):
        validate(success, manifest, summary, "a"*40, True)
    with pytest.raises(ValueError):
        validate(success, manifest, summary+b" ", "a"*40, False)
