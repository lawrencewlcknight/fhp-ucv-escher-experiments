"""Pre-fit phase, no-op control, native equivalence, isolation and single-job tests."""
from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from experiments.fhp.exp17_fhp_critic_budget.fitting import cpu_state, replay_digest
from experiments.fhp.exp22_fhp_critic_objectives.data import frozen_sampler, generate_panel
from experiments.fhp.exp22_fhp_critic_objectives.fitting import fit_objective, optimizer_hash
from experiments.fhp.exp22_fhp_critic_objectives.run import source_identity
from experiments.fhp.exp24_fhp_critic_budget_stages.config import contract, STAGES, UPDATES
from experiments.fhp.exp24_fhp_critic_budget_stages.reference import collect, make_solver
from experiments.fhp.exp24_fhp_critic_budget_stages.states import read_state, restore_reference, load_audit_solver
from experiments.fhp.exp24_fhp_critic_budget_stages.run import main, verify_checkpoint
from experiments.fhp.exp24_fhp_critic_budget_stages.aggregate import aggregate
from gcp.exp24_critic_stages_batch import build_job, actions
from gcp.exp24_critic_stages_gate import validate


@pytest.fixture
def solver():
    torch.set_num_threads(1)
    return make_solver(smoke=True)


def test_protocol_includes_early_prefixes_and_noop():
    assert UPDATES == (0, 100, 500, 1000, 2500, 5000, 10000)
    assert len({contract(stage=s)["random_seed"] for s in STAGES}) == 3
    assert contract()["native_replay_rng"]
    assert contract()["split_roots"] == dict(fit=256, selection=64, assessment=128)


def test_snapshot_does_not_change_reference_trajectory(solver, tmp_path):
    result = collect(solver, tmp_path/"reference", smoke=True)
    actual = source_identity(solver)
    rng = optimizer_hash(solver._capture_rng_state())
    ordinary = make_solver(smoke=True)
    ordinary.iteration()
    ordinary.iteration()
    assert source_identity(ordinary) == actual
    assert optimizer_hash(ordinary._capture_rng_state()) == rng
    assert result["final_iteration"] == 2
    for row in result["stages"]:
        pre, post = row["pre_fit"], row["checkpoint"]
        assert pre["iteration"] == post["iteration"]
        assert pre["nodes"] == post["nodes"]
        assert post["critic_versions"] == [x+1 for x in pre["critic_versions"]]
        assert pre["phase"].startswith("before_ensemble_fit")
        assert not pre["resumable"] and post["resumable"]


def test_full_checkpoint_resume_next_iteration(solver, tmp_path):
    collect(solver, tmp_path/"reference", smoke=True)
    solver.iteration()
    expected = source_identity(solver)
    other = make_solver(smoke=True)
    restore_reference(tmp_path/"reference/checkpoints/six_hours", other)
    other.iteration()
    assert source_identity(other) == expected
    verify_checkpoint(tmp_path, smoke=True)
    with pytest.raises(ValueError, match="phase"):
        restore_reference(tmp_path/"reference/pre_fit/six_hours", other)


def test_prefit_extraction_preserves_models_and_optimizer(solver, tmp_path):
    collect(solver, tmp_path/"reference", smoke=True)
    payload, metadata = read_state(tmp_path/"reference/pre_fit/six_hours", pre_fit=True)
    extracted, _ = load_audit_solver(tmp_path/"reference/pre_fit/six_hours")
    assert extracted.num_iteration == metadata["iteration"]
    assert not hasattr(extracted, "_workers")
    for member, state in zip(extracted.q_value_trainer.members, payload["q_ensemble"]["members"]):
        assert optimizer_hash(member.model.state_dict()) == optimizer_hash(state["model"])
        assert optimizer_hash(member.optimizer.state_dict()) == optimizer_hash(state["optimizer"])
        assert optimizer_hash(member.target_history) == optimizer_hash(state["target_history"])


def test_zero_noop_and_positive_native_fit_equivalence(solver):
    solver.iteration()
    member = solver.q_value_trainer.members[0]
    original = source_identity(solver)
    panel, _ = generate_panel(frozen_sampler(solver), contract(True), seed=0, member=0, split="fit", seen=set())
    saved = []
    rows = fit_objective(member, solver.num_iteration, panel, "replay_td", contract(True), seed=0, fold=0,
                        callback=lambda c, m: saved.append(c))
    zero = saved[0]
    assert rows[0]["fit_prefix_seconds"] == rows[0]["target_cache_seconds"] == 0
    assert optimizer_hash(zero["deployed"]) == optimizer_hash(member.target_model.state_dict())
    assert optimizer_hash(zero["optimizer"]) == optimizer_hash(member.optimizer.state_dict())
    assert zero["sampler_rng"] == member.buffer.rng.bit_generator.state
    for candidate in saved[1:]:
        native = deepcopy(member)
        native.train_steps = candidate["updates"]
        type(native).train_model(native, solver.num_iteration)
        for name, model in (("online", native.model), ("deployed", native.target_model)):
            assert optimizer_hash(candidate[name]) == optimizer_hash(model.state_dict())
    assert source_identity(solver) == original


def test_one_iteration_late_boundary(solver, tmp_path):
    solver.iteration()
    solver._resume_training_elapsed_seconds = 86400.
    result = collect(solver, tmp_path/"late_reference", smoke=True, late=True, lineage=dict(synthetic=True))
    assert result["final_iteration"] == 2 and result["source_iteration"] == 1
    assert result["stages"][0]["pre_fit"]["elapsed_training_seconds"] > 86400
    assert result["stages"][0]["checkpoint"] is None


def test_snapshot_overhead_excluded_from_clock(solver, tmp_path, monkeypatch):
    from experiments.fhp.exp24_fhp_critic_budget_stages import reference
    real_save, original_fit = reference.save_state, solver.q_value_trainer.train_model
    clock = [0.]
    def save(*a, **k):
        value = real_save(*a, **k)
        clock[0] += 50.
        return value
    def fit(*a, **k):
        value = original_fit(*a, **k)
        clock[0] += 10.
        return value
    monkeypatch.setattr(reference.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(reference, "save_state", save)
    monkeypatch.setattr(solver.q_value_trainer, "train_model", fit)
    result = collect(solver, tmp_path/"reference", smoke=True)
    assert result["added_active_training_seconds"] == 20.
    assert result["reference_wall_seconds"] == 220.
    assert result["snapshot_and_upload_seconds"] == 200.


def test_full_smoke_reports_curves_and_prevents_assessment_reselection(tmp_path):
    main(["smoke-local", "--output", str(tmp_path)])
    primary = json.loads((tmp_path/"analysis/exp24/summary.json").read_text())
    secondary = json.loads((tmp_path/"analysis/exp25/summary.json").read_text())
    assert primary["seed_level_confidence_intervals"] is None
    assert len(primary["curves"]["one_hour"]) == 4
    assert len(secondary["curves"]["one_hour"]) == 16
    detail = json.loads((tmp_path/"analysis/per_stage.json").read_text())
    for stage in STAGES[:2]:
        assert detail[stage]["summary"]["candidates"] == 32
        for row in detail[stage]["fitting"]:
            assert row["fixed_fit_panel"]["fixed_fitting_traces"] == 16
            assert set(row["fixed_fit_panel"]["online"]["objectives"]) == {
                "replay_td_probe", "matched_td", "weighted_td", "regret_variance"}
    # Assessment corruption must not be silently accepted by aggregation.
    (tmp_path/"audits/one_hour/assessment.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        aggregate(tmp_path, smoke=True, local=True)


@pytest.mark.parametrize("kind", ("run", "smoke"))
def test_one_batch_job_no_child_jobs_or_global_parallelism(kind, tmp_path):
    args = Namespace(kind=kind, run_id="exp24-critic-test", bucket_root="gs://test-bucket", repo_ref="a"*40,
        service_account="runner@example.iam.gserviceaccount.com", project_id="test", region="europe-west1",
        repo_url="https://example.org/repo.git")
    job = build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 1
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    assert group["taskSpec"]["runnables"][-1]["alwaysRun"]
    for runnable in group["taskSpec"]["runnables"]:
        script = runnable["script"]["text"]
        assert "batch jobs submit" not in script
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    code = actions(args)
    assert code.index(" late ") < code.index(" reference ") < code.index(" verify-checkpoint ") < code.index(" audit ") < code.index(" aggregate ")
    assert "--ray-smoke" in code if kind == "smoke" else "--smoke" not in code


def test_gate_rejects_local_smoke_or_different_commit():
    summary = dict(stages=list(STAGES), checkpoint_verification=dict(ray_actors=8),
        audits={s: dict(source_state_preserved=True, candidates=32) for s in STAGES},
        references=dict(late_reference=dict(stages=[dict(pre_fit=dict(lineage=dict(source_run_id="exp10-features-20261001-161740")))])))
    raw = json.dumps(summary).encode()
    success = dict(status="complete", smoke=True, local_smoke=False, commit="a"*40,
        experiments=[24, 25], stages=list(STAGES), artifacts={"analysis/summary.json": hashlib.sha256(raw).hexdigest()})
    validate(json.dumps(success), raw, "a"*40)
    with pytest.raises(ValueError):
        validate(json.dumps(success), raw, "b"*40)
    success["local_smoke"] = True
    with pytest.raises(ValueError):
        validate(json.dumps(success), raw, "a"*40)


@pytest.mark.skipif(__import__("os").environ.get("FHP_TEST_RAY") != "1", reason="Opt-in eight-actor process test")
def test_eight_actor_reference_checkpoint_smoke(tmp_path):
    main(["smoke-local", "--ray-smoke", "--output", str(tmp_path)])
    assert json.loads((tmp_path/"checkpoint_verification.json").read_text())["ray_actors"] == 8
