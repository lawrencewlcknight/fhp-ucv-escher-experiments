"""Coverage, native-estimator algebra, isolation, matched refits and Batch gates."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from experiments.fhp.exp28_fhp_rare_state_audit.config import contract
from experiments.fhp.exp28_fhp_rare_state_audit.common import digest, verify, file_hash
from experiments.fhp.exp28_fhp_rare_state_audit.panels import (
    build_panel, canonical_board, histories, make_state, from_case, feature_hash, categories, board_classes,
)
from experiments.fhp.exp28_fhp_rare_state_audit.replay import group_replay, training_groups, coverage
from experiments.fhp.exp28_fhp_rare_state_audit.fitting import arm_weights, refit
from experiments.fhp.exp28_fhp_rare_state_audit.continuation import (
    observe_traversal, observe_regret_fit, reservoir_decisions, run_continuation, ObservedWorker, ObservedSolver,
)
from experiments.fhp.exp28_fhp_rare_state_audit.diagnostics import estimator_probe
from experiments.fhp.exp28_fhp_rare_state_audit.run import synthetic_fixture, local_smoke
from experiments.fhp.exp28_fhp_rare_state_audit.aggregate import aggregate
from experiments.fhp.exp28_fhp_rare_state_audit.package import package
from experiments.fhp.exp17_fhp_critic_budget.run import synthetic_solver
from experiments.fhp.exp22_fhp_critic_objectives.run import source_identity
from experiments.fhp.exp22_fhp_critic_objectives.fitting import optimizer_hash
from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
from unbiased_escher.efficient_replay import CompactReservoirBuffer
from unbiased_escher.grouped_parallel_solver import StructuredGroupedTraversalWorker, ParallelStructuredGroupedUCVEscher
from gcp.exp28_rare_state_batch import build_job, actions
from gcp.exp28_rare_state_gate import validate as validate_gate


@pytest.fixture(scope="module")
def panel():
    torch.set_num_threads(1)
    return build_panel({}, contract(True))


def test_holdout_excludes_all_previously_inspected_board_classes(panel):
    dev={tuple(r["board_class"]) for r in panel["rows"] if r["partition"]=="development"}
    test={tuple(r["board_class"]) for r in panel["rows"] if r["partition"]=="assessment"}
    assert test and not dev & test
    prior=dict(boards=[dict(board=list(b)) for b in test],reference_deals=[dict(cards=[0,1,2,3,*list(next(iter(test)))])])
    new=build_panel(prior,contract(True))
    assert not test & {tuple(b) for b in new["holdout_board_classes"]}
    assert len({r["id"] for r in new["rows"]})==len(new["rows"])


@pytest.mark.parametrize("board,hand,pre,post,seat",[
    ((6,10,50),(2,14),(1,2,2,2,1),(1,2,2,2),1),
    ((24,26,27),(12,25),(2,2,2,1),(1,2,2,2),1),
])
def test_known_guaranteed_winner_dominance_all_1081_hidden_hands(board,hand,pre,post,seat):
    remaining=[c for c in range(52) if c not in board+hand]
    gaps=[]
    for other in combinations(remaining,2):
        state=make_state(board,hand,pre,post,seat,other)
        assert state.legal_actions()==[0,1] and state.child(1).is_terminal()
        assert state.child(1).returns()[seat]>0
        gaps.append((state.child(1).returns()[seat]-state.child(0).returns()[seat])/100)
    assert len(gaps)==1081 and set(gaps)=={13.}


def test_card_features_identify_category_and_suit_class(panel):
    encoder=FHPHandBoardFeatureEncoder()
    rows=[r for r in panel["rows"] if r["strict_winner"]]
    f=np.asarray([encoder.information_state(from_case(r),r["seat"]) for r in rows])
    assert categories(f).tolist()==[r["category"] for r in rows]
    assert board_classes(f)==[tuple(r["board_class"]) for r in rows]
    for row in rows[:10]:
        state=from_case(row)
        assert row["guaranteed_call_gain_bb"]==(state.child(1).returns()[row["seat"]]-state.child(0).returns()[row["seat"]])/100


def replay_fixture(panel):
    """Synthetic replay used ONLY for plumbing tests, never the scientific audit."""
    encoder=FHPHandBoardFeatureEncoder()
    targeted=next(r for r in panel["rows"] if r["partition"]=="development" and r["category"]==8)
    ordinary=next(r for r in panel["rows"] if r["partition"]=="development" and r["category"]<7)
    other=next(r for r in panel["rows"] if r["partition"]=="development" and r["feature_hash"]!=ordinary["feature_hash"] and r["category"]<7)
    holdout=next(r for r in panel["rows"] if r["partition"]=="assessment")
    cases=[targeted,targeted,ordinary,other,holdout,holdout]
    f=np.asarray([encoder.information_state(from_case(r),r["seat"]) for r in cases])
    policies=np.array([[0,1,0],[.04,.96,0],[.2,.8,0],[.3,.7,0],[0,1,0],[0,1,0]],dtype=np.float32)
    return dict(size=6,cur_id=1000,infostate=f,q_value=policies,
                q_value_mask=np.tile([1,1,0],(6,1)).astype(np.float32),iteration=np.array([[2],[3],[2],[3],[2],[3]],dtype=np.float32)),cases


def test_grouped_targets_effective_rows_and_holdout_exclusion(panel):
    buffer,cases=replay_fixture(panel)
    groups=group_replay(buffer,4,2)
    h=coverage(groups,panel,1000)
    assert h["retained_rows"]==6 and h["reservoir_stream_rows"]==1000
    row=h["exact"][cases[0]["id"]]
    np.testing.assert_allclose(row["target"],[.04*9/13,1-.04*9/13,0],rtol=1e-6)
    assert row["count"]==2 and row["effective_rows"]==pytest.approx(13**2/(4**2+9**2))
    assert (row["first_iteration"],row["last_iteration"])==(2,3)
    training,targeted,eligibility=training_groups(groups,panel,contract(True))
    assert eligibility["excluded_holdout_rows"]==2 and targeted.sum()==1
    assert not set(board_classes(training.features(slice(None)))) & set(map(tuple,panel["holdout_board_classes"]))
    wrong=deepcopy(buffer);wrong["q_value_mask"][1,1]=0
    with pytest.raises(ValueError):group_replay(wrong,4,2)


def test_weighted_controls_match_added_mass_count_and_scale(panel):
    buffer,_=replay_fixture(panel)
    # More ordinary mass makes the rare upweighting nontrivial.
    for key in ("infostate","q_value","q_value_mask","iteration"):
        buffer[key]=np.concatenate([buffer[key][:2],np.repeat(buffer[key][2:4],30,axis=0)])
    buffer["size"]=62
    groups=group_replay(buffer,4,2)
    groups,targeted,_=training_groups(groups,panel,contract(True))
    weights,spec=arm_weights(groups,targeted,contract(True),0)
    assert spec["target_multiplier"]>1 and spec["target_groups"]==spec["random_groups"]==1
    assert not set(spec["target_group_ids"]) & set(spec["random_group_ids"])
    for w in weights.values():
        assert w@ groups.masses==pytest.approx(groups.masses.sum())
    empty,spec=arm_weights(groups,np.zeros(len(targeted),bool),contract(True),0)
    assert set(empty)=={"ordinary"} and spec["status"].startswith("not_identifiable")


def test_matched_refits_all_arms_source_preservation_and_lock(panel,tmp_path):
    solver,(payload,template,_)=synthetic_fixture(0)
    buffer,_=replay_fixture(panel)
    groups=group_replay(buffer,4,2)
    result=refit(solver,payload,template,groups,panel,contract(True),0,tmp_path)
    assert result["candidates"]==6 and result["source_preserved"] and result["assessment_locked"]
    candidates=json.loads((tmp_path/"candidates.json").read_text())
    for updates in (1,2):
        rows=[r for r in candidates if r["updates"]==updates]
        assert len(rows)==3
        assert len({r["minibatch_stream_sha256"] for r in rows})==1
        assert len({r["initial_model_sha256"] for r in rows})==1
        assert len({r["initial_optimizer_sha256"] for r in rows})==1
    assert json.loads((tmp_path/"assessment.json").read_text())["lock_sha256"]==file_hash(tmp_path/"fitting_lock.json")


@pytest.mark.parametrize("capacity",[1,7,23])
def test_reservoir_preview_matches_native_without_consuming_rng(capacity):
    buffer=CompactReservoirBuffer(capacity,3,3,seed=18)
    offset=0
    for n in (0,3,4,50,1,120):
        f=np.arange(offset,offset+3*n).reshape(n,3).astype(np.float32)
        p=dict(infostates=f,values=f+1,legal_masks=np.ones_like(f),iterations=np.ones((n,1)))
        before=deepcopy(buffer.rng.bit_generator.state)
        dest,accepted,survivors=reservoir_decisions(buffer,p)
        assert buffer.rng.bit_generator.state==before
        buffer.add_batch(p)
        np.testing.assert_array_equal(buffer.infostate_buf[dest[survivors]],p["infostates"][survivors])
        assert set(survivors).issubset(accepted)
        offset+=3*n


def test_frozen_probe_uses_actual_corrected_estimator_and_preserves_source(panel):
    solver=synthetic_solver(4)
    before=source_identity(solver);rng=optimizer_hash(solver._capture_rng_state())
    result=estimator_probe(solver,panel,contract(True),4)
    assert result["rows"] and result["training_rng_preserved"] and result["no_samples_added"]
    for row in result["rows"]:
        for h in row["histories"]:
            assert h["expectation_max_error"]<1e-9
            assert h["call_minus_fold_bb"]>0
    assert source_identity(solver)==before
    assert optimizer_hash(solver._capture_rng_state())==rng


@pytest.mark.parametrize("scores",[(1.,9.,0.),(2.,8.,0.),(1.,2.,0.)])
def test_frozen_probe_normalises_float32_reference_without_changing_estimator(panel,monkeypatch,scores):
    solver=synthetic_solver(4)
    raw=solver.regret_trainers[0].regret_matching(np.array(scores,dtype=np.float32),[0,1])
    saved=raw.copy()
    assert raw.dtype==np.float32
    mass=float(raw.astype(np.float64).sum())
    assert abs(mass-1)>1e-9  # Includes both below-one and above-one rounding.
    for trainer in solver.regret_trainers:
        monkeypatch.setattr(trainer,"get_policy",lambda state,iteration:raw)
    before=source_identity(solver);rng=optimizer_hash(solver._capture_rng_state())
    result=estimator_probe(solver,panel,contract(True),4)
    old_errors=[]
    for row in result["rows"]:
        state=from_case(next(r for r in panel["rows"] if r["id"]==row["case_id"]))
        returns=np.array([state.child(a).returns()[state.current_player()]/solver.max_utility for a in (0,1)])
        for h in row["histories"]:
            np.testing.assert_array_equal(h["policy"],saved.astype(np.float64))
            assert h["policy_legal_mass"]==mass
            normalised=saved.astype(np.float64)/mass
            np.testing.assert_array_equal(h["normalised_policy"],normalised)
            mean=np.asarray(h["proposal"])[:2]@np.asarray(h["sampled_action_estimates"])
            exact=np.zeros(3);exact[:2]=returns-normalised[:2]@returns
            np.testing.assert_allclose(mean,exact,rtol=0,atol=1e-12)
            np.testing.assert_allclose(h["expected_advantage"],exact,rtol=0,atol=1e-12)
            old_exact=returns-saved.astype(np.float64)[:2]@returns
            old_errors.append(float(np.max(np.abs(mean[:2]-old_exact))))
            assert h["expectation_max_error"]<1e-12
    assert max(old_errors)>1e-9  # This regression would fail the original audit.
    np.testing.assert_array_equal(raw,saved)
    assert source_identity(solver)==before
    assert optimizer_hash(solver._capture_rng_state())==rng


@pytest.mark.parametrize("corruption",[1e-6,np.nan])
def test_frozen_probe_still_rejects_estimator_errors_and_reports_context(panel,monkeypatch,corruption):
    from experiments.fhp.exp28_fhp_rare_state_audit import diagnostics
    solver=synthetic_solver(4)
    native=diagnostics.control_variate_advantage
    def corrupted(*args,**kwargs):
        estimate=native(*args,**kwargs)
        advantages=estimate.advantages.copy()
        advantages[0]+=corruption
        return replace(estimate,advantages=advantages)
    monkeypatch.setattr(diagnostics,"control_variate_advantage",corrupted)
    before=source_identity(solver);rng=optimizer_hash(solver._capture_rng_state())
    with pytest.raises(ValueError,match="Native corrected-estimator expectation mismatch:") as exc:
        estimator_probe(solver,panel,contract(True),4)
    details=json.loads(str(exc.value).split(": ",1)[1])
    assert details["tolerance"]==1e-9
    assert details["case_id"] in {r["id"] for r in panel["rows"]}
    assert details["seat"] in (0,1) and details["fold"] in (0,1)
    assert len(details["opponent"])==2 and details["legal_actions"]==[0,1]
    for key in ("policy","normalised_policy","proposal","q","mean_advantage","expected_advantage"):
        assert len(details[key])==3
    assert len(details["returns"])==2
    if np.isfinite(corruption):
        assert details["expectation_max_error"]==pytest.approx(corruption)
    else:
        assert not np.isfinite(details["expectation_max_error"])
    assert source_identity(solver)==before
    assert optimizer_hash(solver._capture_rng_state())==rng


def test_passive_continuation_is_bitwise_native_equivalent(panel,tmp_path):
    reference=synthetic_solver(11)
    reference.iteration()
    expected=source_identity(reference);rng=optimizer_hash(reference._capture_rng_state())
    policy=deepcopy(reference.ave_policy_trainer.buffer)
    observed=synthetic_solver(11)
    run_continuation(observed,panel,contract(True),tmp_path,synthetic=True)
    assert source_identity(observed)==expected
    assert optimizer_hash(observed._capture_rng_state())==rng
    for name in ("infostate_buf","q_value_buf","q_value_mask_buf","iteration_buf"):
        n=min(policy.cur_id,policy.buffer_size)
        np.testing.assert_array_equal(getattr(policy,name)[:n],getattr(observed.ave_policy_trainer.buffer,name)[:n])
    assert policy.rng.bit_generator.state==observed.ave_policy_trainer.buffer.rng.bit_generator.state


def test_rare_node_trace_and_actual_regret_loss_hook(panel):
    solver=synthetic_solver(19)
    row=next(r for r in panel["rows"] if r["strict_winner"] and r["partition"]=="development")
    state=from_case(row)
    player=state.current_player()
    with observe_traversal(solver) as audit:
        solver.dfs(state,player)
    assert len(audit["records"])==1
    record=audit["records"][0]
    assert record["feature_hash"]==row["feature_hash"]
    trainer=solver.regret_trainers[player]
    # A one-row replay forces the native fit to expose the rare target.
    trainer.buffer.reset()
    f=solver.feature_encoder.information_state(state,player)
    trainer.add_data(f,record["corrected_advantage"],state.legal_actions_mask(),solver.num_iteration)
    original=deepcopy(trainer)
    with observe_regret_fit(trainer,solver.num_iteration) as fit:
        trainer.train_model(solver.num_iteration)
    original.train_model(solver.num_iteration)
    assert fit["rows"] and fit["rows"][0]["sampled_count"]==trainer.train_steps
    assert optimizer_hash(original.model.state_dict())==optimizer_hash(trainer.model.state_dict())
    assert optimizer_hash(original.optimizer.state_dict())==optimizer_hash(trainer.optimizer.state_dict())


def worker_kwargs():
    from experiments.fhp.exp10_fhp_hand_board_features.config import smoke_config
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
    kw=_solver_kwargs(7,smoke_config());kw.pop("cache_frozen_critic_targets");kw.pop("logger",None)
    return kw


def snapshot(solver):
    return ParallelStructuredGroupedUCVEscher._inference_snapshot(solver)


def test_native_actor_collection_equals_observed_actor():
    kw=worker_kwargs()
    native=StructuredGroupedTraversalWorker(kw["game_name"],kw,923,4)
    frozen=snapshot(native._solver)
    result=native.collect(4,0,0,frozen,1)
    state=native.continuation_state()
    observed=ObservedWorker(kw["game_name"],kw,923,4)
    actual=observed.collect(4,0,0,frozen,1)
    assert optimizer_hash(state)==optimizer_hash(observed.continuation_state())
    for key in ("regret","average_policy","q_folds","calibration","nodes_touched"):
        assert optimizer_hash(result[key])==optimizer_hash(actual[key])
    assert actual["rare_state_audit"]["counts"]


def test_complete_local_smoke_and_analysis_package(tmp_path):
    root=tmp_path/"run"
    local_smoke(root)
    manifest=verify(root)
    assert manifest["experiment"]==28 and manifest["local_smoke"]
    with pytest.raises(ValueError,match="cloud smoke"):
        validate_gate((root/"SUCCESS.json").read_bytes(),(root/"analysis/summary.json").read_bytes(),manifest["commit"])
    package(root,tmp_path/"analysis")
    assert verify(tmp_path/"analysis")["local_smoke"]
    assert (tmp_path/"analysis/report.md").is_file()
    assert not list((tmp_path/"analysis").rglob("*.pkl"))
    (root/"workers/seed_0/offline_48h/coverage.json").write_text("{}")
    with pytest.raises(ValueError,match="checksum"):
        aggregate(root,smoke=True,local=True)


@pytest.mark.parametrize("kind",["smoke","run"])
def test_single_batch_job_order_limits_pins_and_syntax(kind):
    args=Namespace(kind=kind,run_id="exp28-rare-test",bucket_root="gs://test-bucket",repo_ref="a"*40,
        service_account="runner@example.iam.gserviceaccount.com",project_id="test",region="europe-west1",
        repo_url="https://example.org/repo.git")
    job=build_job(args);group=job["taskGroups"][0]
    assert group["taskCount"]==group["parallelism"]==1
    assert group["taskSpec"]["maxRetryCount"]==0
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"]=="n2-standard-16"
    assert group["taskSpec"]["runnables"][-1]["alwaysRun"]
    for runnable in group["taskSpec"]["runnables"]:
        script=runnable["script"]["text"]
        assert "batch jobs submit" not in script
        subprocess.run(["bash","-n"],input=script,text=True,check=True)
    script=actions(args)
    assert script.index(" panel ")<script.index(" offline ")<script.index(" continue ")<script.index(" gameplay ")<script.index(" aggregate ")
    assert "--pilot" in script and "exp28_fhp_rare_state_audit.package" in script
    assert "for AUDIT_SEED in 0 1 2" in script if kind=="run" else "--smoke" in script
    subprocess.run(["bash","-n","gcp/run_exp28_rare_state_audit.sh"],check=True)


def test_gate_rejects_missing_actors_sources_or_changed_commit():
    row=dict(continuation=dict(ray_actors=8,complete_iterations=1,synthetic=False,diagnostic_samples_added=0),
             gameplay=dict(complete=True,rows=[{}]),historical={})
    for hour,run in (("24","exp10-features-20261001-161740"),("48","exp16-feat48-20261004-182051")):
        row["historical"][hour]=dict(summary=dict(synthetic=False,source=dict(run=run),estimator_expectation_verified=True))
    row["historical"]["48"]["summary"]["refitting"]=dict(source_preserved=True,assessment_locked=True,candidates=2)
    summary=dict(seeds=[0],pilot=False,per_seed=[row])
    raw=json.dumps(summary).encode()
    success=dict(status="complete",experiment=28,smoke=True,local_smoke=False,commit="a"*40,seeds=[0],
        artifacts={"analysis/summary.json":hashlib.sha256(raw).hexdigest()})
    validate_gate(json.dumps(success),raw,"a"*40)
    with pytest.raises(ValueError):validate_gate(json.dumps(success),raw,"b"*40)
    row["continuation"]["ray_actors"]=0
    raw=json.dumps(summary).encode();success["artifacts"]["analysis/summary.json"]=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError,match="eight actors"):validate_gate(json.dumps(success),raw,"a"*40)


@pytest.mark.parametrize("kind",["smoke","run"])
def test_submission_preflight_is_safe_under_macos_bash_nounset(kind):
    script=Path("gcp/run_exp28_rare_state_audit.sh").read_text()
    preflight=script.split("PREFLIGHT_COMMAND=(",1)[1].split("if gcloud storage ls",1)[0]
    # Stub only Python dispatch; no cloud calls, source downloads or files.
    setup='function exp28_python() { printf "%s\\n" "$@"; }; PYTHON=exp28_python; BUCKET_ROOT=gs://test; JOB_DIR=/tmp; '
    result=subprocess.run(["bash","-uc",setup+"KIND="+kind+"; PREFLIGHT_COMMAND=("+preflight],
        check=True,text=True,capture_output=True)
    assert ("--smoke" in result.stdout.splitlines())==(kind=="smoke")
    assert "preflight" in result.stdout.splitlines()


@pytest.mark.skipif(os.environ.get("FHP_TEST_RAY")!="1",reason="Opt-in real Ray actor integration")
def test_ray_driver_passive_merge_is_native_equivalent(tmp_path):
    from experiments.fhp.exp10_fhp_hand_board_features.config import smoke_config
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
    kwargs=_solver_kwargs(7,smoke_config());kwargs.pop("cache_frozen_critic_targets")
    kwargs.update(parallel_num_workers=2,parallel_run_seed=7,parallel_collection_chunk_size=4,
                  parallel_ray_object_store_memory=128*1024**2,num_traversals=6)
    results=[]
    for cls in (ParallelStructuredGroupedUCVEscher,ObservedSolver):
        solver=cls(**kwargs)
        try:
            solver.audit_merges=[]
            solver.evaluation_frequency=0
            solver._maybe_run_training_time_checkpoint=lambda:None
            solver.iteration()
            results.append((source_identity(solver),optimizer_hash(solver._capture_rng_state())))
            if cls is ObservedSolver:assert solver.audit_merges
        finally:
            solver.close()
    assert results[0]==results[1]


@pytest.mark.skipif(os.environ.get("FHP_TEST_RAY")!="1",reason="Opt-in native checkpoint/Ray restore")
def test_native_full_state_restores_into_instrumented_solver(tmp_path):
    from experiments.fhp.exp10_fhp_hand_board_features.config import smoke_config
    from experiments.fhp.exp10_fhp_hand_board_features.worker import _make_solver
    from experiments.fhp.exp10_fhp_hand_board_features import training_state as ts
    from experiments.fhp.exp28_fhp_rare_state_audit.continuation import make_solver
    config=smoke_config()
    native=_make_solver(0,config,smoke=True)
    try:
        native.iteration()
        native.train_average_policy()
        state=ts.build_training_state(native,seed=0,checkpoint_id="time_48h",
            repository_commit="test-reference",config=config,captured_checkpoints=[])
        native.iteration()
        expected=source_identity(native)
        rng=optimizer_hash(native._capture_rng_state())
    finally:
        native.close()
    observed=make_solver(0,config,smoke=True)
    try:
        ts.restore_training_state(observed,state,seed=0,repository_commit="test-reference",config=config)
        observed.iteration()
        assert source_identity(observed)==expected
        assert optimizer_hash(observed._capture_rng_state())==rng
        assert observed.audit_merges
    finally:
        observed.close()
