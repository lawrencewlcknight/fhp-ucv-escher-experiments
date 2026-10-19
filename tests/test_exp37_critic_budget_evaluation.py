"""Protocol counts, real scoring, crossed uncertainty, source checks and recovery."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.exp37_fhp_critic_budget_evaluation import run, statistics as stats
from gcp import exp37_critic_budget_evaluation_batch as batch

ROOT = Path(__file__).resolve().parents[1]


def records():
    return [dict(experiment=c, seed=s, training_hours=h, checkpoint_path=f"/{c}/{s}/{h}.pkl",
        checkpoint_sha256=f"{c}-{s}-{h}", nodes_touched=h*100)
        for c in stats.COHORTS for s in range(3) for h in (6, 12, 18, 24)]


def arguments(tmp_path, **kw):
    options = ["--output-dir", str(tmp_path/"analysis")]
    for c in stats.COHORTS:
        options += ["--"+c+"-run", str(tmp_path/c)]
    args = run.parser().parse_args(options)
    return SimpleNamespace(**(vars(args)|kw))


def fixture_sources(tmp_path, monkeypatch):
    from tests import test_exp20_evaluation as prior
    monkeypatch.setattr(prior.run, "CONFIGS", {c: run.existing.CONFIGS[c] for c in stats.COHORTS})
    return prior.source_fixture(tmp_path, monkeypatch)


def fake_score(task):
    seed = task.get("seed_a", task.get("training_seed", 0))
    score = seed + int(task["kind"] == "direct_crossplay")
    values = {k: np.arange(task["num_deals"], dtype=float)+score for k in ("paired", "player_zero", "player_one")}
    result = dict(task, **run.protocol._match_summary(values, policy_a=task["policy_a_name"], policy_b="fixture"))
    if task["kind"] == "lbr":
        result.update({"_"+k+"_values": v.tolist() for k, v in values.items()})
    return result


def test_full_protocol(tmp_path):
    tasks = run.build_tasks(records(), arguments(tmp_path))
    assert Counter(t["kind"] for t in tasks) == {"direct_crossplay": 270, "rule": 45, "lbr": 900}
    direct = [t for t in tasks if t["kind"] == "direct_crossplay"]
    assert sum(t["num_deals"] for t in direct) == 1_350_000
    assert sum(t["num_deals"] for t in tasks) == 1_809_000
    assert len({(t["cohort_a"], t["cohort_b"], t["seed_a"], t["seed_b"]) for t in direct}) == 27
    assert all(t["training_hours"] == 24 and "exp33" not in t["task_id"] for t in tasks)
    assert len({t["task_id"] for t in tasks}) == len({t["evaluation_seed"] for t in tasks}) == len(tasks)
    reverse = {t["task_id"]: t["evaluation_seed"] for t in run.build_tasks(list(reversed(records())), arguments(tmp_path))}
    assert reverse == {t["task_id"]: t["evaluation_seed"] for t in tasks}
    assert "training_seed" not in direct[0]  # No fictitious nine-seed inference.


def test_shard_pooling_handles_unequal_sizes_and_singletons(tmp_path):
    values = np.array([1., -3., 4., 20., -50., 8., 3.])
    rows = [run.protocol._match_summary({k: v for k in ("paired", "player_zero", "player_one")},
        policy_a="a", policy_b="b") for v in (values[:2], values[2:6], values[6:])]
    rows = run.protocol._json_safe(rows)  # singleton std becomes None in saved JSON
    pooled = stats.pool_shards(rows, 10.)
    assert pooled["mean_mbb_per_hand"] == pytest.approx(np.mean(values)*10)
    assert pooled["std_chips_per_pair"] == pytest.approx(np.std(values, ddof=1))
    assert pooled["se_mbb_per_hand"] == pytest.approx(np.std(values, ddof=1)/np.sqrt(7)*10)


def cells(value=0., se=0.):
    return [dict(cohort_a=a, cohort_b=b, seed_a=i, seed_b=j, num_deal_pairs=50000,
        mean_mbb_per_hand=value, se_mbb_per_hand=se) for a, b in stats.PAIRS for i in range(3) for j in range(3)]


def test_crossed_uncertainty_and_no_forced_winner():
    zero = stats.crossed_summary(cells())
    assert all(r["bootstrap_ci95_low"] == r["bootstrap_ci95_high"] == 0 for r in zero)
    assert stats.leader_screen(zero)["provisional_head_to_head_leader"] is None
    noisy = stats.crossed_summary(cells(se=10.))
    assert all(r["conditional_deal_se_mbb_per_hand"] == pytest.approx(10/3) for r in noisy)
    for r in noisy:
        assert r["bootstrap_family_ci_low"] <= r["bootstrap_ci95_low"] < 0
        assert r["bootstrap_family_ci_high"] >= r["bootstrap_ci95_high"] > 0
    data = cells()
    for r in data:
        r["mean_mbb_per_hand"] = [-100., 0., 100.][r["seed_a"]]
    summaries = stats.crossed_summary(data)
    # Repeated columns are not nine independently trained models.
    assert all(r["bootstrap_ci95_high"] > 90 for r in summaries)
    assert summaries == stats.crossed_summary(data)
    assert stats.leader_screen(stats.crossed_summary(cells(value=100.)))["provisional_head_to_head_leader"] == "exp36"
    cycle = stats.crossed_summary(cells(value=100.))
    cycle[1]["bootstrap_family_ci_low"] = -101.
    cycle[1]["bootstrap_family_ci_high"] = -99.
    assert stats.leader_screen(cycle)["provisional_head_to_head_leader"] is None
    with pytest.raises(ValueError, match="Incomplete"):
        stats.crossed_summary(data[:-1])


@pytest.mark.parametrize("cohort", stats.COHORTS)
def test_strict_original_source_contract(tmp_path, monkeypatch, cohort):
    roots = fixture_sources(tmp_path, monkeypatch)
    assert len(run.existing.discover(roots[cohort], cohort)) == 12
    worker = next((roots[cohort]/"workers").glob("task_*"))
    path = worker/"run_manifest.json"
    payload = json.loads(path.read_text())
    payload["training_config"]["baseline_network_train_steps"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        run.existing.discover(roots[cohort], cohort)


@pytest.mark.parametrize("smoke", (True, False))
def test_reporting_interruption_and_resume(tmp_path, monkeypatch, smoke):
    fixture_sources(tmp_path, monkeypatch)
    args = arguments(tmp_path, workers=1, rule_deals=2, crossplay_deals=3, crossplay_shard_deals=2,
        lbr_deals=3, lbr_shard_deals=2, lbr_rollouts=2, smoke=smoke)
    calls = []
    def fail(task):
        calls.append(task)
        if len(calls) == 3:
            raise RuntimeError("deliberate failure")
        return fake_score(task)
    monkeypatch.setattr(run.protocol, "_evaluation_worker", fail)
    monkeypatch.setattr(run.rare, "run", lambda *a, **kw: {"test_fixture": True})
    with pytest.raises(RuntimeError, match="deliberate"):
        run.run_analysis(args)
    assert len(list((args.output_dir/"task_results").glob("*.json"))) == 2
    assert not (args.output_dir/"SUCCESS.json").exists()
    assert json.loads((args.output_dir/"evaluation_manifest.json").read_text())["status"] == "failed"
    args.resume = True
    monkeypatch.setattr(run.protocol, "_evaluation_worker", fake_score)
    output = run.run_analysis(args)
    summary = json.loads((output/"summary.json").read_text())
    assert summary["head_to_head"][0]["models_per_configuration"] == (1 if smoke else 3)
    assert summary["exact_exploitability"] is False and "AND richer" in summary["interpretation"]
    assert len(list(output.glob("*.png"))) == 3
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda *a: pytest.fail("Cached tasks must be reused"))
    run.run_analysis(args)
    args.crossplay_deals += 1
    with pytest.raises(ValueError, match="resume mismatch"):
        run.run_analysis(args)


def test_rare_subset_recovery(tmp_path, monkeypatch):
    panel = dict(rows=[], scope="test", assessment_board_classes=[])
    monkeypatch.setattr(run.rare, "build_panel", lambda smoke: deepcopy(panel))
    monkeypatch.setattr(run.rare, "assess", lambda r, p: dict(experiment=r["experiment"], seed=r["seed"],
        checkpoint_sha256=r["checkpoint_sha256"], seconds=1,
        metrics={"assessment/all": dict(positions=10, mean_fold_probability=.1, mean_gap_bb=.2,
            p95_gap_bb=.3, max_gap_bb=.4, fraction_fold_over_half=0)}))
    out = run.rare.run(tmp_path, records(), "test", smoke=False, controls=("exp20", "exp10"))
    assert out["policy_count"] == 9
    monkeypatch.setattr(run.rare, "assess", lambda *a: pytest.fail("Reuse diagnostic results"))
    assert run.rare.run(tmp_path, records(), "test", smoke=False, controls=("exp20", "exp10")) == out


def test_parallel_failure_drains_successes_without_dispatching_rest(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from experiments.fhp.exp37_fhp_critic_budget_evaluation import execution
    monkeypatch.setattr(execution, "ProcessPoolExecutor", lambda max_workers, **kw: ThreadPoolExecutor(max_workers=max_workers))
    tasks = run.build_tasks(records(), arguments(tmp_path, smoke=True, crossplay_deals=2))[:3]
    barrier = Barrier(2)
    calls = []
    def worker(task):
        calls.append(task["task_id"])
        barrier.wait(timeout=10)
        if task == tasks[0]:
            raise ValueError("intentional parallel failure")
        return fake_score(task)
    monkeypatch.setattr(execution, "timed_worker", worker)
    # Ensure the failing future and success are returned together, so dispatch
    # order cannot make the test depend on thread scheduling.
    from concurrent.futures import wait as all_wait, ALL_COMPLETED
    monkeypatch.setattr(execution, "wait", lambda f, **kw: all_wait(f, return_when=ALL_COMPLETED))
    with pytest.raises(RuntimeError, match="intentional parallel"):
        execution.run_tasks(tasks, 2, tmp_path/"tasks", "test")
    assert len(calls) == 2
    assert (tmp_path/"tasks"/f"{tasks[1]['task_id']}.json").exists()
    assert not (tmp_path/"tasks"/f"{tasks[0]['task_id']}.json").exists()
    monkeypatch.setattr(execution, "timed_worker", fake_score)
    assert len(execution.run_tasks(tasks, 1, tmp_path/"tasks", "test")) == 3
    path = tmp_path/"tasks"/f"{tasks[0]['task_id']}.json"
    bad = json.loads(path.read_text())
    bad["result"]["mean_mbb_per_hand"] = 999
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="corrupt"):
        execution.run_tasks(tasks, 1, tmp_path/"tasks", "test")


def test_real_native_policy_scoring_and_parallel_shards(tmp_path):
    """Small real OpenSpiel matches, genuine original/richer encoders and spawn."""
    import torch
    from fhp_escher.checkpointing import save_policy_checkpoint
    from fhp_escher.features import make_feature_encoder
    from vr_deep_cfr.solver import MLP
    paths = {}
    for c in ("exp10", "exp36"):
        encoder = make_feature_encoder(run.existing.CONFIGS[c].EXPERIMENT_CONFIG["feature_encoder_id"])
        size = encoder.policy_layout.total_size
        model = MLP(size, [8], 3)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=model), feature_encoder=encoder,
            num_iteration=1, episode=1, nodes_touched=1, infostate_size=size, action_size=3, network_layers=[8])
        paths[c] = str(save_policy_checkpoint(solver, tmp_path/f"{c}.pkl", seed=0, config={}, checkpoint_row={}))
    selected = [r for r in records() if r["training_hours"] == 24]
    for r in selected:
        r["checkpoint_path"] = paths["exp36" if r["experiment"] == "exp36" else "exp10"]
    args = arguments(tmp_path, smoke=True, rule_deals=2, lbr_deals=2, lbr_rollouts=2, crossplay_deals=2)
    tasks = run.build_tasks(selected, args)
    task = next(t for t in tasks if t["kind"] == "direct_crossplay" and t["cohort_a"] == "exp36")
    match = run.protocol._evaluation_worker(task)
    assert match["num_games"] == 4
    assert np.isfinite(match["mean_mbb_per_hand"])
    assert match == run.protocol._evaluation_worker(task)
    examples = [next(t for t in tasks if t["kind"] == k and t.get("experiment") == "exp36") for k in ("rule", "lbr")]
    result = run.run_tasks(examples, 2, tmp_path/"native_tasks", "real-test")
    assert len(result) == 2 and all(r["num_deal_pairs"] == 2 for r in result)
    assert run.run_tasks(examples, 1, tmp_path/"native_tasks", "real-test") == result


def test_batch_and_stdlib_builder(tmp_path):
    cli = ["--run-id", "exp37-test", "--bucket-root", "gs://test-bucket", "--service-account", "test@example.com",
        "--repo-ref", "a"*40, "--output", str(tmp_path/"job.json")]
    args = batch.parser().parse_args(cli)
    for smoke, resume in ((False, False), (True, False), (False, True)):
        args.smoke_only, args.resume = smoke, resume
        job = batch.build_job(args)
        spec = job["taskGroups"][0]["taskSpec"]
        script = spec["runnables"][0]["script"]["text"]
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert spec["maxRunDuration"] == "172800s" and spec["maxRetryCount"] == 0
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
        assert all(v in script for v in batch.SOURCES.values())
        assert "continuation_inputs" in script and "training_states" in script
        assert "--workers 16" in script and "--crossplay-shard-deals 5000" in script
        assert "exp33" not in script
    subprocess.run([sys.executable, "-S", str(ROOT/"gcp/exp37_critic_budget_evaluation_batch.py"), *cli], check=True)
    subprocess.run(["bash", "-n", str(ROOT/"gcp/run_exp37_critic_budget_evaluation.sh")], check=True)
    args.run_id = args.exp10_run_id
    with pytest.raises(ValueError, match="distinct"):
        batch.build_job(args)


@pytest.mark.parametrize("change", ({"crossplay_deals": 1}, {"lbr_deals": 1}, {"workers": 0}, {"base_seed": -1}))
def test_budget_validation(tmp_path, change):
    with pytest.raises(ValueError):
        run.validate(arguments(tmp_path, **change))
