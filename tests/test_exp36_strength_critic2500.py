"""Frozen experiment contract, Batch routing, evaluation and native continuation."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.fhp.exp36_fhp_strength_critic2500 import config, evaluate, rare
from experiments.fhp.exp20_fhp_half_critic_updates import config as exp20
from experiments.fhp.strategic_feature_suite.config import get_experiment, SPECS
from gcp.exp36_strength_critic2500_batch import build_job

ROOT = Path(__file__).resolve().parents[1]


def test_contract():
    changed = {k for k, v in config.EXPERIMENT_CONFIG.items() if v != exp20.EXPERIMENT_CONFIG[k]}
    assert changed == {"feature_encoder_id", "baseline_network_train_steps"}
    assert config.EXPERIMENT_CONFIG["baseline_network_train_steps"] == 2500
    exp33 = get_experiment(33)
    assert {k for k, v in config.EXPERIMENT_CONFIG.items() if v != exp33.EXPERIMENT_CONFIG[k]} == {"baseline_network_train_steps"}
    assert config.encoder().metadata() == exp33.encoder().metadata()
    assert config.PRODUCTION_SEEDS == (0, 1, 2)
    assert config.REMOTE_TASK_ENV == "EXP36_REMOTE_TASK_URI"
    assert config.parallel_settings()["parallel_num_workers"] == 8
    assert config.LEARNER_THREADS == 8
    assert config.smoke_config()["baseline_network_train_steps"] == 2
    assert 36 not in SPECS  # The historical encoder-only suite is unchanged.
    for hours in (24, 48):
        schedule = config.checkpoint_schedule(total_hours=hours)
        assert [r["target_training_seconds"] for r in schedule] == [h * 3600 for h in range(6, hours+1, 6)]
        config.validate_contract(seeds=config.PRODUCTION_SEEDS, schedule=schedule,
                                 config=config.EXPERIMENT_CONFIG, smoke=False, total_hours=hours)
        manifest = config.contract_manifest(total_hours=hours)
        assert manifest["automatic_evaluation"] is False
        assert manifest["training_state_retention"] == "final"
        assert manifest["primary_comparator"] == 20
        assert manifest["baseline"] == exp20.EXPERIMENT_NAME
    config.validate_contract(seeds=config.SMOKE_SEEDS, schedule=config.checkpoint_schedule(smoke=True),
                             config=config.smoke_config(), smoke=True)
    with pytest.raises(ValueError):
        config.validate_contract(seeds=config.PRODUCTION_SEEDS, schedule=config.checkpoint_schedule(),
            config=dict(config.EXPERIMENT_CONFIG, baseline_network_train_steps=5000), smoke=False)


def job_args(kind, extension=False):
    return SimpleNamespace(kind=kind, run_id="exp36-test", repo_ref="a"*40,
        bucket_root="gs://test-bucket", service_account="batch@test-project.iam.gserviceaccount.com",
        project_id="test-project", region="europe-west1", parallelism=3, max_hours=48,
        total_hours=48 if extension else 24, source_run_id="exp36-source" if extension else "",
        repo_url="https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git", resume=False)


@pytest.mark.parametrize("kind", ("controller", "smoke", "train", "aggregate", "evaluate"))
@pytest.mark.parametrize("extension", (False, True))
def test_batch_routing(kind, extension):
    args = job_args(kind, extension)
    if kind == "evaluate" and extension:
        with pytest.raises(ValueError, match="original 24h"):
            build_job(args)
        return
    job = build_job(args)
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    script = spec["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert spec["maxRetryCount"] == 0
    assert "EXP20" not in script and "FEATURE_REMOTE_TASK_URI" not in script
    assert "exp20_fhp_half_critic_updates" not in script
    if kind in ("train", "smoke", "evaluate"):
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    if kind in ("train", "smoke"):
        assert "exp36_fhp_strength_critic2500.run" in script
        assert "capacity-preflight" in script if kind == "smoke" else "EXP36_REMOTE_TASK_URI" in script
    if kind == "train":
        assert "strategic_ae_critic2500_ucv_escher" in script
        assert "stage-source" in script and "EXP36_TOTAL_HOURS" in script
        assert spec["environment"]["variables"]["EXP36_TOTAL_HOURS"] == str(args.total_hours)
    if kind == "evaluate":
        for expected in ("--candidate-run", "--exp20-run", "--exp33-run", "--exp10-run",
                         "exp20-critic24-20261006-164357", "exp33-features-20261012-143026",
                         "exp10-features-20261001-161740", '"$BUCKET_ROOT/$RUN_ID/evaluation"'):
            assert expected in script
        assert "continuation_inputs" in script
    if kind == "controller":
        assert "gcp/run_exp36_strength_critic2500.sh" in script
        assert "export EXP36_REMOTE_CONTROLLER=1" in script


def test_launcher_and_stdlib_builder(tmp_path):
    path = ROOT / "gcp/run_exp36_strength_critic2500.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    script = path.read_text()
    assert "for stage in smoke train aggregate; do ensure_stage" in script
    assert "for stage in smoke train aggregate evaluate" not in script
    import sys
    args = job_args("train")
    command = [sys.executable, "-S", str(ROOT / "gcp/exp36_strength_critic2500_batch.py"), "--output", str(tmp_path/"job.json")]
    for key, value in vars(args).items():
        if key != "resume":
            command += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(command, check=True)
    assert json.loads((tmp_path/"job.json").read_text())["labels"]["stage"] == "train"
    args.source_run_id = "exp20-source"
    with pytest.raises(ValueError, match="Only Experiment 36"):
        build_job(args)


def records():
    return [dict(experiment=exp, seed=seed, training_hours=hour,
        checkpoint_sha256=f"{exp}-{seed}-{hour}", checkpoint_path=f"/{exp}/{seed}/{hour}.pkl",
        nodes_touched=hour*100) for exp in evaluate.CONFIGS for seed in (0, 1, 2) for hour in (6, 12, 18, 24)]


def arguments(tmp_path, **overrides):
    args = evaluate.parser().parse_args(["--candidate-run", str(tmp_path/"exp36"),
        "--exp10-run", str(tmp_path/"exp10"), "--exp20-run", str(tmp_path/"exp20"),
        "--exp33-run", str(tmp_path/"exp33"), "--output-dir", str(tmp_path/"analysis")])
    return SimpleNamespace(**(vars(args) | overrides))


def test_evaluation_endpoints(tmp_path):
    tasks = evaluate.build_tasks(records(), arguments(tmp_path))
    assert Counter(t["kind"] for t in tasks) == {"rule": 60, "lbr": 1200, "direct_crossplay": 9}
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    primary = [t for t in tasks if t.get("primary_endpoint")]
    assert len(primary) == 3 and all(t["experiment"] == "exp36_vs_exp20" for t in primary)
    assert all(t["training_hours"] == 24 for t in tasks)
    direct = [t for t in tasks if t["kind"] == "direct_crossplay"]
    assert all(t["policy_a_path"].startswith("/exp36/") for t in direct)
    assert len({(t["opponent"], t["evaluation_seed"]) for t in tasks if t["kind"] == "rule"}) == 5


def source_fixture(tmp_path, monkeypatch):
    from tests import test_exp20_evaluation as old
    monkeypatch.setattr(old.run, "CONFIGS", evaluate.CONFIGS)
    return old.source_fixture(tmp_path, monkeypatch)


@pytest.mark.parametrize("key", ("exp36", "exp20", "exp33", "exp10"))
def test_source_contracts(tmp_path, monkeypatch, key):
    roots = source_fixture(tmp_path, monkeypatch)
    assert len(evaluate.discover(roots[key], key)) == 12
    worker = next((roots[key] / "workers").glob("task_*"))
    path = worker / "run_manifest.json"
    bad = json.loads(path.read_text())
    bad["training_config"]["baseline_network_train_steps"] += 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        evaluate.discover(roots[key], key)


def test_actual_budget_rejected(tmp_path, monkeypatch):
    roots = source_fixture(tmp_path, monkeypatch)
    worker = next((roots["exp36"] / "workers").glob("task_*"))
    path = worker / "summary.json"
    value = json.loads(path.read_text())
    value["execution_diagnostics"]["critic_train_steps_per_fit"] = [2500, 5000]
    path.write_text(json.dumps(value))
    (worker / "SUCCESS.json").write_text(json.dumps(dict(status="complete", summary_sha256=rare.sha256_file(path))))
    with pytest.raises(ValueError, match="actual critic budget"):
        evaluate.discover(roots["exp36"], "exp36")


@pytest.mark.parametrize("smoke", (True, False))
def test_reporting_and_recovery(tmp_path, monkeypatch, smoke):
    from tests.test_strategic_feature_evaluation import score
    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate.protocol, "_evaluation_worker", score)
    monkeypatch.setattr(rare, "run", lambda *a, **kw: {"test_fixture": True})
    args = arguments(tmp_path, workers=1, smoke=smoke, rule_deals=2, lbr_deals=3, lbr_shard_deals=2, crossplay_deals=2)
    output = evaluate.run_analysis(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["primary_endpoint"][0]["num_training_seeds"] == (1 if smoke else 3)
    assert len(summary["secondary_endpoints"]) == 2
    assert "nan" not in (output / "analysis_summary.md").read_text()
    args.resume = True
    monkeypatch.setattr(evaluate.protocol, "_evaluation_worker", lambda *a: pytest.fail("Must recover saved tasks"))
    evaluate.run_analysis(args)
    args.crossplay_deals += 1
    with pytest.raises(ValueError, match="resume mismatch"):
        evaluate.run_analysis(args)


def test_rare_metrics_and_recovery(tmp_path, monkeypatch):
    panel = dict(rows=[], scope="test fixture", assessment_board_classes=[])
    monkeypatch.setattr(rare, "build_panel", lambda smoke: deepcopy(panel))
    monkeypatch.setattr(rare, "assess", lambda r, p: dict(experiment=r["experiment"], seed=r["seed"],
        checkpoint_sha256=r["checkpoint_sha256"], seconds=1,
        metrics={"assessment/all": dict(positions=10, mean_fold_probability=.1, mean_gap_bb=.2,
            p95_gap_bb=.3, max_gap_bb=.4, fraction_fold_over_half=0)}))
    result = rare.run(tmp_path, records(), "fingerprint", smoke=False)
    assert result["policy_count"] == 12
    assert all(r["mean"] == 0 for r in result["paired_seed_differences"])
    monkeypatch.setattr(rare, "assess", lambda *a: pytest.fail("Must reuse diagnostics"))
    assert rare.run(tmp_path, records(), "fingerprint", smoke=False) == result
    path = tmp_path / "rare_results/exp36_seed0.json"
    saved = json.loads(path.read_text())
    saved["result"]["seed"] = 9
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="resume mismatch"):
        rare.run(tmp_path, records(), "fingerprint", smoke=False)


@pytest.mark.ray
def test_native_cache_restart_and_completed_extension(tmp_path):
    if os.environ.get("RUN_EXP36_RAY_TESTS") != "1":
        pytest.skip("Set RUN_EXP36_RAY_TESTS=1 for real eight-actor training/restart tests")
    from experiments.fhp.exp36_fhp_strength_critic2500.run import main
    from experiments.fhp.strategic_feature_suite.worker import _make_solver
    from experiments.fhp.strategic_feature_suite.diagnostics import execution_diagnostics
    # Production critic update settings, tiny allocation, no 24h training.
    solver = _make_solver(0, dict(config.smoke_config(), baseline_network_train_steps=2500),
                          smoke=True, experiment=config.EXPERIMENT)
    try:
        assert execution_diagnostics(solver)["critic_train_steps_per_fit"] == [2500, 2500]
    finally:
        solver.close()
    main(["smoke", "--output-root", str(tmp_path), "--no-resume"])
    for name in ("cache_validation/cache_validation.json", "restart_validation/restart_validation.json"):
        assert json.loads((tmp_path / name).read_text())["learning_state_bitwise_identical"]
    extended = next((tmp_path / "extension_smoke/workers").glob("task_*"))
    rows = json.loads((extended / "checkpoint_manifest.json").read_text())
    assert len(rows) == 8 and rows[-1]["training_state_sha256"]
    assert not any("training_state_path" in r for r in rows[:-1])


def test_rare_panel_and_terminal_metrics(monkeypatch):
    import numpy as np
    import torch
    panel = rare.build_panel(smoke=True)
    excluded = set(map(tuple, panel["excluded_board_classes"]))
    assert not (set(map(tuple, panel["assessment_board_classes"])) & excluded)
    assert {r["partition"] for r in panel["rows"]} == {"assessment", "development"}
    class Uniform(torch.nn.Module):
        def forward(self, values):
            return torch.zeros((len(values), 3))
    monkeypatch.setattr(rare, "LoadedFHPPolicy", lambda *a: SimpleNamespace(
        feature_encoder=config.encoder(), model=Uniform()))
    row = records()[3]
    result = rare.assess(row, panel)
    for group, values in result["metrics"].items():
        split, category = group.split("/")
        cases = [r for r in panel["rows"] if r["partition"] == split and r["strict_winner"]
                 and (category == "all" or r["category"] == {"quads": 7, "straight_flush": 8}[category])]
        assert values["mean_fold_probability"] == pytest.approx(.5)
        assert values["mean_gap_bb"] == pytest.approx(np.mean([r["guaranteed_call_gain_bb"] for r in cases]) / 2)
    assert all(r["probabilities"] == [.5, .5, 0.] for r in result["positions"])
