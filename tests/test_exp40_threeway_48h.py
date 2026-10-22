"""Protocol, provenance, recovery and native cross-repository integration."""
from copy import deepcopy
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.exp40_fhp_threeway_48h import engine as native, run, sources
from gcp import exp40_threeway_48h_batch as batch

ROOT = run.ROOT
WORKSPACE = ROOT.parents[1]
VR = WORKSPACE/"fhp_vr_deep/fhp-vr-deep-experiments"
SD = WORKSPACE/"fhp_deep_cfr/fhp-deep-cfr-experiments"


@pytest.fixture
def engine():
    # Pure scheduling/report functions do not initialise native packages in
    # pytest's shared process; real package isolation is tested separately.
    spec = importlib.util.spec_from_file_location("exp40_test_engine", VR/native.ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL, module.BASE_SEED, module.LABELS = native.PROTOCOL, 640048, native.LABELS
    return module


def records():
    return [dict(id=f"{a}_{s}", algorithm=a, seed=s, training_hours=48, label=native.LABELS[a],
        path=f"/fixture/{a}_{s}", sha256=f"{a}_{s}", nodes_touched=(s+1)*1000,
        source_run=native.RUN_IDS[a], active_seconds=172800, policy_size_bytes=1024)
        for a in ("sd", "ucv", "vr") for s in range(3)]


def fake_result(task, engine):
    row = dict(task=engine.portable(task), elapsed_seconds=.1, result=dict(
        num_deal_pairs=task["num_deals"], num_games=2*task["num_deals"], mean_chips_per_hand=.1,
        std_chips_per_pair=1., mean_mbb_per_hand=1., policy_a_player0_mean_chips=.2,
        policy_a_player1_mean_chips=0.))
    row["result_sha256"] = engine.digest(row)
    return row


def args(tmp_path, **changes):
    return SimpleNamespace(source_root=tmp_path/"input", output=tmp_path/"analysis", vr_repo=VR,
        sd_repo=SD, workers=2, max_hours=23., deadline_unix=None, source_uri=[], **changes)


def test_full_counts_and_fresh_prespecified_streams(engine):
    tasks = engine.build_tasks(records(), "production", "code")
    assert len(tasks) == 630 and len({t["task_id"] for t in tasks}) == 630
    assert sum(t["num_deals"] for t in tasks) == 3_150_000
    assert sum(t["kind"] == "direct" for t in tasks) == 540
    assert sum(t["kind"] == "rule" for t in tasks) == 90
    assert all(t["a"]["training_hours"] == 48 for t in tasks)
    smoke = engine.build_tasks(records(), "smoke", "code")
    profile = engine.build_tasks(records(), "profile", "code")
    assert len(smoke) == 72 and all(t["num_deals"] == 2 for t in smoke)
    assert {t["evaluation_seed"] for t in tasks}.isdisjoint(t["evaluation_seed"] for t in smoke+profile)
    engine.BASE_SEED = 601048
    old = engine.build_tasks(records(), "production", "code")
    assert {t["evaluation_seed"] for t in tasks}.isdisjoint(t["evaluation_seed"] for t in old)
    with pytest.raises(ValueError):
        engine.build_tasks(records(), "lbr", "code")


def test_reporting_independent_seed_uncertainty(engine, tmp_path):
    tasks = engine.build_tasks(records(), "production", "code")
    result = run.report([fake_result(t,engine) for t in tasks], tmp_path, records(), [], engine)
    assert result["experiment_id"] == 40
    assert result["exact_exploitability"] is False and result["lbr_evaluated"] is False
    assert result["family_confidence"] == pytest.approx(1-.05/3)
    assert len(result["direct_comparisons"]) == 3
    assert all(r["conditional_mc_ci95_low"] < r["mean_mbb_per_hand"] < r["conditional_mc_ci95_high"]
               for r in result["direct_comparisons"])
    matrix = np.arange(3)[:,None]-np.arange(3)[None,:]
    lo, hi = engine.cluster_interval(matrix)
    assert lo < 0 < hi  # Independent algorithms are NOT paired by seed number.
    assert (tmp_path/"final_48h_head_to_head.png").is_file()
    with pytest.raises(ValueError, match="Incomplete"):
        run.report([fake_result(t,engine) for t in tasks[:-1]], tmp_path, records(), [], engine)


class ImmediatePool:
    def __init__(self, **kw): self._processes = {}
    def submit(self, fn, task):
        f = Future()
        try: f.set_result(fn(task))
        except Exception as exc: f.set_exception(exc)
        return f
    def shutdown(self, **kw): pass


def test_shards_resume_and_reject_corruption(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ProcessPoolExecutor", ImmediatePool)
    tasks = engine.build_tasks(records(), "smoke", "code")[:3]
    options = args(tmp_path, smoke_only=False)
    options.workers = 1
    def score(t):
        if t == tasks[1]: raise RuntimeError("interrupted")
        return fake_result(t,engine)
    monkeypatch.setattr(native, "execute_task", score)
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run_tasks(tasks,tmp_path/"shards",options,engine,time.monotonic()+30)
    assert len(list((tmp_path/"shards").glob("*.json"))) == 1
    seen = []
    monkeypatch.setattr(native, "execute_task", lambda t: (seen.append(t["task_id"]) or fake_result(t,engine)))
    result = run.run_tasks(tasks,tmp_path/"shards",options,engine,time.monotonic()+30)
    assert len(result) == 3 and tasks[0]["task_id"] not in seen
    path = tmp_path/"shards"/(tasks[0]["task_id"]+".json")
    path.write_text('{}')
    with pytest.raises(ValueError, match="Corrupt"):
        run.run_tasks(tasks,tmp_path/"shards",options,engine,time.monotonic()+30)


@pytest.mark.parametrize("smoke", [True,False])
def test_pipeline_does_not_abort_on_pessimistic_forecast(engine, tmp_path, monkeypatch, smoke):
    monkeypatch.setattr(native, "load", lambda *a: engine)
    monkeypatch.setattr(sources, "validate", lambda *a: records())
    monkeypatch.setattr(sources, "cost_context", lambda *a: [])
    monkeypatch.setattr(run, "implementation_digest", lambda *a: "code")
    stages = []
    def score(tasks, *a):
        stages.append(tasks[0]["stage"])
        return [fake_result(t,engine) for t in tasks]
    monkeypatch.setattr(run, "run_tasks", score)
    monkeypatch.setattr(engine, "cost_estimate", lambda *a: dict(predicted_hours_with_2x_margin=100000))
    options = args(tmp_path,smoke_only=smoke)
    run.run(options)
    success = engine.read_json(options.output/"SUCCESS.json")
    assert success["smoke"] == smoke and success["status"] == "complete"
    assert stages == (["smoke"] if smoke else ["smoke","profile","production"])
    if not smoke:
        assert engine.read_json(options.output/"summary.json")["hands"] == 6_300_000
        assert engine.read_json(options.output/"timing_forecast.json")["cost_gate_enabled"] is False
    stages.clear()
    run.run(options)
    assert not stages
    options.source_uri = ["changed"]
    with pytest.raises(ValueError,match="Incompatible"):
        run.run(options)


@pytest.mark.parametrize("mutation", [None,"lineage","history","budget","checkpoint","no_progress"])
def test_ucv_continuation_and_native_payload(engine,tmp_path,monkeypatch,mutation):
    from tests.test_exp39_48h_evaluation import fixture_sources
    from experiments.fhp.exp39_fhp_48h_evaluation import sources as old
    import fhp_escher.checkpointing as checkpoints
    roots = fixture_sources(tmp_path,monkeypatch)
    monkeypatch.setattr(checkpoints,"LoadedFHPPolicy",old.native.LoadedFHPPolicy)
    worker = roots["exp38"]/"workers"/sources.contract.worker_name(0)
    if mutation:
        name = "continuation_source.json" if mutation == "lineage" else "checkpoint_manifest.json"
        if mutation == "budget": name = "summary.json"
        path = worker/name
        data = json.loads(path.read_text())
        if mutation == "lineage": data["source_worker"] = "gs://wrong/worker"
        elif mutation == "budget": data["execution_diagnostics"]["critic_train_steps_per_fit"] = [5000,5000]
        elif mutation == "history": data[0]["sha256"] = "f"*64
        elif mutation == "checkpoint": data[-1]["sha256"] = "f"*64
        elif mutation == "no_progress": data[-1]["nodes_touched"] = 2400
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError): sources.validate_ucv(roots["exp38"],roots["exp36"],engine)
    else:
        selected = sources.validate_ucv(roots["exp38"],roots["exp36"],engine)
        assert len(selected) == 3
        assert all(r["source_run"] == native.RUN_IDS["ucv"] and r["training_hours"] == 48 for r in selected)


def test_batch_and_launcher(tmp_path):
    cli = ["--run-id","exp40-test","--bucket","gs://test-output","--service-account",
        "runner@test.iam.gserviceaccount.com","--repo-ref","a"*40,"--output",str(tmp_path/"job.json")]
    options = batch.parser().parse_args(cli)
    downloads = batch.download_script(options)
    sd_block = downloads.split("ALGORITHM="+batch.ALGORITHMS["sd"])[1].split("done\n\nSOURCE=")[0]
    assert "summary.json" not in sd_block  # SD native completion is its timing summary.
    for smoke,resume in ((False,False),(True,False),(False,True)):
        options.smoke_only,options.resume = smoke,resume
        job = batch.build_job(options)
        spec = job["taskGroups"][0]["taskSpec"]
        assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "86400s"
        assert spec["runnables"][1]["alwaysRun"] is True
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
        for runnable in spec["runnables"]:
            text = runnable["script"]["text"]
            subprocess.run(["bash","-n"],input=text,text=True,check=True)
            assert "flock" in text and "SUCCESS[.]json" in text
        script = spec["runnables"][0]["script"]["text"]
        assert all(v in script for v in native.RUN_IDS.values())
        assert "--workers 16" in script and "--source-uri ucv_source36=" in script
        assert "--smoke-only" in script if smoke else "--smoke-only" not in script
        assert "training_states/" not in script
        assert batch.VR_REF == native.VR_REF and batch.SD_REF == native.SD_REF
    subprocess.run([sys.executable,"-S",str(ROOT/"gcp/exp40_threeway_48h_batch.py"),*cli],check=True)
    subprocess.run(["bash","-n",str(ROOT/"gcp/run_exp40_threeway_48h.sh")],check=True)
    options.run_id = native.RUN_IDS["ucv"]
    with pytest.raises(ValueError): batch.build_job(options)


def test_sd_native_cost_summary_uses_completion_marker(engine,tmp_path):
    record = next(r for r in records() if r["algorithm"] == "sd")
    worker = engine._worker(tmp_path/"sd","sd",0)
    worker.mkdir(parents=True)
    engine.write_json(worker/"SUCCESS.json",dict(active_seconds=172803,final_nodes=695507354))
    result = sources.cost_context(tmp_path,[record],engine)
    assert result[0]["source_summary_file"] == "SUCCESS.json"
    assert result[0]["native_training_summary"]["final_nodes"] == 695507354


def test_native_loaders_and_spawn_in_fresh_process(tmp_path):
    result = subprocess.run([sys.executable,str(ROOT/"tests/exp40_native_checks.py"),str(tmp_path)],
        cwd=ROOT,text=True,capture_output=True,timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "Native three-way smoke and cache reuse passed" in result.stdout
