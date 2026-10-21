"""Endpoint/lineage integrity, paired uncertainty, scoring and interruption recovery."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.exp39_fhp_48h_evaluation import run, statistics as stats, sources, rare
from gcp import exp39_48h_evaluation_batch as batch
from tests.test_exp37_critic_budget_evaluation import fake_score

ROOT = Path(__file__).resolve().parents[1]


def records():
    return [dict(experiment=c, seed=s, training_hours=h, checkpoint_path=f"/{c}/{s}/{h}.pkl",
        checkpoint_sha256=f"{c}-{s}-{h}", nodes_touched=h*100)
        for c in stats.COHORTS for s in range(3) for h in range(6, sources.HOURS[c]+1, 6)]


def arguments(tmp_path, **kw):
    options = ["--output-dir", str(tmp_path/"analysis"), "--source10-run", str(tmp_path/"source10"),
        "--rare-panel", str(tmp_path/"panel.json")]
    for c in stats.COHORTS:
        options += ["--"+c+"-run", str(tmp_path/c)]
    return SimpleNamespace(**(vars(run.parser().parse_args(options))|kw))


def fixture_sources(tmp_path, monkeypatch):
    payloads, roots = {}, {c: tmp_path/c for c in ("source10", *stats.COHORTS)}
    for cohort in ("source10", "exp16", "exp36", "exp38"):
        contract = sources.old_contract if cohort in ("source10", "exp16") else sources.new_contract
        config = sources.original if cohort == "source10" else sources.CONFIGS[cohort]
        hours = 24 if cohort == "source10" else sources.HOURS[cohort]
        for seed in range(3):
            worker = roots[cohort]/"workers"/contract.worker_name(seed)
            (worker/"checkpoints").mkdir(parents=True)
            encoder = sources.native.make_feature_encoder(config.EXPERIMENT_CONFIG["feature_encoder_id"])
            manifest = dict(experiment_id=config.EXPERIMENT_ID, experiment_name=config.EXPERIMENT_NAME,
                algorithm_id=config.ALGORITHM_ID, seed=seed, smoke=False,
                training_config=deepcopy(config.EXPERIMENT_CONFIG), training_config_sha256=contract.CONFIG_SHA256,
                repository_commit=contract.TRAINING_REF, execution_backend="ray_parallel", reference_vm=config.REFERENCE_VM,
                feature_encoder=encoder.metadata(), game={"parameters": dict(run.protocol.FHP_GAME_PARAMETERS)},
                training_duration_seconds=hours*3600, training_state_retention="final",
                checkpoint_schedule=list(config.checkpoint_schedule(total_hours=hours)))
            runtime = dict(contract.RUNTIME, reference_vm=config.REFERENCE_VM,
                parallel_settings=config.parallel_settings(), feature_encoder=encoder.metadata())
            rows = []
            for h in range(6, hours+1, 6):
                policy = worker/"checkpoints"/f"time_{h:02d}h.pkl"
                identity = ("source10" if cohort == "exp16" else "exp36") if hours == 48 and h <= 24 else cohort
                policy.write_bytes(f"{identity}-{seed}-{h}".encode())
                payloads[str(policy)] = dict(experiment_name=config.EXPERIMENT_NAME, algorithm_id=config.ALGORITHM_ID,
                    seed=seed, training_config=config.EXPERIMENT_CONFIG, nodes_touched=h*100, outer_iteration=h,
                    checkpoint_target_seconds=h*3600, feature_encoder=encoder.metadata())
                rows.append(dict(checkpoint_id=f"time_{h:02d}h", path=f"checkpoints/{policy.name}",
                    sha256=run.sources.sha256_file(policy), checkpoint_target_seconds=h*3600,
                    actual_training_elapsed_seconds=h*3600+1, outer_iteration=h, nodes_touched=h*100))
            rows[-1].update(training_state_path=f"training_states/time_{hours}h.pt", training_state_sha256="a"*64)
            summary = dict(seed=seed, status="complete", checkpoint_count=hours//6,
                final_training_elapsed_seconds=hours*3600+1, final_nodes_touched=hours*100, final_iteration=hours,
                resumed_from_training_state=hours == 48,
                execution_diagnostics=dict(critic_train_steps_per_fit=[config.EXPERIMENT_CONFIG["baseline_network_train_steps"]]*2))
            for name, value in (("run_manifest.json", manifest), ("runtime_manifest.json", runtime),
                    ("summary.json", summary), ("checkpoint_manifest.json", rows)):
                (worker/name).write_text(json.dumps(value))
            (worker/"SUCCESS.json").write_text(json.dumps(dict(status="complete",
                summary_sha256=run.sources.sha256_file(worker/"summary.json"))))
            if hours == 48:
                source = roots["source10" if cohort == "exp16" else "exp36"]/"workers"/contract.worker_name(seed)
                (worker/"continuation_source.json").write_text(json.dumps(dict(seed=seed, source_total_hours=24,
                    total_hours=48, source_commit=contract.TRAINING_REF,
                    source_state_path="training_states/time_24h.pt", source_state_sha256="a"*64,
                    source_summary_sha256=run.sources.sha256_file(source/"summary.json"),
                    source_worker=f"gs://test-bucket/{contract.SOURCE_RUN_ID}/workers/{contract.worker_name(seed)}")))
    monkeypatch.setattr(sources.native, "LoadedFHPPolicy", lambda game, path: SimpleNamespace(checkpoint=payloads[str(path)]))
    return roots


def test_full_protocol_and_matched_external_deals(tmp_path):
    tasks = run.build_tasks(records(), arguments(tmp_path))
    assert Counter(t["kind"] for t in tasks) == {"direct_crossplay": 180, "rule": 45, "lbr": 900}
    assert sum(t["num_deals"] for t in tasks) == 1_359_000
    assert len({t["task_id"] for t in tasks}) == 1125
    assert len({t["evaluation_seed"] for t in tasks}) == 1110  # 15 intentionally paired rule streams
    rules = {(t["experiment"], t["training_seed"], t["opponent"]): t for t in tasks if t["kind"] == "rule"}
    for s in range(3):
        for opponent in run.protocol.PUBLISHED_AGENT_NAMES:
            assert rules[("exp36", s, opponent)]["evaluation_seed"] == rules[("exp38", s, opponent)]["evaluation_seed"]
    for task in tasks:
        if task["kind"] == "direct_crossplay":
            assert task["hours_a"] == 48 and task["hours_b"] == sources.HOURS[task["cohort_b"]]
            assert task["num_deals"] == 5000
        else:
            assert task["training_hours"] == sources.HOURS[task["experiment"]]
    assert {t["task_id"]: t["evaluation_seed"] for t in tasks} == {t["task_id"]: t["evaluation_seed"] for t in run.build_tasks(list(reversed(records())), arguments(tmp_path))}


def cells(value=0., se=0.):
    return [dict(cohort_a=a, cohort_b=b, seed_a=i, seed_b=j, num_deal_pairs=50000,
        mean_mbb_per_hand=value, se_mbb_per_hand=se) for a, b in stats.PAIRS for i in range(3) for j in range(3)]


def test_continuation_bootstrap_keeps_shared_trajectory_weights():
    data = cells()
    for row in data:
        row["mean_mbb_per_hand"] = 100*(row["seed_a"]-row["seed_b"])
    primary, matched = stats.comparisons(data)
    # Cancelling paired trajectory effects must not acquire artificial variance.
    assert primary[1]["bootstrap_ci95_low"] == pytest.approx(0, abs=1e-12)
    assert primary[1]["bootstrap_ci95_high"] == pytest.approx(0, abs=1e-12)
    assert primary[0]["bootstrap_ci95_high"] > 50  # independent cohort 16
    assert matched["mean_mbb_per_hand"] == 0
    assert stats.comparisons(data) == (primary, matched)
    with pytest.raises(ValueError, match="Incomplete"):
        stats.comparisons(data[:-1])
    noisy, _ = stats.comparisons(cells(se=10))
    assert all(r["conditional_deal_se_mbb_per_hand"] == pytest.approx(10/3) for r in noisy)
    assert all(r["bootstrap_family_ci_low"] <= r["bootstrap_ci95_low"] for r in noisy)
    assert stats.decision_screen(noisy)["automatic_promotion"] is False


def test_diagonal_is_not_all_pair_estimate():
    data = cells()
    for row in data:
        row["mean_mbb_per_hand"] = 90 if row["seed_a"] == row["seed_b"] else 0
    primary, matched = stats.comparisons(data)
    assert primary[1]["mean_mbb_per_hand"] == 30
    assert matched["mean_mbb_per_hand"] == 90
    assert matched["num_deal_pairs"] == 150000


def test_paired_external_changes():
    data = [dict(experiment=c, training_seed=s, mean_mbb_per_hand=100*s+(10 if c == "exp38" else 0))
        for c in stats.COHORTS for s in range(3)]
    rows, summary = stats.paired_changes(data, data)
    assert len(rows) == 6
    assert all(r["ci95_low"] == r["ci95_high"] == r["mean_change_mbb_per_hand"] == 10 for r in summary)


@pytest.mark.parametrize("mutation", [None, "lineage", "imported_history", "config", "wrong_horizon", "state_hash"])
def test_sources_and_continuation_integrity(tmp_path, monkeypatch, mutation):
    roots = fixture_sources(tmp_path, monkeypatch)
    root = roots.pop("source10")
    worker = roots["exp38"]/"workers"/sources.new_contract.worker_name(0)
    if mutation:
        filename = "continuation_source.json" if mutation in ("lineage", "state_hash") else "run_manifest.json"
        if mutation == "imported_history":
            (worker/"checkpoints/time_24h.pkl").write_bytes(b"changed")
        else:
            path = worker/filename
            data = json.loads(path.read_text())
            if mutation == "lineage": data["source_worker"] = "gs://wrong/seed_0"
            elif mutation == "state_hash": data["source_state_sha256"] = "b"*64
            elif mutation == "config": data["training_config"]["baseline_network_train_steps"] = 5000
            elif mutation == "wrong_horizon": data["training_duration_seconds"] = 24*3600
            path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            sources.discover(roots, root)
    else:
        result = sources.discover(roots, root)
        assert len(result) == 9
        assert Counter((r["experiment"], r["training_hours"]) for r in result) == {("exp16", 48): 3, ("exp36", 24): 3, ("exp38", 48): 3}


@pytest.mark.parametrize("smoke", (True, False))
def test_reporting_interruption_and_resume(tmp_path, monkeypatch, smoke):
    fixture_sources(tmp_path, monkeypatch)
    args = arguments(tmp_path, workers=1, rule_deals=2, crossplay_deals=3, crossplay_shard_deals=2,
        lbr_deals=3, lbr_shard_deals=2, lbr_rollouts=2, smoke=smoke)
    monkeypatch.setattr(rare, "load_panel", lambda p: {})
    monkeypatch.setattr(rare, "run", lambda *a, **kw: {"test_fixture": True})
    calls = []
    def fail(task):
        calls.append(task)
        if len(calls) == 3: raise RuntimeError("deliberate failure")
        return fake_score(task)
    monkeypatch.setattr(run.protocol, "_evaluation_worker", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        run.run_analysis(args)
    assert len(list((args.output_dir/"task_results").glob("*.json"))) == 2
    assert not (args.output_dir/"SUCCESS.json").exists()
    args.resume = True
    monkeypatch.setattr(run.protocol, "_evaluation_worker", fake_score)
    output = run.run_analysis(args)
    summary = json.loads((output/"summary.json").read_text())
    assert summary["experiment_id"] == 39 and summary["exact_exploitability"] is False
    assert len(summary["head_to_head"]) == 2
    assert summary["same_trajectory_head_to_head"]["trajectories"] == (1 if smoke else 3)
    assert len(list(output.glob("*.png"))) == 3
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda *a: pytest.fail("Must reuse cached tasks"))
    run.run_analysis(args)
    args.crossplay_deals += 1
    with pytest.raises(ValueError, match="resume mismatch"):
        run.run_analysis(args)


def test_frozen_rare_panel_and_result_recovery(tmp_path, monkeypatch):
    path = tmp_path/"panel.json"
    path.write_text('{"rows": []}')
    with pytest.raises(ValueError, match="checksum"):
        rare.load_panel(path)
    monkeypatch.setattr(rare, "PANEL_SHA256", run.sources.sha256_file(path))
    monkeypatch.setattr(rare.base, "assess", lambda r, p: dict(experiment=r["experiment"], seed=r["seed"],
        checkpoint_sha256=r["checkpoint_sha256"], seconds=1,
        metrics={"assessment/all": dict(positions=10, mean_fold_probability=.1, mean_gap_bb=.2,
            p95_gap_bb=.3, max_gap_bb=.4, fraction_fold_over_half=0)}))
    selected = [r for r in records() if r["training_hours"] == sources.HOURS[r["experiment"]]]
    out = rare.run(tmp_path, selected, "test", panel_path=path, smoke=False)
    assert out["policy_count"] == 9 and "Not fresh holdout" in out["scope"]
    assert (tmp_path/"rare_panel.json").read_bytes() == path.read_bytes()
    monkeypatch.setattr(rare.base, "assess", lambda *a: pytest.fail("Must reuse rare results"))
    assert rare.run(tmp_path, selected, "test", panel_path=path, smoke=False) == out
    result_path = tmp_path/"rare_results/exp38_seed0.json"
    data = json.loads(result_path.read_text())
    data["result"]["checkpoint_sha256"] = "wrong"
    result_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="resume mismatch"):
        rare.run(tmp_path, selected, "test", panel_path=path, smoke=False)


def test_real_policies_scoring_and_parallel_shards(tmp_path):
    import torch
    from fhp_escher.checkpointing import save_policy_checkpoint
    from fhp_escher.features import make_feature_encoder
    from vr_deep_cfr.solver import MLP
    paths = {}
    for cohort in ("exp16", "exp36"):
        encoder = make_feature_encoder(sources.CONFIGS[cohort].EXPERIMENT_CONFIG["feature_encoder_id"])
        size = encoder.policy_layout.total_size
        model = MLP(size, [8], 3)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=model), feature_encoder=encoder,
            num_iteration=1, episode=1, nodes_touched=1, infostate_size=size, action_size=3, network_layers=[8])
        paths[cohort] = str(save_policy_checkpoint(solver, tmp_path/f"{cohort}.pkl", seed=0, config={}, checkpoint_row={}))
    selected = [r for r in records() if r["training_hours"] == sources.HOURS[r["experiment"]]]
    for record in selected:
        record["checkpoint_path"] = paths["exp16" if record["experiment"] == "exp16" else "exp36"]
    tasks = run.build_tasks(selected, arguments(tmp_path, smoke=True, rule_deals=2, lbr_deals=2,
        lbr_rollouts=2, crossplay_deals=2))
    for task in [t for t in tasks if t["kind"] == "direct_crossplay"]:
        match = run.protocol._evaluation_worker(task)
        assert match["num_games"] == 4 and np.isfinite(match["mean_mbb_per_hand"])
        assert match == run.protocol._evaluation_worker(task)
    examples = [next(t for t in tasks if t["kind"] == kind and t.get("experiment") == "exp38") for kind in ("rule", "lbr")]
    result = run.run_tasks(examples, 2, tmp_path/"native_tasks", "real-test")
    assert len(result) == 2 and all(r["num_deal_pairs"] == 2 for r in result)
    assert run.run_tasks(examples, 1, tmp_path/"native_tasks", "real-test") == result


def test_batch_stdlib_and_shell_syntax(tmp_path):
    cli = ["--run-id", "exp39-test", "--bucket-root", "gs://test-bucket", "--service-account", "test@example.com",
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
        assert batch.PANEL_RUN_ID in script and "--source10-run" in script
        assert "exp37_fhp" not in script and "exp20" not in script
        assert "--workers 16" in script and "--exp38-run" in script
    subprocess.run([sys.executable, "-S", str(ROOT/"gcp/exp39_48h_evaluation_batch.py"), *cli], check=True)
    subprocess.run(["bash", "-n", str(ROOT/"gcp/run_exp39_48h_evaluation.sh")], check=True)
    args.run_id = args.exp38_run_id
    with pytest.raises(ValueError, match="distinct"):
        batch.build_job(args)


@pytest.mark.parametrize("change", ({"crossplay_deals": 1}, {"lbr_deals": 1}, {"workers": 0}, {"base_seed": -1}))
def test_budget_validation(tmp_path, change):
    with pytest.raises(ValueError):
        run.validate(arguments(tmp_path, **change))
