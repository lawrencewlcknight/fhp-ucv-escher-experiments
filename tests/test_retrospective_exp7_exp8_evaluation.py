from argparse import Namespace
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.retrospective_exp7_exp8_evaluation import run
from gcp.retrospective_exp7_exp8_evaluation_batch import build_job


def arguments(tmp_path, **overrides):
    args = run._parser().parse_args([
        "--exp7-run", str(tmp_path / "exp7"), "--exp8-run", str(tmp_path / "exp8"),
        "--output-dir", str(tmp_path / "analysis"),
    ])
    return Namespace(**(vars(args) | overrides))


def records():
    return [{"experiment": exp, "seed": seed, "training_hours": hour,
             "checkpoint_path": f"/{exp}/{seed}/{hour}.pkl", "checkpoint_sha256": f"{exp}-{seed}-{hour}",
             "nodes_touched": hour * 100}
            for exp in run.EXPERIMENTS for seed in run.EXPECTED_SEEDS for hour in run.SOURCE_HOURS[exp]]


def test_schedule_budgets_orientation_and_common_random_numbers(tmp_path):
    args = arguments(tmp_path)
    tasks = run._build_tasks(records(), args)
    assert Counter(t["kind"] for t in tasks) == {
        "rule": 90, "lbr": 1800, "temporal_crossplay": 30, "direct_crossplay": 15,
    }
    assert len({t["task_id"] for t in tasks}) == 1935
    assert sum(t["num_deals"] for t in tasks) == 3_168_000
    assert len({t["policy_a_path"] for t in tasks if t["kind"] == "rule"}) == 18
    assert {t["training_hours"] for t in tasks if t["kind"] == "rule" and t["experiment"] == "exp7"} == {24}
    temporal = [t for t in tasks if t["kind"] == "temporal_crossplay"]
    assert len({t["comparison_id"] for t in temporal}) == 10
    assert len([t for t in temporal if t["adjacent_checkpoint"]]) == 12
    assert len([t for t in temporal if t["late_stage_endpoint"]]) == 6
    assert {t["comparison_id"] for t in temporal if t["primary_endpoint"]} == {"exp8_48h_vs_exp8_24h"}
    assert len([t for t in temporal if t["primary_endpoint"]]) == 3
    for task in tasks:
        assert task["policy_a_sha256"]
        if task["kind"] == "direct_crossplay":
            assert task["right_hours"] == 24
            assert task["policy_a_name"].startswith("exp8_")
            assert task["policy_b_name"].startswith("exp7_")
            assert task["evaluation_seed"] == args.base_seed + 3_000_000 + task["left_hours"]
        if task["kind"] in ("direct_crossplay", "temporal_crossplay"):
            assert task["num_deals"] == 50_000
            assert task["policy_b_sha256"]
        if task["kind"] == "temporal_crossplay":
            assert task["left_hours"] > task["right_hours"]
            assert task["evaluation_seed"] == args.base_seed + 2_000_000 + task["right_hours"] * 1000 + task["left_hours"]
    assert len({(t["opponent"], t["evaluation_seed"]) for t in tasks if t["kind"] == "rule"}) == 5
    assert all(t["num_deals"] == 10 and t["lbr_rollouts"] == 4096 for t in tasks if t["kind"] == "lbr")
    assert not any(t["kind"] == "node_matched_crossplay" for t in tasks)
    assert run._run_tasks is run.shared._run_tasks
    assert run.protocol is run.shared.protocol


def test_smoke_exercises_all_hours_and_comparison_types_with_partial_shards(tmp_path):
    tasks = run._build_tasks(records(), arguments(tmp_path, smoke=True, lbr_deals=3, lbr_shard_deals=2))
    assert Counter(t["kind"] for t in tasks) == {
        "rule": 30, "lbr": 12, "temporal_crossplay": 10, "direct_crossplay": 5,
    }
    assert {t["training_seed"] for t in tasks} == {0}
    assert [t["num_deals"] for t in tasks if t["kind"] == "lbr"] == [2, 1] * 6
    assert {t["training_hours"] for t in tasks} == {24, 30, 36, 42, 48}


def source_fixture(tmp_path, monkeypatch):
    payloads = {}
    for exp, config in run.CONFIGS.items():
        for seed in run.EXPECTED_SEEDS:
            worker = tmp_path / exp / "workers" / f"task_{seed:03d}"
            (worker / "checkpoints").mkdir(parents=True)
            contract = {k: v for k, v in run.EXPERIMENTS[exp].items() if k != "label"}
            manifest = dict(contract, smoke=False, seed=seed,
                            training_config=deepcopy(config.EXPERIMENT_CONFIG), repository_commit="a" * 40,
                            checkpoint_schedule=list(config.checkpoint_schedule()),
                            training_duration_seconds=run.SOURCE_HOURS[exp][-1] * 3600,
                            game={"parameters": dict(run.protocol.FHP_GAME_PARAMETERS)})
            runtime = {"reference_vm": {"machine_type": "n2-standard-16"},
                       "torch_intraop_threads": 8, "torch_interop_threads": 8,
                       "frozen_critic_target_cache": False, "traversal_execution": "ray_parallel",
                       "parallel_settings": config.parallel_settings()}
            encoder = run.make_feature_encoder(config.EXPERIMENT_CONFIG.get("feature_encoder_id", run.ENCODER_ID))
            rows = []
            for hour in run.SOURCE_HOURS[exp]:
                checkpoint = worker / "checkpoints" / f"{hour}.pkl"
                checkpoint.write_bytes(f"{exp}-{seed}-{hour}".encode())
                payloads[str(checkpoint)] = dict(contract, seed=seed, training_config=deepcopy(config.EXPERIMENT_CONFIG),
                                                nodes_touched=hour*100, outer_iteration=hour,
                                                checkpoint_target_seconds=hour*3600, feature_encoder=encoder.metadata())
                rows.append({"checkpoint_id": f"time_{hour:02d}h", "checkpoint_target_seconds": hour*3600,
                             "nodes_touched": hour*100, "outer_iteration": hour,
                             "actual_training_elapsed_seconds": hour*3600+1,
                             "path": f"checkpoints/{hour}.pkl", "sha256": run.sha256_file(checkpoint)})
            for filename, content in (("run_manifest.json", manifest), ("runtime_manifest.json", runtime),
                                      ("checkpoint_manifest.json", rows), ("SUCCESS.json", {"status": "complete"})):
                (worker / filename).write_text(json.dumps(content))
    monkeypatch.setattr(run, "LoadedFHPPolicy", lambda game, path: SimpleNamespace(checkpoint=payloads[str(path)]))
    return payloads


def test_source_verification_includes_unscored_early_policies(tmp_path, monkeypatch):
    payloads = source_fixture(tmp_path, monkeypatch)
    assert len(run.discover_checkpoints(tmp_path / "exp7", "exp7")) == 12
    exp8 = run.discover_checkpoints(tmp_path / "exp8", "exp8")
    assert len(exp8) == 24
    assert sum(r["selected_for_evaluation"] for r in exp8) == 15
    payloads[str(tmp_path / "exp8/workers/task_000/checkpoints/6.pkl")]["seed"] = 99
    with pytest.raises(ValueError, match="metadata mismatch"):
        run.discover_checkpoints(tmp_path / "exp8", "exp8")


@pytest.mark.parametrize("change", [
    "smoke", "config", "commit", "schedule", "horizon", "game", "seed", "experiment", "algorithm",
    "cache", "threads", "interop", "workers", "backend", "vm", "success", "continuation",
    "hash", "duplicate_hour", "missing", "target", "elapsed", "nodes", "escape", "encoder",
])
def test_reject_bad_sources(tmp_path, monkeypatch, change):
    payloads = source_fixture(tmp_path, monkeypatch)
    worker = tmp_path / "exp8/workers/task_000"
    manifest_changes = {
        "smoke": ("smoke", True), "config": ("training_config", {}), "commit": ("repository_commit", "main"),
        "schedule": ("checkpoint_schedule", []), "horizon": ("training_duration_seconds", 24*3600),
        "game": ("game", {}), "seed": ("seed", 99), "experiment": ("experiment_name", "wrong"),
        "algorithm": ("algorithm_id", "wrong"),
    }
    runtime_changes = {
        "cache": ("frozen_critic_target_cache", True), "threads": ("torch_intraop_threads", 1),
        "interop": ("torch_interop_threads", 1), "workers": ("parallel_settings", {}),
        "backend": ("traversal_execution", "sequential"), "vm": ("reference_vm", {}),
    }
    if change in manifest_changes or change in runtime_changes:
        filename = "run_manifest.json" if change in manifest_changes else "runtime_manifest.json"
        key, value = (manifest_changes | runtime_changes)[change]
        path = worker / filename
        content = json.loads(path.read_text())
        content[key] = value
        path.write_text(json.dumps(content))
    elif change in ("continuation", "success"):
        path = worker / ("continuation_source.json" if change == "continuation" else "SUCCESS.json")
        path.write_text(json.dumps({"status": "failed"}))
    elif change == "encoder":
        payloads[str(worker / "checkpoints/48.pkl")]["feature_encoder"] = {}
    else:
        path = worker / "checkpoint_manifest.json"
        rows = json.loads(path.read_text())
        if change == "hash":
            rows[0]["sha256"] = "wrong"
        elif change == "duplicate_hour":
            rows[0] = rows[1]
        elif change == "missing":
            rows.pop()
        elif change == "target":
            rows[-1]["checkpoint_target_seconds"] += 1
        elif change == "elapsed":
            rows[-1]["actual_training_elapsed_seconds"] = 1
        elif change == "nodes":
            rows[-1]["nodes_touched"] = 1
            payloads[str(worker / "checkpoints/48.pkl")]["nodes_touched"] = 1
        elif change == "escape":
            outside = tmp_path / "outside.pkl"
            outside.write_bytes(b"outside")
            rows[0]["path"] = str(outside)
        path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        run.discover_checkpoints(tmp_path / "exp8", "exp8")


def test_missing_seed_and_mixed_commit_fail_before_scoring(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    worker = tmp_path / "exp8/workers/task_002"
    worker.rename(worker.with_name("not_a_worker"))
    with pytest.raises(ValueError, match="requires seeds"):
        run.discover_checkpoints(tmp_path / "exp8", "exp8")
    worker.with_name("not_a_worker").rename(worker)
    path = worker / "run_manifest.json"
    content = json.loads(path.read_text())
    content["repository_commit"] = "b" * 40
    path.write_text(json.dumps(content))
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda task: pytest.fail("Must not score"))
    with pytest.raises(ValueError, match="Mixed source commits"):
        run.run_analysis(arguments(tmp_path))


def synthetic_score(task):
    # Synthetic deterministic measurements only; never policy-strength evidence.
    mean = task["training_seed"] + task["training_hours"] / 6
    values = {key: np.full(task["num_deals"], mean) for key in ("paired", "player_zero", "player_one")}
    result = dict(task, **run.protocol._match_summary(values, policy_a=task["policy_a_name"], policy_b="fixture"))
    if task["kind"] == "lbr":
        result.update({"_" + key + "_values": value.tolist() for key, value in values.items()})
    return result


@pytest.mark.parametrize("smoke", [False, True])
def test_end_to_end_reporting_seed_uncertainty_and_resume(tmp_path, monkeypatch, smoke):
    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(run.protocol, "_evaluation_worker", synthetic_score)
    args = arguments(tmp_path, smoke=smoke, workers=1, rule_deals=2, lbr_deals=3,
                     lbr_shard_deals=2, crossplay_deals=2)
    output = run.run_analysis(args)
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["evaluation"]["primary_endpoint"] == "exp8_48h_vs_exp8_24h"
    assert manifest["evaluation"]["num_scored_policies"] == (6 if smoke else 18)
    assert len(manifest["checkpoints"]) == 36
    assert len(list(output.glob("*.png"))) == 7
    assert json.loads((output / "SUCCESS.json").read_text())["smoke"] is smoke
    report = (output / "analysis_summary.md").read_text()
    assert ("SMOKE TEST ONLY" in report) is smoke
    assert ("Not estimable (one seed)" in report) is smoke
    assert "[nan, nan]" not in report
    assert "does not establish equivalence or convergence" in report
    with (output / "paired_metric_differences_aggregate.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    for row in rows:
        expected = (int(row["left_hours"]) - int(row["right_hours"])) / 6 * run.protocol.MILLI_BIG_BLINDS_PER_CHIP
        assert float(row["mean_mbb_per_hand"]) == pytest.approx(expected)
        assert int(row["num_training_seeds"]) == (1 if smoke else 3)
    with (output / "temporal_crossplay_aggregate.csv").open() as handle:
        temporal = list(csv.DictReader(handle))
    assert len(temporal) == 10
    for row in temporal:
        if smoke:
            assert row["training_seed_ci95_low_mbb_per_hand"] == ""
        else:
            expected = run.protocol._aggregate([(s + int(row["left_hours"]) / 6)
                                               * run.protocol.MILLI_BIG_BLINDS_PER_CHIP for s in (0, 1, 2)])
            assert float(row["training_seed_ci95_low_mbb_per_hand"]) == pytest.approx(expected["training_seed_ci95_low_mbb_per_hand"])
    args.resume = True
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda task: pytest.fail("Must reuse tasks"))
    run.run_analysis(args)
    args.crossplay_deals += 1
    with pytest.raises(ValueError, match="Resume rejected"):
        run.run_analysis(args)


def test_failed_score_has_failure_marker_not_success(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    def fail(task):
        raise RuntimeError("synthetic scoring failure")
    monkeypatch.setattr(run.protocol, "_evaluation_worker", fail)
    args = arguments(tmp_path, workers=1, smoke=True)
    with pytest.raises(RuntimeError, match="synthetic scoring failure"):
        run.run_analysis(args)
    assert json.loads((args.output_dir / "evaluation_manifest.json").read_text())["status"] == "failed"
    assert (args.output_dir / "failure.json").exists()
    assert not (args.output_dir / "SUCCESS.json").exists()


def test_cache_detects_corrupt_scores_and_changed_policy_hash(tmp_path, monkeypatch):
    task = run._build_tasks(records(), arguments(tmp_path))[0]
    monkeypatch.setattr(run.protocol, "_evaluation_worker", synthetic_score)
    cache = tmp_path / "cache"
    run._run_tasks([task], 1, cache, "fingerprint")
    with pytest.raises(ValueError, match="Incompatible"):
        run._run_tasks([dict(task, policy_a_sha256="changed")], 1, cache, "fingerprint")
    path = cache / (task["task_id"] + ".json")
    value = json.loads(path.read_text())
    value["result"]["mean_mbb_per_hand"] = -999
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="corrupt"):
        run._run_tasks([task], 1, cache, "fingerprint")


def batch_args(**overrides):
    return Namespace(**({"repo_ref": "a" * 40, "repo_url": "https://example.invalid/repo.git",
                         "run_id": "fhp-eval78-test", "exp7_run_id": "exp7-run", "exp8_run_id": "exp8-run",
                         "bucket_root": "gs://test-bucket", "service_account": "test@example.invalid",
                         "max_hours": 48, "resume": False, "smoke_only": False} | overrides))


@pytest.mark.parametrize("smoke_only", [False, True])
def test_batch_smoke_gate_pin_diagnostics_and_no_training(tmp_path, smoke_only):
    job = build_job(batch_args(resume=True, smoke_only=smoke_only))
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 1
    assert group["taskSpec"]["computeResource"] == {"cpuMilli": 16000, "memoryMib": 62000}
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["maxRunDuration"] == "172800s"
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "uv python install 3.11.16" in script
    assert "--workers 16 --rule-deals 10000" in script
    assert script.index('--output-dir "$OUTPUT_ROOT/smoke"') < script.index('--output-dir "$OUTPUT_ROOT/analysis"')
    assert f"SMOKE_ONLY={int(smoke_only)}" in script
    assert 'if [[ "$SMOKE_ONLY" != 1 ]]' in script
    assert "--smoke" in script and "--resume" in script
    assert "while sleep 300" in script and "batch_diagnostics monitor" in script
    assert script.count("--exclude='.*training_states.*") == 2
    assert "--lbr-rollouts 4096" in script
    assert "exp8_fhp_parallel_48h.run" not in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    launcher = Path(__file__).resolve().parents[1] / "gcp/run_retrospective_exp7_exp8_evaluation.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)


@pytest.mark.parametrize("override", [
    {"run_id": "exp7-run"}, {"repo_ref": "main"}, {"exp8_run_id": "../source"},
    {"bucket_root": "gs://test-bucket/prefix"}, {"max_hours": 0}, {"max_hours": 73},
])
def test_reject_unsafe_cloud_targets(override):
    with pytest.raises(ValueError):
        build_job(batch_args(**override))


@pytest.mark.parametrize("contents, expected", [("", ""), ("--resume", "--resume")])
def test_launcher_optional_array_with_macos_bash(contents, expected):
    launcher = Path(__file__).resolve().parents[1] / "gcp/run_retrospective_exp7_exp8_evaluation.sh"
    expansion = '${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}'
    assert expansion in launcher.read_text()
    result = subprocess.run(["/bin/bash", "-uc", f'RESUME_ARGS=({contents}); printf "%s" {expansion}'],
                            check=True, capture_output=True, text=True)
    assert result.stdout == expected
