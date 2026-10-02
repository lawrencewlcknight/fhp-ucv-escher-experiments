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

from experiments.fhp.retrospective_exp7_exp9_evaluation import run
from gcp.retrospective_exp7_exp9_evaluation_batch import build_job


def arguments(tmp_path, **overrides):
    args = run._parser().parse_args([
        "--exp7-run", str(tmp_path / "exp7"), "--exp9-run", str(tmp_path / "exp9"),
        "--output-dir", str(tmp_path / "analysis"),
    ])
    return Namespace(**(vars(args) | overrides))


def records():
    return [{"experiment": exp, "seed": seed, "training_hours": hour,
             "checkpoint_path": f"/{exp}/{seed}/{hour}.pkl", "checkpoint_sha256": f"{exp}-{seed}-{hour}",
             "nodes_touched": hour * (150 if exp == "exp9" else 100)}
            for exp in run.EXPERIMENTS for seed in run.EXPECTED_SEEDS for hour in run.EXPECTED_HOURS]


def test_same_protocol_as_previous_evaluation_with_correct_new_comparisons(tmp_path):
    args = arguments(tmp_path)
    tasks = run._build_tasks(records(), args)
    assert Counter(t["kind"] for t in tasks) == {
        "rule": 120, "lbr": 2400, "temporal_crossplay": 36,
        "direct_crossplay": 12, "node_matched_crossplay": 3,
    }
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    for task in tasks:
        assert task["policy_a_sha256"]
        if task["kind"] == "direct_crossplay":
            assert task["policy_a_name"].startswith("exp9_")
            assert task["policy_b_name"].startswith("exp7_")
            assert task["num_deals"] == 50_000
        elif task["kind"] == "node_matched_crossplay":
            assert (task["exp9_hours"], task["exp7_hours"]) == (12, 18)
            assert task["exactly_node_matched"] is False
            assert task["relative_node_excess_exp9"] == 0
    rule_rngs = {(t["opponent"], t["evaluation_seed"]) for t in tasks if t["kind"] == "rule"}
    assert len(rule_rngs) == 5  # Common deals across seed, method and checkpoint.
    lbr = [t for t in tasks if t["kind"] == "lbr"]
    assert all(t["num_deals"] == 10 and t["lbr_rollouts"] == 4096 for t in lbr)
    assert sum(t["num_deals"] for t in tasks) == 3_774_000
    assert run._run_tasks is run.shared._run_tasks
    assert run.protocol is run.shared.protocol


def test_node_mismatch_is_reported_not_hidden(tmp_path):
    rows = records()
    for row in rows:
        if row["experiment"] == "exp9":
            row["nodes_touched"] *= 1.1
    nodes = [t for t in run._build_tasks(rows, arguments(tmp_path))
             if t["kind"] == "node_matched_crossplay"]
    assert len(nodes) == 3
    assert all(t["relative_node_excess_exp9"] == pytest.approx(.1) for t in nodes)


def test_smoke_and_partial_shards(tmp_path):
    tasks = run._build_tasks(records(), arguments(tmp_path, smoke=True, lbr_deals=3, lbr_shard_deals=2))
    assert Counter(t["kind"] for t in tasks) == {
        "rule": 20, "lbr": 8, "temporal_crossplay": 2, "direct_crossplay": 2,
    }
    assert {t["training_seed"] for t in tasks} == {0}
    assert [t["num_deals"] for t in tasks if t["kind"] == "lbr"] == [2, 1] * 4


def test_cache_reuses_completed_tasks_and_rejects_changes(tmp_path, monkeypatch):
    tasks = run._build_tasks(records(), arguments(tmp_path))[:2]
    calls = []

    def evaluate(task):
        calls.append(task["task_id"])
        return dict(task, num_deal_pairs=task["num_deals"], mean_mbb_per_hand=1.0)

    monkeypatch.setattr(run.protocol, "_evaluation_worker", evaluate)
    first = run._run_tasks(tasks, 1, tmp_path / "cache", "fingerprint")
    assert len(calls) == 2
    assert run._run_tasks(tasks, 1, tmp_path / "cache", "fingerprint") == first
    assert len(calls) == 2
    moved = [dict(t, policy_a_path="/different/local/root/" + Path(t["policy_a_path"]).name) for t in tasks]
    assert len(run._run_tasks(moved, 1, tmp_path / "cache", "fingerprint")) == 2
    assert len(calls) == 2
    changed = deepcopy(tasks)
    changed[0]["policy_a_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Incompatible"):
        run._run_tasks(changed, 1, tmp_path / "cache", "fingerprint")
    with pytest.raises(ValueError, match="Incompatible"):
        run._run_tasks(tasks, 1, tmp_path / "cache", "different")
    cache_path = tmp_path / "cache" / f"{tasks[0]['task_id']}.json"
    cached = json.loads(cache_path.read_text())
    cached["result"]["mean_mbb_per_hand"] = 100
    cache_path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="corrupt"):
        run._run_tasks(tasks, 1, tmp_path / "cache", "fingerprint")


def source_fixture(tmp_path, monkeypatch):
    payloads = {}
    for exp in run.EXPERIMENTS:
        for seed in run.EXPECTED_SEEDS:
            worker = tmp_path / exp / "workers" / f"task_{seed:03d}"
            (worker / "checkpoints").mkdir(parents=True)
            contract = {k: v for k, v in run.EXPERIMENTS[exp].items() if k != "label"}
            config = run.CONFIGS[exp]
            manifest = dict(contract, smoke=False, seed=seed,
                            training_config=deepcopy(config.EXPERIMENT_CONFIG),
                            repository_commit=("a" if exp == "exp7" else "b") * 40)
            runtime = {"reference_vm": {"machine_type": "n2-standard-16"},
                       "torch_intraop_threads": 8, "frozen_critic_target_cache": exp == "exp9",
                       "traversal_execution": "ray_parallel",
                       "parallel_settings": config.parallel_settings()}
            rows = []
            for hour in run.EXPECTED_HOURS:
                checkpoint = worker / "checkpoints" / f"{hour}.pkl"
                checkpoint.write_bytes(f"{exp}-{seed}-{hour}".encode())
                payloads[str(checkpoint)] = dict(contract, seed=seed, training_config=deepcopy(config.EXPERIMENT_CONFIG),
                                                nodes_touched=hour*100, outer_iteration=hour,
                                                checkpoint_target_seconds=hour*3600)
                rows.append({"checkpoint_id": f"time_{hour:02d}h", "checkpoint_target_seconds": hour*3600,
                             "nodes_touched": hour*100, "outer_iteration": hour,
                             "actual_training_elapsed_seconds": hour*3600+1,
                             "path": f"checkpoints/{hour}.pkl", "sha256": run.sha256_file(checkpoint)})
            for filename, content in (("run_manifest.json", manifest), ("runtime_manifest.json", runtime),
                                      ("checkpoint_manifest.json", rows), ("SUCCESS.json", {"status": "complete"})):
                (worker / filename).write_text(json.dumps(content))
    monkeypatch.setattr(run, "LoadedFHPPolicy", lambda game, path: SimpleNamespace(checkpoint=payloads[str(path)]))
    return payloads


def test_source_verification_and_payload_identity(tmp_path, monkeypatch):
    payloads = source_fixture(tmp_path, monkeypatch)
    assert len(run.discover_checkpoints(tmp_path / "exp7", "exp7")) == 12
    key = next(k for k in payloads if "/exp7/" in k)
    payloads[key]["seed"] = 99
    with pytest.raises(ValueError, match="metadata mismatch"):
        run.discover_checkpoints(tmp_path / "exp7", "exp7")


@pytest.mark.parametrize("change", ["smoke", "hash", "escape", "missing", "duplicate", "workers"])
def test_reject_bad_sources(tmp_path, monkeypatch, change):
    source_fixture(tmp_path, monkeypatch)
    worker = tmp_path / "exp9/workers/task_000"
    if change == "smoke":
        p = worker / "run_manifest.json"
        value = json.loads(p.read_text())
        value["smoke"] = True
    elif change == "workers":
        p = worker / "runtime_manifest.json"
        value = json.loads(p.read_text())
        value["parallel_settings"]["parallel_num_workers"] = 4
    else:
        p = worker / "checkpoint_manifest.json"
        value = json.loads(p.read_text())
        if change == "hash":
            value[0]["sha256"] = "corrupt"
        elif change == "escape":
            escape = tmp_path / "external.pkl"
            escape.write_bytes(b"wrong")
            value[0]["path"] = str(escape)
        elif change == "missing":
            value.pop()
        elif change == "duplicate":
            value[0] = value[1]
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run.discover_checkpoints(tmp_path / "exp9", "exp9")


@pytest.mark.parametrize("change", [
    "learning_config", "cache_config", "runtime_cache", "threads", "backend",
    "missing_commit", "unsuccessful", "wrong_experiment", "wrong_algorithm",
])
def test_reject_noncanonical_training_contracts(tmp_path, monkeypatch, change):
    source_fixture(tmp_path, monkeypatch)
    worker = tmp_path / "exp9/workers/task_000"
    filename = "run_manifest.json"
    if change in ("runtime_cache", "threads", "backend"):
        filename = "runtime_manifest.json"
    elif change == "unsuccessful":
        filename = "SUCCESS.json"
    path = worker / filename
    value = json.loads(path.read_text())
    if change == "learning_config":
        value["training_config"]["num_traversals"] = 999
    elif change == "cache_config":
        value["training_config"]["cache_frozen_critic_targets"] = False
    elif change == "runtime_cache":
        value["frozen_critic_target_cache"] = False
    elif change == "threads":
        value["torch_intraop_threads"] = 1
    elif change == "backend":
        value["traversal_execution"] = "sequential_single_collector"
    elif change == "missing_commit":
        value["repository_commit"] = None
    elif change == "unsuccessful":
        value["status"] = "failed"
    elif change == "wrong_experiment":
        value["experiment_name"] = run.EXPERIMENTS["exp7"]["experiment_name"]
    else:
        value["algorithm_id"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run.discover_checkpoints(tmp_path / "exp9", "exp9")


def test_reject_mixed_commits_within_one_experiment_before_scoring(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    path = tmp_path / "exp9/workers/task_000/run_manifest.json"
    value = json.loads(path.read_text())
    value["repository_commit"] = "c" * 40
    path.write_text(json.dumps(value))
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda task: pytest.fail("Must not score"))
    with pytest.raises(ValueError, match="Mixed source commits within exp9"):
        run.run_analysis(arguments(tmp_path))


def test_failure_records_diagnostics_without_success_marker(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)

    def fail(task):
        raise RuntimeError("synthetic scoring failure")

    monkeypatch.setattr(run.protocol, "_evaluation_worker", fail)
    args = arguments(tmp_path, workers=1, smoke=True)
    with pytest.raises(RuntimeError, match="synthetic scoring failure"):
        run.run_analysis(args)
    manifest = json.loads((args.output_dir / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "synthetic scoring failure" in manifest["error"]
    assert not (args.output_dir / "SUCCESS.json").exists()


def test_end_to_end_aggregation_resume_and_signed_differences(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)

    def evaluate(task):
        # Exact deterministic synthetic measurements, not trained policy results.
        mean = float(task["training_seed"] + (2 if "exp9" in task["policy_a_name"] else 1))
        values = {k: np.full(task["num_deals"], mean) for k in ("paired", "player_zero", "player_one")}
        result = dict(task, **run.protocol._match_summary(values, policy_a=task["policy_a_name"],
                                                         policy_b=task.get("policy_b_name", "opponent")))
        if task["kind"] == "lbr":
            result.update({"_paired_values": values["paired"].tolist(),
                           "_player_zero_values": values["player_zero"].tolist(),
                           "_player_one_values": values["player_one"].tolist()})
        return result

    monkeypatch.setattr(run.protocol, "_evaluation_worker", evaluate)
    args = arguments(tmp_path, workers=1, rule_deals=2, lbr_deals=3, lbr_shard_deals=2, crossplay_deals=2)
    result = run.run_analysis(args)
    manifest = json.loads((result / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["evaluation"]["positive_direct_favours"] == "exp9"
    assert manifest["evaluation"]["primary_endpoint_hours"] == 24
    assert {r["source_repository_commit"] for r in manifest["checkpoints"]} == {"a" * 40, "b" * 40}
    assert len(list(result.glob("*.png"))) == 6
    assert (result / "paired_metric_differences_aggregate.csv").is_file()
    assert json.loads((result / "SUCCESS.json").read_text())["smoke"] is False
    with (result / "paired_metric_differences_aggregate.csv").open() as handle:
        differences = list(csv.DictReader(handle))
    assert all(float(r["mean_mbb_per_hand"]) == pytest.approx(run.protocol.MILLI_BIG_BLINDS_PER_CHIP)
               for r in differences)
    with (result / "temporal_run_mean_aggregate.csv").open() as handle:
        temporal = list(csv.DictReader(handle))
    assert len(temporal) == 2
    assert all(int(r["num_training_seeds"]) == 3 for r in temporal)
    args.resume = True
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda task: pytest.fail("Must reuse cached results"))
    run.run_analysis(args)
    args.rule_deals += 1
    with pytest.raises(ValueError, match="Resume rejected"):
        run.run_analysis(args)


def batch_args(**overrides):
    return Namespace(**({"repo_ref": "a" * 40, "repo_url": "https://example.invalid/repo.git",
                         "run_id": "fhp-eval79-test", "exp7_run_id": "exp7-run", "exp9_run_id": "exp9-run",
                         "bucket_root": "gs://test-bucket", "service_account": "test@example.invalid",
                         "max_hours": 36, "resume": False} | overrides))


def test_single_vm_job_smoke_resume_and_uploads(tmp_path):
    job = build_job(batch_args(resume=True))
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 1
    assert group["taskSpec"]["computeResource"]["cpuMilli"] == 8000
    assert group["taskSpec"]["maxRunDuration"] == "129600s"
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "--workers 8 --rule-deals 10000" in script
    assert "--smoke" in script and "--resume" in script
    assert "while sleep 300" in script
    assert script.count("--exclude='.*training_states.*") == 2
    assert "--lbr-rollouts 4096" in script
    script_path = tmp_path / "batch.sh"
    script_path.write_text(script)
    subprocess.run(["bash", "-n", str(script_path)], check=True)


@pytest.mark.parametrize("override", [
    {"run_id": "exp7-run"}, {"repo_ref": "main"}, {"exp9_run_id": "../source"},
    {"bucket_root": "gs://test-bucket/prefix"}, {"max_hours": 0},
])
def test_reject_unsafe_cloud_targets(override):
    with pytest.raises(ValueError):
        build_job(batch_args(**override))


@pytest.mark.parametrize("contents, expected", [("", ""), ("--resume", "--resume")])
def test_launcher_optional_array_with_bash_nounset(contents, expected):
    # macOS ships Bash 3.2: an ordinary empty-array expansion fails under set -u.
    launcher = Path(__file__).resolve().parents[1] / "gcp/run_retrospective_exp7_exp9_evaluation.sh"
    expansion = '${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}'
    assert expansion in launcher.read_text()
    result = subprocess.run(
        ["/bin/bash", "-uc", f'RESUME_ARGS=({contents}); printf "%s" {expansion}'],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout == expected
