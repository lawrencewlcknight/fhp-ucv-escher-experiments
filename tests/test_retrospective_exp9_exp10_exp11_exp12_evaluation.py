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

from experiments.fhp.retrospective_exp9_exp10_exp11_exp12_evaluation import run
from gcp.retrospective_exp9_exp10_exp11_exp12_evaluation_batch import build_job


def arguments(tmp_path, **overrides):
    args = run._parser().parse_args([
        "--exp9-run", str(tmp_path / "exp9"), "--exp10-run", str(tmp_path / "exp10"),
        "--exp11-run", str(tmp_path / "exp11"), "--exp12-run", str(tmp_path / "exp12"),
        "--output-dir", str(tmp_path / "analysis"),
    ])
    return Namespace(**(vars(args) | overrides))


def records():
    return [{"experiment": exp, "seed": seed, "training_hours": hour,
             "checkpoint_path": f"/{exp}/{seed}/{hour}.pkl", "checkpoint_sha256": f"{exp}-{seed}-{hour}",
             "nodes_touched": hour * (150 if exp == "exp9" else 100)}
            for exp in run.EXPERIMENTS for seed in run.EXPECTED_SEEDS for hour in run.EXPECTED_HOURS]


def test_same_protocol_with_all_six_pairings(tmp_path):
    args = arguments(tmp_path)
    tasks = run._build_tasks(records(), args)
    assert Counter(t["kind"] for t in tasks) == {
        "rule": 240, "lbr": 4800, "temporal_crossplay": 72, "direct_crossplay": 72,
    }
    assert len({t["task_id"] for t in tasks}) == 5184
    direct = [t for t in tasks if t["kind"] == "direct_crossplay"]
    assert {t["pair_id"] for t in direct} == {f"{a}_vs_{b}" for a, b in run.DIRECT_PAIRS}
    assert len([t for t in direct if t["primary_endpoint"]]) == 9
    assert all(t["num_deals"] == 50_000 for t in direct)
    assert {t["evaluation_seed"] for t in direct} == {args.base_seed + 3_000_000 + h for h in run.EXPECTED_HOURS}
    for task in tasks:
        assert task["policy_a_sha256"]
        if task["kind"] == "direct_crossplay":
            assert task["policy_a_name"].startswith(task["left_experiment"] + "_")
            assert task["policy_b_name"].startswith(task["right_experiment"] + "_")
    rule_rngs = {(t["opponent"], t["evaluation_seed"]) for t in tasks if t["kind"] == "rule"}
    assert len(rule_rngs) == 5
    assert all(t["num_deals"] == 10 and t["lbr_rollouts"] == 4096 for t in tasks if t["kind"] == "lbr")
    assert sum(t["num_deals"] for t in tasks) == 9_648_000
    assert run._run_tasks is run.shared._run_tasks
    assert run.protocol is run.shared.protocol


def test_node_exposure_recorded_without_arbitrary_node_matching(tmp_path):
    tasks = run._build_tasks(records(), arguments(tmp_path))
    assert not any(t["kind"] == "node_matched_crossplay" for t in tasks)
    assert all(t["policy_a_nodes_touched"] > 0 for t in tasks)


def test_smoke_and_partial_shards(tmp_path):
    tasks = run._build_tasks(records(), arguments(tmp_path, smoke=True, lbr_deals=3, lbr_shard_deals=2))
    assert Counter(t["kind"] for t in tasks) == {
        "rule": 40, "lbr": 16, "temporal_crossplay": 4, "direct_crossplay": 12,
    }
    assert {t["training_seed"] for t in tasks} == {0}
    assert [t["num_deals"] for t in tasks if t["kind"] == "lbr"] == [2, 1] * 8


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
                            repository_commit=format(int(exp[3:]), "x") * 40)
            runtime = {"reference_vm": {"machine_type": "n2-standard-16"},
                       "torch_intraop_threads": 8, "frozen_critic_target_cache": True,
                       "traversal_execution": "ray_parallel",
                       "parallel_settings": config.parallel_settings()}
            encoder = run.make_feature_encoder(config.EXPERIMENT_CONFIG.get("feature_encoder_id", run.ENCODER_ID))
            runtime["feature_encoder"] = encoder.metadata()
            rows = []
            for hour in run.EXPECTED_HOURS:
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


def test_source_verification_and_payload_identity(tmp_path, monkeypatch):
    payloads = source_fixture(tmp_path, monkeypatch)
    assert len(run.discover_checkpoints(tmp_path / "exp10", "exp10")) == 12
    key = next(k for k in payloads if "/exp10/" in k)
    payloads[key]["seed"] = 99
    with pytest.raises(ValueError, match="metadata mismatch"):
        run.discover_checkpoints(tmp_path / "exp10", "exp10")


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
        value["experiment_name"] = run.EXPERIMENTS["exp10"]["experiment_name"]
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
        mean = float(task["training_seed"] + int(task["policy_a_name"].split("_")[0][3:]) - 8)
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
    assert manifest["evaluation"]["positive_direct_favours"] == "left_experiment"
    assert manifest["evaluation"]["primary_endpoint_hours"] == 24
    assert {r["source_repository_commit"] for r in manifest["checkpoints"]} == {format(i, "x") * 40 for i in range(9, 13)}
    assert len(list(result.glob("*.png"))) == 7
    assert (result / "paired_metric_differences_aggregate.csv").is_file()
    assert json.loads((result / "SUCCESS.json").read_text())["smoke"] is False
    with (result / "paired_metric_differences_aggregate.csv").open() as handle:
        differences = list(csv.DictReader(handle))
    assert all(float(r["mean_mbb_per_hand"]) == pytest.approx(
        (int(r["pair_id"].split("_vs_")[0][3:]) - int(r["pair_id"].split("_vs_")[1][3:]))
        * run.protocol.MILLI_BIG_BLINDS_PER_CHIP) for r in differences)
    with (result / "temporal_run_mean_aggregate.csv").open() as handle:
        temporal = list(csv.DictReader(handle))
    assert len(temporal) == 4
    assert all(int(r["num_training_seeds"]) == 3 for r in temporal)
    args.resume = True
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda task: pytest.fail("Must reuse cached results"))
    run.run_analysis(args)
    args.rule_deals += 1
    with pytest.raises(ValueError, match="Resume rejected"):
        run.run_analysis(args)


def batch_args(**overrides):
    return Namespace(**({"repo_ref": "a" * 40, "repo_url": "https://example.invalid/repo.git",
                         "run_id": "fhp-eval9to12-test", "exp9_run_id": "exp9-run", "exp10_run_id": "exp10-run",
                         "exp11_run_id": "exp11-run", "exp12_run_id": "exp12-run",
                         "bucket_root": "gs://test-bucket", "service_account": "test@example.invalid",
                         "max_hours": 48, "resume": False, "smoke_only": False} | overrides))


def test_single_vm_job_smoke_resume_and_uploads(tmp_path):
    job = build_job(batch_args(resume=True))
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 1
    assert group["taskSpec"]["computeResource"]["cpuMilli"] == 16000
    assert group["taskSpec"]["maxRunDuration"] == "172800s"
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "--workers 16 --rule-deals 10000" in script
    assert "--smoke" in script and "--resume" in script
    assert "while sleep 300" in script
    assert script.count("--exclude='.*training_states.*") == 4
    assert "--lbr-rollouts 4096" in script
    script_path = tmp_path / "batch.sh"
    script_path.write_text(script)
    subprocess.run(["bash", "-n", str(script_path)], check=True)


@pytest.mark.parametrize("override", [
    {"run_id": "exp9-run"}, {"repo_ref": "main"}, {"exp9_run_id": "../source"},
    {"bucket_root": "gs://test-bucket/prefix"}, {"max_hours": 0},
])
def test_reject_unsafe_cloud_targets(override):
    with pytest.raises(ValueError):
        build_job(batch_args(**override))


@pytest.mark.parametrize("contents, expected", [("", ""), ("--resume", "--resume")])
def test_launcher_optional_array_with_bash_nounset(contents, expected):
    # macOS ships Bash 3.2: an ordinary empty-array expansion fails under set -u.
    launcher = Path(__file__).resolve().parents[1] / "gcp/run_retrospective_exp9_exp10_exp11_exp12_evaluation.sh"
    expansion = '${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}'
    assert expansion in launcher.read_text()
    result = subprocess.run(
        ["/bin/bash", "-uc", f'RESUME_ARGS=({contents}); printf "%s" {expansion}'],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout == expected


@pytest.mark.parametrize("experiment", ["exp10", "exp11", "exp12"])
def test_rejects_wrong_feature_encoder_before_scoring(tmp_path, monkeypatch, experiment):
    payloads = source_fixture(tmp_path, monkeypatch)
    key = next(k for k in payloads if f"/{experiment}/" in k)
    payloads[key]["feature_encoder"] = run.make_feature_encoder().metadata()
    with pytest.raises(ValueError, match="metadata mismatch"):
        run.discover_checkpoints(tmp_path / experiment, experiment)


def test_smoke_only_and_fixed_python(tmp_path):
    job = build_job(batch_args(smoke_only=True))
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "SMOKE_ONLY=1" in script
    assert 'if [[ "$SMOKE_ONLY" != 1 ]]; then' in script
    assert "uv python install 3.11.16\n" in script
    assert "uv venv --python 3.11.16 --seed" in script
    assert script.index("--smoke") < script.index("--workers 16")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_discovery_does_not_touch_source_files(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for experiment in run.EXPERIMENTS:
        run.discover_checkpoints(tmp_path / experiment, experiment)
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_real_encoders_support_rule_lbr_and_all_direct_pairs(tmp_path):
    """Tiny untrained fixtures exercise real OpenSpiel inference, not strength."""
    import torch
    from fhp_escher.checkpointing import save_policy_checkpoint
    from fhp_escher.features import StructuredFHPMLP
    torch.set_num_threads(1)
    paths = {}
    with torch.random.fork_rng():
        torch.manual_seed(10)
        for experiment, config in run.CONFIGS.items():
            encoder = run.make_feature_encoder(config.EXPERIMENT_CONFIG.get("feature_encoder_id", run.ENCODER_ID))
            model = StructuredFHPMLP(encoder.policy_layout, (8,), 3, branch_width=4)
            solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=model),
                feature_encoder=encoder, num_iteration=1, episode=1, nodes_touched=100,
                infostate_size=encoder.policy_size, action_size=3, network_layers=(8,))
            paths[experiment] = str(save_policy_checkpoint(
                solver, tmp_path / f"{experiment}.pkl", seed=0, config={"untrained_test_fixture": True},
                checkpoint_row={"experiment_name": config.EXPERIMENT_NAME, "algorithm_id": config.ALGORITHM_ID}))
    tasks = []
    for experiment in run.EXPERIMENTS:
        for kind in ("rule", "lbr"):
            tasks.append({"kind": kind, "task_id": f"{experiment}_{kind}", "policy_a_path": paths[experiment],
                "policy_a_name": experiment, "num_deals": 2, "evaluation_seed": 10,
                "opponent": run.protocol.PUBLISHED_AGENT_NAMES[0], "lbr_rollouts": 8, "lbr_seed": 11})
    for left, right in run.DIRECT_PAIRS:
        tasks.append({"kind": "direct_crossplay", "task_id": f"{left}_vs_{right}",
            "policy_a_path": paths[left], "policy_b_path": paths[right], "policy_a_name": left,
            "policy_b_name": right, "num_deals": 2, "evaluation_seed": 12})
    for task in tasks:
        result = run.protocol._evaluation_worker(task)
        assert result["num_deal_pairs"] == 2
        assert np.isfinite(result["mean_mbb_per_hand"])


def test_smoke_report_handles_single_seed_uncertainty(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    def score(task):
        values = {key: np.array([0., 1.]) for key in ("paired", "player_zero", "player_one")}
        result = dict(task, **run.protocol._match_summary(values, policy_a=task["policy_a_name"], policy_b="fixture"))
        if task["kind"] == "lbr":
            result.update({"_" + key + "_values": value.tolist() for key, value in values.items()})
        return result
    monkeypatch.setattr(run.protocol, "_evaluation_worker", score)
    args = arguments(tmp_path, smoke=True, workers=1, rule_deals=2, lbr_deals=2,
                     lbr_shard_deals=2, crossplay_deals=2)
    output = run.run_analysis(args)
    assert json.loads((output / "SUCCESS.json").read_text())["smoke"]
    assert "SMOKE TEST ONLY" in (output / "analysis_summary.md").read_text()
    assert len(list(output.glob("*.png"))) == 7
