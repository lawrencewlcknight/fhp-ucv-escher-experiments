"""Validate frozen sources, the established scoring protocol and resumable output."""
from argparse import Namespace
from collections import Counter
from copy import deepcopy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _config_sha256
from experiments.fhp.exp20_fhp_half_critic_updates import evaluate as run
from experiments.fhp.exp20_fhp_half_critic_updates.aggregate import task_name


def arguments(tmp_path, **overrides):
    args = run._parser().parse_args([
        "--exp10-run", str(tmp_path / "exp10"), "--exp20-run", str(tmp_path / "exp20"),
        "--output-dir", str(tmp_path / "analysis")])
    return Namespace(**(vars(args) | overrides))


def source_fixture(tmp_path, monkeypatch):
    payloads = {}
    roots = {exp: tmp_path / exp for exp in run.CONFIGS}
    for exp, config in run.CONFIGS.items():
        for seed in run.EXPECTED_SEEDS:
            worker = roots[exp] / "workers" / task_name(seed, seed)
            (worker / "checkpoints").mkdir(parents=True)
            encoder = run.long_eval.make_feature_encoder(config.EXPERIMENT_CONFIG["feature_encoder_id"])
            manifest = dict(experiment_id=config.EXPERIMENT_ID, experiment_name=config.EXPERIMENT_NAME,
                            algorithm_id=config.ALGORITHM_ID, seed=seed, smoke=False,
                            training_config=deepcopy(config.EXPERIMENT_CONFIG),
                            training_config_sha256=_config_sha256(config.EXPERIMENT_CONFIG),
                            repository_commit=run.candidate.BASELINE_REF if exp == "exp10" else "b" * 40,
                            execution_backend="ray_parallel", reference_vm=config.REFERENCE_VM,
                            game={"parameters": dict(run.protocol.FHP_GAME_PARAMETERS)},
                            training_duration_seconds=86400, training_state_retention="final",
                            checkpoint_schedule=list(config.checkpoint_schedule()))
            runtime = dict(reference_vm=config.REFERENCE_VM, torch_intraop_threads=8, torch_interop_threads=8,
                           traversal_execution="ray_parallel", parallel_settings=config.parallel_settings(),
                           frozen_critic_target_cache=True, feature_encoder=encoder.metadata(),
                           python_version="3.11.16", torch_version="2.7.0+cpu", numpy_version="1.26.4",
                           ray_version="2.51.2")
            rows = []
            for hour in run.EXPECTED_HOURS:
                policy = worker / "checkpoints" / f"time_{hour:02d}h.pkl"
                policy.write_bytes(f"{exp}-{seed}-{hour}".encode())
                payloads[str(policy)] = dict(experiment_name=config.EXPERIMENT_NAME,
                    algorithm_id=config.ALGORITHM_ID, seed=seed, training_config=config.EXPERIMENT_CONFIG,
                    nodes_touched=hour * 100, outer_iteration=hour,
                    checkpoint_target_seconds=hour * 3600, feature_encoder=encoder.metadata())
                rows.append(dict(checkpoint_id=f"time_{hour:02d}h", path=f"checkpoints/{policy.name}",
                                 sha256=run.sha256_file(policy), checkpoint_target_seconds=hour * 3600,
                                 actual_training_elapsed_seconds=hour * 3600 + 1,
                                 outer_iteration=hour, nodes_touched=hour * 100))
            rows[-1].update(training_state_path="training_states/time_24h.pt", training_state_sha256="a" * 64)
            summary = dict(seed=seed, status="complete", checkpoint_count=4,
                           final_training_elapsed_seconds=86401,
                           execution_diagnostics={"critic_train_steps_per_fit": [
                               config.EXPERIMENT_CONFIG["baseline_network_train_steps"]] * 2})
            for name, value in (("run_manifest.json", manifest), ("runtime_manifest.json", runtime),
                                ("summary.json", summary), ("checkpoint_manifest.json", rows)):
                (worker / name).write_text(json.dumps(value))
            (worker / "SUCCESS.json").write_text(json.dumps(dict(
                status="complete", summary_sha256=run.sha256_file(worker / "summary.json"))))
    monkeypatch.setattr(run.long_eval, "LoadedFHPPolicy", lambda game, path: SimpleNamespace(checkpoint=payloads[str(path)]))
    return roots


def records():
    return [dict(experiment=exp, seed=seed, training_hours=hour, checkpoint_sha256=f"{exp}-{seed}-{hour}",
                 checkpoint_path=f"/{exp}/{seed}/{hour}.pkl", nodes_touched=hour * 100)
            for exp in run.CONFIGS for seed in run.EXPECTED_SEEDS for hour in run.EXPECTED_HOURS]


def test_protocol_budgets_and_primary_endpoint(tmp_path):
    tasks = run._build_tasks(records(), arguments(tmp_path))
    assert Counter(t["kind"] for t in tasks) == dict(rule=120, lbr=2400, temporal_crossplay=36, direct_crossplay=12)
    assert len({t["task_id"] for t in tasks}) == 2568
    assert sum(t["num_deals"] for t in tasks) == 3_624_000
    assert len([t for t in tasks if t["primary_endpoint"]]) == 3
    direct = [t for t in tasks if t["kind"] == "direct_crossplay"]
    assert all(t["policy_a_path"].startswith("/exp20/") and t["policy_b_path"].startswith("/exp10/") for t in direct)
    assert all(t["training_hours"] == 24 for t in direct if t["primary_endpoint"])
    assert len({(t["opponent"], t["evaluation_seed"]) for t in tasks if t["kind"] == "rule"}) == 5
    assert not any(t["kind"] == "matched_node_crossplay" for t in tasks)


def test_source_contract(tmp_path, monkeypatch):
    roots = source_fixture(tmp_path, monkeypatch)
    rows = [r for exp, root in roots.items() for r in run.discover_checkpoints(root, exp)]
    assert len(rows) == 24
    assert len({r["learning_config_without_critic_budget_sha256"] for r in rows}) == 1


@pytest.mark.parametrize("experiment,recorded_hash", [
    ("exp10", "065734572e18f4597dd8f494ea39230c1f193b9d68aa775138c4a8a3e39eddad"),
    ("exp20", "8c1a01fad8f00628d283a9895ef9b7547e45f3d3b3a71dcadf410c759fc50256"),
])
def test_accept_recorded_training_config_hash(tmp_path, monkeypatch, experiment, recorded_hash):
    # These hashes come from the completed production runs, not the evaluator.
    config = run.CONFIGS[experiment].EXPERIMENT_CONFIG
    assert _config_sha256(config) == recorded_hash
    assert run._digest(config) != recorded_hash
    roots = source_fixture(tmp_path, monkeypatch)
    for worker in (roots[experiment] / "workers").glob("task_*"):
        path = worker / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["training_config_sha256"] = recorded_hash
        path.write_text(json.dumps(manifest))
    assert len(run.discover_checkpoints(roots[experiment], experiment)) == 12


@pytest.mark.parametrize("experiment", ["exp10", "exp20"])
@pytest.mark.parametrize("bad_hash", ["missing", "corrupt", "evaluation_digest"])
def test_reject_invalid_training_config_hash(tmp_path, monkeypatch, experiment, bad_hash):
    roots = source_fixture(tmp_path, monkeypatch)
    path = roots[experiment] / "workers" / task_name(0, 0) / "run_manifest.json"
    manifest = json.loads(path.read_text())
    if bad_hash == "missing":
        manifest.pop("training_config_sha256")
    else:
        manifest["training_config_sha256"] = (
            run._digest(manifest["training_config"]) if bad_hash == "evaluation_digest" else "0" * 64)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"Training configuration checksum mismatch for {experiment} seed 0"):
        run.discover_checkpoints(roots[experiment], experiment)


@pytest.mark.parametrize("experiment", ["exp10", "exp20"])
def test_reject_changed_summary_checksum(tmp_path, monkeypatch, experiment):
    roots = source_fixture(tmp_path, monkeypatch)
    path = roots[experiment] / "workers" / task_name(0, 0) / "summary.json"
    summary = json.loads(path.read_text())
    summary["final_training_elapsed_seconds"] += 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Incomplete or incompatible source metadata"):
        run.discover_checkpoints(roots[experiment], experiment)


@pytest.mark.parametrize("change", ["config", "summary", "commit", "python", "threads", "smoke", "policy", "continuation", "critic"])
def test_reject_incompatible_sources(tmp_path, monkeypatch, change):
    roots = source_fixture(tmp_path, monkeypatch)
    exp = "exp10" if change == "commit" else "exp20"
    worker = roots[exp] / "workers" / task_name(0, 0)
    if change == "continuation":
        (worker / "continuation_source.json").write_text("{}")
    elif change == "policy":
        (worker / "checkpoints/time_24h.pkl").write_bytes(b"corrupt")
    else:
        name = {"config": "run_manifest.json", "commit": "run_manifest.json", "smoke": "run_manifest.json",
                "summary": "summary.json", "critic": "summary.json", "python": "runtime_manifest.json",
                "threads": "runtime_manifest.json"}[change]
        path = worker / name
        value = json.loads(path.read_text())
        if change == "config": value["training_config"]["baseline_network_train_steps"] = 10_000
        elif change == "summary": value["status"] = "failed"
        elif change == "commit": value["repository_commit"] = "a" * 40
        elif change == "python": value["python_version"] = "3.11.17"
        elif change == "threads": value["torch_intraop_threads"] = 4
        elif change == "smoke": value["smoke"] = True
        elif change == "critic": value["execution_diagnostics"]["critic_train_steps_per_fit"] = [5000, 10000]
        path.write_text(json.dumps(value))
        if change == "critic":
            (worker / "SUCCESS.json").write_text(json.dumps(dict(status="complete", summary_sha256=run.sha256_file(path))))
    with pytest.raises(ValueError):
        run.discover_checkpoints(roots[exp], exp)


def synthetic_score(task):
    # Deliberately artificial scores to validate analysis code, not poker strength.
    value = task["training_hours"] / 6 + task["training_seed"] + (task["experiment"] == "exp20")
    values = {key: np.full(task["num_deals"], value) for key in ("paired", "player_zero", "player_one")}
    result = dict(task, **run.protocol._match_summary(values, policy_a=task["policy_a_name"], policy_b="fixture"))
    if task["kind"] == "lbr":
        result.update({"_" + key + "_values": value.tolist() for key, value in values.items()})
    return result


@pytest.mark.parametrize("smoke", [True, False])
def test_reporting_and_resume(tmp_path, monkeypatch, smoke):
    source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(run.protocol, "_evaluation_worker", synthetic_score)
    args = arguments(tmp_path, workers=1, smoke=smoke, rule_deals=2, lbr_deals=3,
                     lbr_shard_deals=2, crossplay_deals=2)
    output = run.run_analysis(args)
    assert json.loads((output / "SUCCESS.json").read_text())["smoke"] is smoke
    assert len(list(output.glob("*.png"))) == 5
    assert "nan" not in (output / "analysis_summary.md").read_text()
    with (output / "metric_differences_aggregate.csv").open() as f:
        changes = list(csv.DictReader(f))
    assert len(changes) == (4 if smoke else 8)
    for row in changes:
        assert float(row["mean_mbb_per_hand"]) == pytest.approx(run.protocol.MILLI_BIG_BLINDS_PER_CHIP)
        assert int(row["num_training_seeds"]) == (1 if smoke else 3)
    args.resume = True
    monkeypatch.setattr(run.protocol, "_evaluation_worker", lambda task: pytest.fail("Should reuse every task"))
    run.run_analysis(args)
    args.crossplay_deals += 1
    with pytest.raises(ValueError, match="Resume rejected"):
        run.run_analysis(args)


def test_failure_has_no_success_marker(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    def fail(task):
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(run.protocol, "_evaluation_worker", fail)
    args = arguments(tmp_path, workers=1, smoke=True)
    with pytest.raises(RuntimeError, match="fixture failure"):
        run.run_analysis(args)
    assert (args.output_dir / "failure.json").exists()
    assert not (args.output_dir / "SUCCESS.json").exists()
