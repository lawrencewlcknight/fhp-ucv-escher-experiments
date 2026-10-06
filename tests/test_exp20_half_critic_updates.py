"""24-hour contract, cloud wiring, and completed-endpoint extension checks."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.fhp.exp10_fhp_hand_board_features import config as baseline
from fhp_escher.hand_board_features import ENCODER_ID, FHPHandBoardFeatureEncoder
from experiments.fhp.exp20_fhp_half_critic_updates import config as exp20
from experiments.fhp.exp20_fhp_half_critic_updates.aggregate import task_name
from experiments.fhp.exp20_fhp_half_critic_updates.run import main
from experiments.fhp.exp20_fhp_half_critic_updates.continuation import stage_source
from experiments.fhp.exp20_fhp_half_critic_updates.training_state import STATE_TYPE, read_training_state
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _config_sha256
from fhp_escher.checkpointing import sha256_file
from gcp import exp20_half_critic_updates_batch as builder
from gcp import exp10_hand_board_features_batch as previous_builder


def test_only_critic_budget_changes_the_learning_configuration():
    expected = dict(baseline.EXPERIMENT_CONFIG, baseline_network_train_steps=5000)
    assert exp20.EXPERIMENT_CONFIG == expected
    assert exp20.EXPERIMENT_CONFIG is not baseline.EXPERIMENT_CONFIG
    assert exp20.parallel_settings() == baseline.parallel_settings()
    assert exp20.REFERENCE_VM == baseline.REFERENCE_VM
    assert exp20.PRODUCTION_SEEDS == (0, 1, 2)
    assert exp20.CHECKPOINT_HOURS == (6, 12, 18, 24)
    assert exp20.checkpoint_schedule() == baseline.checkpoint_schedule()
    assert exp20.checkpoint_schedule()[-1]["target_training_seconds"] == 24 * 3600
    assert len(exp20.checkpoint_schedule(smoke=True)) == 4
    assert len(exp20.checkpoint_schedule(total_hours=48)) == 8
    assert exp20.contract_manifest()["training_hours_per_seed"] == 24
    assert exp20.contract_manifest()["training_state_retention"] == "final"
    assert exp20.contract_manifest()["frozen_critic_target_cache"]
    assert baseline.contract_manifest()["frozen_critic_target_cache"] is True
    assert baseline.EXPERIMENT_CONFIG["baseline_network_train_steps"] == 10_000
    assert exp20.contract_manifest()["baseline"] == baseline.EXPERIMENT_NAME
    representation = exp20.contract_manifest()["representation"]
    assert representation["encoder_id"] == ENCODER_ID
    assert representation["policy_feature_size"] == 213
    assert representation["critic_feature_size"] == 323
    assert baseline.contract_manifest()["representation"]["policy_feature_size"] == 213
    exp20.validate_contract(seeds=(0, 1, 2), config=exp20.EXPERIMENT_CONFIG,
                          schedule=exp20.checkpoint_schedule(), smoke=False)
    with pytest.raises(ValueError, match="must equal"):
        exp20.validate_contract(seeds=(0, 1, 2), config=dict(exp20.EXPERIMENT_CONFIG, num_traversals=80_000),
                              schedule=exp20.checkpoint_schedule(), smoke=False)
    for bad in (18, 25, 24.5, True):
        with pytest.raises(ValueError, match="multiple"):
            exp20.checkpoint_schedule(total_hours=bad)


@pytest.mark.parametrize("actual_steps", [[5000, 5000], [5000, 10000]])
def test_production_solver_forwards_and_checks_both_critic_budgets(monkeypatch, actual_steps):
    from experiments.fhp.exp20_fhp_half_critic_updates import worker
    captured, closed = {}, []
    solver = SimpleNamespace(q_value_trainer=SimpleNamespace(
        members=[SimpleNamespace(train_steps=n) for n in actual_steps]),
        close=lambda: closed.append(True))
    def factory(**kwargs):
        captured.update(kwargs)
        return solver
    monkeypatch.setattr(worker, "ParallelStructuredGroupedUCVEscher", factory)
    monkeypatch.setattr(worker, "install_cache", lambda *args, **kwargs: None)
    if actual_steps != [5000, 5000]:
        with pytest.raises(ValueError, match="Both critics"):
            worker._make_solver(0, exp20.EXPERIMENT_CONFIG)
        assert closed == [True]
    else:
        assert worker._make_solver(0, exp20.EXPERIMENT_CONFIG) is solver
    assert captured["baseline_network_train_steps"] == 5000
    assert captured["parallel_num_workers"] == 8


def job_args(kind, **kwargs):
    return SimpleNamespace(kind=kind, repo_url=builder.REPO_URL, repo_ref="a" * 40,
                           bucket_root="gs://test-bucket", run_id="exp20-features-test",
                           service_account="runner@example.com", parallelism=3,
                           project_id="test-project", region="europe-west1",
                           controller_action="orchestrate", max_hours=48, resume=False, **kwargs)


@pytest.mark.parametrize("kind", ["controller", "smoke", "train", "aggregate"])
@pytest.mark.parametrize("extension", [False, True])
def test_cloud_jobs(kind, extension, tmp_path):
    args = job_args(kind, total_hours=48 if extension else 24,
                    source_run_id="exp20-original" if extension else "")
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert group["taskCountPerNode"] == 1
    env = spec["environment"]["variables"]
    assert env["EXP20_TOTAL_HOURS"] == ("48" if extension else "24")
    assert env["EXP20_SOURCE_RUN_ID"] == ("exp20-original" if extension else "")
    if kind in ("smoke", "train"):
        assert spec["computeResource"] == {"cpuMilli": 16000, "memoryMib": 62000}
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    if kind == "train":
        assert spec["maxRunDuration"] == ("259200s" if extension else "129600s")
        assert spec["maxRetryCount"] == 0
    script = spec["runnables"][0]["script"]["text"]
    assert "exp9_fhp_cached" not in script and "EXP9_" not in script
    if kind == "train":
        assert " stage-source " in script
        assert '--total-hours "$EXP20_TOTAL_HOURS"' in script
    if kind == "smoke":
        assert "capacity-preflight" in script and "checkpoint-smoke" not in script
    path = tmp_path / (kind + ".sh")
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)
    old = previous_builder.build_job(job_args("train"))
    assert old["taskGroups"][0]["taskSpec"]["maxRunDuration"] == "129600s"


def test_cloud_extension_requires_distinct_source():
    for source, hours in (("", 48), ("exp20-features-test", 48), ("exp20-source", 24)):
        with pytest.raises(ValueError):
            builder.build_job(job_args("train", total_hours=hours, source_run_id=source))


def test_launcher_is_isolated_and_shell_valid():
    path = Path(__file__).resolve().parents[1] / "gcp/run_exp20_half_critic_updates.sh"
    source = path.read_text()
    assert "EXP8_" not in source and "exp8_" not in source
    assert 'EXP20_TOTAL_HOURS="${EXP20_TOTAL_HOURS:-24}"' in source
    assert 'for stage in smoke train aggregate; do ensure_stage "$stage"; done' in source
    assert 'for stage in smoke train aggregate evaluate' not in source
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_evaluation_job_is_separate_and_frozen():
    args = job_args("evaluate", total_hours=24, source_run_id="")
    job = builder.build_job(args)
    spec = job["taskGroups"][0]["taskSpec"]
    text = spec["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "172800s"
    assert builder.EVALUATION_MODULE in text
    assert builder.MODULE + " worker" not in text
    assert "--exp10-run" in text and "--exp20-run" in text
    assert builder.BASELINE_RUN_ID in text
    assert '"$BUCKET_ROOT/$RUN_ID/evaluation"' in text
    assert "--lbr-rollouts 4096" in text and "--crossplay-deals 50000" in text
    assert text.index('--output-dir "$OUTPUT_ROOT/smoke"') < text.index('--output-dir "$OUTPUT_ROOT/analysis"')
    assert "training_states" in text and "continuation_inputs" in text


@pytest.mark.parametrize("updates", [{"repo_ref": "main"}, {"run_id": builder.BASELINE_RUN_ID},
    {"run_id": "bad/name"}, {"bucket_root": "gs://bucket/path"}, {"parallelism": 4},
    {"max_hours": 0}, {"total_hours": 48, "source_run_id": builder.BASELINE_RUN_ID}])
def test_reject_unsafe_jobs(updates):
    args = job_args("train", total_hours=24, source_run_id="")
    args.__dict__.update(updates)
    with pytest.raises(ValueError):
        builder.build_job(args)


@pytest.mark.parametrize("parallelism", [1, 2, 3])
def test_controller_budget_covers_seed_waves(parallelism):
    args = job_args("controller", total_hours=24, source_run_id="")
    args.parallelism = parallelism
    job = builder.build_job(args)
    spec = job["taskGroups"][0]["taskSpec"]
    assert spec["maxRetryCount"] == 0
    assert int(spec["maxRunDuration"][:-1]) >= (36 * ((3 + parallelism - 1) // parallelism) + 12) * 3600
    assert "EXP20_REMOTE_CONTROLLER=1" in spec["runnables"][0]["script"]["text"]


@pytest.mark.parametrize("action", ["dry-run", "dry-run-evaluate"])
def test_dry_run_without_cloud_dependencies(tmp_path, action):
    import os
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PROJECT_ID="test-project", REGION="europe-west1", BUCKET="test-bucket",
               SA_EMAIL="runner@example.com", REPO_REF="a" * 40, RUN_ID="exp20-local-dry-run",
               EXP20_TOTAL_HOURS="24", EXP20_SOURCE_RUN_ID="")
    result = subprocess.run(["bash", str(repo / "gcp/run_exp20_half_critic_updates.sh"), action],
                            env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "no cloud actions" in result.stdout


def test_restore_rejects_wrong_state_type_or_cache_mode():
    from experiments.fhp.exp20_fhp_half_critic_updates.training_state import restore_training_state
    solver = SimpleNamespace(q_value_trainer=SimpleNamespace(
        members=[SimpleNamespace(cache_frozen_targets=True)] * 2))
    with pytest.raises(ValueError, match="schema"):
        restore_training_state(solver, {"type": "exp8_fhp_parallel_48h_full_training_state"})
    payload = {"type": STATE_TYPE, "critic_cache": {"contract": {
        "member_flags": [False, False], "lifetime": "this_fit_only"}}}
    with pytest.raises(ValueError, match="critic-cache contract"):
        restore_training_state(solver, payload)


def fake_source(folder):
    folder.mkdir()
    schedule = exp20.checkpoint_schedule()
    rows = []
    for point in schedule:
        path = folder / "checkpoints" / (point["checkpoint_id"] + ".pkl")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(point["checkpoint_id"].encode())
        rows.append({"checkpoint_id": point["checkpoint_id"],
                     "path": str(path.relative_to(folder)), "sha256": sha256_file(path)})
    state = folder / "training_states" / "final.pt"
    state.parent.mkdir()
    state.write_bytes(b"synthetic-state")
    rows[-1].update(training_state_path="training_states/final.pt",
                    training_state_sha256=sha256_file(state))
    manifest = dict(experiment_id=20, experiment_name=exp20.EXPERIMENT_NAME,
                    algorithm_id=exp20.ALGORITHM_ID, seed=0, smoke=False,
                    execution_backend="ray_parallel", reference_vm=exp20.REFERENCE_VM,
                    training_config_sha256=_config_sha256(exp20.EXPERIMENT_CONFIG),
                    repository_commit="test", checkpoint_schedule=list(schedule))
    summary = dict(seed=0, status="complete", checkpoint_count=4,
                   final_training_elapsed_seconds=24 * 3600 + 12)
    for name, data in (("run_manifest.json", manifest), ("summary.json", summary),
                       ("checkpoint_manifest.json", rows), ("runtime_manifest.json", {})):
        (folder / name).write_text(json.dumps(data))
    (folder / "SUCCESS.json").write_text(json.dumps({
        "status": "complete", "summary_sha256": sha256_file(folder / "summary.json")}))
    return state


def test_import_preserves_source_and_checks_integrity(tmp_path):
    source = tmp_path / "source"
    state = fake_source(source)
    original_hash = sha256_file(state)
    destination = tmp_path / "extended"
    result = stage_source(str(source), destination, seed=0, total_hours=48, source_commit="test")
    assert result["source_total_hours"] == 24
    assert result["total_hours"] == 48
    assert sha256_file(state) == original_hash
    assert sha256_file(destination / "continuation_inputs/source_state.pt") == original_hash
    assert not (destination / "training_states").exists()
    assert not (destination / "SUCCESS.json").exists()
    assert not (destination / "run_manifest.json").exists()
    assert len(list((destination / "checkpoints").iterdir())) == 4
    assert stage_source(str(source), destination, seed=0, total_hours=48, source_commit="test") == result
    with pytest.raises(ValueError, match="another continuation"):
        stage_source(str(source), destination, seed=0, total_hours=78, source_commit="test")
    with pytest.raises(ValueError, match="compatible"):
        stage_source(str(source), tmp_path / "bad-seed", seed=1, total_hours=48, source_commit="test")
    with pytest.raises(ValueError, match="add 6 to 48"):
        stage_source(str(source), tmp_path / "too-long", seed=0, total_hours=78, source_commit="test")
    state.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        stage_source(str(source), tmp_path / "bad-state", seed=0, total_hours=48, source_commit="test")
    assert not (tmp_path / "bad-state").exists()


@pytest.mark.ray
def test_completed_endpoint_can_extend_without_retraining(tmp_path, noisy_ray_startup):
    main(["smoke", "--output-root", str(tmp_path), "--no-resume"])
    worker = tmp_path / "workers" / task_name(0, 0)
    extended = tmp_path / "extension_smoke" / "workers" / task_name(0, 0)
    initial = json.loads((worker / "summary.json").read_text())
    final = json.loads((extended / "summary.json").read_text())
    assert initial["checkpoint_count"] == 4 and initial["final_iteration"] == 1
    assert final["checkpoint_count"] == 8 and final["final_iteration"] == 2
    assert final["final_episode"] == 64
    assert final["resumed_from_training_state"]
    initial_rows = json.loads((worker / "checkpoint_manifest.json").read_text())
    final_rows = json.loads((extended / "checkpoint_manifest.json").read_text())
    assert all("training_state_path" not in r for r in initial_rows[:-1])
    assert all("training_state_path" not in r for r in final_rows[:-1])
    assert len(list(worker.rglob("*.pt"))) == 1
    assert len(list(extended.rglob("*.pt"))) == 1
    assert not (extended / "continuation_inputs").exists()
    assert not list((tmp_path / "restart_validation").rglob("*.pt"))
    assert [row["sha256"] for row in initial_rows] == [row["sha256"] for row in final_rows[:4]]
    payload = read_training_state(worker / initial_rows[-1]["training_state_path"])
    assert payload["type"] == STATE_TYPE and len(payload["parallel"]["actors"]) == 8
    assert payload["feature_encoder"] == FHPHandBoardFeatureEncoder().metadata()
    assert (tmp_path / "analysis/SUCCESS.json").is_file()
    assert (tmp_path / "extension_smoke/analysis/SUCCESS.json").is_file()
    assert len(json.loads((tmp_path / "analysis/throughput_summary.json").read_text())["checkpoint_means"]) == 4
    cache = json.loads((tmp_path / "cache_validation/cache_validation.json").read_text())
    assert cache["learning_state_bitwise_identical"]
    assert cache["initial_learning_state_bitwise_identical"]
    assert len(set(noisy_ray_startup)) >= 2
    assert payload["critic_cache"]["contract"]["member_flags"] == [True, True]
    assert final["execution_diagnostics"]["cumulative_cached_critic_fit_calls"] == 4
    assert not list((tmp_path / "cache_validation").rglob("*.pt"))
    assert (tmp_path / "analysis/feature_specification.json").is_file()
    # The larger input layout also round-trips through real inference/evaluation.
    from fhp_escher.checkpointing import LoadedFHPPolicy
    from fhp_escher.game import load_fhp_game
    from fhp_evaluation.duplicate import evaluate_duplicate_match
    game = load_fhp_game()
    policy = LoadedFHPPolicy(game, worker / initial_rows[-1]["path"])
    assert policy.model.input_size == 213
    result = evaluate_duplicate_match(game, policy, policy, num_deals=2, seed=901)
    assert result.num_games == 4
