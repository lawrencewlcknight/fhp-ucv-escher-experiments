"""Continuation provenance, fixed-opponent protocol, cloud wiring and reporting."""
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

from experiments.fhp.exp16_fhp_hand_board_48h import contract as c, evaluate as run
from experiments.fhp.exp8_fhp_parallel_48h import config as exp8
from experiments.fhp.exp9_fhp_cached_parallel_24h import config as exp9
from gcp.exp16_hand_board_48h_batch import build_job, REPO_URL


def arguments(tmp_path, **overrides):
    args = run.parser().parse_args([item for key in (*run.LABELS, "source10")
                                    for item in ("--" + key + "-run", str(tmp_path / key))]
                                  + ["--output-dir", str(tmp_path / "analysis")])
    return Namespace(**(vars(args) | overrides))


def records():
    return [{"experiment": exp, "seed": seed, "training_hours": hour,
             "checkpoint_path": f"/{exp}/{seed}/{hour}.pkl", "checkpoint_sha256": f"{exp}-{seed}-{hour}",
             "nodes_touched": hour * 100}
            for exp in run.LABELS for seed in c.SEEDS
            for hour in (c.HOURS if exp in ("exp16", "exp8") else (6, 12, 18, 24))]


def source_fixture(tmp_path, monkeypatch):
    payloads = {}
    roots = {key: tmp_path / key for key in (*run.LABELS, "source10")}
    for exp, config, hours in (("source10", run.learner, 24), ("exp16", run.learner, 48),
                               ("exp9", exp9, 24), ("exp8", exp8, 48)):
        for seed in c.SEEDS:
            worker = roots[exp] / "workers" / c.worker_name(seed)
            (worker / "checkpoints").mkdir(parents=True)
            encoder = run.long_eval.make_feature_encoder(config.EXPERIMENT_CONFIG.get(
                "feature_encoder_id", run.long_eval.ENCODER_ID))
            manifest = dict(experiment_id=config.EXPERIMENT_ID, experiment_name=config.EXPERIMENT_NAME,
                            algorithm_id=config.ALGORITHM_ID, seed=seed, smoke=False,
                            training_config=deepcopy(config.EXPERIMENT_CONFIG),
                            training_config_sha256=c.digest(config.EXPERIMENT_CONFIG),
                            repository_commit=c.TRAINING_REF, execution_backend="ray_parallel",
                            reference_vm=config.REFERENCE_VM, game={"parameters": dict(run.protocol.FHP_GAME_PARAMETERS)},
                            training_duration_seconds=hours * 3600,
                            checkpoint_schedule=list(config.checkpoint_schedule(total_hours=hours)))
            runtime = {**c.RUNTIME, "reference_vm": config.REFERENCE_VM,
                       "parallel_settings": config.parallel_settings(), "feature_encoder": encoder.metadata(),
                       "frozen_critic_target_cache": exp != "exp8"}
            rows = []
            for hour in range(6, hours + 1, 6):
                policy = worker / "checkpoints" / f"time_{hour:02d}h.pkl"
                identity = "source10" if exp == "exp16" and hour <= 24 else exp
                policy.write_bytes(f"{identity}-{seed}-{hour}".encode())
                payloads[str(policy)] = dict(experiment_name=config.EXPERIMENT_NAME, algorithm_id=config.ALGORITHM_ID,
                                            seed=seed, training_config=config.EXPERIMENT_CONFIG,
                                            nodes_touched=hour * 100, outer_iteration=hour,
                                            checkpoint_target_seconds=hour * 3600, feature_encoder=encoder.metadata())
                rows.append(dict(checkpoint_id=f"time_{hour:02d}h", path=f"checkpoints/{policy.name}",
                                 sha256=run.sha256_file(policy), checkpoint_target_seconds=hour * 3600,
                                 actual_training_elapsed_seconds=hour * 3600 + 1,
                                 outer_iteration=hour, nodes_touched=hour * 100))
            rows[-1].update(training_state_path=f"training_states/time_{hours}h.pt", training_state_sha256="a" * 64)
            summary = dict(seed=seed, status="complete", checkpoint_count=hours // 6,
                           final_training_elapsed_seconds=hours * 3600 + 1)
            for name, value in (("run_manifest.json", manifest), ("runtime_manifest.json", runtime),
                                 ("summary.json", summary), ("checkpoint_manifest.json", rows)):
                (worker / name).write_text(json.dumps(value))
            (worker / "SUCCESS.json").write_text(json.dumps(dict(status="complete", summary_sha256=run.sha256_file(worker / "summary.json"))))
            if exp == "exp16":
                source = roots["source10"] / "workers" / c.worker_name(seed)
                lineage = dict(seed=seed, source_total_hours=24, total_hours=48, source_commit=c.TRAINING_REF,
                               source_state_path="training_states/time_24h.pt", source_state_sha256="a" * 64,
                               source_summary_sha256=run.sha256_file(source / "summary.json"),
                               source_worker=f"gs://test-bucket/{c.SOURCE_RUN_ID}/workers/{c.worker_name(seed)}")
                (worker / "continuation_source.json").write_text(json.dumps(lineage))
    for module in (run.long_eval, run.feature_eval):
        monkeypatch.setattr(module, "LoadedFHPPolicy", lambda game, path: SimpleNamespace(checkpoint=payloads[str(path)]))
    return roots, payloads


def test_config_and_source_contract():
    assert c.digest(run.learner.EXPERIMENT_CONFIG) == c.CONFIG_SHA256
    assert c.SEEDS == (0, 1, 2)
    assert c.HOURS == (6, 12, 18, 24, 30, 36, 42, 48)
    assert c.TOTAL_HOURS == 48


def test_protocol_task_budgets_and_fixed_opponents(tmp_path):
    args = arguments(tmp_path)
    tasks = run.build_tasks(records(), args)
    assert Counter(t["kind"] for t in tasks) == dict(rule=105, lbr=2100, temporal_crossplay=30, direct_crossplay=30)
    assert len({t["task_id"] for t in tasks}) == 2265
    assert sum(t["num_deals"] for t in tasks) == 4_071_000
    assert len({t["policy_a_path"] for t in tasks if t["kind"] == "rule"}) == 21
    assert {t["training_hours"] for t in tasks if t["experiment"] == "exp16"} == set(c.EVALUATED_HOURS)
    direct = [t for t in tasks if t["kind"] == "direct_crossplay"]
    for opponent, hour in (("exp9", 24), ("exp8", 48)):
        group = [t for t in direct if t["right_experiment"] == opponent]
        assert len(group) == 15
        assert {t["right_hours"] for t in group} == {hour}
        assert len({t["evaluation_seed"] for t in group}) == 1
        assert len({t["policy_b_sha256"] for t in group}) == 3
        assert all(t["num_deals"] == 50_000 for t in group)
    temporal = [t for t in tasks if t["kind"] == "temporal_crossplay"]
    assert all(t["left_hours"] > t["right_hours"] for t in temporal)
    assert {t["right_hours"] for t in temporal if t["left_hours"] == 48} == {24, 30, 36, 42}
    assert len({(t["opponent"], t["evaluation_seed"]) for t in tasks if t["kind"] == "rule"}) == 5


def test_source_identity_and_retention_validation(tmp_path, monkeypatch):
    roots, _ = source_fixture(tmp_path, monkeypatch)
    sources = run.discover_sources(roots)
    assert len(sources) == 60
    assert sum(r["selected_for_evaluation"] for r in sources) == 21


def test_cloud_preflight_reads_only_metadata_and_fails_closed(tmp_path, monkeypatch):
    roots, _ = source_fixture(tmp_path, monkeypatch)
    run_to_root = {c.SOURCE_RUN_ID: roots["source10"], c.EXP9_RUN_ID: roots["exp9"], c.EXP8_RUN_ID: roots["exp8"]}
    calls = []

    def cloud(*args):
        calls.append(args)
        if args[0] == "objects":
            return b'{"size": 1234, "generation": "1"}'
        uri = args[1].removeprefix("gs://test-bucket/")
        run_id, relative = uri.split("/", 1)
        root = run_to_root[run_id]
        if args[0] == "ls":
            return "\n".join(f"gs://test-bucket/{run_id}/{p.relative_to(root)}" for p in sorted(root.glob(relative))).encode()
        assert args[0] == "cat" and relative.endswith(".json")
        return (root / relative).read_bytes()

    result = c.cloud_preflight("gs://test-bucket", cloud=cloud)
    assert len(result["source_states"]) == 3
    assert len(result["fixed_panel_policies"]) == 6
    assert len([a for a in calls if a[0] == "objects"]) == 9
    path = roots["source10"] / "workers" / c.worker_name(1) / "runtime_manifest.json"
    runtime = json.loads(path.read_text())
    runtime["python_version"] = "3.11.17"
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="runtime mismatch"):
        c.cloud_preflight("gs://test-bucket", cloud=cloud)


def test_real_policy_loader_and_scoring_across_both_feature_layouts(tmp_path):
    """Small real OpenSpiel score checks; no invented source learning results."""
    import torch
    from fhp_escher.checkpointing import save_policy_checkpoint
    from fhp_escher.features import FHPFeatureEncoder, StructuredFHPMLP
    from fhp_escher.hand_board_features import FHPHandBoardFeatureEncoder
    torch.set_num_threads(1)
    policy_rows = []
    for exp, encoder in (("exp16", FHPHandBoardFeatureEncoder()), ("exp9", FHPFeatureEncoder())):
        size = encoder.policy_layout.total_size
        model = StructuredFHPMLP(encoder.policy_layout, [8], 3, branch_width=8)
        solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=model), feature_encoder=encoder,
                                 num_iteration=1, episode=1, nodes_touched=1, infostate_size=size,
                                 action_size=3, average_policy_network_layers=[8], network_layers=[8])
        path = save_policy_checkpoint(solver, tmp_path / (exp + ".pkl"), seed=0, config={}, checkpoint_row={})
        policy_rows.append(dict(experiment=exp, seed=0, training_hours=24, checkpoint_path=str(path),
                                checkpoint_sha256=run.sha256_file(path), nodes_touched=1))
    identity = dict(task_id="real_test", training_seed=0, training_hours=24, experiment="exp16",
                    num_deals=2, evaluation_seed=123, **run.long_eval._policy_fields(policy_rows[0], "a"))
    tasks = [dict(identity, kind="rule", opponent=run.protocol.PUBLISHED_AGENT_NAMES[0]),
             dict(identity, kind="direct_crossplay", **run.long_eval._policy_fields(policy_rows[1], "b")),
             dict(identity, kind="lbr", lbr_rollouts=2, lbr_seed=12, shard_index=0)]
    for task in tasks:
        result = run.protocol._evaluation_worker(task)
        assert result["num_deal_pairs"] == 2
        assert np.isfinite(result["mean_mbb_per_hand"])


@pytest.mark.parametrize("change", ["state", "summary", "commit", "python", "hours", "seed", "source_run", "policy", "intermediate_state"])
def test_reject_bad_continuation(tmp_path, monkeypatch, change):
    roots, _ = source_fixture(tmp_path, monkeypatch)
    worker = roots["exp16"] / "workers" / c.worker_name(0)
    name = {"commit": "run_manifest.json", "python": "runtime_manifest.json",
            "policy": "checkpoint_manifest.json", "intermediate_state": "checkpoint_manifest.json"}.get(change, "continuation_source.json")
    path = worker / name
    value = json.loads(path.read_text())
    if change == "state": value["source_state_sha256"] = "b" * 64
    elif change == "summary": value["source_summary_sha256"] = "b" * 64
    elif change == "commit": value["repository_commit"] = "b" * 40
    elif change == "python": value["python_version"] = "3.11.17"
    elif change == "hours": value["source_total_hours"] = 30
    elif change == "seed": value["seed"] = 1
    elif change == "source_run": value["source_worker"] = "gs://test-bucket/other/workers/task_0"
    elif change == "policy":
        policy = worker / value[0]["path"]
        policy.write_bytes(b"changed source policy")
        value[0]["sha256"] = run.sha256_file(policy)
    elif change == "intermediate_state": value[4]["training_state_path"] = "training_states/30h.pt"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run.discover_sources(roots)


def synthetic_score(task):
    # Toy scores: test infrastructure only, never evidence of poker performance.
    value = task["training_hours"] / 6 + task["training_seed"]
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
    assert len(list(output.glob("*.png"))) == 6
    report = (output / "analysis_summary.md").read_text()
    assert "nan" not in report
    with (output / "external_improvement_aggregate.csv").open() as f:
        changes = list(csv.DictReader(f))
    assert len(changes) == 40
    for row in changes:
        expected = (int(row["later_hours"]) - int(row["earlier_hours"])) / 6 * run.protocol.MILLI_BIG_BLINDS_PER_CHIP
        assert float(row["mean_mbb_per_hand"]) == pytest.approx(expected)
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


def batch_args(**updates):
    return Namespace(**(dict(kind="train", run_id="exp16-test", repo_ref="a" * 40, repo_url=REPO_URL,
                             bucket_root="gs://test-bucket", service_account="runner@example.com",
                             parallelism=3, project_id="test-project", region="europe-west1",
                             max_hours=48, resume=False) | updates))


@pytest.mark.parametrize("kind", ["controller", "smoke", "train", "aggregate", "evaluate"])
def test_cloud_stages_are_pinned_and_isolated(kind):
    job = build_job(batch_args(kind=kind))
    group = job["taskGroups"][0]
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRetryCount"] == 0
    text = group["taskSpec"]["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    if kind in ("smoke", "train", "aggregate"):
        assert f"REPO_REF={c.TRAINING_REF}" in text
        assert "uv python install 3.11.16" in text
        assert "PINNED_RUNTIME" in text
        assert "exp16_fhp_hand_board_48h" not in text  # Not present at the old learner commit.
    if kind == "train":
        assert group["taskSpec"]["environment"]["variables"]["EXP10_TOTAL_HOURS"] == "48"
        assert group["taskSpec"]["environment"]["variables"]["EXP10_SOURCE_RUN_ID"] == c.SOURCE_RUN_ID
        assert "stage-source" in text and "--total-hours" in text
        assert "continuation_inputs" in text
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    if kind == "controller":
        assert "gcp/run_exp16_hand_board_48h.sh" in text and "EXP16_REMOTE_CONTROLLER=1" in text
        assert f"REPO_REF={'a' * 40}" in text
    if kind == "evaluate":
        assert "--exp16-run" in text and "--source10-run" in text and "--exp9-run" in text
        assert '"$BUCKET_ROOT/$RUN_ID/evaluation"' in text
        assert "--lbr-rollouts 4096" in text and "--crossplay-deals 50000" in text
        assert "--workers 16" in text
        assert text.count("gcloud storage rsync --recursive ") >= 4
        assert "exp10_fhp_hand_board_features.run worker" not in text
        assert text.index('--output-dir "$OUTPUT_ROOT/smoke"') < text.index('--output-dir "$OUTPUT_ROOT/analysis"')


@pytest.mark.parametrize("updates", [{"repo_ref": "main"}, {"run_id": c.SOURCE_RUN_ID},
                                     {"run_id": "bad/name"}, {"bucket_root": "gs://bucket/path"},
                                     {"parallelism": 4}, {"max_hours": 0}])
def test_reject_unsafe_jobs(updates):
    with pytest.raises(ValueError):
        build_job(batch_args(**updates))


def test_launcher_contract():
    path = Path(__file__).resolve().parents[1] / "gcp/run_exp16_hand_board_48h.sh"
    text = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "for stage in smoke train aggregate evaluate; do ensure_stage" in text
    assert "evaluate-resume" in text and "build evaluate --resume" in text
    assert "No training will be restarted automatically" in text
    assert "source_contract.json" in text
