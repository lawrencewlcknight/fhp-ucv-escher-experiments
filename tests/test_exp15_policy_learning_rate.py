import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.fhp.exp4_fhp_average_policy_audit.data import group_replay, split_groups
from experiments.fhp.exp13_fhp_policy_capacity.data import group_digest
from experiments.fhp.exp13_fhp_policy_capacity.fitting import fit_path, write_json, state_hash
from experiments.fhp.exp15_fhp_policy_learning_rate import run, aggregate as aggregation
from experiments.fhp.exp15_fhp_policy_learning_rate.config import contract, arm_name
from experiments.fhp.exp15_fhp_policy_learning_rate.data import load_source, source_row, fetch_source
from experiments.fhp.exp15_fhp_policy_learning_rate.evaluation import build_tasks, policy_names
from fhp_escher.checkpointing import LoadedFHPPolicy, sha256_file
from fhp_escher.game import load_fhp_game


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    return run.synthetic_source(tmp_path / "source")


def test_only_policy_rate_changes_from_frozen_source_configuration():
    from experiments.fhp.exp9_fhp_cached_parallel_24h.config import EXPERIMENT_CONFIG as base
    cfg = contract()
    assert cfg["source_experiment"] == "exp9_fhp_cached_parallel_24h"
    assert cfg["seeds"] == [0, 1, 2] and cfg["replicates"] == [0, 1]
    assert cfg["updates"][-1] == base["average_policy_train_steps"] == 20000
    assert cfg["batch_size"] == base["ave_policy_batch_size"] == 2048
    assert cfg["gamma"] == base["gamma"]
    assert cfg["architectures"]["standard"]["network_layers"] == list(base["average_policy_network_layers"])
    assert cfg["architectures"]["standard"]["branch_width"] == base["structured_branch_width"]
    a, b = cfg["recipes"].values()
    assert {k: v for k, v in a.items() if k != "learning_rate"} == {k: v for k, v in b.items() if k != "learning_rate"}
    assert (a["learning_rate"], b["learning_rate"]) == (.003, .0003)
    assert cfg["lbr_deals"] == 0 and contract(include_lbr=True)["lbr_deals"] == 1000
    assert cfg["training_state_retention"] == "none" and not cfg["exact_exploitability"]


def test_source_validation_and_read_only_loading(source):
    before = sha256_file(source / "state.pt"), sha256_file(source / "policy.pkl")
    with pytest.raises(ValueError, match="Synthetic"):
        load_source(source, 0)
    buffer, iteration, template, provenance = load_source(source, 0, smoke=True)
    assert iteration == 10 and buffer["size"] > 0 and template["experiment_id"] == 9
    assert provenance["source_state_sha256"] == before[0]
    assert before == (sha256_file(source / "state.pt"), sha256_file(source / "policy.pkl"))
    with pytest.raises(ValueError, match="seed"):
        load_source(source, 1, smoke=True)
    manifest = json.loads((source / "run_manifest.json").read_text())
    manifest["experiment_id"] = 2
    write_json(source / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="Experiment 9"):
        source_row(source)


def test_reject_corrupt_input_and_missing_replay(source):
    rows = json.loads((source / "checkpoint_manifest.json").read_text())
    rows[0]["sha256"] = "corrupt"
    write_json(source / "checkpoint_manifest.json", rows)
    with pytest.raises(ValueError, match="checksum"):
        load_source(source, 0, smoke=True)
    rows[0].pop("training_state_path")
    write_json(source / "checkpoint_manifest.json", rows)
    with pytest.raises(ValueError, match="retained full replay"):
        source_row(source)


def test_real_source_contract_and_million_row_requirement(source):
    import pickle
    from experiments.fhp.exp9_fhp_cached_parallel_24h.config import EXPERIMENT_CONFIG
    with (source / "policy.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    payload.pop("audit_synthetic_fixture")
    with (source / "policy.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    state = torch.load(source / "state.pt", weights_only=False)
    state.update(config=dict(EXPERIMENT_CONFIG), repository_commit="source-commit")
    state["solver"]["training_elapsed_seconds"] = 86401
    torch.save(state, source / "state.pt")
    manifest = json.loads((source / "run_manifest.json").read_text())
    manifest.update(training_config=dict(EXPERIMENT_CONFIG), repository_commit="source-commit")
    write_json(source / "run_manifest.json", manifest)
    rows = json.loads((source / "checkpoint_manifest.json").read_text())
    rows[0].update(sha256=sha256_file(source / "policy.pkl"), training_state_sha256=sha256_file(source / "state.pt"))
    write_json(source / "checkpoint_manifest.json", rows)
    for smoke in (False, True):
        with pytest.raises(ValueError, match="1,000,000"):
            load_source(source, 0, smoke=smoke)
    manifest["training_config"]["average_policy_loss"] = "mse"
    write_json(source / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="average_policy_loss"):
        load_source(source, 0)


def test_fetch_only_final_state_and_policy(source, monkeypatch):
    calls = []
    monkeypatch.setattr("experiments.fhp.exp15_fhp_policy_learning_rate.data.subprocess.run", lambda args, **kw: calls.append(args))
    fetch_source("gs://test", "exp9-source", 0, source)
    assert len(calls) == 5
    assert all("task_000_cached_parallel_structured_ucv_escher_seed_0" in args[3] for args in calls)
    assert all("exp2" not in args[3] for args in calls)
    assert sum(args[3].endswith("state.pt") for args in calls) == 1
    rows = json.loads((source / "checkpoint_manifest.json").read_text())
    rows[0]["training_state_path"] = "../escape.pt"
    write_json(source / "checkpoint_manifest.json", rows)
    with pytest.raises(ValueError, match="escapes"):
        fetch_source("gs://test", "exp9-source", 0, source)


def test_matched_sampling_initialisation_split_reload_and_resumption(source, tmp_path):
    buffer, iteration, template, _ = load_source(source, 0, smoke=True)
    groups = group_replay(buffer, iteration)
    cfg = contract(True)
    train, heldout = split_groups(groups, .1, cfg["split_seed"])
    assert not {r.tobytes() for r in train.features} & {r.tobytes() for r in heldout.features}
    assert train.rows + heldout.rows == groups.rows
    fits, paths = [], []
    for recipe in cfg["recipes"]:
        path = tmp_path / recipe
        kwargs = dict(config=cfg, seed=0, replicate=0, architecture="standard", recipe=recipe,
                      updates=cfg["updates"], directory=path, template=template,
                      data_sha256=group_digest(train), phase="diagnostic")
        result = fit_path(train, heldout, **kwargs)
        assert fit_path(train, heldout, **kwargs) == result
        fits.append(result)
        paths.append(path / "4.pkl")
        assert not list(path.rglob("*.pt"))
    for a, b in zip(*fits, strict=True):
        assert a["initial_state_sha256"] == b["initial_state_sha256"]
        assert a["sample_sequence_sha256"] == b["sample_sequence_sha256"]
        assert a["processed_examples"] == b["processed_examples"]
        assert a["validation_data_sha256"] == b["validation_data_sha256"]
    assert state_hash(LoadedFHPPolicy(load_fhp_game(), paths[0]).model) != state_hash(LoadedFHPPolicy(load_fhp_game(), paths[1]).model)


@pytest.mark.parametrize("include_lbr", [False, True])
def test_evaluation_tasks_match_source_replicate_common_deals_and_optional_lbr(source, include_lbr):
    cfg = contract(True, include_lbr)
    policies = {name: source / "policy.pkl" for name in policy_names(cfg)}
    tasks = build_tasks(policies, cfg, 0)
    assert len([t for t in tasks if t["kind"] == "rule"]) == 25
    assert len([t for t in tasks if t["kind"] == "direct_crossplay"]) == 6
    assert len([t for t in tasks if t["kind"] == "lbr"]) == (5 if include_lbr else 0)
    paired = [t for t in tasks if t.get("opponent") == "matched_003"]
    assert len(paired) == 2
    for rep, task in enumerate(paired):
        assert task["policy_a_name"] == arm_name("adam_0003", rep)
        assert task["policy_b_name"] == arm_name("adam_003", rep)
    assert len({t["evaluation_seed"] for t in tasks if t["kind"] == "direct_crossplay"}) == 1


def test_three_source_pipeline_and_reject_incomplete_or_unmatched_results(tmp_path, monkeypatch):
    cfg = contract(True)
    cfg["seeds"] = [0, 1, 2]
    monkeypatch.setattr(run, "contract", lambda *args: cfg)
    monkeypatch.setattr(aggregation, "contract", lambda *args: cfg)
    monkeypatch.setattr(aggregation, "plot", lambda *args: None)
    def fake_evaluate(policies, config, seed, directory, workers, sync):
        rows = [{**task, "arm": task["policy_a_name"], "metric": f"{task['kind']}_{task['opponent']}",
                 "num_deal_pairs": task["num_deals"], "num_games": 2 * task["num_deals"],
                 "mean_mbb_per_hand": seed + int(task["policy_a_name"].startswith("adam_0003"))}
                for task in build_tasks(policies, config, seed)]
        write_json(directory.parent / "evaluation_summary.json", rows)
    monkeypatch.setattr(run, "evaluate", fake_evaluate)
    root = tmp_path / "outputs"
    for seed in cfg["seeds"]:
        source = run.synthetic_source(tmp_path / f"source{seed}", seed)
        run.run_worker(source, root, seed, smoke=True, evaluation_workers=1)
    assert not list(root.rglob("*.pt"))
    output = aggregation.aggregate(root, smoke=True)
    result = json.loads((output / "summary.json").read_text())
    assert all(r["n_source_seeds"] == 3 for r in result["summaries"])
    contrast = next(r for r in result["paired_contrasts"] if r["metric"] == "rule_suite_mean" and r["contrast"] == "adam_0003_minus_adam_003")
    assert contrast["mean"] == 1 and contrast["exact_two_sided_sign_flip_p"] == .25
    path = root / "workers/seed_2/evaluation_summary.json"
    rows = json.loads(path.read_text())
    write_json(path, rows[:-1])
    with pytest.raises(ValueError, match="Incomplete gameplay"):
        aggregation.aggregate(root, smoke=True)
    write_json(path, rows)
    path = root / "workers/seed_2/fit_metrics.json"
    rows = json.loads(path.read_text())
    rows[-1]["sample_sequence_sha256"] = "changed"
    write_json(path, rows)
    with pytest.raises(ValueError, match="identity mismatch|unmatched"):
        aggregation.aggregate(root, smoke=True)


def test_source_cannot_live_in_output_tree(source):
    with pytest.raises(ValueError, match="outside"):
        run.run_worker(source, source / "outputs", 0, smoke=True)


@pytest.mark.parametrize("kind", ["controller", "smoke", "train", "aggregate"])
@pytest.mark.parametrize("lbr", [0, 1])
def test_cloud_stage_routing_runtime_and_retention(tmp_path, kind, lbr):
    root = Path(__file__).parents[1]
    path = tmp_path / "job.json"
    subprocess.run([sys.executable, str(root / "gcp/exp15_policy_learning_rate_batch.py"), "--kind", kind,
                    "--output", str(path), "--run-id", "exp15-test", "--bucket-root", "gs://test",
                    "--service-account", "test@example.com", "--repo-ref", "abc", "--source-run-id", "exp9-source",
                    "--project-id", "test", "--region", "europe-west1", "--include-lbr", str(lbr)], check=True)
    job = json.loads(path.read_text())
    group = job["taskGroups"][0]
    text = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert group["taskCountPerNode"] == 1
    assert "EXP2_RUN_ID" not in text and "EXP4_REMOTE" not in text
    assert "exp4_fhp_average_policy_audit.run" not in text
    if kind != "controller":
        assert "uv python install 3.11.16" in text
        assert "uv venv --python 3.11.16 " in text
        assert ("--include-lbr" in text) == bool(lbr)
    else:
        assert f"export EXP15_INCLUDE_LBR={lbr}" in text
    if kind in ("smoke", "train"):
        assert "EXP9_RUN_ID" in text and "EXP15_REMOTE_WORKER" in text
        assert "training_states" in text and r"\.pt$" in text
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    script = tmp_path / "job.sh"
    script.write_text(text)
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_launcher_order_and_syntax():
    path = Path(__file__).parents[1] / "gcp/run_exp15_policy_learning_rate.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    commands = ['ensure "$SMOKE_JOB" smoke', 'ensure "$TRAIN_JOB" train', 'ensure "$AGGREGATE_JOB" aggregate']
    assert [text.index(c) for c in commands] == sorted(text.index(c) for c in commands)
    assert "reset REPO_REF" in text and "cached_parallel_structured_ucv_escher" in text
