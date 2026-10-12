"""Frozen historical sources, inference units, and interrupted-evaluation recovery."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.strategic_feature_suite import evaluate
from experiments.fhp.strategic_feature_suite.config import get_experiment, SPECS
from experiments.fhp.exp10_fhp_hand_board_features import config as baseline


def source_fixture(tmp_path, monkeypatch, number):
    # Reuse the established source-contract fixture, with different encoders.
    from tests import test_exp20_evaluation as old_tests
    exp = get_experiment(number)
    monkeypatch.setattr(old_tests.run, "CONFIGS", {"exp10": baseline, f"exp{number}": exp})
    roots = old_tests.source_fixture(tmp_path, monkeypatch)
    return exp, roots


def args_for(tmp_path, number, *, smoke=False):
    return SimpleNamespace(candidate_run=tmp_path/f"exp{number}", exp10_run=tmp_path/"exp10",
        output_dir=tmp_path/"analysis", workers=1, smoke=smoke, rule_deals=2,
        lbr_deals=3, lbr_shard_deals=2, lbr_rollouts=2, crossplay_deals=2,
        base_seed=20260922, resume=False)


def score(task):
    value = task["training_seed"] + (task["experiment"] != "exp10")
    values = {k: np.full(task["num_deals"], value) for k in ("paired", "player_zero", "player_one")}
    result = dict(task, **evaluate.protocol._match_summary(values, policy_a=task["policy_a_name"], policy_b="fixture"))
    if task["kind"] == "lbr":
        result.update({"_" + k + "_values": v.tolist() for k, v in values.items()})
    return result


@pytest.mark.parametrize("number", tuple(SPECS))
def test_sources_scoring_resume_and_changed_budget(tmp_path, monkeypatch, number):
    exp, roots = source_fixture(tmp_path, monkeypatch, number)
    assert len(evaluate.discover(roots[f"exp{number}"], exp)) == 12
    assert len(evaluate.discover(roots["exp10"], exp, control=True)) == 12
    monkeypatch.setattr(evaluate.protocol, "_evaluation_worker", score)
    args = args_for(tmp_path, number)
    out = evaluate.run_analysis(args, experiment=exp)
    summary = json.loads((out/"summary.json").read_text())
    assert summary["primary_endpoint"][0]["num_training_seeds"] == 3
    assert "nan" not in (out/"analysis_summary.md").read_text()
    args.resume = True
    monkeypatch.setattr(evaluate.protocol, "_evaluation_worker", lambda task: pytest.fail("Completed tasks must be reused"))
    evaluate.run_analysis(args, experiment=exp)
    args.crossplay_deals += 1
    with pytest.raises(ValueError, match="resume mismatch"):
        evaluate.run_analysis(args, experiment=exp)


@pytest.mark.parametrize("change", ["commit", "python", "summary", "encoder", "policy"])
def test_source_guards(tmp_path, monkeypatch, change):
    exp, roots = source_fixture(tmp_path, monkeypatch, 29)
    control = change == "commit"
    root = roots["exp10" if control else "exp29"]
    worker = next((root/"workers").glob("task_*"))
    if change == "policy":
        (worker/"checkpoints/time_24h.pkl").write_bytes(b"corrupted")
    else:
        filename = {"commit": "run_manifest.json", "python": "runtime_manifest.json",
                    "summary": "summary.json", "encoder": "run_manifest.json"}[change]
        path = worker/filename
        obj = json.loads(path.read_text())
        if change == "commit": obj["repository_commit"] = "c"*40
        elif change == "python": obj["python_version"] = "3.11.17"
        elif change == "summary": obj["seed"] = 100
        else: obj["training_config"]["feature_encoder_id"] = baseline.EXPERIMENT_CONFIG["feature_encoder_id"]
        path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        evaluate.discover(root, exp, control=control)


def test_failure_and_smoke_no_lbr(tmp_path, monkeypatch):
    exp, _ = source_fixture(tmp_path, monkeypatch, 29)
    args = args_for(tmp_path, 29, smoke=True)
    args.lbr_deals = 0
    def fail(task):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(evaluate.protocol, "_evaluation_worker", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate.run_analysis(args, experiment=exp)
    assert not (args.output_dir/"SUCCESS.json").exists()
    assert json.loads((args.output_dir/"evaluation_manifest.json").read_text())["status"] == "failed"
    args.resume = True
    monkeypatch.setattr(evaluate.protocol, "_evaluation_worker", score)
    evaluate.run_analysis(args, experiment=exp)
    assert "not estimated (smoke)" in (args.output_dir/"analysis_summary.md").read_text()


@pytest.mark.parametrize("number", tuple(SPECS))
def test_feature_preflight(tmp_path, number):
    from experiments.fhp.strategic_feature_suite.feature_preflight import run_feature_preflight
    from fhp_escher.relative_strength import board_table
    before = np.random.get_state()
    result = run_feature_preflight(tmp_path, experiment=get_experiment(number), samples=8)
    after = np.random.get_state()
    assert result["status"] == "passed" and result["sampled_states"] == 16
    assert result["candidate_to_baseline_encoding_time_ratio"] > 0
    assert result["board_cache_array_bytes_at_64_entries"] < 25*1024**2
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert board_table.cache_info().currsize == 0


def test_cloud_builder_works_without_site_packages(tmp_path):
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-S", "gcp/exp29_relative_strength_batch.py",
        "--kind", "controller", "--run-id", "exp29-test", "--bucket-root", "gs://test-bucket",
        "--service-account", "runner@example.com", "--repo-ref", "a"*40,
        "--project-id", "test-project", "--region", "europe-west1", "--output", str(tmp_path/"job.json")],
        cwd=root, check=True, capture_output=True, text=True)
