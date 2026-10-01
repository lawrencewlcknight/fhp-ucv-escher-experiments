"""No replay snapshots except the explicitly requested final endpoint."""
import json
from unittest.mock import Mock

import pytest
import torch

from experiments.fhp.exp2_fhp_lossless_structured_ucv import worker as common
from experiments.fhp.exp6_fhp_structured_n2_standard16 import config as cfg
from gcp import exp6_structured_n2_standard16_batch as builder6
from gcp import exp7_parallel_structured_n2_standard16_batch as builder7
from gcp import exp8_parallel_48h_batch as builder8
from types import SimpleNamespace


def run(path, retention, **kwargs):
    torch.set_num_threads(1)
    return common.run_worker(
        seed=0, schedule=cfg.checkpoint_schedule(smoke=True), worker_dir=path,
        config=cfg.smoke_config(), smoke=True, resume=True,
        experiment_id=6, experiment_name=cfg.EXPERIMENT_NAME,
        algorithm_id=cfg.ALGORITHM_ID, contract_validator=cfg.validate_contract,
        training_state_retention=retention, **kwargs)


@pytest.mark.parametrize("retention,count", [("none", 0), ("final", 1), ("all", 4)])
def test_only_expected_states_built_and_saved(tmp_path, retention, count):
    build = Mock(wraps=common.build_training_state)
    save = Mock(wraps=common.save_training_state)
    result = run(tmp_path, retention, training_state_builder=build, training_state_saver=save)
    rows = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert len(rows) == 4 and len(list((tmp_path / "checkpoints").glob("*.pkl"))) == 4
    assert build.call_count == save.call_count == count
    assert len(list(tmp_path.rglob("*.pt"))) == count
    assert sum("training_state_path" in r for r in rows) == count
    assert result["training_state_retention"] == retention
    if retention == "final":
        assert "training_state_path" in rows[-1]
        restored = run(tmp_path, retention)
        assert restored["resumed_from_training_state"]
        assert restored["final_nodes_touched"] == result["final_nodes_touched"]
    if retention == "none":
        cached = run(tmp_path, retention, solver_factory=Mock(side_effect=AssertionError("must not retrain")))
        assert cached["reused_completed_outputs"]


@pytest.mark.parametrize("retention", ["none", "final"])
def test_partial_run_fails_without_silent_retraining(tmp_path, monkeypatch, retention):
    upload = Mock(side_effect=RuntimeError("interrupted after first policy"))
    monkeypatch.setattr(common, "_sync_remote", upload)
    with pytest.raises(RuntimeError, match="interrupted after"):
        run(tmp_path, retention)
    rows = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert len(rows) == 1 and "training_state_path" not in rows[0]
    assert not list(tmp_path.rglob("*.pt"))
    before = (tmp_path / "checkpoint_manifest.json").read_bytes()
    with pytest.raises(RuntimeError, match="new RUN_ID"):
        run(tmp_path, retention, solver_factory=Mock(side_effect=AssertionError("must not retrain")))
    assert (tmp_path / "checkpoint_manifest.json").read_bytes() == before


def test_completed_policy_corruption_rejected(tmp_path):
    run(tmp_path, "none")
    row = json.loads((tmp_path / "checkpoint_manifest.json").read_text())[0]
    (tmp_path / row["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        run(tmp_path, "none")


def test_corrupt_final_state_never_restarts_training(tmp_path):
    run(tmp_path, "final")
    state = next(tmp_path.rglob("*.pt"))
    state.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="no valid training state"):
        run(tmp_path, "final")


def test_retention_does_not_change_policy_weights(tmp_path):
    from fhp_escher.checkpointing import load_checkpoint_payload
    weights = []
    for mode in ("all", "none", "final"):
        run(tmp_path / mode, mode)
        weights.append(load_checkpoint_payload(tmp_path / mode / "final_policy_checkpoint.pkl")["policy_state_dict"])
    for actual in weights[1:]:
        assert all(torch.equal(actual[k], weights[0][k]) for k in actual)


@pytest.mark.parametrize("builder", [builder6, builder7, builder8])
def test_cloud_fetches_partial_metadata_and_excludes_inputs(builder):
    job = builder.build_job(SimpleNamespace(
        kind="train", repo_url=builder.REPO_URL, repo_ref="test", bucket_root="gs://test",
        run_id="retention-test", service_account="test@example.com", parallelism=3,
        project_id="test", region="europe-west1", controller_action="orchestrate"))
    spec = job["taskGroups"][0]["taskSpec"]
    script = spec["runnables"][0]["script"]["text"]
    assert 'elif gcloud storage ls "$REMOTE_TASK/run_manifest.json"' in script
    assert "continuation_inputs" in script and "--exclude=" in script
    assert spec["maxRetryCount"] == 0
    if builder != builder8:
        assert r"\.pt$" in script


def test_sync_excludes_temporary_imports_but_retains_final_state(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP8_REMOTE_TASK_URI", "gs://test/exp8/workers/task_0")
    calls = []
    monkeypatch.setattr(common.subprocess, "run",
                        lambda command, **kwargs: (calls.append(command) or SimpleNamespace(returncode=0)))
    common._sync_remote(tmp_path, remote_task_env="EXP8_REMOTE_TASK_URI")
    command = calls[0]
    import re
    excluded = re.compile(command[command.index("--exclude") + 1])
    assert excluded.search("continuation_inputs/source_state.pt")
    assert excluded.search("training_states/final.pt.tmp")
    assert not excluded.search("training_states/final.pt")
    assert not excluded.search("checkpoints/time_06h.pkl")


def test_failed_final_serialization_keeps_playable_policy_not_partial_state(tmp_path):
    def fail_save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".pt.tmp").write_bytes(b"incomplete")
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        run(tmp_path, "final", training_state_saver=fail_save)
    assert len(list((tmp_path / "checkpoints").glob("*.pkl"))) == 4
    assert not list(tmp_path.rglob("*.pt*"))
    assert not (tmp_path / "SUCCESS.json").exists()
    assert all("training_state_path" not in r for r in json.loads(
        (tmp_path / "checkpoint_manifest.json").read_text()))
    with pytest.raises(RuntimeError, match="new RUN_ID"):
        run(tmp_path, "final")
