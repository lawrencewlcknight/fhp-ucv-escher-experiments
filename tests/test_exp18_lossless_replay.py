import argparse
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from gcp.exp18_lossless_replay_batch import build_job
from unbiased_escher.lossless_replay import EncodedFeatures, CODEBOOK, STORAGE_ID, feature_state
from experiments.fhp.exp18_fhp_lossless_replay import run
from tests.test_lossless_replay import make_solver


def test_logical_hash_independent_of_storage():
    values = np.tile(CODEBOOK, (35_000, 1))
    coded = EncodedFeatures(values.shape)
    coded[:] = values
    assert run.logical_digest(values) == run.logical_digest(feature_state(coded, len(values)))
    coded[3] = np.zeros(len(CODEBOOK))
    assert run.logical_digest(values) != run.logical_digest(feature_state(coded, len(values)))


def test_byte_counts():
    # Exact populated replay allocation formula for the proposed larger pilot.
    dense = 14_000_000 * 880 + 10_000_000 * 3447 + 10_000_000 * 880
    coded = 14_000_000 * 241 + 10_000_000 * 870 + 10_000_000 * 241
    assert dense / 2**30 == pytest.approx(51.77222, rel=1e-6)
    assert coded / 2**30 == pytest.approx(13.48928, rel=1e-6)
    from unbiased_escher.efficient_replay import CompactCircularBuffer, CompactReservoirBuffer
    from unbiased_escher.fhp_structured_solver import EncodedCalibrationBuffer
    assert CompactReservoirBuffer(1, 213, 3, feature_array_factory=EncodedFeatures).nbytes() == 241
    assert CompactCircularBuffer(1, 323, 213, 3, feature_array_factory=EncodedFeatures).nbytes() == 870
    calibration = EncodedCalibrationBuffer(1, 219, 213)
    assert calibration.features.nbytes + calibration.targets.nbytes == 241


def test_failure_gates():
    with pytest.raises(AssertionError, match="learning_state"):
        run.compare({"learning_state_sha256": "a"}, {"learning_state_sha256": "b"})
    with pytest.raises(AssertionError, match="nodes"):
        run.compare({"nodes": 1}, {"nodes": 2})


def test_cloud_contract_and_shell(tmp_path):
    args = SimpleNamespace(repo_url="https://example.com/repo.git", repo_ref="a"*40,
                           bucket_root="gs://test-results", run_id="exp18-test",
                           service_account="runner@test.iam.gserviceaccount.com")
    job = build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 1
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["maxRunDuration"] == "43200s"
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "3.11.16" in script and "run --output-root" in script
    assert '"$WORK_ROOT/audit/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"' in script
    assert '"$WORK_ROOT/audit" "$BUCKET_ROOT' not in script
    shell = tmp_path / "cloud.sh"
    shell.write_text(script)
    subprocess.run(["bash", "-n", str(shell)], check=True)
    subprocess.run(["bash", "-n", "gcp/run_exp18_lossless_replay.sh"], check=True)


def test_unsafe_dense_stress_rejected_before_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_make_solver", lambda *a, **k: pytest.fail("Must reject before allocation"))
    args = argparse.Namespace(output_root=tmp_path, kind="capacity", small=False, rows=10_000_000, storage="dense")
    with pytest.raises(ValueError, match="unsafe dense"):
        run.worker(args)


def test_default_and_opt_in_are_isolated():
    assert isinstance(make_solver().ave_policy_trainer.buffer.infostate_buf, np.ndarray)
    assert isinstance(make_solver(STORAGE_ID).ave_policy_trainer.buffer.infostate_buf, EncodedFeatures)
