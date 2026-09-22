from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.config import (
    CHECKPOINT_TRAINING_SECONDS,
    EXPERIMENT_35_CONFIG,
    PRODUCTION_SEEDS,
    REPLAY_STORAGE,
    checkpoint_schedule,
    smoke_config,
    task_schedule,
    validate_contract,
)
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.float32_solver import (
    Float32LegacyCircularBuffer,
    Float32LegacyReservoirBuffer,
    validate_replay_storage,
)
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.run import _run_task
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.worker import (
    _make_solver,
    _solver_kwargs,
)
from unbiased_escher.grouped_wide_solver import (
    GROUPED_SOFT_TARGET_CROSS_ENTROPY,
    GroupedSoftTargetCrossEntropyAvePolicyTrainer,
    GroupedWideUnbiasedControlVariateEscher,
    TemporallyAveragedCrossFittedQMember,
)
from vr_deep_cfr.solver import CircularBuffer, ReservoirBuffer
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer


def _load_batch_builder():
    path = Path(__file__).parents[1] / "gcp" / "exp1_grouped_wide_batch.py"
    spec = importlib.util.spec_from_file_location("exp1_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contract_is_three_parallel_seeds_and_four_six_hour_checkpoints():
    assert PRODUCTION_SEEDS == (0, 1, 2)
    assert CHECKPOINT_TRAINING_SECONDS == (21_600, 43_200, 64_800, 86_400)
    assert [row["checkpoint_id"] for row in checkpoint_schedule()] == [
        "time_06h",
        "time_12h",
        "time_18h",
        "time_24h",
    ]
    assert task_schedule() == tuple(
        ("grouped_wide_ucv_escher", seed) for seed in PRODUCTION_SEEDS
    )


def test_configuration_matches_selected_leduc_experiment_35_mechanisms():
    config = EXPERIMENT_35_CONFIG
    assert config["q_ensemble_size"] == 2
    assert config["critic_target_average_window"] == 4
    assert config["fixed_control_variate_beta"] == 1.0
    assert config["use_instantaneous_predictor"] is False
    assert config["use_residual_calibration"] is True
    assert config["average_policy_loss"] == GROUPED_SOFT_TARGET_CROSS_ENTROPY
    assert config["average_policy_network_layers"] == (136, 136, 136)
    assert config["average_policy_learning_rate"] == 0.003
    assert config["average_policy_train_steps"] == 20_000
    assert config["average_policy_reset_each_fit"] is True
    assert config["paired_legacy_network_layers"] == (64, 64, 64)
    assert config["evaluation_frequency"] == 0
    assert config["evaluate_initial_policy"] is False
    assert config["early_evaluation_node_thresholds"] == ()


def test_contract_rejects_changed_seed_or_checkpoint_schedule():
    validate_contract(
        seeds=PRODUCTION_SEEDS,
        schedule=checkpoint_schedule(),
        config=EXPERIMENT_35_CONFIG,
        smoke=False,
    )
    with pytest.raises(ValueError, match="Seeds"):
        validate_contract(
            seeds=(0, 1),
            schedule=checkpoint_schedule(),
            config=EXPERIMENT_35_CONFIG,
            smoke=False,
        )


def test_batch_job_is_three_independent_retryable_standard_vms(tmp_path):
    builder = _load_batch_builder()
    args = SimpleNamespace(
        kind="train",
        repo_url=builder.REPO_URL,
        repo_ref="deadbeef",
        bucket_root="gs://test-bucket",
        run_id="exp1-fhp-test",
        service_account="batch@example.iam.gserviceaccount.com",
        parallelism=3,
        project_id="test-project",
        region="europe-west1",
        controller_action="orchestrate",
    )
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == 3
    assert group["parallelism"] == 3
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRetryCount"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "129600s"
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "batch_diagnostics monitor" in script
    assert "training_states/**" in script
    assert "EXP1_REMOTE_TASK_URI" in script


@pytest.mark.smoke
def test_four_checkpoint_policy_and_resume_smoke(tmp_path):
    first = _run_task(
        task_index=0,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    assert first["status"] == "complete"
    assert first["checkpoint_count"] == 4
    worker = tmp_path / "workers" / "task_000_grouped_wide_ucv_escher_seed_0"
    manifest = json.loads(
        (worker / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest) == 4
    assert all((worker / row["path"]).is_file() for row in manifest)
    assert all((worker / row["training_state_path"]).is_file() for row in manifest)
    run_manifest = json.loads(
        (worker / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["replay_storage"] == REPLAY_STORAGE

    resumed = _run_task(
        task_index=0,
        output_root=tmp_path,
        smoke=True,
        resume=True,
    )
    assert resumed["resumed_from_training_state"] is True
    assert resumed["final_nodes_touched"] == first["final_nodes_touched"]
    restored_manifest = json.loads(
        (worker / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert all(row.get("training_state_sha256") for row in restored_manifest)


def test_smoke_configuration_retains_the_selected_mechanisms():
    config = smoke_config()
    assert config["q_ensemble_size"] == 2
    assert config["critic_target_average_window"] == 4
    assert config["fixed_control_variate_beta"] == 1.0
    assert config["average_policy_network_layers"] == (136, 136, 136)
    assert config["average_policy_train_steps"] == 1
    solver = _make_solver(0, config)
    assert solver.average_policy_network_layers == (136, 136, 136)
    assert solver.ave_policy_trainer.network_layers == [136, 136, 136]
    assert solver.q_value_trainer.ensemble_size == 2
    assert all(
        member.target_average_window == 4
        for member in solver.q_value_trainer.members
    )


def test_exp1_uses_float32_values_without_compact_replay_confounds():
    solver = _make_solver(0, smoke_config())

    reservoirs = [
        *(trainer.buffer for trainer in solver.regret_trainers),
        solver.ave_policy_trainer.buffer,
    ]
    for buffer in reservoirs:
        assert buffer.infostate_buf.dtype == np.float32
        assert buffer.q_value_buf.dtype == np.float32
        assert buffer.q_value_mask_buf.dtype == np.float32
        assert buffer.iteration_buf.dtype == np.float32
        assert not hasattr(buffer, "rng")

    for member in solver.q_value_trainer.members:
        buffer = member.buffer
        assert buffer.history_buf.dtype == np.float32
        assert buffer.next_history_buf.dtype == np.float32
        assert buffer.next_state_buf.dtype == np.float32
        assert buffer.reward_buf.dtype == np.float32
        assert buffer.action_buf.dtype == np.dtype(int)
        assert buffer.next_legal_actions_mask_buf.dtype == np.dtype(int)
        assert not hasattr(buffer, "rng")

    assert solver.replay_storage_contract == REPLAY_STORAGE
    assert solver.replay_storage_allocated_bytes == validate_replay_storage(solver)


def test_shared_grouped_solver_factories_retain_legacy_defaults():
    solver = GroupedWideUnbiasedControlVariateEscher(
        **_solver_kwargs(0, smoke_config())
    )
    assert all(
        type(trainer) is VRDCFRPlusRegretTrainer
        for trainer in solver.regret_trainers
    )
    assert (
        type(solver.ave_policy_trainer)
        is GroupedSoftTargetCrossEntropyAvePolicyTrainer
    )
    assert all(
        type(member) is TemporallyAveragedCrossFittedQMember
        for member in solver.q_value_trainer.members
    )
    assert solver.ave_policy_trainer.buffer.infostate_buf.dtype == np.float64
    assert solver.q_value_trainer.members[0].buffer.history_buf.dtype == np.float64


def test_float32_storage_preserves_legacy_replacement_and_sampling_streams():
    def filled_reservoir(buffer_type):
        np.random.seed(719)
        buffer = buffer_type(7, 4, 3)
        for value in range(50):
            buffer.add(
                np.full(4, value),
                np.full(3, value),
                np.ones(3),
                value,
            )
        return buffer

    legacy = filled_reservoir(ReservoirBuffer)
    revised = filled_reservoir(Float32LegacyReservoirBuffer)
    for legacy_array, revised_array in zip(
        (
            legacy.infostate_buf,
            legacy.q_value_buf,
            legacy.q_value_mask_buf,
            legacy.iteration_buf,
        ),
        (
            revised.infostate_buf,
            revised.q_value_buf,
            revised.q_value_mask_buf,
            revised.iteration_buf,
        ),
    ):
        np.testing.assert_array_equal(legacy_array, revised_array)

    random.seed(991)
    legacy_sample = legacy.sample(5)
    random.seed(991)
    revised_sample = revised.sample(5)
    for legacy_tensor, revised_tensor in zip(legacy_sample, revised_sample):
        np.testing.assert_array_equal(legacy_tensor.numpy(), revised_tensor.numpy())

    def filled_circular(buffer_type):
        buffer = buffer_type(7, 8, 4, 3)
        for value in range(12):
            buffer.add(
                np.full(8, value),
                value % 3,
                np.full(8, value + 1),
                np.full(4, value),
                np.ones(3),
                value % 2,
                value % 4 == 0,
                value,
            )
        return buffer

    legacy_circular = filled_circular(CircularBuffer)
    revised_circular = filled_circular(Float32LegacyCircularBuffer)
    random.seed(301)
    legacy_sample = legacy_circular.sample(5)
    random.seed(301)
    revised_sample = revised_circular.sample(5)
    for legacy_tensor, revised_tensor in zip(legacy_sample, revised_sample):
        np.testing.assert_array_equal(legacy_tensor.numpy(), revised_tensor.numpy())
