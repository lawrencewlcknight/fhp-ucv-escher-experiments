from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import (
    EXPERIMENT_CONFIG as EXP2_CONFIG,
    smoke_config as exp2_smoke_config,
)
from experiments.fhp.exp3_fhp_wider_lossless_structured_ucv.config import (
    BASELINE_ONLINE_PARAMETER_COUNT,
    CHECKPOINT_HOURS,
    EXPERIMENT_CONFIG,
    ONLINE_PARAMETER_COUNT,
    PARAMETER_MULTIPLIER,
    PRODUCTION_SEEDS,
    checkpoint_schedule,
    contract_manifest,
    smoke_config,
    task_schedule,
    validate_contract,
)
from experiments.fhp.exp3_fhp_wider_lossless_structured_ucv.run import _run_task
from experiments.fhp.exp3_fhp_wider_lossless_structured_ucv.worker import _make_solver
from fhp_escher.checkpointing import LoadedFHPPolicy, load_checkpoint_payload
from fhp_escher.features import ENCODER_ID
from fhp_escher.game import load_fhp_game


def _load_batch_builder():
    path = Path(__file__).parents[1] / "gcp" / "exp3_wider_structured_batch.py"
    spec = importlib.util.spec_from_file_location("exp3_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _online_parameter_count(solver) -> int:
    models = [
        *(trainer.model for trainer in solver.regret_trainers),
        solver.ave_policy_trainer.model,
        *(member.model for member in solver.q_value_trainer.members),
        solver.calibration_trainer.model,
    ]
    return sum(_parameter_count(model) for model in models)


def test_contract_changes_only_network_width_from_experiment_2():
    validate_contract(
        seeds=PRODUCTION_SEEDS,
        schedule=checkpoint_schedule(),
        config=EXPERIMENT_CONFIG,
        smoke=False,
    )
    changed = {
        key
        for key in set(EXP2_CONFIG) | set(EXPERIMENT_CONFIG)
        if EXP2_CONFIG.get(key) != EXPERIMENT_CONFIG.get(key)
    }
    assert changed == {
        "num_hiddens",
        "average_policy_network_layers",
        "structured_branch_width",
    }
    assert CHECKPOINT_HOURS == (6, 12, 18, 24)
    assert task_schedule() == tuple(
        ("wider_lossless_structured_ucv_escher", seed)
        for seed in PRODUCTION_SEEDS
    )
    manifest = contract_manifest()
    assert manifest["baseline"] == "exp2_fhp_lossless_structured_ucv"
    assert manifest["single_intended_change"] == "network_width"


def test_wider_network_shapes_and_parameter_count():
    solver = _make_solver(0, smoke_config())
    assert solver.structured_branch_width == 96
    assert solver.network_layers == [192, 192]
    assert solver.average_policy_network_layers == (256, 256)
    assert solver.regret_trainers[0].model.branch_width == 96
    assert solver.regret_trainers[0].model.hidden_layers == [192, 192]
    assert solver.q_value_trainer.members[0].model.hidden_layers == [192, 192]
    assert solver.calibration_trainer.model.hidden_layers == [192, 192]
    assert solver.ave_policy_trainer.model.hidden_layers == [256, 256]
    assert _online_parameter_count(solver) == ONLINE_PARAMETER_COUNT
    assert BASELINE_ONLINE_PARAMETER_COUNT == 310_993
    assert 1.9 < PARAMETER_MULTIPLIER < 2.0


def test_smoke_overrides_only_training_scale_in_addition_to_widths():
    exp2_smoke = exp2_smoke_config()
    exp3_smoke = smoke_config()
    changed = {
        key
        for key in set(exp2_smoke) | set(exp3_smoke)
        if exp2_smoke.get(key) != exp3_smoke.get(key)
    }
    assert changed == {
        "num_hiddens",
        "average_policy_network_layers",
        "structured_branch_width",
    }


def test_batch_job_is_three_independent_retryable_vms():
    builder = _load_batch_builder()
    args = SimpleNamespace(
        kind="train",
        repo_url=builder.REPO_URL,
        repo_ref="deadbeef",
        bucket_root="gs://test-bucket",
        run_id="exp3-fhp-test",
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
    assert group["taskSpec"]["maxRetryCount"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "129600s"
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-8"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "exp3_fhp_wider_lossless_structured_ucv" in script
    assert "EXP3_REMOTE_TASK_URI" in script
    assert "wider_lossless_structured_ucv_escher" in script
    assert "EXP2_REMOTE_TASK_URI" not in script


def test_wider_checkpoint_round_trip_and_resume(tmp_path):
    summary = _run_task(
        task_index=0,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    assert summary["status"] == "complete"
    assert summary["experiment_id"] == 3
    assert summary["checkpoint_count"] == 4
    worker = (
        tmp_path
        / "workers"
        / "task_000_wider_lossless_structured_ucv_escher_seed_0"
    )
    checkpoint = worker / "final_policy_checkpoint.pkl"
    payload = load_checkpoint_payload(checkpoint)
    assert payload["feature_encoder"]["id"] == ENCODER_ID
    assert payload["policy_model"]["hidden_layers"] == [256, 256]
    assert payload["policy_model"]["branch_width"] == 96
    policy = LoadedFHPPolicy(load_fhp_game(), checkpoint)
    state = load_fhp_game().new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    assert abs(sum(policy.action_probabilities(state).values()) - 1.0) < 1e-6
    resumed = _run_task(
        task_index=0,
        output_root=tmp_path,
        smoke=True,
        resume=True,
    )
    assert resumed["resumed_from_training_state"] is True
    assert resumed["final_nodes_touched"] == summary["final_nodes_touched"]
