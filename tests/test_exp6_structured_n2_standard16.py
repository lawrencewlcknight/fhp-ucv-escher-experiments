"""Contract, real-checkpoint and cloud-spec checks for the VM-only replication."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.fhp.exp2_fhp_lossless_structured_ucv import config as exp2
from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _make_solver as make_exp2
from experiments.fhp.exp6_fhp_structured_n2_standard16 import config as exp6
from experiments.fhp.exp6_fhp_structured_n2_standard16.aggregate import aggregate_workers, task_name
from experiments.fhp.exp6_fhp_structured_n2_standard16.run import _run_task, main
from experiments.fhp.exp6_fhp_structured_n2_standard16.worker import _make_solver
from experiments.fhp.exp6_fhp_structured_n2_standard16.training_state import read_training_state, STATE_TYPE
from fhp_escher.checkpointing import LoadedFHPPolicy, load_checkpoint_payload
from fhp_escher.game import load_fhp_game
from gcp import exp6_structured_n2_standard16_batch as builder
from gcp import exp2_lossless_structured_batch as old_builder


def args(kind):
    return SimpleNamespace(kind=kind, repo_url=builder.REPO_URL, repo_ref="abc123",
                           bucket_root="gs://test-bucket", run_id="exp6-vm16-test",
                           service_account="runner@example.com", parallelism=3,
                           project_id="test-project", region="europe-west1",
                           controller_action="orchestrate")


def test_only_vm_and_identity_change():
    assert exp6.EXPERIMENT_CONFIG == exp2.EXPERIMENT_CONFIG
    assert exp6.EXPERIMENT_CONFIG is not exp2.EXPERIMENT_CONFIG
    assert exp6.smoke_config() == exp2.smoke_config()
    assert exp6.PRODUCTION_SEEDS == exp2.PRODUCTION_SEEDS == (0, 1, 2)
    assert exp6.CHECKPOINT_HOURS == (6, 12, 18, 24)
    assert exp6.checkpoint_schedule() == exp2.checkpoint_schedule()
    assert exp6.checkpoint_schedule()[-1]["target_training_seconds"] == 24*3600
    assert exp6.REFERENCE_VM["machine_type"] == "n2-standard-16"
    assert exp6.LEARNER_THREADS == 8
    assert exp6.contract_manifest()["single_intended_change"] == "vm_size"
    assert exp6.contract_manifest()["parallel_traversal_workers"] == 0
    assert not exp6.contract_manifest()["frozen_critic_target_cache"]
    assert exp6.task_schedule() == tuple((exp6.ALGORITHM_ID, s) for s in (0, 1, 2))
    exp6.validate_contract(seeds=(0, 1, 2), schedule=exp6.checkpoint_schedule(),
                          config=exp6.EXPERIMENT_CONFIG, smoke=False)
    changed = deepcopy(exp6.EXPERIMENT_CONFIG)
    changed["num_traversals"] *= 2
    with pytest.raises(ValueError, match="num_traversals"):
        exp6.validate_contract(seeds=(0, 1, 2), schedule=exp6.checkpoint_schedule(),
                              config=changed, smoke=False)


def test_learner_matches_experiment2_at_fixed_iterations():
    torch.set_num_threads(1)
    results = []
    for factory in (make_exp2, _make_solver):
        solver = factory(0, exp6.smoke_config())
        assert not any(getattr(m, "cache_frozen_targets", False)
                       for m in solver.q_value_trainer.members)
        for _ in range(2):
            solver.iteration()
        models = [t.model for t in solver.regret_trainers]
        models += [m.model for m in solver.q_value_trainer.members]
        models += [m.target_model for m in solver.q_value_trainer.members]
        models += [solver.calibration_trainer.model, solver.ave_policy_trainer.model]
        states = [deepcopy(m.state_dict()) for m in models]
        results.append((solver.nodes_touched, states))
    assert results[0][0] == results[1][0]
    for first, second in zip(results[0][1], results[1][1]):
        assert all(torch.equal(first[k], second[k]) for k in first)


@pytest.mark.parametrize("kind", ["controller", "smoke", "train", "aggregate"])
def test_cloud_job_contract(kind, tmp_path):
    job = builder.build_job(args(kind))
    group = job["taskGroups"][0]
    spec = group["taskSpec"]
    machine = job["allocationPolicy"]["instances"][0]["policy"]["machineType"]
    assert group["taskCount"] == (3 if kind == "train" else 1)
    assert group["taskCountPerNode"] == 1
    if kind in ("train", "smoke"):
        assert machine == "n2-standard-16"
        assert spec["computeResource"] == {"cpuMilli": 16000, "memoryMib": 62000}
    else:
        assert machine == ("e2-small" if kind == "controller" else "n2-standard-8")
    if kind == "train":
        assert spec["maxRunDuration"] == "129600s"
        assert spec["maxRetryCount"] == 0
    script = spec["runnables"][0]["script"]["text"]
    assert "exp2_fhp_lossless_structured_ucv" not in script
    assert "EXP2_REMOTE" not in script
    assert "checkpoint-smoke" not in script
    if kind in ("train", "smoke"):
        assert "--requested-memory-mib 62000" in script
        assert "export OMP_NUM_THREADS=8" in script
        assert "export MKL_NUM_THREADS=8" in script
    if kind == "train":
        assert "lossless_structured_ucv_escher_n2_16" in script
        assert "EXP6_REMOTE_TASK_URI" in script
    path = tmp_path/f"{kind}.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)
    # Reusing the builder must not mutate historical experiments.
    old = old_builder.build_job(args("train"))
    assert old["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    assert old["taskGroups"][0]["taskSpec"]["computeResource"]["cpuMilli"] == 8000


def test_smoke_checkpoint_resume_and_analysis(tmp_path):
    main(["smoke", "--output-root", str(tmp_path), "--no-resume"])
    worker = tmp_path/"workers"/task_name(0, 0)
    summary = json.loads((worker/"summary.json").read_text())
    assert summary["experiment_id"] == 6
    assert summary["checkpoint_count"] == 4
    assert not summary["resumed_from_training_state"]
    assert summary["training_state_retention"] == "none"
    assert summary["reference_vm"] == exp6.REFERENCE_VM
    manifest = json.loads((worker/"run_manifest.json").read_text())
    assert manifest["experiment_name"] == exp6.EXPERIMENT_NAME
    assert manifest["execution_backend"] == "sequential_seed_worker"
    rows = json.loads((worker/"checkpoint_manifest.json").read_text())
    assert all("training_state_path" not in row for row in rows)
    assert not list(tmp_path.rglob("*.pt"))
    checkpoint = worker/"final_policy_checkpoint.pkl"
    payload = load_checkpoint_payload(checkpoint)
    assert payload["policy_model"]["hidden_layers"] == [192, 192]
    game = load_fhp_game()
    policy = LoadedFHPPolicy(game, checkpoint)
    decision = game.new_initial_state()
    while decision.is_chance_node():
        decision.apply_action(decision.chance_outcomes()[0][0])
    assert sum(policy.action_probabilities(decision).values()) == pytest.approx(1.)
    assert (tmp_path/"analysis/nodes_by_training_time.png").is_file()
    assert json.loads((tmp_path/"analysis/throughput_summary.json").read_text())["smoke"]
    assert (tmp_path/"analysis/SUCCESS.json").is_file()
    assert json.loads((worker/"runtime_manifest.json").read_text())["torch_intraop_threads"] == 1
    manifest["experiment_id"] = 2
    (worker/"run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Not an Experiment 6"):
        aggregate_workers(tmp_path, seeds=(0,))
