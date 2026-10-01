"""Correctness checks for the opt-in frozen critic target cache."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from adaptive_escher.frozen_target_cache import (
    build_target_cache, frozen_targets, replay_indices, transition_batch,
)
from experiments.critic_target_cache_benchmark import (
    aggregate, benchmark, load_components, restore, rng_hash, snapshot, tensor_comparison, write_json,
)
from experiments.fhp.exp5_fhp_critic_target_cache.run import SPEC, make_smoke, _make_solver


@pytest.fixture
def solver():
    torch.set_num_threads(1)
    return make_smoke(SPEC["seeds"][0])


@pytest.mark.parametrize("full_batch", [False, True])
def test_sampler_matches_existing_buffer(solver, full_batch):
    member = solver.q_value_trainer.members[0]
    count = -1 if full_batch else member.batch_size
    saved = snapshot(member)
    restore(member, saved, 999)
    expected = member.buffer.sample(count)
    expected_rng = rng_hash(member)
    restore(member, saved, 999)
    ids = replay_indices(member.buffer, count)
    observed = transition_batch(member.buffer, ids, member.device)
    assert rng_hash(member) == expected_rng
    for a, b in zip(expected, observed):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cache_preserves_rng_terminal_rewards_and_refreshes(solver):
    member = solver.q_value_trainer.members[0]
    before = rng_hash(member)
    cached = build_target_cache(member, solver.num_iteration)
    assert rng_hash(member) == before
    size = len(member.buffer)
    for start in range(0, size, member.batch_size):
        ids = np.resize(np.arange(start, min(size, start+member.batch_size)), member.batch_size)
        expected = frozen_targets(member, transition_batch(member.buffer, ids, member.device), solver.num_iteration)
        torch.testing.assert_close(cached[ids], expected, atol=1e-6, rtol=1e-5)
    terminal = np.flatnonzero(member.buffer.done_buf[:size])
    if len(terminal):
        torch.testing.assert_close(cached[terminal], torch.tensor(member.buffer.reward_buf[terminal], dtype=torch.float32))
    with torch.no_grad():
        list(member.target_model.parameters())[-1].add_(1.)
    refreshed = build_target_cache(member, solver.num_iteration+1)
    nonterminal = np.flatnonzero(member.buffer.done_buf[:size] == 0)
    assert len(nonterminal) > 0
    assert not torch.equal(cached[nonterminal], refreshed[nonterminal])


def test_rejects_stochastic_target_network(solver):
    member = solver.q_value_trainer.members[0]
    member.target_model = torch.nn.Sequential(member.target_model, torch.nn.Dropout(.1))
    with pytest.raises(ValueError, match="row-independent"):
        build_target_cache(member, solver.num_iteration)


def test_cache_preserves_subsequent_whole_solver_iterations():
    results = []
    for cached in (False, True):
        torch.set_num_threads(1)
        learner = make_smoke(SPEC["seeds"][0])
        for member in learner.q_value_trainer.members:
            member.cache_frozen_targets = cached
        for _ in range(2):
            learner.iteration()
        models = [t.model for t in learner.regret_trainers]
        models += [learner.calibration_trainer.model, learner.ave_policy_trainer.model]
        models += [m.model for m in learner.q_value_trainer.members]
        models += [m.target_model for m in learner.q_value_trainer.members]
        state = {f"{index}/{name}": value.detach().clone()
                 for index, model in enumerate(models)
                 for name, value in model.state_dict().items()}
        results.append((learner.nodes_touched, state,
                        [rng_hash(m) for m in learner.q_value_trainer.members]))
    assert results[0][0] == results[1][0]
    assert results[0][2] == results[1][2]
    assert tensor_comparison(results[0][1], results[1][1])["allclose"]


def test_paired_benchmark_equivalence_and_reuse(solver, tmp_path):
    for member in solver.q_value_trainer.members:
        member.train_steps = 4
    manifest = {"test": True}
    benchmark(solver.q_value_trainer.members, solver.num_iteration, seed=0,
              directory=tmp_path, manifest=manifest, smoke=True)
    result = json.loads((tmp_path/"summary.json").read_text())
    assert result["equivalence_passed"]
    assert len(result["records"]) == 2
    assert all(row["sample_rng_equal"] and row["cache_audit"]["construction_preserved_rng"]
               for row in result["records"])
    previous = (tmp_path/"repeat_0.json").read_bytes()
    benchmark(solver.q_value_trainer.members, solver.num_iteration, seed=0,
              directory=tmp_path, manifest=manifest, smoke=True)
    assert (tmp_path/"repeat_0.json").read_bytes() == previous
    with pytest.raises(ValueError, match="another source"):
        benchmark(solver.q_value_trainer.members, solver.num_iteration, seed=0,
                  directory=tmp_path, manifest={"different": True}, smoke=True)


def test_large_minibatch_tail_and_temporal_target_average(solver):
    """Use the production minibatch shape and eight CPU threads, including a tail."""
    torch.set_num_threads(8)
    member = solver.q_value_trainer.members[0]
    for name in ("history", "next_history", "next_state", "reward", "action",
                 "next_legal_actions_mask", "next_player", "done"):
        original = getattr(member.buffer, name+"_buf")[:len(member.buffer)]
        ids = np.arange(2051) % len(original)
        setattr(member.buffer, name+"_buf", original[ids].copy())
    member.buffer.size = member.buffer.buffer_size = 2051
    member.batch_size, member.train_steps = 2048, 3
    saved = snapshot(member)
    results = []
    for cached in (False, True):
        restore(member, saved, 76)
        member.cache_frozen_targets = cached
        member.train_model(solver.num_iteration)
        results.append(snapshot(member))
    assert tensor_comparison(results[0]["model"], results[1]["model"])["allclose"]
    assert tensor_comparison(results[0]["target"], results[1]["target"])["allclose"]
    assert results[0]["version"] == results[1]["version"]
    torch.set_num_threads(1)


def test_lightweight_source_restore_retains_replay_and_models(solver):
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.config import EXPERIMENT_CONFIG
    config = deepcopy(EXPERIMENT_CONFIG)
    # Only constructors use this config; actual saved buffers below remain tiny.
    members = solver.q_value_trainer.members
    payload = {"seed": SPEC["seeds"][0], "config": config,
               "solver": {"num_iteration": solver.num_iteration},
               "regret_trainers": [{"model": t.model.state_dict()} for t in solver.regret_trainers],
               "q_ensemble": {"members": []}}
    for m in members:
        state = {"model": m.model.state_dict(), "target_model": m.target_model.state_dict(),
                 "optimizer": m.optimizer.state_dict(), "target_version": m.target_version,
                 "target_history": list(m.target_history),
                 "buffer": {"size": len(m.buffer), "cur_id": m.buffer.cur_id}}
        for name in ("history", "next_history", "next_state", "reward", "action",
                     "next_legal_actions_mask", "next_player", "done"):
            state["buffer"][name] = getattr(m.buffer, name+"_buf")[:len(m.buffer)]
        payload["q_ensemble"]["members"].append(state)
    restored, iteration, _ = load_components(payload, _make_solver)
    assert iteration == solver.num_iteration
    for old, new in zip(members, restored):
        assert np.shares_memory(old.buffer.history_buf, new.buffer.history_buf)
        assert tensor_comparison(old.model.state_dict(), new.model.state_dict())["bitwise_equal"]
        assert tensor_comparison(old.target_model.state_dict(), new.target_model.state_dict())["bitwise_equal"]


def test_generated_cloud_jobs_are_bounded_and_syntax_valid(tmp_path):
    root = Path(__file__).parents[1]
    for kind in ("controller", "smoke", "worker", "aggregate"):
        output = tmp_path/f"{kind}.json"
        subprocess.run([sys.executable, str(root/"gcp/frozen_critic_target_efficiency_batch.py"),
                        "--kind", kind, "--output", str(output), "--project", "test", "--region", "europe-west1",
                        "--bucket", "gs://bucket", "--run-id", "cache-test", "--source-run-id", "source-test",
                        "--service-account", "worker@example.com", "--repo-ref", "abc"], check=True)
        payload = json.loads(output.read_text())
        group = payload["taskGroups"][0]
        assert group["taskCount"] == (3 if kind == "worker" else 1)
        assert group["taskCountPerNode"] == 1
        assert group["taskSpec"]["maxRetryCount"] == 0
        if kind == "worker":
            assert group["taskSpec"]["maxRunDuration"] == "43200s"
        script = group["taskSpec"]["runnables"][0]["script"]["text"]
        assert "{MODULE}" not in script
        if kind != "controller":
            assert "experiments.fhp.exp5_fhp_critic_target_cache.run" in script
        path = tmp_path/f"{kind}.sh"
        path.write_text(script)
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_three_seed_aggregate_rejects_mixed_or_incomplete_results(tmp_path):
    manifest = {"experiment": SPEC, "smoke": False, "code_sha256": "code",
                "commit": "commit", "torch_version": "test", "threads": 8,
                "repetitions": 3, "atol": 1e-6, "rtol": 1e-5}
    for seed in SPEC["seeds"]:
        folder = tmp_path/"workers"/f"seed_{seed}"
        rows = [{"seed": seed, "repeat": repeat, "fold": fold,
                 "updates": 10000, "batch_size": 2048, "equivalence_passed": True,
                 "arms": {"recompute": {"fit_seconds": 2.}, "cache": {"fit_seconds": 1.}}}
                for repeat in range(3) for fold in range(2)]
        write_json(folder/"manifest.json", dict(manifest, source={"seed": seed}))
        write_json(folder/"summary.json", {"seed": seed, "records": rows,
                                          "equivalence_passed": True, "peak_process_rss_mib": 1.})
        write_json(folder/"SUCCESS.json", {"seed": seed})
    analysis = aggregate(tmp_path, SPEC["seeds"])
    result = json.loads((analysis/"summary.json").read_text())
    assert result["all_equivalence_checks_passed"]
    assert result["mean_speedup"] == 2.
    assert (analysis/"critic_cache_efficiency.png").is_file()
    last = tmp_path/"workers"/f"seed_{SPEC['seeds'][-1]}"
    metadata = json.loads((last/"manifest.json").read_text())
    write_json(last/"manifest.json", dict(metadata, code_sha256="wrong"))
    with pytest.raises(ValueError, match="Inconsistent benchmark manifests"):
        aggregate(tmp_path, SPEC["seeds"])
    write_json(last/"manifest.json", metadata)
    summary = json.loads((last/"summary.json").read_text())
    summary["records"].pop()
    write_json(last/"summary.json", summary)
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate(tmp_path, SPEC["seeds"])
