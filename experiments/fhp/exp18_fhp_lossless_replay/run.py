"""One-VM engineering audit. No poker-quality or convergence claim is tested."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

import numpy as np
import psutil
import torch

from unbiased_escher.lossless_replay import CODEBOOK, STORAGE_ID, EncodedFeatures, CHUNK_ROWS
from experiments.fhp.exp10_fhp_hand_board_features.config import EXPERIMENT_CONFIG, smoke_config
from experiments.fhp.exp10_fhp_hand_board_features.worker import _make_solver
from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import execution_diagnostics
from experiments.fhp.exp10_fhp_hand_board_features import training_state as states
from experiments.fhp.exp10_fhp_hand_board_features.smoke import learning_state
from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.training_state import (
    save_sharded_training_state, read_sharded_training_state,
)

MODULE = "experiments.fhp.exp18_fhp_lossless_replay.run"
GIB = 1024**3


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def logical_digest(value):
    """Hash decoded logical state in bounded chunks, independent of storage."""
    digest = hashlib.sha256()

    def array(value, *, encoded=False):
        shape = ((len(value["codes"]), value["codes"].shape[1] + value["tail"].shape[1])
                 if encoded else value.shape)
        dtype = np.dtype(np.float32) if encoded else value.dtype
        digest.update(str((shape, str(dtype))).encode())
        if not shape:
            digest.update(value.tobytes())
            return
        for start in range(0, shape[0], CHUNK_ROWS):
            rows = slice(start, min(start + CHUNK_ROWS, shape[0]))
            chunk = (np.concatenate((CODEBOOK[value["codes"][rows]], value["tail"][rows]), axis=1)
                     if encoded else value[rows])
            digest.update(np.ascontiguousarray(chunk).tobytes())

    def visit(item):
        if isinstance(item, dict) and item.get("encoding") == STORAGE_ID:
            array(item, encoded=True)
        elif isinstance(item, torch.Tensor):
            array(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            array(item)
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


class Monitor:
    """Whole-VM pressure, plus driver RSS; do not sum shared Ray RSS pages."""
    def __init__(self, output):
        self.output = output
        self.stage = "startup"
        self.records = []
        self.stop = threading.Event()

    def sample(self):
        memory = psutil.virtual_memory()
        record = {"time": time.time(), "stage": self.stage,
                  "vm_total_bytes": memory.total, "vm_available_bytes": memory.available,
                  "vm_unavailable_bytes": memory.total - memory.available,
                  "driver_rss_bytes": psutil.Process().memory_info().rss}
        self.records.append(record)
        with self.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def __enter__(self):
        def loop():
            while not self.stop.wait(.5):
                self.sample()
        self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()

    @contextmanager
    def phase(self, name, timings):
        self.stage = name
        self.sample()
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = time.perf_counter() - start
            self.sample()

    def summary(self):
        return {"minimum_vm_available_gib": min(r["vm_available_bytes"] for r in self.records) / GIB,
                "peak_vm_unavailable_gib": max(r["vm_unavailable_bytes"] for r in self.records) / GIB,
                "peak_driver_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                    (GIB if sys.platform == "darwin" else 1024**2),
                "sampling_seconds": .5,
                "interpretation": "VM unavailable=total-available (not sum of process RSS); includes other VM processes."}


def snapshot(solver):
    return states.build_training_state(solver, seed=0, config={}, repository_commit="exp18-audit",
                                      checkpoint_id="validation", captured_checkpoints=[])


def restore(solver, payload):
    states.restore_training_state(solver, payload, seed=0, config={}, repository_commit="exp18-audit")


def synthetic_fill(solver, monitor):
    """Touch every allocated row; high-diversity inputs are NOT poker samples."""
    rng = np.random.default_rng(8128)
    buffers = [solver.ave_policy_trainer.buffer, *(t.buffer for t in solver.regret_trainers),
               *(m.buffer for m in solver.q_value_trainer.members), solver.calibration_trainer.buffer]
    for number, buffer in enumerate(buffers):
        monitor.stage = f"fill_buffer_{number}"
        capacity = getattr(buffer, "buffer_size", getattr(buffer, "capacity", None))
        features = ([("infostate_buf", buffer.infostate_buf)] if hasattr(buffer, "infostate_buf") else
                    [(name, getattr(buffer, name)) for name in ("history_buf", "next_history_buf", "next_state_buf")]
                    if hasattr(buffer, "history_buf") else [("features", buffer.features)])
        for start in range(0, capacity, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, capacity)
            rows = slice(start, stop)
            for _, target in features:
                # Exact same pseudo-data/RNG consumption in both storage arms.
                values = CODEBOOK[rng.integers(len(CODEBOOK), size=(stop-start, target.shape[1]), dtype=np.uint8)]
                target[rows] = values
            if hasattr(buffer, "infostate_buf"):
                buffer.q_value_buf[rows] = np.float32(1 / 3)
                buffer.q_value_mask_buf[rows] = 1
                buffer.iteration_buf[rows] = 1
            elif hasattr(buffer, "history_buf"):
                buffer.action_buf[rows] = 1
                buffer.next_legal_actions_mask_buf[rows] = 1
                buffer.next_player_buf[rows] = 0
                buffer.done_buf[rows] = 0
                buffer.reward_buf[rows] = .125
            else:
                buffer.targets[rows] = .125
        if hasattr(buffer, "infostate_buf"):
            buffer.cur_id = capacity
        else:
            buffer.size = capacity
            if hasattr(buffer, "cur_id"):
                buffer.cur_id = 0
            else:
                buffer.cursor = 0


def replay_bytes(solver):
    calibration = solver.calibration_trainer.buffer
    return (solver.ave_policy_trainer.buffer.nbytes() +
            sum(t.buffer.nbytes() for t in solver.regret_trainers) +
            sum(m.buffer.nbytes() for m in solver.q_value_trainer.members) +
            calibration.features.nbytes + calibration.targets.nbytes)


def worker(args):
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    small = args.kind == "smoke" or args.small
    if args.kind == "capacity":
        if args.rows <= 0:
            raise ValueError("Capacity must be positive")
        if args.rows > 1_000_000 and args.storage == "dense":
            raise ValueError("Do not allocate the unsafe dense 10m-row control")
        if args.rows >= 10_000_000 and psutil.virtual_memory().total < 60 * GIB:
            raise RuntimeError("Full capacity stress requires the 64-GiB VM; use --small locally")
    config = dict(smoke_config() if small else EXPERIMENT_CONFIG, replay_storage=args.storage)
    if args.kind == "capacity":
        config.update(ave_policy_buffer_size=args.rows, baseline_buffer_size=args.rows,
                      calibration_buffer_size=args.rows, advantage_buffer_size=max(1, args.rows // 5),
                      ave_policy_batch_size=32 if small else 2048,
                      average_policy_train_steps=3 if small else 100,
                      baseline_network_train_steps=3 if small else 20,
                      calibration_train_steps=3 if small else 20)
    torch.set_num_threads(1 if small else 8)
    timings, result = {}, {"kind": args.kind, "storage": args.storage, "config": config,
                          "seed": 0, "synthetic": args.kind == "capacity"}
    with Monitor(output / "memory.jsonl") as monitor:
        solver = None
        try:
            with monitor.phase("initialise", timings):
                solver = _make_solver(0, config, smoke=small)
            result["initial_state_sha256"] = logical_digest(learning_state(solver))
            if args.kind == "capacity":
                with monitor.phase("fill_all_replay", timings):
                    synthetic_fill(solver, monitor)
                result["replay_bytes"] = replay_bytes(solver)
                solver.num_iteration = 1
                # Exercise real Ray transfer and admission while persistent
                # replay is full, not just idle actors next to allocated arrays.
                original_traversals = solver.num_traversals
                solver.num_traversals = solver._parallel_collection_chunk_size
                with monitor.phase("full_replay_collection_and_merge", timings):
                    for player in range(solver.num_players):
                        solver.collect_training_data(player)
                result["collection_traversals_per_player"] = solver.num_traversals
                solver.num_traversals = original_traversals
                with monitor.phase("policy_grouping_and_fit", timings):
                    solver.ave_policy_trainer.train_model(1)
                result["unique_policy_groups"] = solver.ave_policy_trainer.grouped_num_information_sets
                with monitor.phase("critic_cache_and_fit", timings):
                    for member in solver.q_value_trainer.members:
                        member.train_model(1)
                with monitor.phase("calibration_fit", timings):
                    solver.calibration_trainer.train_model()
            else:
                with monitor.phase("fixed_training", timings):
                    for _ in range(3 if small else 2):
                        solver.iteration()
                    solver.ave_policy_trainer.train_model(solver.num_iteration)
                result["nodes"] = solver.nodes_touched
                result["iterations"] = solver.num_iteration
                result["replay_bytes"] = replay_bytes(solver)
            result["phase_diagnostics"] = execution_diagnostics(solver)
            result["learning_state_sha256"] = logical_digest(learning_state(solver))
            # Exercise output storage without uploading multi-GB states.
            with TemporaryDirectory(prefix="exp18-resume-") as temporary:
                state_path = Path(temporary) / "state"
                with monitor.phase("streamed_save", timings):
                    payload = snapshot(solver)
                    expected = logical_digest(learning_state(solver))
                    save_sharded_training_state(state_path, payload)
                    del payload
                result["checkpoint_bytes"] = sum(p.stat().st_size for p in state_path.rglob("*") if p.is_file())
                if args.kind == "capacity":
                    # Corrupt a resident row before the restore to exercise copying.
                    solver.ave_policy_trainer.buffer.infostate_buf[0] = np.zeros(solver.infostate_size)
                    with monitor.phase("streamed_restore", timings):
                        restored = read_sharded_training_state(state_path)
                        restore(solver, restored)
                        del restored
                    result["restore_bitwise_identical"] = logical_digest(learning_state(solver)) == expected
                else:
                    with monitor.phase("uninterrupted_continuation", timings):
                        solver.iteration()
                        solver.ave_policy_trainer.train_model(solver.num_iteration)
                        expected_next = logical_digest(learning_state(solver))
                    solver.close()
                    del solver
                    solver = None
                    gc.collect()
                    with monitor.phase("fresh_solver_state_restore", timings):
                        # Fresh solver/actors; same process to retain pinned runtime.
                        solver = _make_solver(0, config, smoke=small)
                        restored = read_sharded_training_state(state_path)
                        restore(solver, restored)
                        del restored
                    result["restore_bitwise_identical"] = logical_digest(learning_state(solver)) == expected
                    with monitor.phase("resumed_continuation", timings):
                        solver.iteration()
                        solver.ave_policy_trainer.train_model(solver.num_iteration)
                    result["continuation_sha256"] = logical_digest(learning_state(solver))
                    result["continuation_bitwise_identical"] = result["continuation_sha256"] == expected_next
            if not result["restore_bitwise_identical"] or not result.get("continuation_bitwise_identical", True):
                raise AssertionError("State restoration/continued training differs")
            result["status"] = "passed"
        except BaseException as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if solver is not None:
                solver.close()
            result.update(timings=timings, memory=monitor.summary())
            write_json(output / "result.json", result)


def compare(left, right):
    for field in ("initial_state_sha256", "learning_state_sha256", "continuation_sha256", "nodes", "iterations"):
        if left.get(field) != right.get(field):
            raise AssertionError(f"Dense/coded mismatch: {field}")


def orchestrate(args):
    root = args.output_root
    if (root / "analysis").exists():
        raise ValueError("Use a fresh output directory; audit results are immutable")
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    runtime = {"experiment": 18, "python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "platform": platform.platform(),
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "mode": args.command, "scope": "One-seed engineering audit, not a policy-quality experiment"}
    write_json(analysis / "manifest.json", runtime)

    def run(kind, storage, name, rows=0):
        destination = analysis / name
        command = [sys.executable, "-m", MODULE, "worker", "--kind", kind, "--storage", storage,
                   "--output-root", str(destination), "--rows", str(rows)]
        if args.command == "smoke":
            command.append("--small")
        subprocess.run(command, check=True)
        return json.loads((destination / "result.json").read_text())

    summary = {"status": "running", "runtime": runtime}
    try:
        parity = [run("smoke", storage, "parity_" + storage) for storage in ("dense", STORAGE_ID)]
        compare(*parity)
        summary["eight_actor_parity_and_resume"] = "bitwise_passed"
        timings = []
        if args.command != "smoke":
            for repeat in range(3):
                # Alternate order to reduce systematic first-arm bias; one seed.
                order = ("dense", STORAGE_ID) if repeat % 2 == 0 else (STORAGE_ID, "dense")
                pair = {storage: run("training", storage, f"timing_{repeat}_{storage}") for storage in order}
                compare(pair["dense"], pair[STORAGE_ID])
                timings.append({storage: pair[storage]["timings"]["fixed_training"] for storage in order})
            summary["training_timing_seconds"] = timings
            summary["median_speedup_dense_over_coded"] = statistics.median(
                r["dense"] / r[STORAGE_ID] for r in timings)
            summary["training_slowdown_warning"] = summary["median_speedup_dense_over_coded"] < 1
        count = 1000 if args.command == "smoke" else 1_000_000
        matched = [run("capacity", storage, "matched_" + storage, count) for storage in ("dense", STORAGE_ID)]
        compare(*matched)
        summary["matched_capacity"] = matched
        if args.command != "smoke":
            stress = run("capacity", STORAGE_ID, "capacity_10m", 10_000_000)
            summary["capacity_10m"] = stress
            summary["64gib_headroom_pass"] = stress["memory"]["minimum_vm_available_gib"] >= 8
            if not summary["64gib_headroom_pass"]:
                raise RuntimeError("Capacity run completed but did not retain 8 GiB of VM headroom")
        summary["status"] = "passed"
        summary["interpretation"] = ("Exactness gate passed for tested workloads. Timing is descriptive; "
            "synthetic high-diversity capacity stress is not evidence of poker quality or long-run speedup. "
            "Do not promote by memory savings alone; inspect collection/fitting/checkpoint costs.")
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(analysis / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "worker"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("smoke", "training", "capacity"), default="smoke")
    parser.add_argument("--storage", choices=("dense", STORAGE_ID), default=STORAGE_ID)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
