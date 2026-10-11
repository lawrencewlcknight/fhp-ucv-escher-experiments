"""An unchanged learning trajectory; snapshotting never performs diagnostic fits."""
from copy import deepcopy
from pathlib import Path
import time

from experiments.critic_target_cache_benchmark import write_json
from experiments.fhp.exp10_fhp_hand_board_features import config as baseline
from experiments.fhp.exp22_fhp_critic_objectives.fitting import optimizer_hash
from .states import save_state


def make_solver(*, smoke=False, ray_smoke=False):
    config = baseline.smoke_config() if smoke else baseline.EXPERIMENT_CONFIG
    if not smoke or ray_smoke:
        from experiments.fhp.exp10_fhp_hand_board_features.worker import _make_solver
        return _make_solver(0, config, smoke=smoke)
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
    from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import install_cache
    from unbiased_escher.fhp_structured_solver import StructuredFHPGroupedWideUCVEscher
    kwargs = _solver_kwargs(0, config)
    kwargs.pop("cache_frozen_critic_targets")
    solver = StructuredFHPGroupedWideUCVEscher(**kwargs)
    install_cache(solver, enabled=True)
    return solver


def collect(solver, root, *, smoke=False, late=False, lineage=None, sync=lambda: None):
    """First pre-fit boundary at/after each time; checkpoint after that same fit.

    Stage clock counts normal iteration work only. All snapshot/hash/upload work
    is excluded. A completed-iteration boundary may overshoot the nominal time.
    Smoke uses iterations, not an unreliable tiny wall-time threshold.
    """
    if getattr(solver, "_parallelize_independent_learners", False):
        raise ValueError("Reference requires Experiment 10's sequential central learner order")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "reference_manifest.json").exists():
        raise FileExistsError("Use a fresh reference directory")
    lineage = deepcopy(lineage or dict(synthetic=smoke, new_reference=True))
    stages = [("late_24h_plus_one_iteration", 0.)] if late else [("one_hour", 3600.), ("six_hours", 21600.)]
    history_elapsed = float(getattr(solver, "_resume_training_elapsed_seconds", 0.)) if late else 0.
    manifest = dict(seed=0, smoke=smoke, late=late, lineage=lineage, stages=stages,
        clock="sum of complete iteration compute, excluding all persistence and diagnostics",
        nominal_training_hours=None if late else 6, baseline="unchanged Experiment 10 learning configuration",
        no_policy_extraction=True, source_iteration=solver.num_iteration,
        historical_training_seconds=history_elapsed)
    write_json(root / "reference_manifest.json", manifest)
    ensemble = solver.q_value_trainer
    original_fit = ensemble.train_model
    original_time = solver._training_elapsed_seconds
    original_checkpoint = solver._maybe_run_training_time_checkpoint
    original_evaluation = solver.evaluation_frequency
    elapsed, started, paused = 0., None, 0.
    records, pending = [], []
    start_iteration = solver.num_iteration

    def active_seconds():
        return history_elapsed + elapsed + (time.perf_counter()-started-paused if started is not None else 0.)

    def capture(stage, pre_fit):
        nonlocal paused
        active = active_seconds()
        before = time.perf_counter()
        rng = solver._capture_rng_state()
        rng_digest = optimizer_hash(rng)
        versions = [m.target_version for m in ensemble.members]
        folder = root / ("pre_fit" if pre_fit else "checkpoints") / stage
        record = save_state(solver, folder, stage=stage, smoke=smoke, elapsed=active,
            pre_fit=pre_fit, lineage=lineage)
        if rng_digest != optimizer_hash(solver._capture_rng_state()) or versions != [m.target_version for m in ensemble.members]:
            raise RuntimeError("State capture changed training RNG or critic versions")
        sync()
        record["capture_and_upload_seconds"] = time.perf_counter()-before
        if started is not None:
            paused += record["capture_and_upload_seconds"]
        return record

    def observed_fit(iteration):
        if pending:
            raise RuntimeError("More than one ensemble fit in the iteration")
        next_index = len(records)
        stage, target = stages[next_index]
        due = (late or (solver.num_iteration-start_iteration >= next_index+1 if smoke
                        else active_seconds()-history_elapsed >= target))
        if due:
            pending.append(capture(stage, True))
        # No experimental fitting, sampling, target refresh or RNG use here.
        return original_fit(iteration)

    ensemble.train_model = observed_fit
    solver._training_elapsed_seconds = active_seconds
    solver._maybe_run_training_time_checkpoint = lambda: None
    solver.evaluation_frequency = 0
    began = time.perf_counter()
    try:
        while len(records) < len(stages):
            if solver.num_iteration-start_iteration >= int(baseline.EXPERIMENT_CONFIG["max_num_iterations"]):
                raise RuntimeError("Reference iteration cap reached before stage capture")
            paused, started = 0., time.perf_counter()
            solver.iteration()
            elapsed += time.perf_counter()-started-paused
            started = None
            if solver._stop_requested:
                raise RuntimeError("Unexpected partial-iteration stop in reference")
            if pending:
                pre = pending.pop()
                if [m.target_version for m in ensemble.members] != [v+1 for v in pre["critic_versions"]]:
                    raise RuntimeError("Snapshot did not precede exactly one fit of BOTH critics")
                checkpoint = None if late else capture(pre["stage"], False)
                records.append(dict(stage=pre["stage"], pre_fit=pre, checkpoint=checkpoint,
                                    completed_iteration=solver.num_iteration,
                                    completed_training_seconds=active_seconds()))
                write_json(root / "stages.json", records)
            print(f"reference iteration={solver.num_iteration} nodes={solver.nodes_touched} active_seconds={active_seconds():.3f}", flush=True)
        result = dict(status="complete", seed=0, smoke=smoke, stages=records,
            added_active_training_seconds=elapsed, reference_wall_seconds=time.perf_counter()-began,
            snapshot_and_upload_seconds=sum(r["pre_fit"]["capture_and_upload_seconds"] +
                (r["checkpoint"]["capture_and_upload_seconds"] if r["checkpoint"] else 0) for r in records),
            source_iteration=start_iteration, final_iteration=solver.num_iteration,
            final_nodes=solver.nodes_touched, no_diagnostic_samples_in_learning=True)
        write_json(root / "SUCCESS.json", result)
        sync()
        return result
    finally:
        ensemble.train_model = original_fit
        solver._training_elapsed_seconds = original_time
        solver._maybe_run_training_time_checkpoint = original_checkpoint
        solver.evaluation_frequency = original_evaluation
