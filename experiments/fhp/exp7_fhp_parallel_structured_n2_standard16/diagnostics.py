"""Non-mutating parallel timing and memory diagnostics."""
from pathlib import Path
import resource
import sys

COUNTERS = (
    "_cumulative_parallel_collection_seconds", "_cumulative_worker_collection_seconds",
    "_cumulative_worker_total_seconds", "_cumulative_worker_snapshot_load_seconds",
    "_parallel_worker_snapshot_reload_count", "_cumulative_parallel_sync_seconds",
    "_cumulative_parallel_merge_seconds", "_cumulative_parallel_learner_seconds",
    "_cumulative_parallel_regret_seconds", "_cumulative_parallel_holdout_seconds",
    "_cumulative_parallel_worker_imbalance_seconds", "_parallel_peak_result_bytes",
    "_parallel_dispatch_count", "_effective_parallel_learner_threads",
    "_cumulative_experience_collection_seconds", "_parallel_snapshot_token",
)


def execution_diagnostics(solver):
    result = {name.lstrip("_"): getattr(solver, name, 0) for name in COUNTERS}
    actors = solver._ray.get([worker.resource_summary.remote() for worker in solver._workers])
    if any(actor["torch_threads"] != 1 for actor in actors):
        raise RuntimeError("Traversal actors must each use one Torch thread")
    result["parallel_num_workers"] = len(actors)
    # Sum of per-process peaks is NOT a simultaneous whole-VM memory peak.
    result["parallel_sum_actor_peak_rss_mib"] = sum(actor["peak_rss_mib"] for actor in actors)
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result["parallel_driver_peak_rss_mib"] = rss / (1024**2 if sys.platform == "darwin" else 1024)
    for label, paths in (
        ("current", ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")),
        ("peak", ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes")),
    ):
        result[f"parallel_cgroup_{label}_mib"] = None
        for path in paths:
            try:
                result[f"parallel_cgroup_{label}_mib"] = int(Path(path).read_text()) / 1024**2
                break
            except (FileNotFoundError, PermissionError, ValueError):
                pass
    return result
