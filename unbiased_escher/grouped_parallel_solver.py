"""Synchronous parallel adapters for the selected FHP grouped-policy solvers.

These adapters select the SAME solver for the driver and traversal workers.
The archived parallel classes alone do not implement the grouped-policy fit
or temporally averaged critics. No experiment launcher is changed here.
"""

from experiments.fhp.exp1_fhp_grouped_wide_ucv_baseline.float32_solver import (
    Float32GroupedWideUnbiasedControlVariateEscher,
)
from .fhp_structured_solver import StructuredFHPGroupedWideUCVEscher
from .parallel_solver import (
    ParallelUnbiasedControlVariateEscher,
    UCVEscherTraversalWorker,
)


class GroupedWideTraversalWorker(UCVEscherTraversalWorker):
    SOLVER_CLASS = Float32GroupedWideUnbiasedControlVariateEscher


class StructuredGroupedTraversalWorker(UCVEscherTraversalWorker):
    SOLVER_CLASS = StructuredFHPGroupedWideUCVEscher


class _CompletedIterationCheckpoints:
    def _after_collection_chunk(self):
        # Match the selected sequential solvers: a time boundary must not save
        # a partially updated outer iteration. Early-node diagnostics retain
        # their existing optional hook.
        self._maybe_run_early_node_checkpoint()

    def iteration(self):
        super().iteration()
        self._maybe_run_training_time_checkpoint()


class ParallelGroupedWideUCVEscher(
    _CompletedIterationCheckpoints,
    ParallelUnbiasedControlVariateEscher,
    Float32GroupedWideUnbiasedControlVariateEscher,
):
    """Parallel traversal for the raw-input Experiment 1 configuration."""

    def _traversal_worker_class(self):
        return GroupedWideTraversalWorker


class ParallelStructuredGroupedUCVEscher(
    _CompletedIterationCheckpoints,
    ParallelUnbiasedControlVariateEscher,
    StructuredFHPGroupedWideUCVEscher,
):
    """Parallel traversal for the structured Experiment 2/3 configurations."""

    def _traversal_worker_class(self):
        return StructuredGroupedTraversalWorker
