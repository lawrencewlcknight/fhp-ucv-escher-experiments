"""Native loaders plus a cohort-aware, fixed-budget match schedule."""
import importlib.util
import itertools
from pathlib import Path
from experiments.fhp.exp41_fhp_matched_representation_48h import engine as previous

VR_REF, SD_REF = previous.VR_REF, previous.SD_REF
EXPERIMENTS = {"sd": 42, "vr": 43}
_ENGINE = None


def configure(engine, algorithm):
    experiment = EXPERIMENTS[algorithm]
    engine.PROTOCOL = f"exp{experiment}_fhp_{algorithm}_rich_vs_original_48h_v1"
    engine.BASE_SEED = 600_048 + experiment * 1000
    engine.DIRECT_PAIRS, engine.SHARD_PAIRS, engine.RULE_PAIRS = 100_000, 5_000, 0
    engine.FAMILY_CONFIDENCE = .975  # Two prespecified representation comparisons.
    engine.EXPERIMENT_ID = experiment
    engine.REPRESENTATION_ALGORITHM = algorithm
    engine.COMPARISONS = ((f"{algorithm}_rich_vs_original", "rich", "original"),)
    engine.build_tasks = lambda records, stage, implementation: build_tasks(engine, records, stage, implementation)
    return engine


def load(vr_repo, sd_repo, ucv_repo, algorithm):
    global _ENGINE
    # Reuse import-origin and file-hash checks, but keep source/schedule globals
    # separate from the Experiment 41 module cached in the same process.
    previous.load(vr_repo, sd_repo, ucv_repo)
    if _ENGINE is None:
        spec = importlib.util.spec_from_file_location("_representation_native_engine", Path(vr_repo)/previous.ENGINE_PATH)
        _ENGINE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ENGINE)
        _ENGINE.setup_repositories(sd_repo, ucv_repo)
    return configure(_ENGINE, algorithm)


# Scoring consumes the complete task; workers need no experiment-specific globals.
initialise_worker = previous.initialise_worker
execute_task = previous.execute_task


def build_tasks(engine, records, stage, implementation):
    if stage not in {"smoke", "profile", "production"}:
        raise ValueError("Unknown evaluation stage")
    index = {(r["cohort"], r["seed"]): r for r in records}
    if (len(records) != 6 or set(index) != set(itertools.product(("rich", "original"), range(3)))
            or any(r["algorithm"] != engine.REPRESENTATION_ALGORITHM for r in records)):
        raise ValueError("Expected three rich and three original policies of one algorithm")
    total = 100_000 if stage == "production" else (128 if stage == "profile" else 2)
    size = 5_000 if stage == "production" else total
    comparison = engine.COMPARISONS[0][0]
    tasks = []
    for left, right in itertools.product(range(3), repeat=2):
        cell = f"{comparison}_{left}_{right}"
        for shard, start in enumerate(range(0, total, size)):
            tasks.append(dict(task_id=f"{cell}_{shard:03d}", cell_id=cell, kind="direct",
                comparison_id=comparison, a=index["rich", left], b=index["original", right],
                num_deals=min(size, total-start), evaluation_seed=engine._seed(stage, 100, left, right, shard),
                stage=stage, protocol=engine.PROTOCOL, implementation=implementation))
    return tasks
