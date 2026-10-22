"""Reuse the pinned native three-way engine without changing its old experiment.

Explicit imports avoid collisions between the repositories' ``experiments``,
``vr_deep_cfr`` and ``fhp_evaluation`` packages. Spawned workers initialise the
same imports before loading a policy. No model or archive is converted.
"""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

VR_REF = "236bf0c5e94ae90ce57545d82bbabc8bec9b7f07"
SD_REF = "bb689252d6b322adb3de6102d095a49c3ed87250"
ENGINE_PATH = "experiments/fhp/retrospective_threeway_48h_evaluation/run.py"
ENGINE_SHA256 = "5dbc3823bec798efe779c3cb3f785b7c6290d9e8b258cd4f36a02343a36d3595"
PROTOCOL = "exp40_fhp_selected_48h_threeway_v1"
RUN_IDS = {"ucv": "exp38-rich48-20261019-230016",
           "ucv_source36": "exp36-rich2500-20261018-205125",
           "vr": "vr6-ray48-20261003-221425", "sd": "sdcfr6-48h-20261002-161544"}
LABELS = {"ucv": "UCV-Escher Exp38 (rich AE / critic 2500)",
          "vr": "VR-DeepPDCFR+ Exp6 (Ray8)", "sd": "SD-CFR Exp6 (full uniform history)"}
_ENGINE = None
_ROOTS = None


def load(vr_repo, sd_repo, ucv_repo):
    global _ENGINE, _ROOTS
    roots = tuple(str(Path(p).resolve()) for p in (vr_repo, sd_repo, ucv_repo))
    if _ENGINE is not None:
        if roots != _ROOTS:
            raise ValueError("Native repositories cannot change inside a process")
        return _ENGINE
    vr, sd, ucv = map(Path, roots)
    path = vr / ENGINE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENGINE_SHA256:
        raise ValueError("Native three-way engine differs from the pinned implementation")
    # Keep UCV's experiments package, but select VR's compatible network
    # primitives and SD's evaluation-suite copy (with episode-mixture hooks).
    previous = sys.path[:]
    try:
        sys.path[:0] = [str(vr), str(sd), str(ucv)]
        for package, repo in (("vr_deep_cfr", vr), ("fhp_vr_deep", vr),
                              ("deep_cfr_poker", sd), ("fhp_evaluation", sd),
                              ("fhp_escher", ucv)):
            module = importlib.import_module(package)
            if Path(module.__file__).resolve() != repo / package / "__init__.py":
                raise ValueError(f"Wrong {package} package already imported; use a fresh process")
    finally:
        sys.path[:] = previous
    spec = importlib.util.spec_from_file_location("_exp40_native_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL = PROTOCOL
    module.BASE_SEED = 640_048  # Fresh deals, independent of the previous league.
    module.LABELS = LABELS
    _ENGINE, _ROOTS = module, roots
    module.setup_repositories(sd, ucv)
    return module


def initialise_worker(vr_repo, sd_repo, ucv_repo):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    load(vr_repo, sd_repo, ucv_repo)


def execute_task(task):
    if _ENGINE is None:
        raise RuntimeError("Native worker has not been initialised")
    return _ENGINE.execute_task(task)
