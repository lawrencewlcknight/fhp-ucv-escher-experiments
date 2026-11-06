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

VR_REF = "70abec44561c5e141edf870512ec4237cf9fe4ad"
SD_REF = "ac56342ae92824e06b206dd909a99c897529e373"
ENGINE_PATH = "experiments/fhp/retrospective_threeway_48h_evaluation/run.py"
ENGINE_SHA256 = "5dbc3823bec798efe779c3cb3f785b7c6290d9e8b258cd4f36a02343a36d3595"
PROTOCOL = "exp41_fhp_rich_ae_48h_threeway_v1"
RUN_IDS = {"ucv": "exp38-rich48-20261019-230016",
           "ucv_source36": "exp36-rich2500-20261018-205125",
           "vr": "vr8-rich48-20261101-234215", "sd": "sd8-rich48-20261101-234234"}
LABELS = {"ucv": "UCV-Escher Exp38 (rich AE / critic 2500)",
          "vr": "VR-DeepPDCFR+ Exp8 (rich AE)", "sd": "SD-CFR Exp8 (rich AE, full uniform history)"}
_ENGINE = None
_ROOTS = None


SOURCE_NAMES = {
    "sd": ("exp8_sd_cfr_rich_ae_48h", "rich_ae_uniform_sd_cfr_48h"),
    "vr": ("exp8_fhp_vr_deep_rich_ae_48h", "rich_ae_vr_deep_pdcfr_plus_48h"),
}


def configure(module):
    """Adapt only this isolated module; historical experiments stay unchanged."""
    from .sources import PINS
    from experiments.fhp.exp38_fhp_strength_critic2500_48h import contract
    module.PROTOCOL, module.BASE_SEED, module.LABELS = PROTOCOL, 641_048, LABELS
    module.RUN_IDS = RUN_IDS
    module.SOURCE_NAMES.update(SOURCE_NAMES)
    module.SOURCE_COMMITS.update(sd=SD_REF, vr=VR_REF)
    module.CONFIG_SHA256.update({a: PINS[a]["config_sha256"] for a in ("sd", "vr")})
    module.SOURCE_NAMES["ucv"] = (contract.LEARNER_NAME, contract.ALGORITHM_ID)
    module.SOURCE_COMMITS["ucv"] = contract.TRAINING_REF
    module.CONFIG_SHA256["ucv"] = contract.CONFIG_SHA256
    module.RULE_PAIRS = 0
    original = module.build_tasks
    # Native smoke/profile schedules include rule agents independently of
    # RULE_PAIRS; explicitly filter those stages as well as production.
    module.build_tasks = lambda *a, **kw: [t for t in original(*a, **kw) if t["kind"] == "direct"]


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
    spec = importlib.util.spec_from_file_location("_exp41_native_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configure(module)
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
