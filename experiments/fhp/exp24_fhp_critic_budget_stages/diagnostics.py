"""Fixed training-panel curves, distinct from selection and final assessment."""
import hashlib
import time

import numpy as np
import torch

from adaptive_escher.frozen_target_cache import frozen_targets, transition_batch
from experiments.fhp.exp22_fhp_critic_objectives.data import assess
from experiments.fhp.exp22_fhp_critic_objectives.estimator import (
    recursive_estimates, paired_variance_loss, root_td_loss,
)


class FitPanelDiagnostics:
    """One fixed replay probe per member; all fresh fitting traces are assessed.

    Replay probe losses are training diagnostics, not held-out evidence. Panel
    regeneration, optimisation, budget selection and assessment never use them.
    """
    def __init__(self):
        self.replay = {}

    @torch.no_grad()
    def __call__(self, candidate, model, panel, member, iteration, config):
        started = time.perf_counter()
        fold = candidate["member"]
        if fold not in self.replay:
            rng = np.random.default_rng(np.random.SeedSequence([config["random_seed"], fold, 808]))
            ids = rng.choice(len(member.buffer), min(config["fixed_replay_probe_rows"], len(member.buffer)), replace=False)
            batch = transition_batch(member.buffer, ids, member.device)
            labels = frozen_targets(member, batch, iteration)
            self.replay[fold] = (batch, labels, hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest())
        replay, labels, digest = self.replay[fold]
        scores = {}
        batch_size = config["evaluation_batch_size"]
        if batch_size % 2:
            raise ValueError("Fixed-panel evaluation must keep independent pairs together")
        for representation in ("online", "deployed"):
            model.load_state_dict(candidate[representation])
            model.eval()
            means = dict(matched_td=0., weighted_td=0., regret_variance=0.)
            for start in range(0, panel.trace_count, batch_size):
                ids = np.arange(start, min(start+batch_size, panel.trace_count))
                batch = panel.batch(ids)
                _, regrets = recursive_estimates(model, batch)
                for name, loss in (("matched_td", root_td_loss(model, batch, weighted=False)),
                                   ("weighted_td", root_td_loss(model, batch, weighted=True)),
                                   ("regret_variance", paired_variance_loss(regrets, batch["mask"][:, 0]))):
                    means[name] += float(loss)*len(ids)/panel.trace_count
            predictions = model(replay[0]).gather(1, replay[6][:, None]).squeeze(1)
            means["replay_td_probe"] = float((predictions-labels).square().mean())
            scores[representation] = dict(objectives=means,
                corrected_regret=assess(model, panel, batch_size)["summary"])
        if not all(np.isfinite(v) for r in scores.values() for v in r["objectives"].values()):
            raise FloatingPointError("Nonfinite fixed-panel diagnostic")
        return dict(**scores, seconds=time.perf_counter()-started,
                    fixed_replay_indices_sha256=digest, fixed_replay_rows=len(labels),
                    fixed_fitting_traces=panel.trace_count,
                    interpretation="in-sample diagnostics, never selection or final performance evidence")
