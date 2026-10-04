"""Independent frozen-policy diagnostics; never call the replay-writing DFS.

Both arms follow identical actions/chance draws. Their recursive corrected
returns differ, so the complete implemented estimator (not just a one-step
residual) is compared. Variances are computed within histories, then averaged.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib

import numpy as np
import torch

from unbiased_escher.estimator import control_variate_advantage


class FrozenRollouts:
    def __init__(self, solver, snapshots, old_targets, updates):
        if (solver.fixed_control_variate_beta != 1 or len(solver.q_value_trainer.members) != 2
                or solver.use_instantaneous_predictor):
            raise ValueError("Audit requires the selected fixed-beta, two-fold, non-predictive core")
        self.solver, self.updates = solver, updates
        self.deployed, self.online, self.old = {}, {}, {}
        for fold, member in enumerate(solver.q_value_trainer.members):
            for budget in updates:
                for destination, key in ((self.deployed, "deployed"), (self.online, "online")):
                    model = deepcopy(member.model).eval()
                    model.load_state_dict(snapshots[fold][budget][key])
                    destination[fold, budget] = model
            self.old[fold] = deepcopy(member.target_model).eval()
            self.old[fold].load_state_dict(old_targets[fold])
        self.cache = OrderedDict()

    @torch.no_grad()
    def predictions(self, state, models):
        features = torch.as_tensor(self.solver.get_history_tensor(state), dtype=torch.float32,
                                   device=self.solver.device)
        return np.stack([model(features).cpu().numpy() for model in models]).astype(np.float64)

    def chance(self, state, rng):
        while state.is_chance_node():
            actions, probabilities = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(actions, p=probabilities)))
        return state

    def info(self, state, traverser, fold):
        key = (tuple(state.history()), traverser, fold)
        if key in self.cache:
            return self.cache[key]
        player = state.current_player()
        mask = np.asarray(state.legal_actions_mask(), dtype=np.float64)
        policy = np.asarray(self.solver.regret_trainers[player].get_policy(state, self.solver.num_iteration), dtype=np.float64)
        # The native two-member leave-one-fold-out ensemble uses the OTHER
        # member, so ensemble disagreement is identically zero in both arms.
        member = 1 - fold
        q = self.predictions(state, [self.deployed[member, u] for u in self.updates])
        q *= (1. if traverser == 0 else -1.) * mask
        if player == traverser:
            means, variances, _ = self.solver.calibration_trainer.predict_all(
                self.solver.get_infostate_tensor(state, traverser), self.solver.num_iteration,
                np.zeros_like(mask), traverser)
            proposal = self.solver._traverser_sampling_policy(
                q_values=q[0], beta=np.ones_like(mask), residual_means=means,
                predicted_variances=variances, policy=policy, legal_mask=mask)
        else:
            proposal = np.where(mask > 0, policy, 0.)
            proposal /= proposal.sum()
        if not np.isfinite(q).all() or not np.isfinite(proposal).all():
            raise FloatingPointError("Nonfinite diagnostic predictions/proposals")
        if player == traverser and np.any(proposal[mask > 0] <= 0):
            raise ValueError("Traverser proposal lost full support")
        result = (policy, mask, proposal, q)
        if len(self.cache) >= 8192:
            self.cache.popitem(last=False)
        self.cache[key] = result
        return result

    def rollout(self, state, traverser, fold, rng):
        if state.is_terminal():
            value = state.returns()[traverser] / self.solver.max_utility
            return np.full(2, value), None
        policy, mask, proposal, q = self.info(state, traverser, fold)
        action = int(rng.choice(len(proposal), p=proposal))
        child = self.chance(state.child(action), rng)
        returns, _ = self.rollout(child, traverser, fold, rng)
        estimates = [control_variate_advantage(q[i], beta=np.ones_like(mask),
            sampled_action=action, sample_probability=float(proposal[action]),
            sampled_return=float(returns[i]), policy=policy, legal_actions_mask=mask) for i in range(2)]
        return np.asarray([e.policy_value for e in estimates]), np.stack([e.advantages for e in estimates])

    def select_state(self, traverser, fold, rng):
        for _ in range(100):
            state = self.chance(self.solver.game.new_initial_state(), rng)
            candidates = []
            while not state.is_terminal():
                if state.current_player() == traverser:
                    candidates.append(state.clone())
                _, _, proposal, _ = self.info(state, traverser, fold)
                state = self.chance(state.child(int(rng.choice(len(proposal), p=proposal))), rng)
            if candidates:
                return candidates[int(rng.integers(len(candidates)))]
        raise RuntimeError("Could not sample a traverser decision")

    @torch.no_grad()
    def heldout_td(self, state, traverser, fold, rng):
        """One independent transition scored against the common pre-fit TD target."""
        _, _, proposal, _ = self.info(state, traverser, fold)
        action = int(rng.choice(len(proposal), p=proposal))
        child = self.chance(state.child(action), rng)
        reward = child.returns()[0] / self.solver.max_utility
        if child.is_terminal():
            target = reward
        else:
            policy = self.solver.regret_trainers[child.current_player()].get_policy(child, self.solver.num_iteration)
            target = reward + float(self.predictions(child, [self.old[1-fold]])[0] @ policy)
        return action, target


def evaluate(solver, snapshots, old_targets, config, *, seed, boundary):
    audit = FrozenRollouts(solver, snapshots, old_targets, config["updates"])
    rows = []
    rng = np.random.default_rng(np.random.SeedSequence([config["diagnostic_seed"], seed, boundary]))
    for traverser in (0, 1):
        for fold in (0, 1):
            for probe in range(config["probes_per_player_fold"]):
                audit.cache.clear()
                state = audit.select_state(traverser, fold, rng)
                policy, mask, proposal, _ = audit.info(state, traverser, fold)
                legal = mask > 0
                member = 1 - fold
                online = audit.predictions(state, [audit.online[member, u] for u in config["updates"]])
                deployed = audit.predictions(state, [audit.deployed[member, u] for u in config["updates"]])
                samples, errors_online, errors_deployed = [], [], []
                for _ in range(config["repeats_per_probe"]):
                    _, advantages = audit.rollout(state, traverser, fold, rng)
                    samples.append(advantages)
                    # Separate fresh transitions; never reused as training examples.
                    action, target = audit.heldout_td(state, traverser, fold, rng)
                    errors_online.append((online[:, action] - target)**2)
                    errors_deployed.append((deployed[:, action] - target)**2)
                values = np.asarray(samples)
                variance = values.var(axis=0, ddof=1)
                if not np.isfinite(values).all() or not np.isfinite(errors_online).all():
                    raise FloatingPointError("Nonfinite rollout diagnostics")
                rows.append(dict(
                    traverser=traverser, trajectory_fold=fold, heldout_critic=member, probe=probe,
                    history=list(state.history()), history_sha256=hashlib.sha256(state.serialize().encode()).hexdigest(),
                    legal_actions=np.flatnonzero(legal).tolist(), policy=policy.tolist(), proposal=proposal.tolist(),
                    repeats=config["repeats_per_probe"], arms={str(u): dict(
                        mean_legal_action_advantage_variance=float(variance[i, legal].mean()),
                        action_advantage_variance=variance[i].tolist(),
                        mean_advantage=values[:, i].mean(axis=0).tolist(),
                        online_heldout_td_mse=float(np.mean(errors_online, axis=0)[i]),
                        deployed_heldout_td_mse=float(np.mean(errors_deployed, axis=0)[i]))
                        for i, u in enumerate(config["updates"])},
                    paired_advantage_mean_difference=(values[:, 0] - values[:, 1]).mean(axis=0).tolist(),
                    max_policy_centred_residual=float(np.max(np.abs(values @ policy))),
                ))
    return rows
