"""Inference only: load native models without constructing a learner or Ray actors."""
from types import SimpleNamespace
import time
import numpy as np
import torch

from fhp_escher.checkpointing import LoadedFHPPolicy
from fhp_escher.features import StructuredFHPMLP
from fhp_escher.game import load_fhp_game
from vr_deep_cfr.solver import RegretTrainer
from experiments.fhp.exp28_fhp_rare_state_audit.diagnostics import panel_arrays, probabilities, metrics
from experiments.fhp.exp28_fhp_rare_state_audit.common import digest


def expected_config(experiment):
    if experiment == 10:
        from experiments.fhp.exp10_fhp_hand_board_features.config import EXPERIMENT_CONFIG
        return EXPERIMENT_CONFIG
    from experiments.fhp.strategic_feature_suite.config import get_experiment
    return get_experiment(experiment).EXPERIMENT_CONFIG


def validate_pair(state, policy, source):
    config = expected_config(source['experiment'])
    if (state['seed'] != source['seed'] or policy['seed'] != source['seed']
            or state['checkpoint_id'] != 'time_24h'
            or policy['checkpoint_target_seconds'] != 86400
            or digest(state['config']) != digest(config)
            or digest(policy['training_config']) != digest(config)
            or state['feature_encoder'] != policy['feature_encoder']
            or state['solver']['num_iteration'] != source['iteration']
            or policy['outer_iteration'] != source['iteration']
            or state['solver']['nodes_touched'] != source['nodes']
            or policy['nodes_touched'] != source['nodes']):
        raise ValueError('Wrong checkpoint horizon, configuration, encoder or source identity')
    saved = state['average_policy_trainer']['model']
    if saved.keys() != policy['policy_state_dict'].keys() or any(
            not torch.equal(saved[k], v) for k, v in policy['policy_state_dict'].items()):
        raise ValueError('Full state and deployed policy do not match')
    if len(state['regret_trainers']) != 2 or config['use_instantaneous_predictor']:
        raise ValueError('Audit only supports these two-player nonpredictive source configurations')


@torch.no_grad()
def current_predictions(models, features, masks, seats, argmax):
    scores = np.zeros_like(masks, dtype=np.float32)
    for seat, model in enumerate(models):
        ids = np.flatnonzero(np.asarray(seats)==seat)
        for start in range(0, len(ids), 1024):
            batch = ids[start:start+1024]
            scores[batch] = model(torch.as_tensor(features[batch])).numpy()*masks[batch]
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite regret predictions')
    # Call the native rule, including its argmax fallback when all regrets <=0.
    native = SimpleNamespace(use_regret_matching_argmax=argmax)
    policies = np.asarray([RegretTrainer.regret_matching(native, r, np.flatnonzero(m))
                           for r, m in zip(scores, masks)])
    return scores, policies


def measure(policy, models, panel, config):
    cases, states, features, masks = panel_arrays(panel, policy.feature_encoder)
    deployed = probabilities(policy.model, features, masks)
    scores, current = current_predictions(models, features, masks,
        [r['seat'] for r in cases], config['use_regret_matching_argmax'])
    for p in (deployed, current):
        if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError('Invalid policy distribution')
    rows = [dict(case_id=r['id'], deployed=p.tolist(), current_regret_policy=c.tolist(),
                 current_regret_scores=s.tolist()) for r,p,c,s in zip(cases,deployed,current,scores)]
    primary = np.asarray([i for i,r in enumerate(cases) if r['category'] in (7,8)], dtype=int)
    primary_cases = [cases[i] for i in primary]
    return dict(rows=rows, metrics=dict(deployed=metrics(primary_cases, deployed[primary]),
                current_regret=metrics(primary_cases, current[primary])),
        warning='Current regret policy is not the historical average teacher. Differences do not '
                'identify compression error. Cases are not independent training replicates.')


def audit_source(source, directory, panel):
    started = time.perf_counter()
    torch.set_num_threads(1)
    policy = LoadedFHPPolicy(load_fhp_game(), directory/'policy.pkl')
    # Only trusted, pinned project checkpoints are deserialised. One at a time.
    state = torch.load(directory/'state.pt', map_location='cpu', weights_only=False)
    validate_pair(state, policy.checkpoint, source)
    config = state['config']
    models = []
    for item in state['regret_trainers']:
        model = StructuredFHPMLP(policy.feature_encoder.policy_layout,
            [config['num_hiddens']]*config['num_layers'], 3,
            branch_width=config['structured_branch_width'])
        model.load_state_dict(item['model'])
        models.append(model.eval())
    # No replay grouping, optimizer construction, fitting or new traversal.
    del state
    loaded = time.perf_counter()
    result = measure(policy, models, panel, config)
    result.update(source=source, source_load_seconds=loaded-started,
                  inference_seconds=time.perf_counter()-loaded)
    return result
