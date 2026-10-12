"""Read-only case evaluation and exact expectation of the native terminal estimator."""
from copy import deepcopy
from contextlib import contextmanager
import numpy as np
import torch

from experiments.fhp.exp22_fhp_critic_objectives.fitting import optimizer_hash
from experiments.fhp.exp22_fhp_critic_objectives.data import frozen_sampler
from unbiased_escher.estimator import control_variate_advantage
from .panels import from_case


@contextmanager
def diagnostic_threads():
    """Small inference is faster with one thread; restore native fitting threads."""
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@torch.no_grad()
def probabilities(model, features, masks):
    output=[]
    for start in range(0,len(features),1024):
        logits=model(torch.as_tensor(features[start:start+1024],dtype=torch.float32))
        mask=torch.as_tensor(masks[start:start+1024],dtype=torch.bool)
        p=torch.softmax(logits.masked_fill(~mask,-torch.inf),-1).numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite policy")
        output.append(p)
    return np.concatenate(output)


def panel_arrays(panel, encoder, split=None):
    cases=[r for r in panel["rows"] if split is None or r["partition"]==split]
    states=[from_case(r) for r in cases]
    return cases,states,np.asarray([encoder.information_state(s,s.current_player()) for s in states],dtype=np.float32),np.asarray([s.legal_actions_mask() for s in states],dtype=np.float32)


def metrics(cases,probs):
    result={}
    for split in ("development","assessment"):
        for cat in ("all","quads","straight_flush"):
            ids=[i for i,r in enumerate(cases) if r["partition"]==split and r["strict_winner"]
                 and (cat=="all" or r["category"]=={ "quads":7,"straight_flush":8}[cat])]
            if not ids:
                continue
            folds=probs[ids,0]
            gaps=folds*np.array([cases[i]["guaranteed_call_gain_bb"] for i in ids])
            result[split+"/"+cat]=dict(positions=len(ids),mean_fold_probability=float(folds.mean()),
                mean_gap_bb=float(gaps.mean()),p95_gap_bb=float(np.quantile(gaps,.95)),max_gap_bb=float(gaps.max()),
                fraction_fold_over_half=float(np.mean(folds>.5)))
    return result


@torch.no_grad()
def audit_case_predictions(solver,panel,coverage):
    cases,states,f,m=panel_arrays(panel,solver.feature_encoder)
    deployed=probabilities(solver.ave_policy_trainer.model,f,m)
    rows=[]
    for case,s,p in zip(cases,states,deployed):
        player=s.current_player(); t=solver.regret_trainers[player]
        current=t.get_policy(s,solver.num_iteration)
        retained=coverage["exact"].get(case["id"])
        # No automatic causal verdict: current strategy and average teacher differ.
        rows.append(dict(**case,deployed=p.tolist(),current_regret_policy=current.tolist(),
            current_regret_scores=t.get_regrets(s).tolist(),retained_target=retained,
            target_status="observed_empirical_target" if retained else "not_retained_unidentified",
            deployed_minus_replay_fold=None if retained is None else float(p[0]-retained["target"][0])))
    return dict(rows=rows,metrics=metrics(cases,deployed),
        note="Current regret strategy is not the teacher for historical average strategy.")


@torch.no_grad()
def estimator_probe(solver,panel,config,seed):
    """Exact action expectation, separately for sampled hidden histories/folds.

    Primary roots are guaranteed winners. Each full history has two terminal
    actions, so no rollout approximation or clairvoyant action maximisation.
    A local RNG draws nuisance hidden hands; no training sampler is consumed.
    """
    rng_state=deepcopy(solver._capture_rng_state())
    before=optimizer_hash(rng_state)
    versions=[m.target_version for m in solver.q_value_trainer.members]
    sampler=frozen_sampler(solver)
    candidates=[r for r in panel["rows"] if r["partition"]=="development" and r["strict_winner"]]
    rng=np.random.default_rng(config["random_seed"]+8100+seed)
    # Stratify across category and seat before deterministic budget truncation.
    buckets=[[r for r in candidates if r["category"]==cat and r["seat"]==p] for cat in (7,8) for p in (0,1)]
    selected=[]
    for b in buckets:
        if b:
            selected.extend(b[i] for i in rng.choice(len(b),min(len(b),max(1,config["probe_positions"]//4)),replace=False))
    rows=[]
    try:
        for r in selected:
            remaining=[c for c in range(52) if c not in r["board"]+r["hand"]]
            histories=[]
            for _ in range(config["probe_hidden_hands"]):
                other=tuple(map(int,rng.choice(remaining,2,replace=False)))
                state=from_case(r,other); player=r["seat"]
                for fold in (0,1):
                    pi,mask,proposal,q=sampler.info(state,player,fold)
                    q=q[0]  # the frozen sampler exposes one copy per audit update
                    legal=np.flatnonzero(mask)
                    if set(legal)!={0,1} or np.any(proposal[legal]<=0):
                        raise ValueError("Probe requires full-support terminal call/fold")
                    returns=np.array([state.child(a).returns()[player]/solver.max_utility for a in legal])
                    if returns[1]<=returns[0]:
                        raise ValueError("Declared guaranteed winner failed OpenSpiel check")
                    estimates=[]
                    for a,g in zip(legal,returns):
                        estimates.append(control_variate_advantage(q,beta=np.ones(3),sampled_action=int(a),
                            sample_probability=float(proposal[a]),sampled_return=float(g),policy=pi,
                            legal_actions_mask=mask).advantages)
                    samples=np.asarray(estimates)
                    mean=proposal[legal]@samples
                    exact=np.zeros(3);exact[legal]=returns-float(pi[legal]@returns)
                    error=float(np.max(np.abs(mean-exact)))
                    if error>1e-9:
                        raise ValueError("Native corrected-estimator expectation mismatch")
                    variance=proposal[legal]@((samples-mean)**2)
                    histories.append(dict(opponent=list(other),fold=fold,policy=pi.tolist(),proposal=proposal.tolist(),
                        q=q.tolist(),sampled_action_estimates=samples.tolist(),expected_advantage=exact.tolist(),
                        action_variance=variance.tolist(),expectation_max_error=error,
                        call_minus_fold_bb=float((returns[1]-returns[0])*solver.max_utility/100)))
            rows.append(dict(case_id=r["id"],histories=histories))
    finally:
        if before!=optimizer_hash(solver._capture_rng_state()) or versions!=[m.target_version for m in solver.q_value_trainer.members]:
            solver._restore_rng_state(rng_state)
            raise RuntimeError("Frozen diagnostic changed training RNG/state")
    return dict(rows=rows,training_rng_preserved=True,no_samples_added=True,
        scope="exact conditional action expectation; sampled compatible hidden histories, both cross-fit folds",
        units="native normalised advantages; explicit conversion to BB supplied",beta=1.)
