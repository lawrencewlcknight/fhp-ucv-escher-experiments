"""Matched warm-start policy fits; no critic/regret updates or oracle labels."""
from copy import deepcopy
import pickle
import random
from pathlib import Path
import time
import numpy as np
import torch

from experiments.fhp.exp22_fhp_critic_objectives.fitting import optimizer_hash
from .common import write_json, file_hash, digest
from .config import ARMS
from .diagnostics import panel_arrays, probabilities, metrics, diagnostic_threads
from .replay import training_groups


def arm_weights(groups,targeted,config,seed):
    n=int(targeted.sum()); mass=groups.masses; total=float(mass.sum())
    if not n:
        return {"ordinary":np.ones(len(mass))},dict(status="not_identifiable_no_supported_sensible_rare_targets")
    rng=np.random.default_rng(config["random_seed"]+900+seed)
    available=np.flatnonzero(~targeted)
    control=np.zeros(len(mass),dtype=bool)
    control[rng.choice(available,n,replace=False)]=True
    target_mass=float(mass[targeted].sum())
    fraction=config["targeted_mass_fraction"]
    multiplier=max(1.,min(10.,fraction*(total-target_mass)/((1-fraction)*target_mass)))
    additional=(multiplier-1)*target_mass
    t=np.ones(len(mass));t[targeted]=multiplier
    r=np.ones(len(mass));r[control]+=additional/float(mass[control].sum())
    normalizer=total/(total+additional)
    return {"ordinary":np.ones(len(mass)),"targeted":t*normalizer,"random_control":r*normalizer},dict(
        status="available",target_groups=n,random_groups=n,target_multiplier=multiplier,
        target_mass_before=target_mass,added_mass_before_normalisation=additional,
        random_group_ids=np.flatnonzero(control).tolist(),target_group_ids=np.flatnonzero(targeted).tolist(),
        objective_scale_preserved=True,
        note="Uniform group draws shared across arms; deliberately reweighted CE, capped at 10x before global normalisation.")


def save_policy(template,model,path,metadata):
    payload=dict(template,experiment_id=28,experiment_name="exp28_fhp_rare_state_audit",
        policy_state_dict={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
        offline_fitting=metadata)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".pkl.tmp")
    with open(tmp,"wb") as stream:
        pickle.dump(payload,stream,protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


@torch.no_grad()
def fidelity(model,groups,ids):
    ids=np.asarray(ids,dtype=int)
    if not len(ids):
        return None
    p=probabilities(model,groups.features(ids),groups.masks[ids])
    target=groups.targets[ids]
    ce=-(target*np.log(np.maximum(p,1e-30))).sum(1)
    entropy=-(target*np.log(np.maximum(target,1e-30))).sum(1)
    return dict(groups=len(ids),replay_weighted_kl=float(np.average(np.maximum(0,ce-entropy),weights=groups.masses[ids])),
        weighted_l1=float(np.average(np.abs(p-target).sum(1),weights=groups.masses[ids])),
        scope="fixed uniform group subsample; replay-mass weighted descriptive diagnostic")


def refit(solver,payload,template,groups,panel,config,seed,directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    training,targeted,eligibility=training_groups(groups,panel,config)
    multipliers,weight_spec=arm_weights(training,targeted,config,seed)
    write_json(root/"eligibility.json",dict(**eligibility,weighting=weight_spec))
    if not len(training.codes):
        raise ValueError("Holdout excluded the entire replay")
    base_model=solver.ave_policy_trainer.model
    original_model=optimizer_hash(base_model.state_dict())
    original_optimizer=optimizer_hash(payload["average_policy_trainer"]["optimizer"])
    base_rng=optimizer_hash(solver._capture_rng_state())
    dev,_,dev_f,dev_m=panel_arrays(panel,solver.feature_encoder,"development")
    # Sample once before fitting; never select diagnostics by observed fit error.
    diagnostic_ids=np.random.default_rng(config["random_seed"]+40+seed).choice(
        len(training.codes),min(8192,len(training.codes)),replace=False)
    candidates=[]
    for replicate in range(config["fit_replicates"]):
        for arm in ARMS:
            if arm not in multipliers:
                continue
            model=deepcopy(base_model)
            opt=torch.optim.Adam(model.parameters(),lr=payload["config"]["average_policy_learning_rate"])
            opt.load_state_dict(deepcopy(payload["average_policy_trainer"]["optimizer"]))
            # Python's native without-replacement group sampler, independent of
            # training RNGs and identical minibatch prefixes for all arms.
            draw=random.Random(config["random_seed"]+10000*seed+replicate)
            n=len(training.codes);batch=min(config["batch_size"],n)
            weights=training.masses*n/training.rows*multipliers[arm]
            fit_seconds=0.;sample_counts=np.zeros(n,dtype=np.int64)
            import hashlib
            stream_hash=hashlib.sha256()
            for step in range(1,config["updates"][-1]+1):
                begin=time.perf_counter()
                ids=np.asarray(draw.sample(range(n),batch),dtype=np.int64)
                stream_hash.update(ids.tobytes());np.add.at(sample_counts,ids,1)
                f=torch.from_numpy(training.features(ids));mask=torch.from_numpy(training.masks[ids]).bool()
                target=torch.from_numpy(training.targets[ids]);w=torch.as_tensor(weights[ids],dtype=torch.float32)
                logits=model(f).masked_fill(~mask,-torch.inf)
                # Illegal logits are -inf, so zero target times -inf must be
                # masked BEFORE multiplying (otherwise this produces NaNs).
                logs=torch.log_softmax(logits,-1).masked_fill(~mask,0.)
                loss=(-(target*logs).sum(1)*w).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {arm} fitting loss")
                opt.zero_grad();loss.backward();opt.step()
                fit_seconds+=time.perf_counter()-begin
                if step in config["updates"]:
                    name=f"{arm}_rep{replicate}_{step}"
                    row=dict(name=name,arm=arm,replicate=replicate,updates=step,seed=seed,
                        initial_model_sha256=original_model,initial_optimizer_sha256=original_optimizer,
                        fitting_seconds=fit_seconds,processed_examples=step*batch,
                        minibatch_stream_sha256=stream_hash.hexdigest(),
                        target_group_sample_draws=int(sample_counts[targeted].sum()),
                        target_groups_ever_sampled=int(np.count_nonzero(sample_counts[targeted])),
                        development=metrics(dev,probabilities(model,dev_f,dev_m)),
                        replay_fidelity=fidelity(model,training,diagnostic_ids),
                        same_architecture=True,source_architecture_sha256=digest(template["policy_model"]))
                    save_policy(template,model,root/(name+".pkl"),row)
                    row["policy_sha256"]=file_hash(root/(name+".pkl"))
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler_rng=draw.getstate(),
                                    metadata=row),root/(name+".pt"))
                    candidates.append(row)
                    write_json(root/"candidates.json",candidates)
                if step%1000==0:
                    print(f"exp28 seed={seed} {arm} replicate={replicate} update={step}",flush=True)
    if (optimizer_hash(base_model.state_dict())!=original_model or
            optimizer_hash(payload["average_policy_trainer"]["optimizer"])!=original_optimizer or
            optimizer_hash(solver._capture_rng_state())!=base_rng):
        raise RuntimeError("Policy-only fitting modified the source model/RNG")
    # All candidates are locked before any assessment policy query.
    lock=dict(seed=seed,config=config,candidates=[{k:r[k] for k in ("name","policy_sha256")} for r in candidates],
              no_selection=True,eligibility=eligibility,source_model=original_model)
    write_json(root/"fitting_lock.json",lock)
    cases,_,f,m=panel_arrays(panel,solver.feature_encoder)
    scores=[];case_scores=[]
    def assess(name,model):
        with diagnostic_threads():
            p=probabilities(model,f,m)
        scores.append(dict(name=name,metrics=metrics(cases,p)))
        case_scores.append(dict(name=name,probabilities=p.tolist()))
    assess("unchanged",base_model)
    from fhp_escher.checkpointing import LoadedFHPPolicy
    rng=deepcopy(solver._capture_rng_state())
    try:
        for row in candidates:
            model=LoadedFHPPolicy(solver.game,root/(row["name"]+".pkl")).model
            assess(row["name"],model)
    finally:
        # Loading an architecture initialises weights before overwriting them;
        # that implementation detail must not consume a learner RNG stream.
        solver._restore_rng_state(rng)
    write_json(root/"assessment_case_predictions.json",dict(
        lock_sha256=file_hash(root/"fitting_lock.json"),case_ids=[r["id"] for r in cases],scores=case_scores))
    write_json(root/"assessment.json",dict(lock_sha256=file_hash(root/"fitting_lock.json"),scores=scores,
        note="All fixed candidates, not a test-selected winner; source training may have seen these boards historically."))
    return dict(candidates=len(candidates),eligibility=eligibility,wall_seconds=time.perf_counter()-started,
        source_preserved=True,assessment_locked=True,scores=scores)
