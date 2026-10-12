"""Passive Ray/learner instrumentation. Native collection and loss code still run."""
from contextlib import contextmanager
from copy import deepcopy
from collections import Counter
import time
import numpy as np
import torch

from unbiased_escher.grouped_parallel_solver import StructuredGroupedTraversalWorker, ParallelStructuredGroupedUCVEscher
from experiments.fhp.exp22_fhp_critic_objectives.fitting import optimizer_hash
from .common import write_json, complete
from .panels import categories, feature_hash, from_case
from .diagnostics import diagnostic_threads


@contextmanager
def observe_traversal(solver):
    """One actor process, no threads: intercept native estimator without replacing it.

    Recursion tracks the CURRENT node, not its sampled child's hidden state.
    Records are capped, counters are not; samples never feed back into learning.
    """
    import unbiased_escher.solver as module
    native_dfs=solver.dfs;native_estimator=module.control_variate_advantage
    stack=[];counts=Counter();records=[];eligible=0
    def dfs(state,traverser,*args,**kwargs):
        context=None
        if not state.is_terminal():
            f=solver.get_infostate_tensor(state)
            cat=int(categories(np.asarray([f]))[0])
            role="regret" if state.current_player()==traverser else "policy"
            counts[f"{role}/category_{cat}"]+=1
            if cat>=7 and state.legal_actions()==[0,1] and state.child(1).is_terminal():
                context=dict(history=list(state.history()),acting=state.current_player(),traverser=traverser,
                    feature_hash=feature_hash(f),category=cat,role=role,
                    iteration=solver.num_iteration,episode=solver.episode)
        stack.append(context)
        try:
            return native_dfs(state,traverser,*args,**kwargs)
        finally:
            stack.pop()
    def corrected(q_values,**kwargs):
        nonlocal eligible
        result=native_estimator(q_values,**kwargs)
        if stack and stack[-1] is not None:
            eligible+=1
            if len(records)<256:
                rows={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in kwargs.items()}
                records.append(dict(**stack[-1],q_values=np.asarray(q_values).tolist(),**rows,
                    corrected_q=result.q_values.tolist(),corrected_advantage=result.advantages.tolist(),
                    importance_correction=result.importance_correction))
        return result
    solver.dfs=dfs;module.control_variate_advantage=corrected
    result=dict(counts=counts,records=records)
    try:
        yield result
    finally:
        solver.dfs=native_dfs;module.control_variate_advantage=native_estimator
        result["eligible_trace_count"]=eligible
        result["truncated_trace_count"]=max(0,eligible-len(records))


class ObservedWorker(StructuredGroupedTraversalWorker):
    def collect(self,*args,**kwargs):
        with observe_traversal(self._solver) as audit:
            result=super().collect(*args,**kwargs)
        result["rare_state_audit"]=audit
        return result


def reservoir_decisions(buffer,payload):
    """Replay the Algorithm-R DRAW on a private copy, without consuming the sampler.

    Reports accepted-at-arrival and survivors after this batch separately. A
    later insertion may evict a survivor; completed-iteration counts report that.
    """
    count=len(payload["iterations"]);start=int(buffer.cur_id);cap=buffer.buffer_size
    direct=min(count,max(0,cap-start))
    destinations=np.full(count,-1,dtype=np.int64)
    destinations[:direct]=np.arange(start,start+direct)
    if direct<count:
        rng=np.random.default_rng();rng.bit_generator.state=deepcopy(buffer.rng.bit_generator.state)
        draw=rng.integers(0,np.arange(start+direct+1,start+count+1,dtype=np.int64),dtype=np.int64)
        destinations[direct:]=np.where(draw<cap,draw,-1)
    accepted=np.flatnonzero(destinations>=0)
    if len(accepted):
        _,rev=np.unique(destinations[accepted][::-1],return_index=True)
        survivors=accepted[len(accepted)-1-rev]
    else:
        survivors=np.empty(0,dtype=np.int64)
    return destinations,accepted,survivors


def category_counts(features):
    cats=categories(features)
    return {str(int(c)):int(np.sum(cats==c)) for c in np.unique(cats)}


class ObservedSolver(ParallelStructuredGroupedUCVEscher):
    def _traversal_worker_class(self):
        return ObservedWorker

    def _merge_worker_result(self,traverser,result):
        audit=result["rare_state_audit"]
        row=dict(iteration=self.num_iteration,traverser=traverser,visit_counts=dict(audit["counts"]),
                 terminal_traces=audit["records"],truncated_traces=audit["truncated_trace_count"])
        checks=[]
        for label,buffer in (("regret",self.regret_trainers[traverser].buffer),
                             ("average_policy",self.ave_policy_trainer.buffer)):
            p=result[label];dest,accepted,survivors=reservoir_decisions(buffer,p)
            row[label]=dict(generated=len(p["iterations"]),
                generated_categories=category_counts(p["infostates"]),
                accepted_categories=category_counts(p["infostates"][accepted]),
                batch_survivor_categories=category_counts(p["infostates"][survivors]))
            checks.append((buffer,p,dest,survivors))
        # Unmodified authoritative merge, including its original RNG draws.
        super()._merge_worker_result(traverser,result)
        for buffer,p,dest,ids in checks:
            if len(ids) and (not np.array_equal(buffer.infostate_buf[dest[ids]],p["infostates"][ids])
                            or not np.array_equal(buffer.q_value_buf[dest[ids]],p["values"][ids])):
                raise RuntimeError("Observed reservoir destinations disagree with native merge")
        self.audit_merges.append(row)


def make_solver(seed,config,*,smoke=False):
    from experiments.fhp.exp10_fhp_hand_board_features.config import parallel_settings
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.worker import _solver_kwargs
    from experiments.fhp.exp10_fhp_hand_board_features.diagnostics import install_cache
    kwargs=_solver_kwargs(seed,config);enabled=kwargs.pop("cache_frozen_critic_targets")
    solver=ObservedSolver(**kwargs,**parallel_settings(smoke=smoke),parallel_run_seed=seed)
    install_cache(solver,enabled=enabled)
    solver.audit_merges=[]
    solver.evaluation_frequency=0;solver._maybe_run_training_time_checkpoint=lambda:None
    solver._maybe_run_early_node_checkpoint=lambda:None
    solver.max_wall_clock_seconds=None;solver.target_nodes_touched=None
    return solver


@contextmanager
def observe_regret_fit(trainer,iteration):
    """Capture ACTUAL native MSE targets, not a second implementation of the loss."""
    original=trainer.compute_loss
    summary=dict(iteration=iteration,updates=0,sampled_categories=Counter(),rows={},truncated_row_occurrences=0)
    active=[]
    def loss_input(module,args):
        predicted,target=args
        features,masks,immediate=active[-1]
        f=features.detach().cpu().numpy();cats=categories(f)
        summary["sampled_categories"].update(map(int,cats))
        for i in np.flatnonzero(cats>=7):
            h=feature_hash(f[i])
            if h not in summary["rows"] and len(summary["rows"])>=512:
                summary["truncated_row_occurrences"]+=1
                continue
            previous=summary["rows"].get(h,{})
            summary["rows"][h]=dict(feature_hash=h,features=f[i].tolist(),
                mask=masks[i].detach().cpu().tolist(),category=int(cats[i]),
                sampled_count=previous.get("sampled_count",0)+1,
                last_sampled_immediate=immediate[i].detach().cpu().tolist(),
                actual_regression_target=target[i].detach().cpu().tolist(),
                pre_update_prediction=predicted[i].detach().cpu().tolist())
    hook=trainer.loss_fn.register_forward_pre_hook(loss_input)
    def observed(samples,T):
        active.append((samples[0],samples[2],samples[1]));summary["updates"]+=1
        try:
            return original(samples,T)
        finally:
            active.pop()
    trainer.compute_loss=observed
    try:
        yield summary
    finally:
        trainer.compute_loss=original;hook.remove()
        with torch.no_grad():
            for row in summary["rows"].values():
                f=torch.tensor([row["features"]],dtype=torch.float32)
                mask=torch.tensor([row["mask"]],dtype=torch.float32)
                row["post_fit_prediction"]=trainer.predict(trainer.model,f,mask)[0].tolist()
        summary["sampled_categories"]={str(k):v for k,v in summary["sampled_categories"].items()}
        summary["rows"]=list(summary["rows"].values())


@torch.no_grad()
def current_predictions(solver,panel):
    result=[]
    for r in panel["rows"]:
        if r["partition"]!="development":
            continue
        s=from_case(r);t=solver.regret_trainers[r["seat"]]
        result.append(dict(id=r["id"],scores=t.get_regrets(s).tolist(),
                           policy=t.get_policy(s,solver.num_iteration).tolist()))
    return result


def run_continuation(solver,panel,config,directory,*,synthetic=False):
    """No policy extraction during continuation; ten COMPLETE native iterations."""
    from pathlib import Path
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/"manifest.json").exists():
        raise FileExistsError("No silent partial continuation reuse")
    initial=solver.num_iteration;began=time.perf_counter()
    write_json(directory/"manifest.json",dict(config=config,synthetic=synthetic,
        initial_iteration=initial,learning_config_changed=bool(config["smoke"])))
    original_fit=solver.train_regret
    original_evaluation=solver.evaluation_frequency
    original_checkpoint=solver._maybe_run_training_time_checkpoint
    solver.evaluation_frequency=0;solver._maybe_run_training_time_checkpoint=lambda:None
    records=[];fits=[]
    def observed_fit(player):
        trainer=solver.regret_trainers[player]
        with observe_regret_fit(trainer,solver.num_iteration) as record:
            original_fit(player)
        record["player"]=player;fits.append(record)
    solver.train_regret=observed_fit
    try:
        for index in range(config["continuation_iterations"]):
            if time.perf_counter()-began>config["continuation_cap_seconds"]:
                raise TimeoutError("Continuation safety cap; incomplete scientific stage")
            with diagnostic_threads():
                before=current_predictions(solver,panel)
            if hasattr(solver,"audit_merges"):
                solver.audit_merges.clear();solver.iteration();merges=list(solver.audit_merges)
            else:
                # Synthetic local smoke only; cloud smoke exercises real Ray actors.
                with observe_traversal(solver) as obs:
                    solver.iteration()
                merges=[dict(local_synthetic=True,visit_counts=dict(obs["counts"]),terminal_traces=obs["records"])]
            with diagnostic_threads():
                after=current_predictions(solver,panel)
            if solver._stop_requested:
                raise RuntimeError("Partial-iteration stop")
            size=min(solver.ave_policy_trainer.buffer.cur_id,solver.ave_policy_trainer.buffer.buffer_size)
            retained=Counter()
            for start in range(0,size,8192):
                retained.update(category_counts(solver.ave_policy_trainer.buffer.infostate_buf[start:min(size,start+8192)]))
            row=dict(index=index+1,iteration=solver.num_iteration,nodes=solver.nodes_touched,
                before=before,after=after,regret_fits=list(fits),merges=merges,
                retained_policy_categories=dict(retained),elapsed_seconds=time.perf_counter()-began)
            fits.clear();records.append({k:row[k] for k in ("index","iteration","nodes","elapsed_seconds")})
            write_json(directory/f"iteration_{index+1:02d}.json",row)
            # Small model/optimizer snapshots make prediction changes replayable;
            # these are not full resumable solver checkpoints.
            torch.save(dict(iteration=solver.num_iteration,
                regrets=[dict(model=t.model.state_dict(),optimizer=t.optimizer.state_dict()) for t in solver.regret_trainers]),
                directory/f"regrets_{index+1:02d}.pt")
            print(f"exp28 continuation {index+1}/{config['continuation_iterations']}",flush=True)
        result=dict(initial_iteration=initial,final_iteration=solver.num_iteration,
            complete_iterations=len(records),records=records,synthetic=synthetic,
            ray_actors=len(getattr(solver,"_workers",[])),wall_seconds=time.perf_counter()-began,
            diagnostic_samples_added=0,policy_refitting_during_continuation=False,
            retention_note="arrival acceptance, within-batch survival, and later retained counts are distinct")
        write_json(directory/"summary.json",result)
        complete(directory,smoke=config["smoke"],synthetic=synthetic)
        return result
    finally:
        solver.train_regret=original_fit;solver.evaluation_frequency=original_evaluation
        solver._maybe_run_training_time_checkpoint=original_checkpoint
