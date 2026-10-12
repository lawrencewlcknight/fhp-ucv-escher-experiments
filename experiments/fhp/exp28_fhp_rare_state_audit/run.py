"""Exp28 offline audit -> refits -> passive continuation; source files read-only."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from .common import write_json, file_hash, fresh, complete, revision, verify, timed, digest
from .config import contract, SOURCES, SEEDS, TRAINING_REF


def runtime(smoke=False):
    import torch
    from .source import validate_runtime
    if not smoke:
        validate_runtime()
        import importlib.metadata
        if importlib.metadata.version("open_spiel")!="1.6.3":
            raise ValueError("Pinned OpenSpiel 1.6.3 required")
    torch.set_num_threads(1 if smoke else 8)
    if not smoke and torch.get_num_interop_threads()!=8:
        torch.set_num_interop_threads(8)


def attach_policy(solver,payload):
    solver.ave_policy_trainer.model.load_state_dict(payload["average_policy_trainer"]["model"])
    solver.ave_policy_trainer.optimizer.load_state_dict(payload["average_policy_trainer"]["optimizer"])
    return solver


def offline(source,seed,hours,panel,config,root,*,fixture=None):
    import torch
    from experiments.fhp.exp22_fhp_critic_objectives.source import extract_verified_snapshot
    from .source import load
    from .replay import group_replay,coverage,critic_coverage
    from .diagnostics import audit_case_predictions,estimator_probe,diagnostic_threads
    from .fitting import refit
    directory=fresh(root);costs={}
    with timed(costs,"source_load_seconds"):
        payload,template,provenance=load(source,seed,hours) if fixture is None else fixture
        solver=attach_policy(extract_verified_snapshot(payload),payload)
    # The previous audit's untouched boards are not consumed. Our new holdout
    # is query-locked until AFTER every refit has been saved.
    development=dict(panel,rows=[r for r in panel["rows"] if r["partition"]=="development"])
    write_json(directory/"manifest.json",dict(seed=seed,hours=hours,config=config,source=provenance,
        commit=revision(),synthetic=fixture is not None,panel_sha256=digest(panel)))
    with timed(costs,"replay_grouping_seconds"):
        buffer=payload["average_policy_trainer"]["buffer"]
        groups=group_replay(buffer,payload["solver"]["num_iteration"],payload["config"]["gamma"])
        covered=coverage(groups,development,buffer["cur_id"])
    with timed(costs,"critic_replay_scan_seconds"):
        critic=critic_coverage(payload)
    with timed(costs,"case_prediction_seconds"), diagnostic_threads():
        predictions=audit_case_predictions(solver,development,covered)
    with timed(costs,"estimator_probe_seconds"), diagnostic_threads():
        probe=estimator_probe(solver,development,config,seed)
    write_json(directory/"coverage.json",covered)
    write_json(directory/"critic_coverage.json",critic)
    write_json(directory/"case_predictions.json",predictions)
    write_json(directory/"estimator_probe.json",probe)
    fitted=None
    if hours==48:
        with timed(costs,"refits_and_assessment_seconds"):
            fitted=refit(solver,payload,template,groups,panel,config,seed,directory/"refits")
    result=dict(seed=seed,hours=hours,smoke=config["smoke"],source=provenance,costs=costs,
        coverage={k:v for k,v in covered.items() if k!="exact"},development=predictions["metrics"],refitting=fitted,
        estimator_expectation_verified=True,synthetic=fixture is not None,
        peak_driver_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=="darwin" else 1024))
    write_json(directory/"summary.json",result)
    complete(directory,seed=seed,hours=hours,smoke=config["smoke"],synthetic=fixture is not None)
    return result


def continuation(source,seed,panel,config,root):
    from experiments.fhp.exp10_fhp_hand_board_features import config as baseline,training_state as ts
    from .source import load
    from .continuation import make_solver,run_continuation
    payload,_,provenance=load(source,seed,48)
    solver=make_solver(seed,baseline.EXPERIMENT_CONFIG)
    try:
        ts.restore_training_state(solver,payload,seed=seed,repository_commit=TRAINING_REF,config=baseline.EXPERIMENT_CONFIG)
        del payload;gc.collect()
        if config["smoke"]:
            # Real dimensions, source, all eight actors; explicitly non-scientific workload.
            solver.num_traversals=16
            for t in solver.regret_trainers:t.train_steps=2
            solver.calibration_trainer.train_steps=2
            for m in solver.q_value_trainer.members:m.train_steps=2
        write_json(Path(root).parent/"continuation_source.json",provenance)
        return run_continuation(solver,panel,config,root)
    finally:
        solver.close()


def gameplay(root,source,seed,config,opponent=None):
    """Modest guardrail, not a powered demonstration of equilibrium improvement."""
    from fhp_escher.game import load_fhp_game
    from fhp_escher.checkpointing import LoadedFHPPolicy
    from fhp_evaluation.duplicate import evaluate_duplicate_match
    from .source import validate
    import torch
    torch.set_num_threads(1)  # separate inference-only subprocess
    root=Path(root);refits=root/"offline_48h/refits"
    verify(root/"offline_48h")
    # The source .pkl is read-only, and was hash-verified in source extraction.
    _,path,_=validate(source,seed,48)
    game=load_fhp_game();original=LoadedFHPPolicy(game,path)
    class Uniform:
        def action_probabilities(self,state,player_id=None):
            legal=state.legal_actions();return {a:1/len(legal) for a in legal}
    panel={"original":original,"uniform":Uniform()}
    if opponent:
        from .source import OPPONENT_SHA256
        if file_hash(opponent)!=OPPONENT_SHA256:
            raise ValueError("Guardrail opponent differs from fixed policy")
        panel["frozen_exp9_seed0_24h"]=LoadedFHPPolicy(game,opponent)
    candidates=json.loads((refits/"candidates.json").read_text())
    chosen=[r for r in candidates if r["updates"]==config["updates"][-1]]
    rows=[];started=time.perf_counter()
    policies=[("unchanged",original)]+[(r["name"],LoadedFHPPolicy(game,refits/(r["name"]+".pkl"))) for r in chosen]
    for name,policy in policies:
        for index,(label,target) in enumerate(panel.items()):
            if name=="unchanged" and label=="original":continue
            # Same deals/action seed across interventions for each source and opponent.
            result=evaluate_duplicate_match(game,policy,target,num_deals=config["gameplay_deal_pairs"],
                seed=config["random_seed"]+70000+100*seed+index,policy_a_name=name,policy_b_name=label)
            rows.append(result.to_dict())
            write_json(root/"gameplay.json",dict(seed=seed,rows=rows,complete=False))
    write_json(root/"gameplay.json",dict(seed=seed,rows=rows,complete=True,seconds=time.perf_counter()-started,
        scope="fixed final-update endpoints; all replicates, no assessment-selected winner; exploratory guardrail"))


def synthetic_fixture(seed):
    from experiments.fhp.exp17_fhp_critic_budget.run import synthetic_solver
    from experiments.fhp.exp10_fhp_hand_board_features import config as baseline
    from experiments.fhp.exp2_fhp_lossless_structured_ucv.training_state import build_training_state
    from fhp_escher.checkpointing import save_policy_checkpoint,load_checkpoint_payload
    import tempfile
    solver=synthetic_solver(seed);solver.train_average_policy()
    config=baseline.smoke_config()
    state=build_training_state(solver,seed=seed,checkpoint_id="synthetic",repository_commit=revision(),config=config,captured_checkpoints=[])
    with tempfile.TemporaryDirectory(prefix="exp28-policy-") as tmp:
        path=Path(tmp)/"fixture.pkl"
        save_policy_checkpoint(solver,path,seed=seed,config=config,checkpoint_row={})
        template=load_checkpoint_payload(path)
    return solver,(state,template,dict(synthetic=True,seed=seed))


def local_smoke(root):
    from .panels import build_panel
    from .continuation import run_continuation
    from .aggregate import aggregate
    root=fresh(root);config=contract(True);runtime(True)
    panel=build_panel({},config);write_json(root/"panel.json",panel)
    write_json(root/"protocol.json",dict(config=config,commit=revision(),synthetic=True))
    solver,fixture=synthetic_fixture(0)
    for hours in (24,48):
        offline(None,0,hours,panel,config,root/f"workers/seed_0/offline_{hours}h",fixture=fixture)
    run_continuation(solver,panel,config,root/"workers/seed_0/continuation",synthetic=True)
    aggregate(root,smoke=True,local=True)
    return root


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("smoke-local","fetch","fetch-opponent","panel","offline","continue","gameplay","aggregate","preflight"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source",type=Path)
    p.add_argument("--root",type=Path)
    p.add_argument("--bucket")
    p.add_argument("--seed",type=int,choices=SEEDS,default=0)
    p.add_argument("--hours",type=int,choices=(24,48),default=48)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--opponent",type=Path)
    p.add_argument("--pilot",action="store_true",help="Validate seed 0 and write an intermediate report; not final success")
    args=p.parse_args(argv);cfg=contract(args.smoke)
    try:
        if args.action=="smoke-local":
            local_smoke(args.output);return
        if args.action=="fetch":
            from .source import fetch
            fetch(args.bucket,args.seed,args.hours,args.output);return
        if args.action=="fetch-opponent":
            from .source import fetch_opponent
            fetch_opponent(args.bucket,args.output);return
        if args.action=="preflight":
            from .source import metadata
            rows=[]
            for seed in cfg["seeds"]:
                for hours in SOURCES:
                    uri,_,data=metadata(args.bucket,seed,hours)
                    final=data["checkpoint_manifest.json"][-1]
                    obj=json.loads(subprocess.check_output(["gcloud","storage","objects","describe",
                        uri+"/"+final["training_state_path"],"--format=json"]))
                    if int(obj.get("size",0))<=0:raise ValueError("Missing source state")
                    rows.append(dict(seed=seed,hours=hours,state_sha256=final["training_state_sha256"],size=obj["size"]))
            write_json(args.output,dict(sources=rows));return
        if args.action=="panel":
            from .source import fetch_library
            from .panels import build_panel
            root=fresh(args.output);started=time.perf_counter()
            library=fetch_library(args.bucket,root/"prior_library.json")
            write_json(root/"panel.json",build_panel(library,cfg))
            write_json(root/"protocol.json",dict(config=cfg,commit=revision(),synthetic=False,
                prior_library_sha256=file_hash(root/"prior_library.json"),panel_creation_seconds=time.perf_counter()-started));return
        if args.action=="aggregate":
            from .aggregate import aggregate
            aggregate(args.output,smoke=args.smoke,pilot=args.pilot);return
        if args.source is None or args.root is None:
            p.error("offline/continue/gameplay require --source and --root")
        runtime(False)  # Real-source cloud smoke also checks the pinned runtime.
        panel=json.loads((args.root/"panel.json").read_text())
        if panel["protocol"]!=json.loads(json.dumps(cfg)):
            raise ValueError("Frozen panel protocol differs")
        if args.action=="offline":offline(args.source,args.seed,args.hours,panel,cfg,args.output)
        elif args.action=="continue":continuation(args.source,args.seed,panel,cfg,args.output)
        else:gameplay(args.output,args.source,args.seed,cfg,args.opponent)
    except BaseException as error:
        if args.action not in ("preflight",):
            failure=args.output.with_name(args.output.name+".failure.json") if args.action=="fetch-opponent" else args.output/"failure.json"
            write_json(failure,dict(action=args.action,error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    main()
