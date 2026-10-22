"""Isolated real-OpenSpiel integration, including spawned native policy loading."""
from pathlib import Path
import sys
from types import SimpleNamespace
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(output):
    from experiments.fhp.exp40_fhp_threeway_48h import engine as native, run
    workspace = ROOT.parents[1]
    vr = workspace/"fhp_vr_deep/fhp-vr-deep-experiments"
    sd = workspace/"fhp_deep_cfr/fhp-deep-cfr-experiments"
    engine = native.load(vr,sd,ROOT)
    import torch
    from fhp_escher.checkpointing import save_policy_checkpoint
    from fhp_escher.features import make_feature_encoder
    from fhp_vr_deep.features import FHPFeatureEncoder
    from vr_deep_cfr.solver import MLP
    from vr_deep_cfr.policy_snapshots import save_policy_snapshot
    from deep_cfr_poker.game import load_fhp_game
    from deep_cfr_poker.networks import build_network
    from deep_cfr_poker.fhp_features import FHPFeatureEncoder as SDEncoder
    from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy
    torch.set_num_threads(1)
    game = load_fhp_game()
    paths = {}
    for algorithm,encoder in (("ucv",make_feature_encoder("fhp_hand_board_strategic_ae_v1")),
                              ("vr",FHPFeatureEncoder())):
        size = encoder.policy_layout.total_size
        model = MLP(size,[8],3)
        solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=model),feature_encoder=encoder,
            num_iteration=1,episode=1,nodes_touched=1,infostate_size=size,action_size=3,network_layers=[8])
        if algorithm == "ucv":
            paths[algorithm] = save_policy_checkpoint(solver,output/"ucv.pkl",seed=0,config={},checkpoint_row={})
        else:
            paths[algorithm] = save_policy_snapshot(solver,output/"vr.pt",seed=0,config={},algorithm_id="fixture",algorithm_label="fixture")
    network_type = "residual_layer_norm_centered_advantage_mlp"
    model = build_network(network_type,183,[32]*8,3)
    solver = SimpleNamespace(archive=SimpleNamespace(metadata={"feature_encoder":SDEncoder().metadata()}),_num_players=2,
        _advantage_networks=[model,model],_game=game,_num_actions=3,_embedding_size=183,
        _advantage_network_type=network_type,_advantage_network_layers=[32]*8)
    archive = DiskSDCFRArchive(solver,output/"archive",chunk_iterations=2)
    for iteration in range(1,4):
        with torch.no_grad():
            next(model.parameters()).add_(.01)
        for player in range(2): archive.capture_from_solver(solver,player,iteration)
    paths["sd"] = archive.checkpoint(output/"archive/final.json")
    reader = DiskArchiveReader(paths["sd"],game)
    policy = DiskSampledPolicy(reader)
    selected = set()
    for seed in range(100):
        policy.begin_episode(seed=seed)
        selected.update(policy.selected_iterations.values())
    assert selected == {1,2,3}  # The archive is not reduced to its last member.
    records = [dict(algorithm=a,seed=s,label=a,path=str(paths[a]),sha256=engine.sha256(paths[a]),
        training_hours=48,nodes_touched=1) for a in engine.ALGORITHMS for s in range(3)]
    tasks = engine.build_tasks(records,"smoke","native-test")
    options = SimpleNamespace(workers=2,vr_repo=vr,sd_repo=sd)
    results = run.run_tasks(tasks,output/"tasks",options,engine,time.monotonic()+120)
    assert len(results) == 72
    again = run.run_tasks(tasks,output/"tasks",options,engine,time.monotonic()+120)
    assert sorted(r["result_sha256"] for r in results) == sorted(r["result_sha256"] for r in again)
    assert all(r["result"]["num_games"] == 4 for r in results)
    print("Native three-way smoke and cache reuse passed")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
