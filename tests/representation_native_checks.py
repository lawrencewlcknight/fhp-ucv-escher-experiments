"""Exercise both encoders in real OpenSpiel matches, including spawn and SD history."""
from pathlib import Path
import sys
from types import SimpleNamespace
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(output):
    from experiments.fhp.representation_head_to_head import engine as native, run
    workspace = ROOT.parents[1]
    vr = workspace/'fhp_vr_deep/fhp-vr-deep-experiments'
    sd = workspace/'fhp_deep_cfr/fhp-deep-cfr-experiments'
    engine = native.load(vr, sd, ROOT, 'sd')
    import torch
    from fhp_vr_deep.rich_ae import make_encoder as vr_rich
    from fhp_vr_deep.features import FHPFeatureEncoder as VROld
    from vr_deep_cfr.solver import MLP
    from vr_deep_cfr.policy_snapshots import save_policy_snapshot
    from deep_cfr_poker.game import load_fhp_game
    from deep_cfr_poker.networks import build_network
    from deep_cfr_poker.rich_ae import make_encoder as sd_rich
    from deep_cfr_poker.fhp_features import FHPFeatureEncoder as SDOld
    from deep_cfr_poker.sd_cfr_disk import DiskSDCFRArchive, DiskArchiveReader, DiskSampledPolicy
    torch.set_num_threads(1)
    game = load_fhp_game()
    paths = {}
    readers = []
    for cohort, encoder in [('original', VROld()), ('rich', vr_rich())]:
        size = encoder.metadata()["policy_layout"]["total_size"]
        model = MLP(size, [8], 3)
        solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=model),feature_encoder=encoder,
            num_iteration=1,episode=1,nodes_touched=1,infostate_size=size,action_size=3,network_layers=[8])
        paths['vr',cohort] = save_policy_snapshot(solver,output/f'vr-{cohort}.pt',seed=0,config={},
            algorithm_id='fixture',algorithm_label='fixture')
    for cohort, encoder in [('original', SDOld()), ('rich', sd_rich())]:
        size = encoder.metadata()["policy_layout"]["total_size"]
        kind = 'residual_layer_norm_centered_advantage_mlp'
        model = build_network(kind,size,[32]*8,3)
        solver = SimpleNamespace(archive=SimpleNamespace(metadata={'feature_encoder':encoder.metadata()}),
            _num_players=2,_advantage_networks=[model,model],_game=game,_num_actions=3,
            _embedding_size=size,_advantage_network_type=kind,_advantage_network_layers=[32]*8)
        directory = output/f'sd-{cohort}'
        archive = DiskSDCFRArchive(solver,directory,chunk_iterations=2)
        for iteration in range(1,4):
            with torch.no_grad(): next(model.parameters()).add_(.01)
            for player in range(2): archive.capture_from_solver(solver,player,iteration)
        paths['sd',cohort] = archive.checkpoint(directory/'final.json')
        reader = DiskArchiveReader(paths['sd',cohort],game)
        readers.append(reader)
        policy = DiskSampledPolicy(reader)
        selected = set()
        for seed in range(100):
            policy.begin_episode(seed=seed)
            selected.update(policy.selected_iterations.values())
        assert selected == {1,2,3} and reader.count == 3
        assert reader.feature_encoder.metadata()["policy_layout"]["total_size"] == size
        policy.begin_episode(seed=17)
        chosen = dict(policy.selected_iterations)
        state = game.new_initial_state()
        while state.is_chance_node(): state.apply_action(state.chance_outcomes()[0][0])
        policy.action_probabilities(state)
        policy.action_probabilities(state)
        assert policy.selected_iterations == chosen
    all_seeds = set()
    for algorithm in ('sd','vr'):
        native.configure(engine, algorithm)
        records = [dict(algorithm=algorithm,cohort=c,seed=s,label=c,path=str(paths[algorithm,c]),
            sha256=engine.sha256(paths[algorithm,c]),training_hours=48,nodes_touched=1)
            for c in ('rich','original') for s in range(3)]
        tasks = engine.build_tasks(records,'smoke','native-test')
        assert all_seeds.isdisjoint(t['evaluation_seed'] for t in tasks)
        all_seeds.update(t['evaluation_seed'] for t in tasks)
        options = SimpleNamespace(workers=2,vr_repo=vr,sd_repo=sd)
        results = run.run_tasks(tasks,output/algorithm,options,engine,time.monotonic()+120)
        assert len(results) == 9
        direct = engine.execute_task(tasks[0])
        spawned = next(r for r in results if r['task']['task_id']==tasks[0]['task_id'])
        assert direct['result'] == spawned['result']
        again = run.run_tasks(tasks,output/algorithm,options,engine,time.monotonic()+120)
        assert sorted(r['result_sha256'] for r in results)==sorted(r['result_sha256'] for r in again)
        assert all(r['result']['num_games']==4 for r in results)
    for reader in readers:
        chunk=reader.path.parent/reader.chunks[0]['path']
        chunk.rename(chunk.with_suffix('.missing'))
        try: DiskArchiveReader(reader.path,game)
        except (OSError,ValueError): pass
        else: raise AssertionError('Missing historical strategy chunk accepted')
    print('Both mixed-encoder native comparisons passed')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
