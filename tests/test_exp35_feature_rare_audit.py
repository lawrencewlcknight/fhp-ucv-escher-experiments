"""Read-only inference, source identity, fresh boards and recoverable Batch plumbing."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pickle
import subprocess

import numpy as np
import pytest
import torch

from experiments.fhp.exp35_fhp_feature_rare_audit import audit, panel, run
from experiments.fhp.exp28_fhp_rare_state_audit.common import digest, file_hash, write_json
from experiments.fhp.exp28_fhp_rare_state_audit.panels import from_case, canonical_board
from fhp_escher.features import StructuredFHPMLP, make_feature_encoder
from unbiased_escher.fhp_structured_solver import StructuredFHPRegretTrainer
from gcp.exp35_feature_rare_audit_batch import build_job


@pytest.fixture(scope='module')
def cases():
    torch.set_num_threads(1)
    return panel.build_panel(run.inputs(),smoke=True)


def test_sources_are_complete_and_fixed():
    sources=run.sources()
    assert {(s['experiment'],s['seed']) for s in sources}=={
        (e,s) for e in (10,29,30,31,32,33) for s in range(3)}
    assert len(run.sources(True))==6
    for s in sources:
        assert len(s['state_sha256'])==len(s['policy_sha256'])==64
        assert 'time_24h' in s['state'] and s['state_bytes']>5_000_000_000


def test_fresh_boards_and_known_examples(cases):
    prior=set(map(tuple,run.inputs()['prior_board_classes']))
    fresh=[r for r in cases['rows'] if r['partition']=='assessment']
    assert not prior & {tuple(r['board_class']) for r in fresh}
    assert {r['category'] for r in fresh if r['strict_winner']}=={7,8}
    assert len({r['id'] for r in cases['rows']})==len(cases['rows'])
    dev=[r for r in cases['rows'] if r['partition']=='development']
    for hand in [(2,14),(12,25),(25,48)]:
        assert any(tuple(r['hand'])==hand and r['strict_winner'] for r in dev)
    for r in cases['rows']:
        state=from_case(r)
        assert state.legal_actions()==[0,1] and state.child(1).is_terminal()
        if r['strict_winner']:
            gap=(state.child(1).returns()[r['seat']]-state.child(0).returns()[r['seat']])/100
            assert gap==r['guaranteed_call_gain_bb']>0


@pytest.mark.parametrize('experiment',[10,29,30,31,32,33])
def test_all_encoders_match_native_regret_policy_and_do_not_leak(experiment,cases):
    cfg=audit.expected_config(experiment)
    encoder=make_feature_encoder(cfg['feature_encoder_id'])
    trainers=[StructuredFHPRegretTrainer(encoder.policy_size,3,[128,128],.001,1,1,1,None,True,
        'cpu',2.3,feature_encoder=encoder,branch_width=64,replay_seed=s) for s in range(2)]
    chosen=[]
    for seat in (0,1):
        for cat in (7,8):
            chosen.append(next(r for r in cases['rows'] if r['seat']==seat and r['category']==cat))
    small=dict(cases,rows=chosen)
    _,states,f,m=audit.panel_arrays(small,encoder)
    scores,p=audit.current_predictions([t.model for t in trainers],f,m,[r['seat'] for r in chosen],True)
    for row,st,actual in zip(chosen,states,p):
        np.testing.assert_allclose(actual,trainers[row['seat']].get_policy(st,10),atol=1e-6)
        cards=[c for c in range(52) if c not in row['hand']+row['board']]
        alternate=from_case(row,cards[-2:])
        np.testing.assert_array_equal(encoder.information_state(st),encoder.information_state(alternate))
    policy=SimpleNamespace(feature_encoder=encoder,
        model=StructuredFHPMLP(encoder.policy_layout,[192,192],3,branch_width=64).eval())
    before={k:v.clone() for k,v in policy.model.state_dict().items()}
    result=audit.measure(policy,[t.model for t in trainers],small,cfg)
    assert set(result['metrics'])=={'deployed','current_regret'}
    assert all(torch.equal(v,policy.model.state_dict()[k]) for k,v in before.items())


def test_native_regret_fallbacks():
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.tensor([[-2.,-1.,999.]]).repeat(len(x),1)
    f=np.zeros((2,213),np.float32);m=np.array([[1,1,0]]*2,np.float32)
    _,p=audit.current_predictions([Constant(),Constant()],f,m,[0,1],True)
    np.testing.assert_array_equal(p,[[0,1,0]]*2)
    _,p=audit.current_predictions([Constant(),Constant()],f,m,[0,1],False)
    np.testing.assert_array_equal(p,[[.5,.5,0]]*2)


@pytest.mark.parametrize('experiment',[10,29,30,31,32,33])
def test_checkpoint_roundtrip_actual_worker_path(experiment,cases,tmp_path):
    from fhp_escher.checkpointing import CHECKPOINT_TYPE,CHECKPOINT_VERSION
    from fhp_escher.game import serialisable_game_definition
    s=next(r for r in run.sources() if r['experiment']==experiment and r['seed']==0)
    cfg=audit.expected_config(experiment);encoder=make_feature_encoder(cfg['feature_encoder_id'])
    model=StructuredFHPMLP(encoder.policy_layout,[192,192],3,branch_width=64)
    policy=dict(type=CHECKPOINT_TYPE,version=CHECKPOINT_VERSION,game=serialisable_game_definition(),
        seed=0,checkpoint_target_seconds=86400,training_config=cfg,feature_encoder=encoder.metadata(),
        outer_iteration=s['iteration'],nodes_touched=s['nodes'],policy_state_dict=model.state_dict(),
        policy_model=model.checkpoint_metadata(),input_size=encoder.policy_size,num_actions=3,
        policy_network_layers=[192,192])
    state=dict(seed=0,checkpoint_id='time_24h',config=cfg,feature_encoder=encoder.metadata(),
        solver=dict(num_iteration=s['iteration'],nodes_touched=s['nodes']),
        average_policy_trainer={'model':model.state_dict()},
        regret_trainers=[{'model':StructuredFHPMLP(encoder.policy_layout,[128,128],3,
                        branch_width=64).state_dict()} for _ in range(2)])
    with (tmp_path/'policy.pkl').open('wb') as stream:pickle.dump(policy,stream)
    torch.save(state,tmp_path/'state.pt')
    before=[file_hash(tmp_path/p) for p in ['policy.pkl','state.pt']]
    small=dict(cases,rows=[r for r in cases['rows'] if r['strict_winner']][:4])
    result=audit.audit_source(s,tmp_path,small)
    assert result['source']==s and len(result['rows'])==4
    assert before==[file_hash(tmp_path/p) for p in ['policy.pkl','state.pt']]


def test_new_board_certificates_against_all_1081_hidden_hands(cases):
    from itertools import combinations
    for cat in (7,8):
        r=next(r for r in cases['rows'] if r['partition']=='assessment' and r['category']==cat and r['strict_winner'])
        hands=list(combinations([c for c in range(52) if c not in r['board']+r['hand']],2))
        assert len(hands)==1081
        for hand in hands:
            state=from_case(r,hand)
            assert state.child(1).returns()[r['seat']]>0
            assert (state.child(1).returns()[r['seat']]-state.child(0).returns()[r['seat']])/100==r['guaranteed_call_gain_bb']


def test_pair_rejects_wrong_horizon_encoder_and_weights():
    s=run.sources()[0];cfg=audit.expected_config(s['experiment'])
    policy=dict(seed=s['seed'],checkpoint_target_seconds=86400,training_config=cfg,
        feature_encoder={'test':True},outer_iteration=s['iteration'],nodes_touched=s['nodes'],
        policy_state_dict={'a':torch.ones(1)})
    state=dict(seed=s['seed'],checkpoint_id='time_24h',config=cfg,feature_encoder=policy['feature_encoder'],
        solver=dict(num_iteration=s['iteration'],nodes_touched=s['nodes']),
        average_policy_trainer={'model':{'a':torch.ones(1)}},regret_trainers=[{},{}])
    audit.validate_pair(state,policy,s)
    for change in [dict(checkpoint_id='time_48h'),dict(feature_encoder={}),
                   dict(average_policy_trainer={'model':{'a':torch.zeros(1)}})]:
        with pytest.raises(ValueError):audit.validate_pair(dict(state,**change),policy,s)


def test_aggregate_seed_unit_and_resume_receipts(tmp_path):
    for source in run.sources():
        directory=tmp_path/'models'/run.key(source)
        metric=dict(positions=100,mean_fold_probability=.1 if source['experiment']==10 else .05,
            mean_gap_bb=1.,p95_gap_bb=2.,max_gap_bb=3.,fraction_fold_over_half=.1)
        result=dict(source=source,metrics={t:{'assessment/all':metric} for t in ['deployed','current_regret']},
            download_seconds=0.,source_load_seconds=0.,inference_seconds=0.)
        write_json(directory/'result.json',result)
        write_json(directory/'SUCCESS.json',dict(fingerprint='f',result_sha256=file_hash(directory/'result.json')))
    run.aggregate(tmp_path,run.sources(),'f',False)
    summary=json.loads((tmp_path/'summary.json').read_text())
    primary=summary['comparisons'][0]['candidate_minus_exp10']
    assert primary['seeds']==3 and primary['mean']==pytest.approx(-.05)
    assert json.loads((tmp_path/'SUCCESS.json').read_text())['models']==18
    with pytest.raises(ValueError):run.verified_result(tmp_path,run.sources()[0],'wrong')
    path=tmp_path/'models'/run.key(run.sources()[0])/'result.json'
    path.write_text('{}')
    with pytest.raises(ValueError):run.aggregate(tmp_path,run.sources(),'f',False)


def test_batch_single_vm_no_learning_or_short_timeout(tmp_path):
    args=SimpleNamespace(kind='run',run_id='exp35-rare-test',bucket_root='gs://test-bucket',
        service_account='test@example.com',repo_ref='a'*40,project_id='test-project',region='europe-west1',
        repo_url='https://github.com/example/repo',resume_run_id='exp35-rare-previous')
    job=build_job(args);group=job['taskGroups'][0]
    assert group['taskCount']==group['parallelism']==1
    assert group['taskSpec']['maxRunDuration']=='43200s'
    assert job['allocationPolicy']['instances'][0]['policy']['machineType']=='n2-standard-16'
    script=group['taskSpec']['runnables'][0]['script']['text']
    assert 'exp35-rare-previous/analysis' in script and 'timeout ' not in script
    assert '--smoke' not in script and 'exp35_fhp_feature_rare_audit.run run' in script
    for i,r in enumerate(group['taskSpec']['runnables']):
        p=tmp_path/f'script{i}.sh';p.write_text(r['script']['text'])
        subprocess.run(['bash','-n',str(p)],check=True)


def test_bucket_validation():
    assert run.bucket_root('gs://test-bucket/')=='gs://test-bucket'
    for bad in ['gs://bucket/path','/tmp/foo','gs://x; echo bad']:
        with pytest.raises(ValueError):run.bucket_root(bad)
