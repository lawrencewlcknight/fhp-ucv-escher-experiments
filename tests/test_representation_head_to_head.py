"""Protocol, provenance, recovery and native cross-repository integration."""
from copy import deepcopy
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.representation_head_to_head import engine as native, run, sources
from gcp import representation_head_to_head_batch as batch

ROOT = run.ROOT
WORKSPACE = ROOT.parents[1]
VR = WORKSPACE/"fhp_vr_deep/fhp-vr-deep-experiments"
SD = WORKSPACE/"fhp_deep_cfr/fhp-deep-cfr-experiments"


@pytest.fixture(params=["sd", "vr"])
def engine(request):
    # Pure scheduling/report functions do not initialise native packages in
    # pytest's shared process; real package isolation is tested separately.
    spec = importlib.util.spec_from_file_location("exp41_test_engine", VR/native.previous.ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native.configure(module, request.param)
    return module


def records(engine):
    a = engine.REPRESENTATION_ALGORITHM
    return [dict(id=f"{a}_{c}_{s}", algorithm=a, cohort=c, seed=s, training_hours=48, label=f"{a} {c}",
        path=f"/fixture/{a}_{c}_{s}", sha256=f"{a}_{c}_{s}", nodes_touched=(s+1)*1000,
        source_run=sources.PINS[a][c]["run_id"], active_seconds=172800)
        for c in ("rich", "original") for s in range(3)]


def fake_result(task, engine):
    row = dict(task=engine.portable(task), elapsed_seconds=.1, result=dict(
        num_deal_pairs=task["num_deals"], num_games=2*task["num_deals"], mean_chips_per_hand=.1,
        std_chips_per_pair=1., mean_mbb_per_hand=1., policy_a_player0_mean_chips=.2,
        policy_a_player1_mean_chips=0.))
    row["result_sha256"] = engine.digest(row)
    return row


def args(tmp_path, engine, **changes):
    return SimpleNamespace(algorithm=engine.REPRESENTATION_ALGORITHM, source_root=tmp_path/"input", output=tmp_path/"analysis", vr_repo=VR,
        sd_repo=SD, workers=2, max_hours=23., deadline_unix=None, source_uri=[], **changes)


def test_full_counts_and_fresh_prespecified_streams(engine):
    tasks = engine.build_tasks(records(engine), "production", "code")
    assert len(tasks) == 180 and len({t["task_id"] for t in tasks}) == 180
    assert sum(t["num_deals"] for t in tasks) == 900_000
    assert sum(t["kind"] == "direct" for t in tasks) == 180
    assert sum(t["kind"] == "rule" for t in tasks) == 0
    from collections import Counter
    budgets = Counter()
    for task in tasks:
        budgets[task["cell_id"]] += task["num_deals"]
    assert len(budgets) == 9 and set(budgets.values()) == {100_000}
    assert all(t["a"]["training_hours"] == 48 for t in tasks)
    smoke = engine.build_tasks(records(engine), "smoke", "code")
    profile = engine.build_tasks(records(engine), "profile", "code")
    assert len(smoke) == 9 and all(t["num_deals"] == 2 for t in smoke)
    assert {t["evaluation_seed"] for t in tasks}.isdisjoint(t["evaluation_seed"] for t in smoke+profile)
    engine.BASE_SEED = 640048
    old = engine.build_tasks(records(engine), "production", "code")
    assert {t["evaluation_seed"] for t in tasks}.isdisjoint(t["evaluation_seed"] for t in old)
    with pytest.raises(ValueError):
        engine.build_tasks(records(engine), "lbr", "code")


def test_reporting_independent_seed_uncertainty(engine, tmp_path):
    tasks = engine.build_tasks(records(engine), "production", "code")
    result = run.report([fake_result(t,engine) for t in tasks], tmp_path, records(engine), [], engine)
    assert result["experiment_id"] == engine.EXPERIMENT_ID
    assert result["exact_exploitability"] is False and result["lbr_evaluated"] is False
    assert result["original_player_features"] == 183 and result["rich_player_features"] == 231
    assert result["family_confidence"] == pytest.approx(.975)
    assert len(result["direct_comparisons"]) == 1
    assert all(r["conditional_mc_ci95_low"] < r["mean_mbb_per_hand"] < r["conditional_mc_ci95_high"]
               for r in result["direct_comparisons"])
    matrix = np.arange(3)[:,None]-np.arange(3)[None,:]
    lo, hi = engine.cluster_interval(matrix)
    assert lo < 0 < hi  # Independent algorithms are NOT paired by seed number.
    assert (tmp_path/"final_48h_head_to_head.png").is_file()
    with pytest.raises(ValueError, match="duplicate"):
        run.report([fake_result(t,engine) for t in tasks]+[fake_result(tasks[0],engine)],tmp_path,records(engine),[],engine)
    with pytest.raises(ValueError, match="Incomplete"):
        run.report([fake_result(t,engine) for t in tasks[:-1]], tmp_path, records(engine), [], engine)


class ImmediatePool:
    def __init__(self, **kw): self._processes = {}
    def submit(self, fn, task):
        f = Future()
        try: f.set_result(fn(task))
        except Exception as exc: f.set_exception(exc)
        return f
    def shutdown(self, **kw): pass


def test_shards_resume_and_reject_corruption(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ProcessPoolExecutor", ImmediatePool)
    tasks = engine.build_tasks(records(engine), "smoke", "code")[:3]
    options = args(tmp_path, engine, smoke_only=False)
    options.workers = 1
    def score(t):
        if t == tasks[1]: raise RuntimeError("interrupted")
        return fake_result(t,engine)
    monkeypatch.setattr(native, "execute_task", score)
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run_tasks(tasks,tmp_path/"shards",options,engine,time.monotonic()+30)
    assert len(list((tmp_path/"shards").glob("*.json"))) == 1
    seen = []
    monkeypatch.setattr(native, "execute_task", lambda t: (seen.append(t["task_id"]) or fake_result(t,engine)))
    result = run.run_tasks(tasks,tmp_path/"shards",options,engine,time.monotonic()+30)
    assert len(result) == 3 and tasks[0]["task_id"] not in seen
    path = tmp_path/"shards"/(tasks[0]["task_id"]+".json")
    path.write_text('{}')
    with pytest.raises(ValueError, match="Corrupt"):
        run.run_tasks(tasks,tmp_path/"shards",options,engine,time.monotonic()+30)


@pytest.mark.parametrize("smoke", [True,False])
def test_pipeline_does_not_abort_on_pessimistic_forecast(engine, tmp_path, monkeypatch, smoke):
    monkeypatch.setattr(native, "load", lambda *a: engine)
    monkeypatch.setattr(sources, "validate", lambda *a: records(engine))
    monkeypatch.setattr(sources, "cost_context", lambda *a: [])
    monkeypatch.setattr(run, "implementation_digest", lambda *a: "code")
    stages = []
    def score(tasks, *a):
        stages.append(tasks[0]["stage"])
        return [fake_result(t,engine) for t in tasks]
    monkeypatch.setattr(run, "run_tasks", score)
    monkeypatch.setattr(engine, "cost_estimate", lambda *a: dict(predicted_hours_with_2x_margin=100000))
    options = args(tmp_path, engine,smoke_only=smoke)
    run.run(options)
    success = engine.read_json(options.output/"SUCCESS.json")
    assert success["smoke"] == smoke and success["status"] == "complete"
    assert stages == (["smoke"] if smoke else ["smoke","profile","production"])
    if not smoke:
        assert engine.read_json(options.output/"summary.json")["hands"] == 1_800_000
        assert engine.read_json(options.output/"timing_forecast.json")["cost_gate_enabled"] is False
        from gcp import download_representation_analysis as downloader
        import shutil
        def copy(uri,target):
            shutil.copyfile(options.output/uri.rsplit("/",1)[1],target)
        downloaded=downloader.download("gs://test-bucket","representation-test",tmp_path/"download",copy)
        assert set(p.name for p in downloaded.iterdir()) == downloader.ALLOWED | {"SUCCESS.json"}
    stages.clear()
    run.run(options)
    assert not stages
    options.source_uri = ["changed"]
    with pytest.raises(ValueError,match="Incompatible"):
        run.run(options)



@pytest.mark.parametrize("algorithm,number", [("sd",42),("vr",43)])
def test_batch_and_launcher(tmp_path, algorithm, number):
    cli = ["--algorithm",algorithm,"--run-id",f"exp{number}-test","--bucket","gs://test-output",
        "--service-account","runner@test.iam.gserviceaccount.com","--repo-ref","a"*40,
        "--output",str(tmp_path/"job.json")]
    options = batch.parser().parse_args(cli)
    for smoke,resume in ((False,False),(True,False),(False,True)):
        options.smoke_only, options.resume = smoke,resume
        job = batch.build_job(options)
        spec = job["taskGroups"][0]["taskSpec"]
        assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "86400s"
        assert spec["runnables"][1]["alwaysRun"] is True
        assert job["labels"]["workload"] == f"fhp-exp{number}"
        for runnable in spec["runnables"]:
            text = runnable["script"]["text"]
            subprocess.run(["bash","-n"],input=text,text=True,check=True)
            assert "flock" in text and "SUCCESS[.]json" in text
        script = spec["runnables"][0]["script"]["text"]
        assert all(p["run_id"] in script for p in sources.PINS[algorithm].values())
        assert not any(p["run_id"] in script for p in sources.PINS["vr" if algorithm=="sd" else "sd"].values())
        assert f"--algorithm {algorithm}" in script and "--workers 16" in script
        assert ("--smoke-only" in script) == smoke
        assert "training_states/" not in script
        assert script.index("representation_head_to_head.sources") < script.index(batch.policy_download_script(options))
        groups = dict(batch.metadata_uris(options))
        assert set(groups) == {"original","rich"}
        assert all(len(v)==(9 if algorithm=="sd" else 12) for v in groups.values())
        if algorithm == "sd":
            assert "time_48h[.]json$|chunk_.*[.]npy$" in script
        else:
            assert "time_48h.pt" in script and "storage rsync --recursive --exclude='^(?!time" not in script
    subprocess.run([sys.executable,"-S",str(ROOT/"gcp/representation_head_to_head_batch.py"),*cli],check=True)
    subprocess.run(["bash","-n",str(ROOT/f"gcp/run_exp{number}_{algorithm}_representation_48h.sh")],check=True)
    options.run_id = sources.PINS[algorithm]["original"]["run_id"]
    with pytest.raises(ValueError): batch.build_job(options)


def test_missing_duplicate_or_wrong_cohort_rejected(engine):
    for rows in (records(engine)[:-1], records(engine)+[records(engine)[0]]):
        with pytest.raises(ValueError): engine.build_tasks(rows,"smoke","code")
    rows=records(engine); rows[0]["algorithm"]="ucv"
    with pytest.raises(ValueError): engine.build_tasks(rows,"smoke","code")


def test_contract_restored_after_failure(engine):
    original = engine.SOURCE_NAMES.copy()
    with pytest.raises(RuntimeError):
        with sources.cohort_contract(engine,engine.REPRESENTATION_ALGORITHM,"original"):
            raise RuntimeError("bad source")
    assert engine.SOURCE_NAMES == original


def test_native_loaders_and_spawn_in_fresh_process(tmp_path):
    result = subprocess.run([sys.executable,str(ROOT/"tests/representation_native_checks.py"),str(tmp_path)],
        cwd=ROOT,text=True,capture_output=True,timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "Both mixed-encoder native comparisons passed" in result.stdout


@pytest.mark.parametrize('mutation', [None, 'bytes', 'encoder', 'config', 'clock', 'endpoint', 'failure'])
def test_metadata_pins_and_configuration_audit(engine, tmp_path, monkeypatch, mutation):
    monkeypatch.syspath_prepend(str(VR))
    from fhp_vr_deep.game import FHP_GAME_PARAMETERS
    algorithm = engine.REPRESENTATION_ALGORITHM
    pins = deepcopy(sources.PINS)
    directories = {}
    for cohort in ('original', 'rich'):
        pin = pins[algorithm][cohort]
        config = {'fixed_setting': 10}
        if cohort == 'rich': config['feature_encoder_id'] = 'fhp_hand_board_strategic_ae_v1'
        pin['config_sha256'] = engine.canonical_config_hash(config)
        for seed in range(3):
            worker = tmp_path/cohort/algorithm/'workers'/f'task_{seed:03d}_{pin["source_names"][1]}_seed_{seed}'
            directories[cohort,seed] = worker
            manifest = dict(repository_commit=pin['source_commit'],experiment_name=pin['source_names'][0],
                algorithm_id=pin['source_names'][1],seed=seed,game={'parameters':dict(FHP_GAME_PARAMETERS)},
                reference_vm={'machine_type':'n2-standard-16'},config=config,
                feature_encoder=deepcopy(pin['feature_encoder']),training_clock='fixed active time',torch_threads=1)
            rows = [dict(checkpoint_target_hours=h,actual_training_elapsed_seconds=h*3600,nodes_touched=h,
                         outer_iteration=h) for h in range(6,49,6)]
            files={'run_manifest.json':manifest,'checkpoint_manifest.json':rows,'SUCCESS.json':{},'summary.json':{}}
            for name in pin['metadata'][str(seed)]:
                engine.write_json(worker/name,files[name])
                pin['metadata'][str(seed)][name] = engine.sha256(worker/name)
    if mutation:
        worker = directories['rich',0]
        name = 'checkpoint_manifest.json' if mutation=='endpoint' else 'run_manifest.json'
        data = engine.read_json(worker/name)
        if mutation=='bytes': data['seed']=99
        elif mutation=='encoder': data['feature_encoder']['policy_layout']['total_size']=183
        elif mutation=='config':
            data['config']['fixed_setting']=20
            pins[algorithm]['rich']['config_sha256']=engine.canonical_config_hash(data['config'])
            # Keep each cohort internally consistent, but change more than representation.
            for seed in (1,2):
                other=directories['rich',seed]/name
                m=engine.read_json(other);m['config']['fixed_setting']=20
                engine.write_json(other,m)
                pins[algorithm]['rich']['metadata'][str(seed)][name]=engine.sha256(other)
        elif mutation=='clock': data['training_clock']='different clock'
        elif mutation=='endpoint': data[-1]['checkpoint_target_hours']=42
        elif mutation=='failure': engine.write_json(worker/'FAILURE.json',{})
        engine.write_json(worker/name,data)
        if mutation!='bytes': pins[algorithm]['rich']['metadata']['0'][name]=engine.sha256(worker/name)
    monkeypatch.setattr(sources,'PINS',pins)
    if mutation:
        with pytest.raises(ValueError): sources.validate_metadata(tmp_path,engine)
    else:
        audit=sources.validate_metadata(tmp_path,engine)
        assert len(audit)==6 and {r['cohort'] for r in audit}=={'rich','original'}


@pytest.mark.parametrize('algorithm',["sd","vr"])
def test_preflight_only_needs_selected_source_bucket(tmp_path, algorithm):
    from gcp import representation_access_preflight as preflight
    options=batch.parser().parse_args(['--algorithm',algorithm,'--run-id','representation-test',
        '--bucket','gs://output-bucket','--service-account','runner@test.iam.gserviceaccount.com',
        '--repo-ref','a'*40,'--output',str(tmp_path/'job.json')])
    seen=[]
    def cloud(arguments, project):
        seen.append(arguments)
        if arguments[:3]==['storage','buckets','get-iam-policy']:
            return {'bindings':[{'role':'roles/storage.objectAdmin',
                'members':['serviceAccount:'+options.service_account]}]}
        return [{'type':'cloud_object','url':uri} for uri in arguments[3:]]
    receipt=preflight.check(batch,options,'project',cloud)
    assert len(receipt['bucket_grants'])==2
    assert len(receipt['source_metadata'])==2
    assert {r['bucket'] for r in receipt['bucket_grants']}=={options.bucket,getattr(options,algorithm+'_bucket')}


@pytest.mark.parametrize('problem',['smoke','extra','checksum'])
def test_analysis_downloader_rejects_unverified_outputs(tmp_path,problem):
    from gcp import download_representation_analysis as downloader
    import hashlib
    receipt=dict(experiment_id=42,status='complete',smoke=False,duplicate_pairs=900_000,
        files={name:hashlib.sha256(b'{}').hexdigest() for name in downloader.ALLOWED})
    if problem=='smoke':receipt['smoke']=True
    if problem=='extra':receipt['files']['model.pt']='0'*64
    if problem=='checksum':receipt['files']['summary.json']='0'*64
    seen=[]
    def copy(uri,target):
        seen.append(uri)
        target.write_text(json.dumps(receipt) if uri.endswith('/SUCCESS.json') else '{}')
    with pytest.raises(ValueError):
        downloader.download('gs://test-bucket','representation-test',tmp_path/'output',copy)
    assert not (tmp_path/'output').exists()
    assert not any(uri.endswith('model.pt') for uri in seen)
