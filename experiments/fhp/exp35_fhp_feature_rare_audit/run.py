"""Single-VM, restartable audit of 18 existing 24h model pairs; never trains."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from experiments.fhp.exp28_fhp_rare_state_audit.common import write_json, digest, file_hash, revision

HERE = Path(__file__).resolve().parent
MODULE = 'experiments.fhp.exp35_fhp_feature_rare_audit.run'


def inputs():
    return json.loads((HERE/'inputs.json').read_text())


def sources(smoke=False):
    return [s for s in inputs()['sources'] if not smoke or s['seed']==0]


def bucket_root(value):
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+/?', value):
        raise ValueError('Expected a GCS bucket root, without subdirectories')
    return value.rstrip('/')


def remote_source(bucket, source, field):
    return f"{bucket_root(bucket)}/{source['run']}/workers/{source['worker']}/{source[field]}"


def preflight(bucket, smoke=False):
    rows = []
    for s in sources(smoke):
        for field in ('policy', 'state'):
            uri = remote_source(bucket, s, field)
            raw = json.loads(subprocess.check_output(['gcloud','storage','objects','describe',uri,'--format=json']))
            size = int(raw['size'])
            if size <= 0 or (field=='state' and size != s['state_bytes']):
                raise ValueError(f'Source size mismatch: {uri}')
            rows.append(dict(uri=uri, size=size))
    return dict(sources=rows, total_bytes=sum(r['size'] for r in rows))


def download(bucket, source, directory):
    start = time.perf_counter()
    for field, filename in [('policy','policy.pkl'), ('state','state.pt')]:
        path = directory/filename
        subprocess.run(['gcloud','storage','cp',remote_source(bucket,source,field),str(path)],check=True)
        if file_hash(path) != source[field+'_sha256']:
            raise ValueError(f'Checksum mismatch before deserialisation: {field}')
    return time.perf_counter()-start


def key(source):
    return f"exp{source['experiment']}_seed{source['seed']}"


def verified_result(root, source, fingerprint):
    path = root/'models'/key(source)/'result.json'
    receipt = path.with_name('SUCCESS.json')
    if not receipt.exists():
        return None
    proof = json.loads(receipt.read_text())
    if (proof['fingerprint'] != fingerprint or not path.exists()
            or proof['result_sha256'] != file_hash(path)):
        raise ValueError(f'Resume receipt mismatch: {receipt}')
    result = json.loads(path.read_text())
    if result['source'] != source:
        raise ValueError('Resume source differs')
    return result


def interval(values):
    mean = statistics.mean(values)
    if len(values) != 3:
        return dict(mean=mean, ci95=None, seeds=len(values))
    half = 4.302652729911275*statistics.stdev(values)/math.sqrt(3)
    return dict(mean=mean, ci95=[mean-half,mean+half], seeds=3)


def aggregate(root, selected, fingerprint, smoke):
    rows = [verified_result(root,s,fingerprint) for s in selected]
    if any(r is None for r in rows):
        raise ValueError('Cannot mark incomplete model set successful')
    baseline = {r['source']['seed']:r for r in rows if r['source']['experiment']==10}
    comparisons = []
    for exp in (29,30,31,32,33):
        candidates = [r for r in rows if r['source']['experiment']==exp]
        for strategy in ('deployed','current_regret'):
            for group in candidates[0]['metrics'][strategy]:
                for metric in ('mean_fold_probability','mean_gap_bb','p95_gap_bb',
                               'max_gap_bb','fraction_fold_over_half'):
                    differences = [r['metrics'][strategy][group][metric]-
                        baseline[r['source']['seed']]['metrics'][strategy][group][metric]
                        for r in candidates]
                    comparisons.append(dict(experiment=exp,strategy=strategy,group=group,metric=metric,
                        candidate_minus_exp10=interval(differences),per_seed_differences=differences))
    summary = dict(smoke=smoke, models=len(rows), fingerprint=fingerprint, comparisons=comparisons,
        interpretation='Negative differences favour candidate. Pointwise exploratory three-training-seed '
            'intervals, not adjusted for multiple tests. Same seed labels are not identical trajectories. '
            'Conditional decision losses are not exploitability or ordinary-game loss rates. '
            'Current regret policy is not the historical average teacher. No causal verdict or automatic selection.',
        per_model=[{k:r[k] for k in ('source','metrics','source_load_seconds','inference_seconds','download_seconds')}
                   for r in rows])
    write_json(root/'summary.json',summary)
    lines = ['# Experiment 35: strategic-feature rare-decision audit', '', summary['interpretation'], '',
             f'Model pairs: {len(rows)}. Smoke: {smoke}. No training or policy modification.', '',
             '## Fresh-board guaranteed winners: deployed fold probability', '',
             '| Candidate | Mean difference vs Exp10 | Training-seed 95% interval |',
             '|---|---:|---|']
    for r in comparisons:
        if r['strategy']=='deployed' and r['group']=='assessment/all' and r['metric']=='mean_fold_probability':
            v=r['candidate_minus_exp10']
            lines.append(f"| Exp{r['experiment']} | {v['mean']:.6f} | {v['ci95']} |")
    lines += ['', 'Per-model result files contain every case, both strategies and regret scores. ',
              'Panel labels distinguish known examples from new board-class tests. Ordinary-hand controls ',
              'are descriptive, not labelled using uniform-range equity. See summary.json for categories, ',
              'tails, per-seed differences and costs. Source states/checkpoints are not re-uploaded.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'SUCCESS.json',dict(status='complete',smoke=smoke,models=len(rows),
        fingerprint=fingerprint,summary_sha256=file_hash(root/'summary.json')))


def run(args):
    from .panel import build_panel
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    args.work.mkdir(parents=True,exist_ok=True)
    selected=sources(args.smoke)
    contract=dict(version=1,source_manifest_sha256=file_hash(HERE/'inputs.json'),
                  revision=revision(),smoke=args.smoke,sources=selected)
    protocol=root/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != contract:
        raise ValueError('Existing results use a different protocol/source revision')
    write_json(protocol,contract)
    panel_path=root/'panel.json'
    if not panel_path.exists():
        write_json(panel_path,build_panel(inputs(),args.smoke))
    panel=json.loads(panel_path.read_text())
    fingerprint=digest(dict(contract=contract,panel=panel))
    sync=lambda: subprocess.run(['gcloud','storage','rsync','--recursive','--exclude=.*\\.tmp$',
                                 str(root),args.remote],check=True) if args.remote else None
    sync()  # panel locked and persisted before the first prediction
    for index,s in enumerate(selected):
        if verified_result(root,s,fingerprint) is not None:
            print(f'Reusing verified {key(s)}',flush=True);continue
        directory=root/'models'/key(s);directory.mkdir(parents=True,exist_ok=True)
        print(json.dumps(dict(stage='source',model=key(s),index=index+1,total=len(selected))),flush=True)
        with tempfile.TemporaryDirectory(prefix=key(s)+'-',dir=args.work) as tmp:
            elapsed=download(args.bucket,s,Path(tmp))
            # Separate process releases numpy replay and PyTorch arenas completely.
            subprocess.run([sys.executable,'-m',MODULE,'worker','--output',str(directory/'result.json'),
                '--source',str(Path(tmp)),'--panel',str(panel_path),'--experiment',str(s['experiment']),
                '--seed',str(s['seed'])],check=True)
        result=json.loads((directory/'result.json').read_text())
        result['download_seconds']=elapsed
        write_json(directory/'result.json',result)
        write_json(directory/'SUCCESS.json',dict(fingerprint=fingerprint,
                   result_sha256=file_hash(directory/'result.json')))
        sync()
    aggregate(root,selected,fingerprint,args.smoke)
    sync()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','worker','preflight'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bucket')
    p.add_argument('--work',type=Path)
    p.add_argument('--remote')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--source',type=Path)
    p.add_argument('--panel',type=Path)
    p.add_argument('--experiment',type=int,choices=[10,29,30,31,32,33])
    p.add_argument('--seed',type=int,choices=[0,1,2])
    args=p.parse_args()
    if args.action=='preflight':
        write_json(args.output,preflight(args.bucket,args.smoke))
    elif args.action=='worker':
        from .audit import audit_source
        s=next(s for s in sources() if s['experiment']==args.experiment and s['seed']==args.seed)
        write_json(args.output,audit_source(s,args.source,json.loads(args.panel.read_text())))
    else:
        if args.work is None or args.bucket is None:p.error('run requires --work and --bucket')
        bucket_root(args.bucket)
        try:
            run(args)
        except BaseException as error:
            write_json(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
