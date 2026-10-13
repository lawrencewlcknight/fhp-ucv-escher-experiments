"""One CPU VM, serial source loading, durable per-model inference results."""
import argparse
from copy import copy
import json
from pathlib import Path

try:
    import exp1_grouped_wide_batch as base
except ModuleNotFoundError:
    from gcp import exp1_grouped_wide_batch as base

MODULE='experiments.fhp.exp35_fhp_feature_rare_audit.run'


def build_job(args):
    proxy=copy(args);proxy.kind='smoke';proxy.parallelism=1
    result=base.build_job(proxy)
    bootstrap=(base._bootstrap(args).replace('exp1-fhp','exp35-rare-features')
        .replace('uv python install 3.11','uv python install 3.11.16')
        .replace('uv venv --python 3.11 ','uv venv --python 3.11.16 '))
    resume=''
    if getattr(args,'resume_run_id',None):
        resume=f'gcloud storage rsync --recursive {base._q(args.bucket_root.rstrip("/")+"/"+args.resume_run_id+"/analysis")} "$OUTPUT_ROOT/analysis"'
    action=f'''
mkdir -p "$OUTPUT_ROOT/analysis" "$WORK_ROOT/sources"
{resume}
python -m {MODULE} run --bucket "$BUCKET_ROOT" --work "$WORK_ROOT/sources" \
  --output "$OUTPUT_ROOT/analysis" --remote "$BUCKET_ROOT/$RUN_ID/analysis" {'--smoke' if args.kind=='smoke' else ''}
'''
    script='#!/usr/bin/env bash\nset -Eeuo pipefail\n'+bootstrap+'\n'+base._diagnostic_wrapper(
        action,'$OUTPUT_ROOT/diagnostics','$BUCKET_ROOT/$RUN_ID/diagnostics',64000)
    tail=f'''#!/usr/bin/env bash
set -u
OUT="/workspace/exp35-rare-features-${{BATCH_TASK_RETRY_ATTEMPT:-0}}/output"
mkdir -p "$OUT/diagnostics"
(date -u; free -h; df -h; dmesg 2>&1 | tail -100) > "$OUT/diagnostics/final_system.txt"
gcloud storage rsync --recursive --exclude='.*\\.tmp$' "$OUT" {base._q(args.bucket_root.rstrip('/')+'/'+args.run_id)}
'''
    group=result['taskGroups'][0]
    group.update(taskCount=1,parallelism=1)
    group['taskSpec'].update(maxRetryCount=0,maxRunDuration='43200s',
        computeResource=dict(cpuMilli=16000,memoryMib=64000),
        runnables=[dict(script=dict(text=script)),dict(alwaysRun=True,script=dict(text=tail))])
    result['allocationPolicy']['instances'][0]['policy'].update(machineType='n2-standard-16',
        bootDisk=dict(sizeGb=100,type='pd-balanced'))
    result['labels']={'experiment':'exp35-feature-rare-audit','stage':args.kind}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=['smoke','run'])
    p.add_argument('--output',type=Path,required=True)
    for name in ('run-id','bucket-root','service-account','repo-ref','project-id','region'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--repo-url',default=base.REPO_URL)
    p.add_argument('--resume-run-id')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(build_job(args),indent=2)+'\n')


if __name__=='__main__':main()
