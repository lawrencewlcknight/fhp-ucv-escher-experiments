#!/usr/bin/env python3
"""Download and checksum-verify only completed Exp42/43 analysis artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ALLOWED = {'evaluation_manifest.json', 'summary.json', 'matchups.csv', 'direct_comparison_summary.csv',
           'checkpoint_index.json', 'checkpoint_index.csv', 'training_cost_context.json',
           'final_48h_head_to_head.png', 'analysis_summary.md', 'timing_forecast.json'}


def copy_object(uri, target):
    subprocess.run(['gcloud','storage','cp',uri,str(target)],check=True)


def download(bucket, run_id, output, copy=copy_object):
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+',bucket):
        raise ValueError('Provide a gs:// bucket without a subdirectory')
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,43}[a-z0-9]',run_id):
        raise ValueError('Invalid run ID')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    source = f'{bucket}/{run_id}/analysis'
    with tempfile.TemporaryDirectory(prefix='representation-analysis-',dir=output.parent) as directory:
        staging = Path(directory)
        copy(source+'/SUCCESS.json',staging/'SUCCESS.json')
        receipt = json.loads((staging/'SUCCESS.json').read_text())
        if (receipt.get('experiment_id') not in (42,43) or receipt.get('status') != 'complete'
                or receipt.get('smoke') is not False or receipt.get('duplicate_pairs') != 900_000):
            raise ValueError('Expected a complete production evaluation, not a smoke or partial run')
        files = receipt.get('files',{})
        if set(files) != ALLOWED:
            raise ValueError('Unexpected analysis file set; refusing models, shards or unknown artifacts')
        for name,expected in files.items():
            copy(source+'/'+name,staging/name)
            if hashlib.sha256((staging/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Analysis checksum mismatch: {name}')
        summary = json.loads((staging/'summary.json').read_text())
        if summary.get('experiment_id') != receipt['experiment_id'] or summary.get('hands') != 1_800_000:
            raise ValueError('Production summary disagrees with completion receipt')
        output.mkdir(parents=True,exist_ok=True)
        for name in files:
            shutil.copyfile(staging/name,output/name)
        shutil.copyfile(staging/'SUCCESS.json',output/'SUCCESS.json')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bucket','run-id','output'):
        p.add_argument('--'+key,required=True)
    args = p.parse_args()
    print(download(args.bucket,args.run_id,args.output))


if __name__ == '__main__':
    main()
