"""Factor every national official name observation in supervised physical phases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb

from scripts.project_retrieval_v94_virtual_national_names import verified_projection
from src.xgb_matcher.retrieval_national_name_facts_v95 import partition,factor_bucket,invert
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v95_national_names.json'


def source(cfg,full=True):
    manifest=json.loads(verify(cfg['source_manifest']).read_text())
    launch=json.loads(verify(manifest['launch']).read_text())
    for ref in launch['implementation'].values(): verify(ref)
    if manifest['stats']['name_observations']!=cfg['expected_observations']: raise ValueError('Whole national source required')
    if full: return manifest,verified_projection(manifest)[1]
    # The controller fully hashes the original source before and after all
    # phases; each bucket reads only its own hashed source-only partition.
    return manifest,None


def worker(output,phase,bucket=None,fact_offset=None):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; c=None; budget=None
    for ref in launch['implementation'].values(): verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            c=duckdb.connect(); c.execute('SET threads=2'); c.execute('SET preserve_insertion_order=false')
            c.execute('SET memory_limit=?',[cfg['duckdb_memory']]); c.execute('SET temp_directory=?',[str(output/'spill')])
            c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            n=cfg['expected_observations']
            if phase=='partition':
                _,sql=source(cfg); stats=partition(c,sql,output,n,lambda row:print(json.dumps(row),flush=True))
                partitions={str(b):[record(p) for p in sorted((output/'staging'/f'owner_bucket={b}').glob('*.parquet'))] for b in range(16)}
                stats['partitions']=partitions
            elif phase=='bucket':
                source(cfg,full=False); ps=json.loads((output/'partition.json').read_text())['stats']
                for ref in ps['partitions'][str(bucket)]: verify(ref)
                stats=factor_bucket(c,output,bucket,fact_offset,n,ps['bucket_observations'][str(bucket)])
            else:
                parts=json.loads((output/'parts.json').read_text())['parts']
                for part in parts: verify(part['artifact'])
                stats=invert(output,parts,n,lambda row:print(json.dumps(row),flush=True))
            c.close(); c=None
        for ref in launch['implementation'].values(): verify(ref)
        if phase=='bucket' and stats['facts']: stats['artifact']=record(stats['path'])
        filename=f'bucket{bucket:02d}.json' if phase=='bucket' else phase+'.json'
        write_json(output/filename,dict(stats=stats,resources=budget.report(),truth_opened=False,crm_opened=False))
        print(json.dumps(dict(phase=phase,bucket=bucket,stats=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        if c is not None: c.close()
        write_json(output/f'stop_{phase}_{bucket}.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v95_national_names.md',ROOT/'scripts/project_retrieval_v94_virtual_national_names.py',
        ROOT/'scripts/project_retrieval_v93_national_names.py',ROOT/'src/xgb_matcher/retrieval_national_name_facts_v95.py',
        ROOT/'src/xgb_matcher/retrieval_name_facts_v85.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False,crm_opened=False))
    def execute(phase,bucket=None,offset=None):
        command=[sys.executable,'-u','-m','scripts.compile_retrieval_v95_national_names','worker','--output',str(output),'--phase',phase]
        if bucket is not None: command.extend(['--bucket',str(bucket),'--fact-offset',str(offset)])
        with (output/f'{phase}_{bucket}.log').open('x') as log:
            subprocess.run(command,check=True,timeout=min(90,max(1,900-(time.monotonic()-began))),
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    try:
        source(cfg); execute('partition'); parts=[]; nf=0
        ps=json.loads((output/'partition.json').read_text())
        if not ps['resources']['pass_resource_policy']: raise ValueError('Partition resource guard failed')
        for bucket in range(16):
            execute('bucket',bucket,nf); result=json.loads((output/f'bucket{bucket:02d}.json').read_text())
            if not result['resources']['pass_resource_policy']: raise ValueError('Bucket resource guard failed')
            part=result['stats']
            if part['bucket']!=bucket or part['fact_start']!=nf: raise ValueError('Fact catalog order drift')
            if part['facts']: parts.append(part); nf+=part['facts']
            print(json.dumps(dict(buckets_completed=bucket+1,facts=nf)),flush=True)
        if sum(p['observations'] for p in parts)!=cfg['expected_observations']: raise ValueError('All-human-independent source denominator drift')
        write_json(output/'parts.json',dict(parts=parts)); execute('inverse'); source(cfg)
        inverse=json.loads((output/'inverse.json').read_text())
        if not inverse['resources']['pass_resource_policy']: raise ValueError('Inverse resource guard failed')
        launch=json.loads((output/'launch.json').read_text())
        for ref in launch['implementation'].values(): verify(ref)
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        artifacts={str(p.relative_to(output)):record(p) for p in output.rglob('*') if p.is_file() and p.suffix in ('.npy','.parquet')}
        write_json(output/'manifest.json',dict(status='SEALED_ALL_NATIONAL_SCOPED_NAME_FACTS_AND_OBSERVATIONS',
            launch=record(output/'launch.json'),source_manifest=cfg['source_manifest'],
            stats=dict(observations=cfg['expected_observations'],lexical_facts_including_context=nf,parts=parts,inverse=inverse['stats']),
            phases={p.name:record(p) for p in output.glob('*.json') if p.name not in ('launch.json','manifest.json')},
            artifacts=artifacts,output_bytes=size,seconds=time.monotonic()-began,truth_opened=False,crm_opened=False,
            model_fit=False,lexical_index_built=False,recall_measured=False,fold0_opened=False,fold1_opened=False,
            original_metadata_in_verified_source=True,ready_for_fold1=False))
        print(json.dumps(dict(complete=True,facts=nf,seconds=time.monotonic()-began,output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,seconds=time.monotonic()-began)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--phase',choices=['partition','bucket','inverse'])
    p.add_argument('--bucket',type=int); p.add_argument('--fact-offset',type=int); a=p.parse_args()
    if a.mode=='worker':
        if a.phase is None or (a.phase=='bucket' and (a.bucket not in range(16) or a.fact_offset is None or a.fact_offset<0)):
            p.error('Invalid physical phase')
        worker(a.output,a.phase,a.bucket,a.fact_offset)
    else: run(a.output)
