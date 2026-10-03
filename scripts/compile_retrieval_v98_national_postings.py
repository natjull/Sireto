"""Full national literal-word/owner/metadata indexes in bounded physical phases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb

from scripts.project_retrieval_v94_virtual_national_names import verified_projection
from src.xgb_matcher.retrieval_national_postings_v98 import prepare,word_bucket
from src.xgb_matcher.retrieval_name_postings_v86 import build_owners,build_metadata
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v98_national_postings.json'


def facts(cfg,full=True):
    m=json.loads(verify(cfg['facts_manifest']).read_text())
    if (m['status']!='SEALED_ALL_NATIONAL_SCOPED_NAME_FACTS_AND_OBSERVATIONS' or m['truth_opened'] or m['crm_opened'] or
            m['stats']['observations']!=cfg['expected_observations'] or m['stats']['lexical_facts_including_context']!=cfg['expected_facts']):
        raise ValueError('Complete official national facts required')
    launch=json.loads(verify(m['launch']).read_text())
    for ref in launch['implementation'].values(): verify(ref)
    if full:
        for part in m['stats']['parts']: verify(part['artifact'])
        for ref in m['artifacts'].values(): verify(ref)
    return m


def worker(output,phase,bucket=None):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; budget=None; c=None
    for ref in launch['implementation'].values(): verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            m=facts(cfg,full=phase in ('prepare','owners')); parts=m['stats']['parts']
            c=duckdb.connect(); c.execute('SET threads=2'); c.execute('SET preserve_insertion_order=false')
            c.execute('SET memory_limit=?',[cfg['duckdb_memory']]); c.execute('SET temp_directory=?',[str(output/'spill')])
            c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            if phase=='prepare':
                stats=prepare(c,parts,output)
                stats['partitions']={str(b):[record(p) for p in sorted((output/'word_staging'/f'token_bucket={b}').glob('*.parquet'))] for b in range(32)}
            elif phase=='word':
                prepared=json.loads((output/'prepare.json').read_text())['stats']
                for ref in prepared['partitions'][str(bucket)]: verify(ref)
                stats=word_bucket(c,output,bucket,cfg['expected_facts'])
            elif phase=='owners': stats=build_owners(parts,output)
            else:
                source_manifest=json.loads(verify(m['source_manifest']).read_text())
                source,_=verified_projection(source_manifest)
                stats=build_metadata(source,output)
                stats['original_provenance_strings_still_in_v82_source']=False
                stats['original_provenance_strings_in_full_v94_source']=True
                verify(source_manifest['stats']['name_projection']['source'])
            c.close(); c=None
        for ref in launch['implementation'].values(): verify(ref)
        filename=f'word{bucket:02d}.json' if phase=='word' else phase+'.json'
        write_json(output/filename,dict(stats=stats,resources=budget.report(),truth_opened=False,crm_opened=False))
        print(json.dumps(dict(phase=phase,bucket=bucket,stats=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        if c is not None: c.close()
        write_json(output/f'stop_{phase}_{bucket}.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v98_national_postings.md',ROOT/'src/xgb_matcher/retrieval_national_postings_v98.py',
        ROOT/'src/xgb_matcher/retrieval_name_postings_v86.py',ROOT/'src/xgb_matcher/retrieval_name_facts_v85.py',
        ROOT/'src/xgb_matcher/retrieval_proof_v56.py',ROOT/'scripts/project_retrieval_v94_virtual_national_names.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    def execute(phase,bucket=None):
        command=[sys.executable,'-u','-m','scripts.compile_retrieval_v98_national_postings','worker','--output',str(output),'--phase',phase]
        if bucket is not None: command.extend(['--bucket',str(bucket)])
        with (output/f'{phase}_{bucket}.log').open('x') as log:
            subprocess.run(command,check=True,timeout=min(90,max(1,900-(time.monotonic()-began))),
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        path=output/(f'word{bucket:02d}.json' if phase=='word' else phase+'.json')
        result=json.loads(path.read_text())
        if not result['resources']['pass_resource_policy']: raise ValueError('Phase resource drift')
        return result['stats']
    try:
        facts(cfg); prepared=execute('prepare'); buckets=[]
        for bucket in range(32):
            stats=execute('word',bucket)
            if stats is not None: buckets.append(stats)
            print(json.dumps(dict(word_buckets_complete=bucket+1)),flush=True)
        owners=execute('owners'); metadata=execute('metadata'); facts(cfg)
        launch=json.loads((output/'launch.json').read_text())
        for ref in launch['implementation'].values(): verify(ref)
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        postings={k:v for k,v in prepared.items() if k!='partitions'}
        postings.update(buckets=buckets,word_postings=sum(b['postings'] for b in buckets),vocabulary_words=sum(b['words'] for b in buckets))
        write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_NATIONAL_NAME_WORD_ACCESS',
            launch=record(output/'launch.json'),facts_source=cfg['facts_manifest'],postings=postings,owners=owners,metadata=metadata,
            artifacts={str(p.relative_to(output)):record(p) for p in output.rglob('*') if p.is_file() and p.suffix in ('.npy','.parquet')},
            phase_authorities={p.name:record(p) for p in output.glob('*.json') if p.name not in ('launch.json','manifest.json')},
            output_bytes=size,seconds=time.monotonic()-began,truth_opened=False,crm_opened=False,model_fit=False,
            recall_measured=False,fold0_opened=False,fold1_opened=False,ready_for_fold1=False,identifier_features=False,
            full_national_source_scope=True,serving_not_yet_integrated=True))
        print(json.dumps(dict(complete=True,seconds=time.monotonic()-began,postings=postings,owners=owners,output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,seconds=time.monotonic()-began)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--phase',choices=['prepare','word','owners','metadata'])
    p.add_argument('--bucket',type=int); a=p.parse_args()
    if a.mode=='worker':
        if a.phase is None or (a.phase=='word' and a.bucket not in range(32)): p.error('Invalid physical phase')
        worker(a.output,a.phase,a.bucket)
    else: run(a.output)
