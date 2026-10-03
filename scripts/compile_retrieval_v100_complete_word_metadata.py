"""Seal completed full word/owner phases with equivalent vectorized metadata."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.compile_retrieval_v98_national_postings import facts
from scripts.project_retrieval_v94_virtual_national_names import verified_projection
from src.xgb_matcher.retrieval_metadata_v100 import build
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v100_complete_word_metadata.json'
OWNER_ARRAYS=('fact_owners','fact_parents','fact_subjects','fact_scopes','parent_owner_keys','parent_owner_offsets',
              'parent_owner_facts','site_owner_keys','site_owner_offsets','site_owner_facts')


def completed_words(cfg):
    root=Path(cfg['word_root']); old_launch=json.loads((root/'launch.json').read_text())
    if old_launch['truth_opened']: raise ValueError('Official-only word phases required')
    for ref in old_launch['implementation'].values(): verify(ref)
    fact_manifest=facts(old_launch['config'])
    refs=dict(original_launch=record(root/'launch.json')); results={}
    for name in ['prepare','owners']+[f'word{b:02d}' for b in range(32)]:
        path=root/(name+'.json'); result=json.loads(path.read_text())
        if result['truth_opened'] or result['crm_opened'] or not result['resources']['pass_resource_policy']:
            raise ValueError('Complete successful word and owner phases required')
        results[name]=result['stats']; refs[name]=record(path)
    prepared=results['prepare']; buckets=[results[f'word{b:02d}'] for b in range(32) if results[f'word{b:02d}'] is not None]
    if prepared['facts']!=cfg['expected_facts']: raise ValueError('National fact denominator drift')
    for b in range(32):
        s=results[f'word{b:02d}']
        if s is not None and s['bucket']!=b: raise ValueError('Word storage bucket drift')
        for ref in prepared['partitions'][str(b)]: verify(ref)
    postings={k:v for k,v in prepared.items() if k!='partitions'}
    postings.update(buckets=buckets,word_postings=sum(b['postings'] for b in buckets),vocabulary_words=sum(b['words'] for b in buckets))
    artifacts={name+'.npy':record(root/(name+'.npy')) for name in OWNER_ARRAYS}
    for b in buckets:
        artifacts[f'word_vocabulary{b["bucket"]:02d}.parquet']=record(b['vocabulary'])
        artifacts[f'word_postings{b["bucket"]:02d}.npy']=record(b['postings_path'])
    if sum(s['facts'] for s in results['owners'].values())!=cfg['expected_facts']:
        raise ValueError('Owner/full fact denominator drift')
    source_manifest=json.loads(verify(fact_manifest['source_manifest']).read_text())
    original,_=verified_projection(source_manifest)
    if source_manifest['stats']['name_observations']!=cfg['expected_observations']:
        raise ValueError('Full original metadata denominator drift')
    return dict(postings=postings,owners=results['owners'],artifacts=artifacts,phase_authorities=refs,
        facts_manifest=old_launch['config']['facts_manifest'],source=record(original),owner_directory=str(root))


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; budget=None
    for ref in launch['implementation'].values(): verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            source=verify(launch['inputs']['source']); stats=build(source,output); verify(launch['inputs']['source'])
            if stats['observations']!=cfg['expected_observations']: raise ValueError('Original metadata denominator drift')
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values(): verify(ref)
        write_json(output/'metadata.json',dict(stats=stats,artifacts={p.name:record(p) for p in output.glob('*.npy')},
            resources=budget.report(),truth_opened=False))
        print(json.dumps(dict(metadata=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text()); inputs=completed_words(cfg)
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v100_complete_word_metadata.md',ROOT/'src/xgb_matcher/retrieval_metadata_v100.py',
        ROOT/'scripts/compile_retrieval_v98_national_postings.py',ROOT/'src/xgb_matcher/retrieval_national_postings_v98.py',
        ROOT/'src/xgb_matcher/retrieval_name_postings_v86.py',ROOT/'scripts/project_retrieval_v94_virtual_national_names.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    launch=dict(config=cfg,inputs=inputs,implementation={str(p):record(p) for p in paths},truth_opened=False)
    write_json(output/'launch.json',launch)
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v100_complete_word_metadata','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        if inputs!=completed_words(cfg): raise ValueError('Completed phases/source drift')
        metadata=json.loads((output/'metadata.json').read_text())
        if not metadata['resources']['pass_resource_policy']: raise ValueError('Metadata resource drift')
        for ref in launch['implementation'].values(): verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_NATIONAL_NAME_WORD_ACCESS',launch=record(output/'launch.json'),
            facts_source=inputs['facts_manifest'],owner_directory=inputs['owner_directory'],
            postings=inputs['postings'],owners=inputs['owners'],metadata=metadata['stats'],
            artifacts={**inputs['artifacts'],**metadata['artifacts']},phase_authorities=inputs['phase_authorities'],
            metadata_authority=record(output/'metadata.json'),resources=metadata['resources'],seconds=time.monotonic()-began,
            truth_opened=False,crm_opened=False,model_fit=False,recall_measured=False,fold0_opened=False,fold1_opened=False,
            ready_for_fold1=False,identifier_features=False,full_national_source_scope=True,serving_not_yet_integrated=True,
            old_partial_metadata_used=False,completed_words_not_recomputed=True))
        print(json.dumps(dict(complete=True,seconds=time.monotonic()-began,postings=inputs['postings']['word_postings'],
            words=inputs['postings']['vocabulary_words'],resources=metadata['resources'])),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,seconds=time.monotonic()-began)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); a=p.parse_args(); (run if a.mode=='run' else worker)(a.output)
