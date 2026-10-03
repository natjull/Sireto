"""Seal complete V95 tuple phases using memory-isolated exhaustive provenance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.compile_retrieval_v95_national_names import source
from src.xgb_matcher.retrieval_name_inverse_v97 import inverse_part,assemble
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v97_complete_provenance.json'


def completed_phases(cfg):
    root=Path(cfg['tuple_root']); launch=json.loads((root/'launch.json').read_text())
    if launch['truth_opened'] or launch['crm_opened']: raise ValueError('Official-only tuple phases required')
    for ref in launch['implementation'].values(): verify(ref)
    if launch['config']['expected_observations']!=cfg['expected_observations']: raise ValueError('Full source denominator drift')
    source(launch['config'])
    partition=json.loads((root/'partition.json').read_text())
    if not partition['resources']['pass_resource_policy'] or partition['stats']['observations']!=cfg['expected_observations']:
        raise ValueError('Complete successful partition required')
    parts=[]; nf=0; refs=dict(original_launch=record(root/'launch.json'),partition=record(root/'partition.json'))
    for bucket in range(16):
        path=root/f'bucket{bucket:02d}.json'; result=json.loads(path.read_text()); p=result['stats']
        if (result['truth_opened'] or result['crm_opened'] or not result['resources']['pass_resource_policy'] or
                p['bucket']!=bucket or p['fact_start']!=nf or
                p['observations']!=partition['stats']['bucket_observations'][str(bucket)]):
            raise ValueError('Incomplete/unverified tuple phase')
        verify(p['artifact']); parts.append(p); nf+=p['facts']; refs[f'bucket{bucket:02d}']=record(path)
        for ref in partition['stats']['partitions'][str(bucket)]: verify(ref)
    if sum(p['observations'] for p in parts)!=cfg['expected_observations'] or nf!=cfg['expected_facts']:
        raise ValueError('Whole exact tuple/source denominator drift')
    return dict(parts=parts,phase_authorities=refs,original_source_manifest=launch['config']['source_manifest'],
                original_mapping=record(root/'observation_facts.npy'))


def worker(output,bucket):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; budget=None
    for ref in launch['implementation'].values(): verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            mapping=verify(launch['inputs']['original_mapping'])
            for ref in launch['inputs']['phase_authorities'].values(): verify(ref)
            parts=launch['inputs']['parts']
            if bucket==-1:
                for p in parts:
                    artifact=json.loads((output/f'part{p["bucket"]:02d}.json').read_text())
                    if not artifact['resources']['pass_resource_policy']: raise ValueError('Inverse part resource drift')
                    for ref in artifact['artifacts'].values(): verify(ref)
                stats=assemble(mapping,parts,output,cfg['expected_observations'])
            else:
                part=next(p for p in parts if p['bucket']==bucket); verify(part['artifact'])
                stats=inverse_part(mapping,part,output,cfg['expected_observations'])
        for ref in launch['implementation'].values(): verify(ref)
        filename='assembly.json' if bucket==-1 else f'part{bucket:02d}.json'
        paths=(output/'fact_offsets.npy',output/'fact_observations.npy') if bucket==-1 else (
            output/f'inverse_offsets{bucket:02d}.npy',output/f'inverse_ordinals{bucket:02d}.npy')
        write_json(output/filename,dict(stats=stats,artifacts={p.name:record(p) for p in paths},
            resources=budget.report(),truth_opened=False))
        print(json.dumps(dict(bucket=bucket,stats=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/f'stop{bucket}.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    inputs=completed_phases(cfg)
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v97_complete_provenance.md',ROOT/'src/xgb_matcher/retrieval_name_inverse_v97.py',
        ROOT/'scripts/compile_retrieval_v95_national_names.py',ROOT/'src/xgb_matcher/retrieval_national_name_facts_v95.py',
        ROOT/'scripts/project_retrieval_v94_virtual_national_names.py',ROOT/'scripts/project_retrieval_v93_national_names.py',
        ROOT/'src/xgb_matcher/retrieval_name_facts_v85.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    launch=dict(config=cfg,inputs=inputs,implementation={str(p):record(p) for p in paths},truth_opened=False,crm_opened=False)
    write_json(output/'launch.json',launch)
    try:
        for bucket in [p['bucket'] for p in inputs['parts']]+[-1]:
            with (output/f'worker{bucket}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v97_complete_provenance','worker',
                    '--output',str(output),'--bucket',str(bucket)],check=True,
                    timeout=min(90,max(1,900-(time.monotonic()-began))),stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            print(json.dumps(dict(inverse_bucket_completed=bucket)),flush=True)
        completed_phases(cfg); assembly=json.loads((output/'assembly.json').read_text())
        if not assembly['resources']['pass_resource_policy']: raise ValueError('Assembly resource drift')
        for ref in launch['implementation'].values(): verify(ref)
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        artifacts={p.name:record(p) for p in output.glob('*.npy')}; artifacts['observation_facts.npy']=inputs['original_mapping']
        write_json(output/'manifest.json',dict(status='SEALED_ALL_NATIONAL_SCOPED_NAME_FACTS_AND_OBSERVATIONS',
            launch=record(output/'launch.json'),source_manifest=inputs['original_source_manifest'],
            stats=dict(observations=cfg['expected_observations'],lexical_facts_including_context=cfg['expected_facts'],
                parts=inputs['parts'],inverse=assembly['stats']),artifacts=artifacts,
            phase_authorities=inputs['phase_authorities'],output_bytes=size,seconds=time.monotonic()-began,
            truth_opened=False,crm_opened=False,recall_measured=False,lexical_index_built=False,model_fit=False,
            fold0_opened=False,fold1_opened=False,ready_for_fold1=False,original_metadata_in_verified_source=True,
            old_partial_inverse_used=False,original_tuples_not_recomputed=True))
        print(json.dumps(dict(complete=True,seconds=time.monotonic()-began,output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,seconds=time.monotonic()-began)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--bucket',type=int); a=p.parse_args()
    if a.mode=='worker':
        if a.bucket not in range(-1,16): p.error('Invalid inverse physical bucket')
        worker(a.output,a.bucket)
    else: run(a.output)
