"""Complete typed name-owner CSR, from every sealed V130 physical bucket."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from scripts.compile_retrieval_v130_variant_dictionary import stage_authority as original_authority
from src.xgb_matcher.retrieval_name_owners_v132 import build,FIELDS
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v132_name_owners.json'
PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    p=json.loads(verify(l['plan']).read_text());verify(l['config']['source_manifest'])
    original_authority(Path(l['config']['source_manifest']['path']).parent)
    return l,p


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text());s=json.loads(verify(cfg['source_manifest']).read_text())
    original_authority(Path(cfg['source_manifest']['path']).parent)
    if (s['status']!='SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY' or s['names']!=cfg['expected_names']
        or s['facts']!=cfg['expected_facts'] or s['original_observations']!=cfg['expected_observations']
        or any(s.get(k) is not False for k in PROTECTED) or s['complete_retrieval_index'] or s['ready_for_fold1']):
        raise ValueError('Protected whole official V130 dictionary required')
    names=0;facts=0
    with Budget(cfg['resources']) as budget:
        for i,b in enumerate(s['buckets']):
            if b['bucket']!=i or b['name_start']!=names or b['fact_start']!=facts:raise ValueError('Every original contiguous dictionary bucket required')
            for key,n in (('names.parquet',b['names']),('facts.parquet',b['facts'])):
                if pq.ParquetFile(verify(b['artifacts'][key])).metadata.num_rows!=n:raise ValueError('Every source name/owner row required')
            names+=b['names'];facts+=b['facts']
        if names!=cfg['expected_names'] or facts!=cfg['expected_facts']:raise ValueError('Complete national owner coverage required')
    write_json(output/'plan.json',dict(status='SEALED_V132_ALL_OFFICIAL_OWNER_PHYSICAL_PLAN',source=cfg['source_manifest'],
        buckets=s['buckets'],names=names,facts=facts,source_full_hash_audit=budget.report(),**{k:False for k in PROTECTED}))
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v132_name_owners.md',ROOT/'src/xgb_matcher/retrieval_name_owners_v132.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),implementation={str(x):record(x) for x in paths},**{k:False for k in PROTECTED}))
    print(json.dumps(dict(names=names,facts=facts,buckets=len(s['buckets']),source_audit=budget.report())),flush=True)


def bucket_dir(output,index):return output/'owners'/f'bucket{index:03d}'


def worker(output,index):
    l,p=load(output);b=p['buckets'][index];dest=bucket_dir(output,index);budget=None
    if any((dest/k).exists() for k in ('manifest.json','worker_stop.json')) or any(dest.glob('*.npy')):
        raise ValueError('Immutable owner bucket already started')
    try:
        with Budget(l['config']['resources']) as budget:
            source=verify(b['artifacts']['facts.parquet']);stats=build(source,dest,b['names'],b['facts'])
            verify(b['artifacts']['facts.parquet']);load(output)
            arrays={x.name:record(x) for x in dest.glob('*.npy')};size=sum(x['size_bytes'] for x in arrays.values())
            if size>l['config']['max_bucket_output_bytes']:raise ValueError('STOP_OWNER_BUCKET_SIZE')
        write_json(dest/'manifest.json',dict(status='SEALED_V132_EXACT_ALL_OWNER_BUCKET',launch=record(output/'launch.json'),
            bucket=b,stats=stats,artifacts=arrays,resources=budget.report(),output_bytes=size,
            complete_retrieval_index=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(bucket=index,stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,index):
    _,p=load(output)
    if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(p['buckets']):raise ValueError('Planned owner bucket required')
    dest=bucket_dir(output,index);dest.mkdir(parents=True,exist_ok=False)
    try:
        with (dest/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v132_name_owners','worker','--output',str(output),'--index',str(index)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((dest/'worker.log').read_text()[-6000:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def phase_result(path,output):
    m=json.loads(path.read_text())
    if (m['status']!='SEALED_V132_EXACT_ALL_OWNER_BUCKET' or m['launch']!=record(output/'launch.json')
        or not m['resources']['pass_resource_policy'] or any(m.get(k) is not False for k in PROTECTED)
        or m['complete_retrieval_index'] or m['stats']['identifiers_as_features']):raise ValueError('Protected complete owner bucket required')
    return m


def seal(output):
    l,p=load(output)
    if (output/'manifest.json').exists():raise ValueError('Immutable owners already sealed')
    refs=[];buckets=[];names=0;facts=0;size=0
    for b in p['buckets']:
        dest=bucket_dir(output,b['bucket']);path=dest/'manifest.json'
        if not path.exists():raise ValueError('UNFINISHED_ALL_OWNER_BUCKETS')
        m=phase_result(path,output);s=m['stats']
        if (m['bucket']!=b or s['names']!=b['names'] or s['facts']!=b['facts'] or not s['every_original_fact_reconstructed_exactly']
            or not s['every_raw_name_has_original_owner'] or not s['owners_outside_candidate_catalog_preserved']
            or s['site_cartesian_product_materialized'] or s['owner_or_name_or_period_cap'] is not None):raise ValueError('Every uncapped original owner required')
        if set(m['artifacts'])!={*(k+'.npy' for k in FIELDS),'name_fact_offsets.npy'}:raise ValueError('Every packed owner array required')
        for key,dtype in {**FIELDS,'name_fact_offsets':np.uint64}.items():
            x=verify(m['artifacts'][key+'.npy']);a=np.load(x,mmap_mode='r',allow_pickle=False)
            if x.resolve()!=(dest/(key+'.npy')).resolve() or a.dtype!=np.dtype(dtype) or a.shape!=((b['names']+1,) if key=='name_fact_offsets' else (b['facts'],)):
                raise ValueError('Whole original typed owner array required')
            if key=='name_fact_offsets' and (a[0]!=0 or a[-1]!=b['facts'] or np.any(a[1:]<=a[:-1])):raise ValueError('Every nonempty raw-name owner CSR required')
        buckets.append(dict(bucket=b['bucket'],name_start=b['name_start'],fact_start=b['fact_start'],names=b['names'],facts=b['facts'],artifacts=m['artifacts']))
        refs.append(record(path));names+=b['names'];facts+=b['facts'];size+=m['output_bytes']
    if names!=p['names'] or facts!=p['facts'] or size>l['config']['max_total_output_bytes']:raise ValueError('Complete national owner size/coverage required')
    write_json(output/'manifest.json',dict(status='SEALED_V132_COMPLETE_ALL_OFFICIAL_NAME_OWNER_CSR',launch=record(output/'launch.json'),
        source=p['source'],names=names,facts=facts,buckets=buckets,phase_authorities=refs,output_bytes=size,
        every_owner_fact_reconstructed_exactly=True,complete_retrieval_index=False,ready_for_fold1=False,
        **{k:False for k in PROTECTED}))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=['init','bucket','worker','seal']);a.add_argument('--output',required=True,type=Path);a.add_argument('--index',type=int);v=a.parse_args()
    if v.mode in ('bucket','worker'):
        if v.index is None:a.error('Planned owner bucket required')
        (execute if v.mode=='bucket' else worker)(v.output,v.index)
    else:(init if v.mode=='init' else seal)(v.output)
