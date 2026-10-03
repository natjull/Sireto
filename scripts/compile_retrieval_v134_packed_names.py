"""V131 complete postings reused unchanged; exact typed/vectorized packing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import numpy as np

from scripts.compile_retrieval_v131_name_access import complete_stage,part_dir
from src.xgb_matcher.retrieval_packed_names_v134 import pack
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v134_packed_names.json'
PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


def source(cfg):
    path=verify(cfg['source_stage']);_,p,s=complete_stage(path.parent)
    if (record(path)!=record(path.parent/'staging_manifest.json') or s['names']!=cfg['expected_names']
        or s['postings']!=cfg['expected_postings'] or any(s.get(k) is not False for k in PROTECTED)
        or s['complete_retrieval_index'] or p['term_buckets']!=cfg['term_buckets']):raise ValueError('Every unchanged official V131 staged name/channel required')
    return s


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    cfg=l['config']
    if duckdb.__version__!=cfg['duckdb_version'] or np.__version__!=cfg['numpy_version']:raise ValueError('Pinned storage/array library drift')
    return l,source(cfg)


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text());s=source(cfg)
    if duckdb.__version__!=cfg['duckdb_version'] or np.__version__!=cfg['numpy_version']:raise ValueError('Pinned storage/array library drift')
    with Budget(cfg['resources']) as budget:
        for items in s['buckets'].values():
            for item in items:verify(item['artifact'])
        for item in s['features']:verify(item['artifact'])
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v134_packed_names.md',ROOT/'src/xgb_matcher/retrieval_packed_names_v134.py',
        ROOT/'src/xgb_matcher/retrieval_name_access_v131.py',ROOT/'scripts/compile_retrieval_v131_name_access.py',
        ROOT/'src/xgb_matcher/retrieval_variant_factor_v119.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
        ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},source_audit=budget.report(),**{k:False for k in PROTECTED}))
    print(json.dumps(dict(names=s['names'],postings=s['postings'],source_audit=budget.report())),flush=True)


def bucket_dir(output,channel,bucket):return output/'access'/f'channel{channel}'/f'bucket{bucket:03d}'


def worker(output,channel,bucket):
    l,s=load(output);cfg=l['config'];dest=bucket_dir(output,channel,bucket);budget=None;c=None
    if any((dest/k).exists() for k in ('manifest.json','worker_stop.json')) or any(dest.glob('*.npy')):
        raise ValueError('Immutable packed channel bucket already started')
    try:
        with Budget(cfg['resources']) as budget:
            c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET temp_directory=?',[str(dest/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            items=s['buckets'][str(bucket)];files=[];source_root=Path(cfg['source_stage']['path']).parent
            for item in items:
                path=verify(item['artifact']);allowed=part_dir(source_root,item['part_id'])/'staging'/f'term_bucket={bucket}'
                if not path.resolve().is_relative_to(allowed.resolve()):raise ValueError('Unchanged pinned whole channel path required')
                files.append(path)
            expected=sum(i['channel_counts'][str(channel)] for i in items)
            stats=pack(c,files,dest,channel,bucket,cfg['term_buckets'],expected,s['names'])
            for i in items:verify(i['artifact'])
            c.close();c=None;load(output);artifacts={p.name:record(p) for p in dest.glob('*.npy')}
            size=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
            if size>cfg['max_phase_output_bytes']:raise ValueError('STOP_PACKED_CHANNEL_OUTPUT_SIZE')
        write_json(dest/'manifest.json',dict(status='SEALED_V134_EXACT_COMPLETE_CHANNEL_BUCKET',launch=record(output/'launch.json'),
            source_stage=cfg['source_stage'],stats=stats,artifacts=artifacts,resources=budget.report(),output_bytes=size,
            complete_retrieval_index=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        if c is not None:c.close()
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,channel,bucket):
    l,_=load(output)
    if (not isinstance(channel,int) or isinstance(channel,bool) or not 0<=channel<4
        or not isinstance(bucket,int) or isinstance(bucket,bool) or not 0<=bucket<l['config']['term_buckets']):raise ValueError('Actual complete planned channel/bucket required')
    dest=bucket_dir(output,channel,bucket);dest.mkdir(parents=True,exist_ok=False)
    try:
        with (dest/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v134_packed_names','worker','--output',str(output),'--channel',str(channel),'--bucket',str(bucket)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((dest/'worker.log').read_text()[-5500:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def phase_result(path,output):
    m=json.loads(path.read_text())
    if (m['status']!='SEALED_V134_EXACT_COMPLETE_CHANNEL_BUCKET' or m['launch']!=record(output/'launch.json')
        or not m['resources']['pass_resource_policy'] or any(m.get(k) is not False for k in PROTECTED)
        or m['complete_retrieval_index'] or m['stats']['raw_alias_ids_not_features'] is not True):raise ValueError('Protected exact complete channel required')
    return m


def seal(output):
    l,s=load(output);cfg=l['config']
    if (output/'manifest.json').exists():raise ValueError('Immutable packed access already sealed')
    refs=[];access=[];posts=0;terms=0;size=s['output_bytes']
    for ch in range(4):
        for b in range(cfg['term_buckets']):
            dest=bucket_dir(output,ch,b);path=dest/'manifest.json'
            if not path.exists():raise ValueError('UNFINISHED_ALL_CHANNEL_BUCKETS')
            m=phase_result(path,output);st=m['stats']
            if (m['source_stage']!=cfg['source_stage'] or st['channel']!=ch or st['bucket']!=b
                or st['postings']!=sum(i['channel_counts'][str(ch)] for i in s['buckets'][str(b)])
                or not st['every_original_posting_and_token_reconstructed_exactly'] or not st['source_semantics_unchanged']
                or not st['native_unsigned_batched_csr_audit']):raise ValueError('Every unchanged exact source channel required')
            specs={'post_ids':(np.uint32,st['postings']),'post_offsets':(np.uint64,st['terms']+1),
                'token_data':(np.uint8,st['token_bytes']),'token_offsets':(np.uint64,st['terms']+1)}
            if set(m['artifacts'])!={k+'.npy' for k in specs}:raise ValueError('Every packed name array required')
            for name,(dtype,n) in specs.items():
                p=verify(m['artifacts'][name+'.npy']);a=np.load(p,mmap_mode='r',allow_pickle=False)
                if p.resolve()!=(dest/(name+'.npy')).resolve() or a.dtype!=np.dtype(dtype) or a.shape!=(n,):raise ValueError('Whole typed packed array required')
                if name.endswith('offsets') and (a[0]!=0 or a[-1]!=(st['postings'] if name=='post_offsets' else st['token_bytes']) or np.any(a[1:]<=a[:-1])):
                    raise ValueError('Whole strictly increasing packed CSR required')
            access.append(dict(channel=ch,bucket=b,postings=st['postings'],terms=st['terms'],artifacts=m['artifacts']));refs.append(record(path))
            posts+=st['postings'];terms+=st['terms'];size+=m['output_bytes']
    if posts!=s['postings'] or size>cfg['max_total_output_bytes_including_source_stage']:raise ValueError('Whole packed source denominator/size required')
    write_json(output/'manifest.json',dict(status='SEALED_V134_COMPLETE_ALL_OFFICIAL_NAME_CHANNEL_ACCESS',launch=record(output/'launch.json'),
        source_stage=cfg['source_stage'],names=s['names'],postings=posts,terms=terms,features=s['features'],access=access,
        phase_authorities=refs,output_bytes_including_source_stage=size,source_stage_bytes=s['output_bytes'],
        source_semantics_unchanged=True,complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','bucket','worker','seal']);p.add_argument('--output',required=True,type=Path);p.add_argument('--channel',type=int);p.add_argument('--bucket',type=int);a=p.parse_args()
    if a.mode in ('bucket','worker'):
        if a.channel is None or a.bucket is None:p.error('Actual planned channel/bucket required')
        (execute if a.mode=='bucket' else worker)(a.output,a.channel,a.bucket)
    else:(init if a.mode=='init' else seal)(a.output)
