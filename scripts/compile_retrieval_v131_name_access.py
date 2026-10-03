"""Complete official name access: bounded physical groups, compact channel CSR.

No query, CRM, label, model fit, fixed candidate pool or retrieval metric.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pyarrow.parquet as pq

from scripts.compile_retrieval_v130_variant_dictionary import stage_authority as original_authority
from src.xgb_matcher.retrieval_name_access_v131 import stage,pack
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v131_name_access.json'
PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if duckdb.__version__!=l['config']['duckdb_version']:raise ValueError('Storage hash library drift')
    p=json.loads(verify(l['plan']).read_text());verify(l['config']['source_manifest'])
    original_authority(Path(l['config']['source_manifest']['path']).parent)
    return l,p


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text());source=json.loads(verify(cfg['source_manifest']).read_text())
    original_authority(Path(cfg['source_manifest']['path']).parent)
    if (source['status']!='SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY' or source['names']!=cfg['expected_names']
        or source['facts']!=cfg['expected_facts'] or source['original_observations']!=cfg['expected_observations']
        or source['complete_retrieval_index'] or source['ready_for_fold1'] or any(source.get(k) is not False for k in PROTECTED)
        or not 1<=cfg['term_buckets']<=254 or duckdb.__version__!=cfg['duckdb_version']):raise ValueError('Complete official V130 authority required')
    parts=[];names=0
    with Budget(cfg['resources']) as budget:
        for b in source['buckets']:
            path=verify(b['artifacts']['names.parquet']);f=pq.ParquetFile(path)
            if b['name_start']!=names or f.metadata.num_rows!=b['names']:raise ValueError('Whole raw name dictionary required')
            begin=0
            for group in range(f.metadata.num_row_groups):
                stop=begin+f.metadata.row_group(group).num_rows
                parts.append(dict(part_id=len(parts),source_bucket=b['bucket'],row_group=group,row_start=begin,row_stop=stop,
                    name_start=names,source=b['artifacts']['names.parquet']))
                begin=stop
            if begin!=b['names']:raise ValueError('Every source rowgroup required')
            names+=b['names']
        if names!=cfg['expected_names']:raise ValueError('Whole national name coverage required')
    p=dict(status='SEALED_V131_ALL_RAW_NAMES_PHYSICAL_PLAN',source=cfg['source_manifest'],parts=parts,names=names,
        term_buckets=cfg['term_buckets'],source_full_name_hash_audit=budget.report(),**{k:False for k in PROTECTED})
    write_json(output/'plan.json',p)
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v131_name_access.md',ROOT/'src/xgb_matcher/retrieval_name_access_v131.py',
        ROOT/'src/xgb_matcher/retrieval_coupled_blocks_v121.py',ROOT/'src/xgb_matcher/retrieval_variants_v118.py',
        ROOT/'src/xgb_matcher/retrieval_name_postings_v86.py',ROOT/'src/xgb_matcher/retrieval_variant_factor_v119.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),implementation={str(x):record(x) for x in paths},**{k:False for k in PROTECTED}))
    print(json.dumps(dict(names=names,parts=len(parts),source_audit=budget.report())),flush=True)


def part_dir(output,index):return output/'parts'/f'part{index:05d}'
def access_dir(output,channel,bucket):return output/'access'/f'channel{channel}'/f'bucket{bucket:03d}'


def database(cfg,destination):
    c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
    c.execute('SET temp_directory=?',[str(destination/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']]);return c


def complete_stage(output):
    l,p=load(output);d=json.loads((output/'access_launch.json').read_text())
    if d['launch']!=record(output/'launch.json'):raise ValueError('Access namespace launch drift')
    s=json.loads(verify(d['staging_manifest']).read_text())
    if s['status']!='SEALED_V131_ALL_OFFICIAL_NAME_CHANNEL_POSTINGS' or s['names']!=p['names']:raise ValueError('Every official name staged required')
    return l,p,s


def worker(output,phase,index,channel):
    l,p=load(output);cfg=l['config'];dest=part_dir(output,index) if phase=='part' else access_dir(output,channel,index)
    if any((dest/name).exists() for name in ('manifest.json','worker_stop.json','staging')) or any(dest.glob('*.npy')) or any(dest.glob('*.parquet')):
        raise ValueError('Immutable physical phase already started')
    c=None;budget=None
    try:
        with Budget(cfg['resources']) as budget:
            c=database(cfg,dest)
            if phase=='part':
                spec=p['parts'][index];path=verify(spec['source']);table=pq.ParquetFile(path).read_row_group(spec['row_group'],use_threads=False)
                if len(table)!=spec['row_stop']-spec['row_start']:raise ValueError('Every original group row required')
                stats=stage(c,table,dest,spec['name_start'],spec['row_start'],cfg['term_buckets']);verify(spec['source'])
                files=[Path(x) for x in stats.pop('files')];artifacts={str(x.relative_to(dest)):record(x) for x in [dest/'features.parquet',*files]}
                counts={str(b):{str(ch):0 for ch in range(4)} for b in range(cfg['term_buckets'])}
                for x in files:
                    b=x.parent.name.removeprefix('term_bucket=');t=pq.ParquetFile(x).read(columns=['channel']);vals=t['channel'].to_numpy(zero_copy_only=False)
                    for ch in range(4):counts[b][str(ch)]+=int((vals==ch).sum())
                if sum(sum(v.values()) for v in counts.values())!=stats['postings']:raise ValueError('Every channel/storage posting required')
                stats['bucket_channel_counts']=counts;authority=dict(part=spec)
            else:
                _,_,s=complete_stage(output);items=s['buckets'][str(index)];files=[]
                for item in items:
                    x=verify(item['artifact']);allowed=part_dir(output,item['part_id'])/'staging'/f'term_bucket={index}'
                    if not x.resolve().is_relative_to(allowed.resolve()):raise ValueError('Pinned storage path required')
                    files.append(x)
                expected=sum(i['channel_counts'][str(channel)] for i in items)
                stats=pack(c,files,dest,channel,index,cfg['term_buckets'],expected,p['names'])
                for i in items:verify(i['artifact'])
                artifacts={x.name:record(x) for x in sorted(dest.glob('*.npy'))};authority=dict(channel=channel,bucket=index,staging_manifest=record(output/'staging_manifest.json'))
            c.close();c=None;load(output);size=sum(x.stat().st_size for x in dest.rglob('*') if x.is_file())
            if size>cfg['max_phase_output_bytes']:raise ValueError('STOP_PHASE_OUTPUT_SIZE')
        m=dict(status='SEALED_V131_OFFICIAL_NAME_CHANNEL_PART' if phase=='part' else 'SEALED_V131_COMPLETE_PACKED_CHANNEL_BUCKET',
            launch=record(output/'launch.json'),authority=authority,stats=stats,artifacts=artifacts,resources=budget.report(),output_bytes=size,
            complete_retrieval_index=False,identifiers_as_features=False,**{k:False for k in PROTECTED})
        write_json(dest/'manifest.json',m);print(json.dumps(m),flush=True)
    except BaseException as exc:
        if c is not None:c.close()
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,phase,index,channel=None):
    l,p=load(output);maximum=len(p['parts']) if phase=='part' else l['config']['term_buckets']
    if phase not in ('part','bucket') or not isinstance(index,int) or isinstance(index,bool) or not 0<=index<maximum:raise ValueError('Planned physical phase required')
    if phase=='bucket':
        if channel not in range(4) or isinstance(channel,bool):raise ValueError('Four actual channels only')
        complete_stage(output)
    dest=part_dir(output,index) if phase=='part' else access_dir(output,channel,index);dest.mkdir(parents=True,exist_ok=False)
    try:
        with (dest/'worker.log').open('x') as log:
            args=[sys.executable,'-u','-m','scripts.compile_retrieval_v131_name_access','worker','--output',str(output),'--phase',phase,'--index',str(index)]
            if channel is not None:args+=['--channel',str(channel)]
            subprocess.run(args,check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((dest/'worker.log').read_text()[-5500:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def phase_result(path,output,status):
    m=json.loads(path.read_text())
    if (m['status']!=status or m['launch']!=record(output/'launch.json') or not m['resources']['pass_resource_policy']
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] or m['identifiers_as_features']):raise ValueError('Protected complete phase required')
    return m


def seal_parts(output):
    l,p=load(output);path=output/'staging_manifest.json'
    if path.exists():raise ValueError('Immutable staging already sealed')
    refs=[];features=[];buckets={str(b):[] for b in range(l['config']['term_buckets'])};names=0;posts=0;size=0;empty=0
    for spec in p['parts']:
        dest=part_dir(output,spec['part_id']);mp=dest/'manifest.json'
        if not mp.exists():raise ValueError('UNFINISHED_ALL_NAME_PARTS')
        m=phase_result(mp,output,'SEALED_V131_OFFICIAL_NAME_CHANNEL_PART');s=m['stats']
        if m['authority']['part']!=spec or s['names']!=spec['row_stop']-spec['row_start'] or not all(s[k] for k in ('all_scalar_feature_texts_exact','all_scalar_channel_sets_exact','all_exported_postings_exact','all_raw_names_retained')):raise ValueError('Every exact raw name/channel set required')
        feature=verify(m['artifacts']['features.parquet'])
        if feature.resolve()!=(dest/'features.parquet').resolve() or pq.ParquetFile(feature).metadata.num_rows!=s['names']:raise ValueError('Whole feature projection required')
        features.append(dict(part=spec,artifact=m['artifacts']['features.parquet']))
        seen={str(b):{str(ch):0 for ch in range(4)} for b in range(l['config']['term_buckets'])}
        for relative,ref in m['artifacts'].items():
            if relative=='features.parquet':continue
            x=verify(ref);b=x.parent.name.removeprefix('term_bucket=')
            if x.resolve()!=(dest/relative).resolve() or b not in buckets or not x.resolve().is_relative_to((dest/'staging'/f'term_bucket={b}').resolve()):raise ValueError('All physical posting paths required')
            counts={str(ch):0 for ch in range(4)}
            for batch in pq.ParquetFile(x).iter_batches(batch_size=131072,columns=['channel'],use_threads=False):
                vals=batch.column(0).to_numpy(zero_copy_only=False)
                if (vals>3).any():raise ValueError('Known actual channel required')
                for ch in range(4):counts[str(ch)]+=int((vals==ch).sum())
            for ch in range(4):seen[b][str(ch)]+=counts[str(ch)]
            buckets[b].append(dict(part_id=spec['part_id'],artifact=ref,channel_counts=counts))
        if seen!=s['bucket_channel_counts'] or sum(sum(v.values()) for v in seen.values())!=s['postings']:raise ValueError('Every whole channel posting required')
        names+=s['names'];posts+=s['postings'];empty+=s['feature_empty'];size+=m['output_bytes'];refs.append(record(mp))
    if names!=p['names'] or size>l['config']['max_total_output_bytes']:raise ValueError('Complete national names/size required')
    write_json(path,dict(status='SEALED_V131_ALL_OFFICIAL_NAME_CHANNEL_POSTINGS',launch=record(output/'launch.json'),source=p['source'],
        phase_authorities=refs,features=features,buckets=buckets,names=names,feature_empty=empty,postings=posts,output_bytes=size,
        **{k:False for k in PROTECTED},complete_retrieval_index=False))
    write_json(output/'access_launch.json',dict(launch=record(output/'launch.json'),staging_manifest=record(path)))


def seal_access(output):
    l,p,s=complete_stage(output)
    if (output/'manifest.json').exists():raise ValueError('Immutable name access already sealed')
    refs=[];access=[];posts=0;terms=0;size=s['output_bytes']
    for ch in range(4):
        for b in range(l['config']['term_buckets']):
            dest=access_dir(output,ch,b);mp=dest/'manifest.json'
            if not mp.exists():raise ValueError('UNFINISHED_ALL_CHANNEL_ACCESS')
            m=phase_result(mp,output,'SEALED_V131_COMPLETE_PACKED_CHANNEL_BUCKET');st=m['stats']
            if (m['authority']!=dict(channel=ch,bucket=b,staging_manifest=record(output/'staging_manifest.json')) or st['channel']!=ch or st['bucket']!=b
                or st['postings']!=sum(i['channel_counts'][str(ch)] for i in s['buckets'][str(b)]) or not st['every_original_posting_and_token_reconstructed_exactly']):raise ValueError('Every exact complete channel bucket required')
            for name,ref in m['artifacts'].items():
                x=verify(ref)
                if x.resolve()!=(dest/name).resolve():raise ValueError('Packed array path drift')
            access.append(dict(channel=ch,bucket=b,postings=st['postings'],terms=st['terms'],artifacts=m['artifacts']));refs.append(record(mp))
            posts+=st['postings'];terms+=st['terms'];size+=m['output_bytes']
    if posts!=s['postings'] or size>l['config']['max_total_output_bytes']:raise ValueError('Whole access denominator/size required')
    write_json(output/'manifest.json',dict(status='SEALED_V131_COMPLETE_OFFICIAL_NAME_LEXICAL_EXACT_ACCESS',launch=record(output/'launch.json'),
        source=p['source'],staging_manifest=record(output/'staging_manifest.json'),phase_authorities=refs,features=s['features'],access=access,
        names=p['names'],postings=posts,terms=terms,feature_empty=s['feature_empty'],output_bytes=size,
        complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','part','bucket','worker','seal-parts','seal']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--phase',choices=['part','bucket']);p.add_argument('--index',type=int);p.add_argument('--channel',type=int);a=p.parse_args()
    if a.mode in ('part','bucket','worker'):
        if a.index is None or a.mode=='worker' and a.phase is None:p.error('Planned physical phase required')
        (worker if a.mode=='worker' else execute)(a.output,a.phase if a.mode=='worker' else a.mode,a.index,a.channel)
    else:{'init':init,'seal-parts':seal_parts,'seal':seal_access}[a.mode](a.output)
