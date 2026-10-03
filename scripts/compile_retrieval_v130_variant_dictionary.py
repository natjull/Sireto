"""All V129 observations, bounded immutable storage parts and global dictionaries.

Not a retrieval experiment. No queries, labels, embeddings or candidate pools.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import duckdb
import pyarrow.parquet as pq

from scripts.project_retrieval_v129_national_variants import verify_plan as verify_original_plan
from src.xgb_matcher.retrieval_variant_dictionary_v130 import partition,dictionary
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v130_variant_dictionary.json'
PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


def load(output):
    launch=json.loads((output/'launch.json').read_text())
    for ref in launch['implementation'].values():verify(ref)
    if duckdb.__version__!=launch['config']['duckdb_version']:raise ValueError('Physical hash library version drift')
    plan=json.loads(verify(launch['plan']).read_text());verify(launch['config']['source_manifest'])
    verify_original_plan(Path(launch['config']['source_manifest']['path']).parent)
    return launch,plan


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text());source_ref=cfg['source_manifest']
    if not isinstance(cfg['buckets'],int) or isinstance(cfg['buckets'],bool) or not 1<=cfg['buckets']<=254:raise ValueError('Positive bounded storage bucket quantum required')
    source=json.loads(verify(source_ref).read_text());original_root=Path(source_ref['path']).parent
    if (source['status']!='SEALED_COMPLETE_ALL_OFFICIAL_VARIANT_SOURCE_PROJECTION' or not source['every_source_role_and_row_complete']
        or source['source_rows']!=cfg['expected_source_rows'] or source['observations']!=cfg['expected_observations']
        or source['truth_opened'] or source['crm_opened'] or source['recall_measured'] or source['complete_retrieval_index']
        or source['ready_for_fold1'] or duckdb.__version__!=cfg['duckdb_version']):raise ValueError('Complete official-only V129 authority required')
    _,original=verify_original_plan(original_root);parts=[];cursor=0;source_rows=0
    if set(original['role_order'])!={'history_ul','history_site','current_ul','current_site','official_names'}:raise ValueError('All official roles required')
    with Budget(cfg['resources']) as budget:
        for role in original['role_order']:
            for spec in original['sources'][role]['parts']:
                ref=record(original_root/role/f'part{spec["part_id"]:05d}'/'manifest.json')
                if source['parts'][len(parts)]!=ref:raise ValueError('Original all-part order/authority drift')
                m=json.loads(verify(ref).read_text())
                if (m['status']!='SEALED_OFFICIAL_VARIANT_PHYSICAL_PART_ONLY' or m['role']!=role or m['part']!=spec
                    or any(m.get(k) is not False for k in PROTECTED) or not m['resources']['pass_resource_policy']
                    or not m['every_source_row_retained'] or m['source_rows']!=spec['source_stop']-spec['source_start']):
                    raise ValueError('Complete protected physical source part required')
                path=verify(m['artifact']);f=pq.ParquetFile(path)
                if path.resolve()!=(Path(ref['path']).parent/'observations.parquet').resolve() or f.metadata.num_rows!=m['observations']:
                    raise ValueError('Every original observation file required')
                stop=cursor+m['observations']
                parts.append(dict(part_id=len(parts),role=role,original_part=spec['part_id'],source_manifest=ref,
                    source=m['artifact'],observation_start=cursor,observation_stop=stop,
                    observations=m['observations'],lexical_observations=m['lexical_observations']))
                cursor=stop;source_rows+=m['source_rows']
        if cursor!=cfg['expected_observations'] or source_rows!=cfg['expected_source_rows'] or len(parts)!=len(source['parts']) or cursor>=2**32:
            raise ValueError('Every whole-source row/observation/role required')
        if sum(p['lexical_observations'] for p in parts)!=cfg['expected_lexical_observations']:raise ValueError('Source lexical cardinality drift')
    plan=dict(status='SEALED_V130_EVERY_ORIGINAL_OBSERVATION_STORAGE_PLAN',source=source_ref,parts=parts,
        observations=cursor,lexical_observations=sum(p['lexical_observations'] for p in parts),
        physical_buckets=cfg['buckets'],source_rows=source_rows,source_full_output_hash_audit=budget.report(),
        **{k:False for k in PROTECTED})
    write_json(output/'plan.json',plan)
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v130_variant_dictionary.md',ROOT/'src/xgb_matcher/retrieval_variant_dictionary_v130.py',
        ROOT/'src/xgb_matcher/retrieval_variant_factor_v119.py',ROOT/'src/xgb_matcher/retrieval_variants_v118.py',
        ROOT/'scripts/project_retrieval_v129_national_variants.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
        ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),implementation={str(p):record(p) for p in paths},
        **{k:False for k in PROTECTED}))
    print(json.dumps(dict(planned_parts=len(parts),observations=cursor,buckets=cfg['buckets'],source_audit=budget.report())),flush=True)


def stage_dir(output,part):return output/'parts'/f'part{part:05d}'
def bucket_dir(output,bucket):return output/'dictionary'/f'bucket{bucket:03d}'


def database(cfg,destination):
    c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
    c.execute('SET temp_directory=?',[str(destination/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
    return c


def stage_authority(output):
    launch,plan=load(output);d=json.loads((output/'dictionary_launch.json').read_text())
    if d['launch']!=record(output/'launch.json'):raise ValueError('Dictionary launch drift')
    staged=json.loads(verify(d['staging_manifest']).read_text())
    if staged['status']!='SEALED_V130_COMPLETE_NAME_STORAGE_ALL_ORIGINAL_PARTS' or staged['observations']!=plan['observations']:
        raise ValueError('All source storage parts required')
    return launch,plan,staged


def worker(output,phase,index):
    launch,plan=load(output);cfg=launch['config'];destination=stage_dir(output,index) if phase=='part' else bucket_dir(output,index)
    if (destination/'manifest.json').exists() or (destination/'worker_stop.json').exists() or (destination/'staging').exists() or any(destination.glob('*.parquet')):
        raise ValueError('Immutable physical phase already started')
    budget=None;c=None
    try:
        with Budget(cfg['resources']) as budget:
            c=database(cfg,destination)
            if phase=='part':
                spec=plan['parts'][index];verify(spec['source_manifest']);source=verify(spec['source'])
                stats=partition(c,source,destination,spec['observation_start'],spec['observations'],spec['lexical_observations'],
                    cfg['buckets'],lambda value:print(json.dumps(value),flush=True))
                verify(spec['source']);files=[Path(p) for p in stats.pop('files')]
                artifacts={str(p.relative_to(destination)):record(p) for p in files}
                counts={str(b):sum(pq.ParquetFile(p).metadata.num_rows for p in files if p.parent.name==f'bucket={b}') for b in range(cfg['buckets'])}
                if sum(counts.values())!=spec['lexical_observations']:raise ValueError('Every source row staged exactly once')
                stats.update(bucket_counts=counts);authority=dict(source_manifest=spec['source_manifest'],part=spec)
            else:
                _,_,staged=stage_authority(output);inputs=staged['buckets'][str(index)];files=[]
                for item in inputs:
                    path=verify(item['artifact']);part=plan['parts'][item['part_id']]
                    if not path.resolve().is_relative_to((stage_dir(output,item['part_id'])/'staging'/f'bucket={index}').resolve()):
                        raise ValueError('Pinned physical bucket path required')
                    if pq.ParquetFile(path).metadata.num_rows!=item['observations']:raise ValueError('Every staged observation required')
                    files.append(path)
                expected=sum(i['observations'] for i in inputs)
                stats=dictionary(c,files,destination,index,cfg['buckets'],expected,lambda value:print(json.dumps(value),flush=True))
                for item in inputs:verify(item['artifact'])
                artifacts={p.name:record(p) for p in sorted(destination.glob('*.parquet'))}
                authority=dict(staging_manifest=record(output/'staging_manifest.json'),bucket=index)
            c.close();c=None;load(output)
            size=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
            if size>cfg['max_phase_output_bytes']:raise ValueError('STOP_PHASE_OUTPUT_SIZE')
        result=dict(status='SEALED_V130_OFFICIAL_NAME_STORAGE_PART' if phase=='part' else 'SEALED_V130_COMPLETE_DICTIONARY_BUCKET',
            launch=record(output/'launch.json'),authority=authority,stats=stats,artifacts=artifacts,resources=budget.report(),
            output_bytes=size,complete_retrieval_index=False,identifiers_as_features=False,**{k:False for k in PROTECTED})
        write_json(destination/'manifest.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        if c is not None:c.close()
        write_json(destination/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,phase,index):
    launch,plan=load(output);maximum=len(plan['parts']) if phase=='part' else launch['config']['buckets']
    if phase not in ('part','bucket') or not isinstance(index,int) or isinstance(index,bool) or not 0<=index<maximum:raise ValueError('Planned physical phase required')
    if phase=='bucket':stage_authority(output)
    destination=stage_dir(output,index) if phase=='part' else bucket_dir(output,index);destination.mkdir(parents=True,exist_ok=False)
    try:
        with (destination/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v130_variant_dictionary','worker','--output',str(output),
                '--phase',phase,'--index',str(index)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((destination/'worker.log').read_text()[-5500:],flush=True)
    except BaseException as exc:
        write_json(destination/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def phase_manifest(path,launch,status):
    m=json.loads(path.read_text())
    if (m['status']!=status or m['launch']!=record(launch) or not m['resources']['pass_resource_policy']
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] or m['identifiers_as_features']):
        raise ValueError('Protected complete physical phase required')
    return m


def seal_parts(output):
    launch,plan=load(output);path=output/'staging_manifest.json'
    if path.exists():raise ValueError('Immutable complete staging already sealed')
    refs=[];buckets={str(b):[] for b in range(launch['config']['buckets'])};total=0;eligible=0;size=0
    for spec in plan['parts']:
        directory=stage_dir(output,spec['part_id']);mfile=directory/'manifest.json'
        if not mfile.exists():raise ValueError('UNFINISHED_SOURCE_STORAGE_PARTS')
        m=phase_manifest(mfile,output/'launch.json','SEALED_V130_OFFICIAL_NAME_STORAGE_PART');stats=m['stats']
        if m['authority']['part']!=spec or stats['original_observations']!=spec['observations'] or stats['lexical_observations']!=spec['lexical_observations'] or not stats['all_eligible_rows_unique_and_exact']:
            raise ValueError('Every original source count/ordinal required')
        seen={str(b):0 for b in range(launch['config']['buckets'])}
        for relative,ref in m['artifacts'].items():
            p=verify(ref)
            if p.resolve()!=(directory/relative).resolve():raise ValueError('Physical storage artifact path drift')
            b=p.parent.name.removeprefix('bucket=')
            if b not in buckets or not p.resolve().is_relative_to((directory/'staging'/f'bucket={b}').resolve()):raise ValueError('Known storage bucket required')
            count=pq.ParquetFile(p).metadata.num_rows;seen[b]+=count
            buckets[b].append(dict(part_id=spec['part_id'],artifact=ref,observations=count))
        if seen!=stats['bucket_counts'] or sum(seen.values())!=spec['lexical_observations']:raise ValueError('Every staged observation/file required')
        total+=spec['observations'];eligible+=spec['lexical_observations'];size+=m['output_bytes'];refs.append(record(mfile))
    if total!=plan['observations'] or eligible!=plan['lexical_observations'] or size>launch['config']['max_total_output_bytes']:raise ValueError('Complete storage denominator/size required')
    write_json(path,dict(status='SEALED_V130_COMPLETE_NAME_STORAGE_ALL_ORIGINAL_PARTS',launch=record(output/'launch.json'),
        source=plan['source'],phase_authorities=refs,buckets=buckets,observations=total,lexical_observations=eligible,
        context_observations=total-eligible,output_bytes=size,**{k:False for k in PROTECTED},complete_retrieval_index=False))
    write_json(output/'dictionary_launch.json',dict(launch=record(output/'launch.json'),staging_manifest=record(path)))


def seal_dictionary(output):
    launch,plan,staged=stage_authority(output)
    if (output/'manifest.json').exists():raise ValueError('Immutable dictionary already sealed')
    refs=[];results=[];names=0;facts=0;obs=0;size=staged['output_bytes']
    for b in range(launch['config']['buckets']):
        directory=bucket_dir(output,b);path=directory/'manifest.json'
        if not path.exists():raise ValueError('UNFINISHED_DICTIONARY_BUCKETS')
        m=phase_manifest(path,output/'launch.json','SEALED_V130_COMPLETE_DICTIONARY_BUCKET');s=m['stats']
        if (s['bucket']!=b or m['authority']!=dict(staging_manifest=record(output/'staging_manifest.json'),bucket=b)
            or s['observations']!=sum(i['observations'] for i in staged['buckets'][str(b)])
            or not s['every_owner_and_name_reconstructed_exactly'] or not s['every_eligible_observation_linked_once']):
            raise ValueError('All exact bucket observations required')
        for name,count in [('names.parquet',s['names']),('facts.parquet',s['facts']),('observation_links.parquet',s['observations'])]:
            ref=m['artifacts'][name];p=verify(ref)
            if p.resolve()!=(directory/name).resolve() or pq.ParquetFile(p).metadata.num_rows!=count:raise ValueError('Whole dictionary output count/path drift')
        results.append(dict(bucket=b,name_start=names,fact_start=facts,names=s['names'],facts=s['facts'],observations=s['observations'],artifacts=m['artifacts']))
        names+=s['names'];facts+=s['facts'];obs+=s['observations'];size+=m['output_bytes'];refs.append(record(path))
    if obs!=plan['lexical_observations'] or max(names,facts)>=2**32 or size>launch['config']['max_total_output_bytes']:raise ValueError('Whole dictionary denominator/size/range drift')
    write_json(output/'manifest.json',dict(status='SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY',launch=record(output/'launch.json'),
        plan=record(output/'plan.json'),source=plan['source'],staging_manifest=record(output/'staging_manifest.json'),
        phase_authorities=refs,buckets=results,names=names,facts=facts,lexical_observations=obs,
        original_observations=plan['observations'],original27_fields_in_sealed_source=True,output_bytes=size,
        complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','part','bucket','worker','seal-parts','seal']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--phase',choices=['part','bucket']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.mode in ('part','bucket','worker'):
        if a.index is None or a.mode=='worker' and a.phase is None:p.error('Physical phase index required')
        (worker if a.mode=='worker' else execute)(a.output,a.phase if a.mode=='worker' else a.mode,a.index)
    else:{'init':init,'seal-parts':seal_parts,'seal':seal_dictionary}[a.mode](a.output)
