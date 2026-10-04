"""Exhaustive batched seal of existing successful V168/V166 address parts."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

import scripts.compile_retrieval_v168_address_access as backend
from src.xgb_matcher.retrieval_direct_alias_v172 import SCHEMA
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v175_address_seal.json'
STATUS='SEALED_V175_ALL_WHOLE_ADDRESS_CHANNEL_POSTINGS'
FLAGS={k:False for k in PROTECTED}
COUNT_FIELDS=('aliases','facts','sites','original_observations','lexical_observations')


def config():
    cfg=json.loads(CONFIG.read_text())
    if (cfg['resources']!=LIMITS or any(type(cfg[k]) is not int or cfg[k]<=0
        for k in (*COUNT_FIELDS,'parts','parts_per_batch','term_buckets'))
        or cfg['term_buckets']!=64 or cfg['parts_per_batch']!=16):
        raise ValueError('Exact bounded complete-source seal policy required')
    return cfg


def source(cfg):
    root=verify(cfg['source_launch']).parent;l,p=backend.load(root)
    if (record(verify(cfg['source_plan']))!=l['plan'] or len(p['parts'])!=cfg['parts']
        or any(p[k]!=cfg[k] for k in COUNT_FIELDS) or l['config']['term_buckets']!=cfg['term_buckets']):
        raise ValueError('Same entire original address source and parts required')
    cursor=0
    for i,spec in enumerate(p['parts']):
        start=spec['alias_start']+spec['row_start'];stop=spec['alias_start']+spec['row_stop']
        if spec['part_id']!=i or start!=cursor or stop<=start:
            raise ValueError('Every original physical alias interval exactly once required')
        cursor=stop
    if cursor!=cfg['aliases']:raise ValueError('Whole official address alias denominator required')
    return root,l,p


def epochs(refs):
    # Artifact hashes here are the original part's pinned generation claims.
    # The bounded batch verifies EVERY one in full before/after use. Doing
    # that global rehash here would recreate the unbounded monolithic seal.
    return [dict(reference=r,fingerprint=backend.fingerprint(r['path'])) for r in refs]


def check_epochs(items):
    for item in items:
        if backend.fingerprint(item['reference']['path'])!=item['fingerprint']:
            raise ValueError('STOP_ORIGINAL_PART_OR_ARTIFACT_EPOCH_DRIFT')


def original_directory(root,p,index):
    item=next((x for x in p['resumed_successful_parts'] if x['index']==index),None)
    return Path(item['manifest']['path']).parent if item is not None else backend.directory(root,'part',index)


def implementation():
    files=[*backend.implementation_paths(),Path(__file__),CONFIG,
        ROOT/'src/xgb_matcher/retrieval_direct_alias_v172.py',ROOT/'docs/retrieval_v175_address_seal_contract.md']
    return {str(f):record(f) for f in files}


def init(output):
    cfg=config()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only seal root required')
    if output.exists():raise FileExistsError('Immutable seal root exists')
    gate=quiet(cfg['quiet_seconds'],50,55);budget=None
    try:
        with Budget(LIMITS) as budget:
            root,l,p=source(cfg);ledger=[]
            for spec in p['parts']:
                dest=original_directory(root,p,spec['part_id'])
                if any((dest/n).exists() for n in ('worker_stop.json','status.json','supervisor_stop.json')):
                    raise ValueError('STOP_SOURCE_PART_DO_NOT_IMPORT_OR_RESTART')
                parent_exit(dest)
                ref=record(dest/'manifest.json');m=json.loads(verify(ref).read_text())
                refs=[ref,record(dest/'supervision.json'),record(dest/'preflight.json'),*m['artifacts'].values()]
                # INIT fixes every byte authority/epoch, but does not yet
                # certify the channel reread or global staging completeness.
                ledger.append(dict(index=spec['part_id'],manifest=ref,epochs=epochs(refs)))
            backend.revalidate(root,l,p)
        output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
        write_json(output/'ledger.json',dict(parts=ledger,resources=budget.report(),complete_retrieval_index=False,**FLAGS))
        write_json(output/'launch.json',dict(config=cfg,ledger=record(output/'ledger.json'),preflight=record(output/'preflight.json'),
            implementation=implementation(),complete_retrieval_index=False,**FLAGS))
    except BaseException as exc:
        if not output.exists():output.mkdir(parents=True,exist_ok=False)
        write_json(output/'init_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**FLAGS));raise


def context(output):
    if (output/'init_stop.json').exists() or (output/'seal_stop.json').exists():raise ValueError('STOP_SEAL_DO_NOT_RESTART')
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config']
    if (cfg!=config() or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())
        or any(launch.get(k) is not False for k in PROTECTED) or launch['complete_retrieval_index']):
        raise ValueError('Immutable protected source-bound seal launch required')
    for ref in launch['implementation'].values():verify(ref)
    gate=json.loads(verify(launch['preflight']).read_text())
    if (verify(launch['preflight']).resolve()!=output/'preflight.json' or gate['policy_id']!=POLICY
        or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']):
        raise ValueError('Actual authorized pressure gate required')
    ledger=json.loads(verify(launch['ledger']).read_text());validate_report(ledger['resources'])
    if (len(ledger['parts'])!=cfg['parts'] or [x['index'] for x in ledger['parts']]!=list(range(cfg['parts']))
        or any(ledger.get(k) is not False for k in PROTECTED) or ledger['complete_retrieval_index']):
        raise ValueError('Every original fixed part byte authority required')
    return launch,ledger


def directory(output,index):return output/'batches'/f'batch{index:04d}'


def indices(cfg,index):
    if type(index) is not int or not 0<=index<(cfg['parts']+cfg['parts_per_batch']-1)//cfg['parts_per_batch']:
        raise ValueError('Actual physical audit batch required')
    return range(index*cfg['parts_per_batch'],min(cfg['parts'],(index+1)*cfg['parts_per_batch']))


def audited_part(root,l,p,item,buckets):
    index=item['index'];check_epochs(item['epochs'])
    m=backend.phase_record(root,'part',index,context=(l,p));spec=p['parts'][index];s=m['stats']
    dest=Path(m['preflight']['path']).parent
    if (item['manifest']!=record(dest/'manifest.json') or m['inputs']!=dict(part=spec)
        or s['addresses']!=spec['row_stop']-spec['row_start']
        or not all(s[k] for k in ('all_scalar_feature_texts_exact','all_scalar_channel_sets_exact',
            'all_exported_postings_exact','all_raw_addresses_retained','source_address_normalization_unchanged'))
        or s['address_fact_attributes_or_scope_changed'] or s['alias_or_proof_cap'] is not None):
        raise ValueError('Every unchanged original whole-address scalar/channel proof required')
    feature=m['artifacts']['features.parquet'];table=pq.ParquetFile(verify(feature)).read(use_threads=False)
    begin=spec['alias_start']+spec['row_start'];stop=spec['alias_start']+spec['row_stop']
    if (table.schema!=SCHEMA or len(table)!=s['addresses'] or table['feature_text'].null_count
        or table['alias_id'].null_count or not np.array_equal(table['alias_id'].to_numpy(),np.arange(begin,stop,dtype=np.uint32))):
        raise ValueError('Every typed raw alias feature retained exactly once required')
    del table
    observed={str(b):{str(ch):0 for ch in range(4)} for b in range(buckets)};posts=[]
    for name,ref in m['artifacts'].items():
        if name=='features.parquet':continue
        path=verify(ref);b=path.parent.name.removeprefix('term_bucket=')
        if (b not in observed or not path.resolve().is_relative_to((dest/'staging'/f'term_bucket={b}').resolve())
            or any(x['bucket']==b for x in posts)):
            raise ValueError('Every original posting path/bucket exactly once required')
        counts={str(ch):0 for ch in range(4)}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=131072,columns=['channel'],use_threads=False):
            values=batch.column(0).to_numpy(zero_copy_only=False)
            if values.dtype!=np.uint8 or np.any(values>3):raise ValueError('Every unchanged known typed channel required')
            for ch in range(4):counts[str(ch)]+=int(np.count_nonzero(values==ch))
        observed[b]=counts;posts.append(dict(bucket=b,part_id=index,artifact=ref,channel_counts=counts))
    total=sum(sum(x.values()) for x in observed.values())
    if observed!=s['bucket_channel_counts'] or total!=s['postings']:
        raise ValueError('Every original channel/posting count required')
    for ref in m['artifacts'].values():verify(ref)  # Full after-use hash, not just stat.
    check_epochs(item['epochs'])
    return dict(index=index,manifest=item['manifest'],feature=dict(part=spec,artifact=feature),posts=posts,
        aliases=s['addresses'],postings=total,feature_empty=s['feature_empty'],output_bytes=m['output_bytes'])


def worker(output,index):
    dest=directory(output,index);budget=None
    if any(f.name not in ('preflight.json','worker.log') for f in dest.iterdir()):raise ValueError('Immutable batch, no retry')
    try:
        with Budget(LIMITS) as budget:
            launch,ledger=context(output);cfg=launch['config'];selected=indices(cfg,index)
            gate=json.loads((dest/'preflight.json').read_text())
            if gate['policy_id']!=POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
                raise ValueError('Actual authorized batch pressure gate required')
            root,l,p=source(cfg)
            rows=[audited_part(root,l,p,ledger['parts'][i],cfg['term_buckets']) for i in selected]
            backend.revalidate(root,l,p)
            for i in selected:check_epochs(ledger['parts'][i]['epochs'])
            for ref in launch['implementation'].values():verify(ref)
            verify(launch['ledger']);verify(cfg['source_launch']);verify(cfg['source_plan'])
        write_json(dest/'manifest.json',dict(status='SEALED_V175_EXHAUSTIVE_PART_AUDIT_BATCH',index=index,
            launch=record(output/'launch.json'),preflight=record(dest/'preflight.json'),rows=rows,resources=budget.report(),
            every_used_artifact_full_hash_before_and_after=True,complete_retrieval_index=False,**FLAGS))
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**FLAGS));raise


def batch_record(output,index):
    launch,ledger=context(output);dest=directory(output,index)
    if any((dest/n).exists() for n in ('worker_stop.json','status.json','supervisor_stop.json')):
        raise ValueError('STOP_BATCH_DO_NOT_RESTART')
    parent_exit(dest);m=json.loads((dest/'manifest.json').read_text());validate_report(m['resources'])
    if (m['status']!='SEALED_V175_EXHAUSTIVE_PART_AUDIT_BATCH' or m['index']!=index
        or m['launch']!=record(output/'launch.json') or m['preflight']!=record(dest/'preflight.json')
        or not m['every_used_artifact_full_hash_before_and_after'] or m['complete_retrieval_index']
        or any(m.get(k) is not False for k in PROTECTED)
        or [x['index'] for x in m['rows']]!=list(indices(launch['config'],index))):
        raise ValueError('Protected whole original supervised audit batch required')
    for row in m['rows']:
        item=ledger['parts'][row['index']];check_epochs(item['epochs'])
        if row['manifest']!=item['manifest'] or record(verify(row['manifest']))!=item['manifest']:
            raise ValueError('Same original certified part required')
        original=json.loads(verify(item['manifest']).read_text());s=original['stats']
        expected_posts=[]
        for name,ref in original['artifacts'].items():
            if name=='features.parquet':continue
            b=Path(ref['path']).parent.name.removeprefix('term_bucket=')
            expected_posts.append(dict(bucket=b,part_id=row['index'],artifact=ref,
                channel_counts=s['bucket_channel_counts'][b]))
        expected=dict(index=row['index'],manifest=item['manifest'],
            feature=dict(part=original['inputs']['part'],artifact=original['artifacts']['features.parquet']),
            posts=expected_posts,aliases=s['addresses'],postings=s['postings'],
            feature_empty=s['feature_empty'],output_bytes=original['output_bytes'])
        if row!=expected:raise ValueError('Every audited field equals its unchanged original part authority')
    gate=json.loads(verify(m['preflight']).read_text())
    if gate['policy_id']!=POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual authorized batch preflight required')
    return m


def execute(output,index):
    launch,_=context(output);cfg=launch['config'];indices(cfg,index);dest=directory(output,index)
    dest.mkdir(parents=True,exist_ok=False)
    try:
        write_json(dest/'preflight.json',quiet(cfg['phase_quiet_seconds'],50,55))
        with (dest/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.seal_retrieval_v175_address_parts','worker',
                '--output',str(output),'--index',str(index)],log,
                {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},dest/'supervisor_stop.json')
        write_json(dest/'supervision.json',s);batch_record(output,index)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**FLAGS));raise


def remaining(output):
    launch,_=context(output);cfg=launch['config']
    for i in range((cfg['parts']+cfg['parts_per_batch']-1)//cfg['parts_per_batch']):
        if directory(output,i).exists():batch_record(output,i)
        else:execute(output,i)


def seal_worker(output):
    launch,ledger=context(output);cfg=launch['config']
    if (output/'staging_manifest.json').exists():raise FileExistsError('Immutable full stage already sealed')
    count=(cfg['parts']+cfg['parts_per_batch']-1)//cfg['parts_per_batch']
    budget=None
    try:
        with Budget(LIMITS) as budget:
            gate=json.loads((output/'seal'/'preflight.json').read_text())
            if gate['policy_id']!=POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
                raise ValueError('Actual authorized seal pressure gate required')
            root,l,p=source(cfg);buckets={str(b):[] for b in range(cfg['term_buckets'])}
            features=[];part_refs=[];audit_refs=[];cursor=posts=empty=size=0
            for i in range(count):
                m=batch_record(output,i);audit_refs.append(record(directory(output,i)/'manifest.json'))
                for row in m['rows']:
                    spec=p['parts'][row['index']]
                    if (row['feature']['part']!=spec or spec['alias_start']+spec['row_start']!=cursor
                        or row['aliases']!=spec['row_stop']-spec['row_start']
                        or sum(sum(x['channel_counts'].values()) for x in row['posts'])!=row['postings']):
                        raise ValueError('Every original alias and channel denominator required')
                    features.append(row['feature']);part_refs.append(row['manifest'])
                    for item in row['posts']:
                        b=item['bucket'];buckets[b].append({k:v for k,v in item.items() if k!='bucket'})
                    cursor+=row['aliases'];posts+=row['postings'];empty+=row['feature_empty'];size+=row['output_bytes']
            if cursor!=p['aliases'] or len(features)!=cfg['parts'] or size>l['config']['max_total_bytes']:
                raise ValueError('Entire original address/staging denominator and disk budget required')
            _,rm=backend.audit_record(root,'seal','reverse')
            if any(rm[k]!=p[k] for k in ('dictionary','catalog',*COUNT_FIELDS,'parts','source_epochs')):
                raise ValueError('Same complete full-hash original source at final seal required')
            for item in ledger['parts']:check_epochs(item['epochs'])
            backend.revalidate(root,l,p)
            for ref in launch['implementation'].values():verify(ref)
            verify(launch['ledger'])
            data=dict(status=STATUS,launch=record(output/'launch.json'),source=p['source_store'],
                aliases=cursor,postings=posts,feature_empty=empty,features=features,buckets=buckets,
                phase_authorities=part_refs,audit_authorities=audit_refs,output_bytes=size,
                sealed_source_audits={kind:backend.audit_record(root,'seal',kind)[0] for kind in ('prefix','reverse')},
                sealed_source_audit_supervisions={kind:record(root/'source_audits'/'seal'/kind/'supervision.json') for kind in ('prefix','reverse')},
                every_original_whole_alias_and_posting_retained=True,complete_retrieval_index=False,ready_for_fold1=False,**FLAGS)
        write_json(output/'staging_manifest.json',dict(data,seal_resources=budget.report()))
    except BaseException as exc:
        write_json(output/'seal_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**FLAGS));raise


def seal(output):
    launch,_=context(output);cfg=launch['config'];dest=output/'seal'
    if (output/'staging_manifest.json').exists() or dest.exists():raise FileExistsError('Immutable seal already started')
    # No source rehash until ALL original parts have bounded complete audits.
    count=(cfg['parts']+cfg['parts_per_batch']-1)//cfg['parts_per_batch']
    for i in range(count):batch_record(output,i)
    root=verify(cfg['source_launch']).parent
    try:
        if not (root/'source_audits'/'seal').exists():backend.run_audits(root,'seal')
        for kind in ('prefix','reverse'):backend.audit_record(root,'seal',kind)
        dest.mkdir(parents=True,exist_ok=False)
        write_json(dest/'preflight.json',quiet(cfg['phase_quiet_seconds'],50,55))
        with (dest/'worker.log').open('x') as log:
            s=supervise([sys.executable,'-u','-m','scripts.seal_retrieval_v175_address_parts','seal-worker',
                '--output',str(output)],log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},dest/'supervisor_stop.json')
        write_json(dest/'supervision.json',s);parent_exit(dest)
        m=json.loads((output/'staging_manifest.json').read_text());validate_report(m['seal_resources'])
        if m['status']!=STATUS or m['launch']!=record(output/'launch.json'):
            raise ValueError('Exact supervised complete stage certificate required')
    except BaseException as exc:
        # A worker manifest without a confirmed parent exit is never accepted.
        write_json(output/'seal_stop.json',dict(status='STOP',reason=str(exc),**FLAGS));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','phase','worker','remaining','seal','seal-worker'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--index',type=int);a=p.parse_args()
    if a.mode in ('phase','worker'):
        if a.index is None:p.error('Actual physical audit batch required')
        (execute if a.mode=='phase' else worker)(a.output,a.index)
    else:{'init':init,'remaining':remaining,'seal':seal,'seal-worker':seal_worker}[a.mode](a.output)
