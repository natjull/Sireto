"""Full V160 store -> lexical access under authorized V153 pressure policy.

V152 semantics are unchanged: all official observations before any worker,
all whole aliases and exact original channel postings, no candidate pool.
Historical V148/V152 resource reports are never reinterpreted.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import numpy as np
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_address_access_v163 import STATUS,physical_plan,stage_addresses
from src.xgb_matcher.retrieval_address_authority_v163 import STATUS as RESUMED_STORE,store as resumed_store,epochs as resumed_epochs
from src.xgb_matcher.retrieval_packed_names_v134 import pack
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report
from src.xgb_matcher.retrieval_supervisor_v157 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v163_address_access.json'
STAGE='SEALED_V163_ALL_WHOLE_ADDRESS_CHANNEL_POSTINGS'


def fingerprint(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def source_epochs(reference,m,d):
    """Every original source epoch, including supervised actual exits."""
    return resumed_epochs(reference,m,d)


def config():
    cfg=json.loads(CONFIG.read_text())
    if (duckdb.__version__!=cfg['duckdb_version'] or np.__version__!=cfg['numpy_version']
        or any(not isinstance(cfg[k],int) or isinstance(cfg[k],bool) or cfg[k]<=0 for k in
            ('term_buckets','batch_rows','expected_sites','expected_original_observations','expected_source_rows'))
        or cfg['term_buckets']>254 or cfg['resources']!=LIMITS):
        raise ValueError('Pinned whole-source typed policy and authorized pressure/RSS limits required')
    return cfg


def implementation_paths():
    return [Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_address_access_v163.py',
        ROOT/'src/xgb_matcher/retrieval_address_authority_v163.py',ROOT/'scripts/resume_retrieval_v160_reverse_store.py',
        ROOT/'config/retrieval_v160_reverse_resume.json',ROOT/'src/xgb_matcher/retrieval_address_stage_v159.py',
        ROOT/'scripts/compile_retrieval_v154_address_store.py',ROOT/'scripts/compile_retrieval_v152_address_access.py',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v153.py',ROOT/'src/xgb_matcher/retrieval_supervisor_v157.py',
        ROOT/'src/xgb_matcher/retrieval_address_access_v152.py',*[ROOT/('src/xgb_matcher/'+s+'.py') for s in
        ('retrieval_name_access_v131','retrieval_packed_names_v134','retrieval_national_names_v138',
         'retrieval_coupled_blocks_v121','retrieval_variants_v118','retrieval_name_postings_v86',
         'retrieval_variant_factor_v119','retrieval_system_quiet_v143','retrieval_address_reverse_v147','retrieval_v20_workbench')]]


def audit_context(output):
    a=json.loads((output/'audit_launch.json').read_text());cfg=a['config']
    if (cfg!=config() or any(a.get(k) is not False for k in PROTECTED)
        or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())):
        raise ValueError('Immutable complete source audit launch required')
    for ref in a['implementation'].values():verify(ref)
    return a,cfg


def audit_record(output,tag,kind):
    from src.xgb_matcher.retrieval_address_authority_v163 import parent_exit
    dest=output/'source_audits'/tag/kind
    if any((dest/n).exists() for n in ('worker_stop.json','status.json')):
        raise ValueError('STOP_SOURCE_AUDIT_DO_NOT_RESTART')
    parent_exit(dest)
    ref=record(dest/'manifest.json');m=json.loads(verify(ref).read_text())
    validate_report(m['resources']);a,cfg=audit_context(output)
    if (m['launch']!=record(output/'audit_launch.json') or m['tag']!=tag or m['kind']!=kind
        or m['source_store']!=a['source_store'] or any(m.get(k) is not False for k in PROTECTED)
        or m['complete_retrieval_index'] or m['ready_for_fold1']
        or m['status']!='SEALED_V163_COMPLETE_ORIGINAL_'+kind.upper()+'_AUDIT'):
        raise ValueError('Exact full source audit certificate required')
    for epoch in m['source_epochs']:
        if fingerprint(epoch['reference']['path'])!=epoch['fingerprint']:
            raise ValueError('Source epoch drift between complete audits')
    return ref,m


def audit_worker(output,tag,kind):
    import scripts.resume_retrieval_v160_reverse_store as backend
    a,cfg=audit_context(output);dest=output/'source_audits'/tag/kind
    if kind not in ('prefix','reverse') or tag not in ('init','seal'):
        raise ValueError('Only complete original prefix/reverse audits')
    if any(f.name not in ('preflight.json','worker.log') for f in dest.iterdir()):
        raise ValueError('Immutable source audit, no retry')
    budget=None
    try:
        gate=json.loads(verify(record(dest/'preflight.json')).read_text())
        if not gate['pass_system_quiet_policy'] or gate['policy_id']!=POLICY:
            raise ValueError('Actual pressure preflight required')
        with Budget(cfg['resources']) as budget:
            if kind=='prefix':
                verify(a['source_store']);root=Path(a['source_store']['path']).parent
                l,p=backend.load(root)
                epochs=[*p['prefix_epochs']]
                for ref in (a['source_store'],record(root/'launch.json'),l['plan']):
                    epochs.append(dict(reference=ref,fingerprint=fingerprint(ref['path'])))
                audited=backend.prefix(l['config'])
                catalog,d=audited[1]['catalog'],audited[2]
                del audited
                backend.load(root)
                for epoch in epochs:
                    if fingerprint(epoch['reference']['path'])!=epoch['fingerprint']:
                        raise ValueError('Full original prefix source changed during audit')
                payload=dict(catalog=catalog,dictionary=p['dictionary'],source_epochs=epochs)
            else:
                ref,prefix=audit_record(output,tag,'prefix')
                m,d=resumed_store(a['source_store'],cfg,ref)
                parts=physical_plan(d['buckets'],d['aliases'],cfg['batch_rows'])
                epochs=source_epochs(a['source_store'],m,d)
                audit_record(output,tag,'prefix')
                payload=dict(dictionary=m['dictionary'],catalog=m['catalog'],aliases=d['aliases'],facts=d['facts'],
                    sites=m['sites'],original_observations=m['original_observations'],
                    lexical_observations=m['lexical_observations'],parts=parts,source_epochs=epochs,
                    prefix_audit=ref)
            audit_context(output)
        write_json(dest/'manifest.json',dict(status='SEALED_V163_COMPLETE_ORIGINAL_'+kind.upper()+'_AUDIT',
            launch=record(output/'audit_launch.json'),tag=tag,kind=kind,source_store=a['source_store'],**payload,
            resources=budget.report(),complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
            complete_retrieval_index=False,**{k:False for k in PROTECTED}));raise


def run_audits(output,tag):
    a,cfg=audit_context(output)
    for kind in ('prefix','reverse'):
        dest=output/'source_audits'/tag/kind
        dest.mkdir(parents=True,exist_ok=False)
        try:
            write_json(dest/'preflight.json',quiet(cfg['phase_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds']))
            args=[sys.executable,'-u','-m','scripts.compile_retrieval_v163_address_access','audit-worker',
                '--output',str(output),'--audit-tag',tag,'--audit-kind',kind]
            with (dest/'worker.log').open('x') as log:
                supervision=supervise(args,log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            write_json(dest/'supervision.json',supervision)
            audit_record(output,tag,kind)
        except BaseException as exc:
            write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise
    return audit_record(output,tag,'prefix'),audit_record(output,tag,'reverse')


def init(output,reference):
    cfg=config()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only output root required')
    if output.exists():raise FileExistsError('Immutable address access root already exists')
    if not Path(reference['path']).is_file():raise ValueError('UNFINISHED_COMPLETE_ADDRESS_STORE_NO_WORKER_LAUNCHED')
    output.mkdir(parents=True,exist_ok=False)
    try:
        gate=quiet(cfg['initial_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
        write_json(output/'preflight.json',gate)
        paths=implementation_paths()
        # V160 launch already pins all upstream code; no GLOBAL object is retained.
        root=verify(reference).parent
        inherited=json.loads((root/'launch.json').read_text())['implementation']
        write_json(output/'audit_launch.json',dict(config=cfg,source_store=reference,
            implementation={**inherited,**{str(p):record(p) for p in paths}},**{k:False for k in PROTECTED}))
        (pref,pm),(rev,rm)=run_audits(output,'init')
        write_json(output/'plan.json',dict(status='SEALED_V163_ALL_RAW_ADDRESS_PHYSICAL_PLAN',
            source_store=reference,**{k:rm[k] for k in ('dictionary','catalog','aliases','facts','sites',
                'original_observations','lexical_observations','parts','source_epochs')},
            source_audit_resources=rm['resources'],prefix_audit_resources=pm['resources'],
            source_audits=dict(prefix=pref,reverse=rev),audit_launch=record(output/'audit_launch.json'),
            source_audit_supervisions={kind:record(output/'source_audits'/'init'/kind/'supervision.json') for kind in ('prefix','reverse')},
            full_source_hashes_at_init_and_seal_not_each_phase=True,**{k:False for k in PROTECTED}))
        a,_=audit_context(output)
        write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),preflight=record(output/'preflight.json'),
            implementation=a['implementation'],**{k:False for k in PROTECTED}))
    except BaseException as exc:
        write_json(output/'init_stop.json',dict(status='STOP',reason=str(exc),source_store=reference,
            complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}));raise


def load(output):
    if (output/'init_stop.json').exists() or any(output.glob('seal_*_stop.json')):raise ValueError('STOP_SEAL_DO_NOT_RESTART')
    l=json.loads((output/'launch.json').read_text());cfg=l['config']
    if (cfg!=config() or any(l.get(k) is not False for k in PROTECTED) or duckdb.__version__!=cfg['duckdb_version']
        or np.__version__!=cfg['numpy_version'] or cfg['resources']!=LIMITS
        or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())):
        raise ValueError('Protected immutable address access launch required')
    for ref in l['implementation'].values():verify(ref)
    p=json.loads(verify(l['plan']).read_text());m=json.loads(verify(p['source_store']).read_text())
    validate_report(p['source_audit_resources']);validate_report(p['prefix_audit_resources'])
    if p['audit_launch']!=record(output/'audit_launch.json'):
        raise ValueError('Immutable supervised original audit launch required')
    for kind,ref in p['source_audits'].items():
        actual,_=audit_record(output,'init',kind)
        if actual!=ref:raise ValueError('Same complete supervised source audit required')
        if p['source_audit_supervisions'][kind]!=record(output/'source_audits'/'init'/kind/'supervision.json'):
            raise ValueError('Same actual source audit supervision required')
    verify(p['dictionary']);verify(p['catalog'])
    for epoch in p['source_epochs']:
        if fingerprint(epoch['reference']['path'])!=epoch['fingerprint']:raise ValueError('Address source epoch drift, full rebuild forbidden')
    if (p['status']!='SEALED_V163_ALL_RAW_ADDRESS_PHYSICAL_PLAN' or any(p.get(k) is not False for k in PROTECTED)
        or p['dictionary']!=m['dictionary'] or p['catalog']!=m['catalog'] or p['sites']!=m['sites']
        or p['aliases']!=m['aliases'] or p['facts']!=m['facts'] or p['original_observations']!=m['original_observations']
        or p['lexical_observations']!=m['lexical_observations']
        or not p['full_source_hashes_at_init_and_seal_not_each_phase']
        or not p['source_audit_resources']['pass_resource_policy']):
        raise ValueError('Same entire raw-address physical plan required')
    return l,p


def directory(output,phase,index,channel=None):
    return output/'parts'/f'part{index:05d}' if phase=='part' else output/'access'/f'channel{channel}'/f'bucket{index:03d}'


def phase_record(output,phase,index,channel=None):
    from src.xgb_matcher.retrieval_address_authority_v163 import parent_exit
    path=directory(output,phase,index,channel)/'manifest.json'
    if not path.is_file():raise ValueError('UNFINISHED_ALL_ADDRESS_'+phase.upper()+'_PHASES')
    if any((path.parent/name).exists() for name in ('worker_stop.json','status.json')):
        raise ValueError('STOP_PHASE_SUPERVISION_OR_WORKER_DO_NOT_RESTART')
    parent_exit(path.parent)
    m=json.loads(path.read_text())
    validate_report(m['resources'])
    if (m['status']!='SEALED_V163_ADDRESS_'+phase.upper() or m['phase']!=phase or m['index']!=index
        or m['channel']!=channel or m['launch']!=record(output/'launch.json')
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index']
        or not m['resources']['pass_resource_policy']):
        raise ValueError('Protected exact complete address phase required')
    gate=json.loads(verify(m['preflight']).read_text())
    if (not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed'] or gate['policy_id']!=POLICY
        or verify(m['preflight']).resolve()!=(path.parent/'preflight.json').resolve()):raise ValueError('Explicit authorized pressure phase gate required')
    for name,ref in m['artifacts'].items():
        if verify(ref).resolve()!=(path.parent/name).resolve():raise ValueError('Exact phase artifact path required')
    return m


def checked_request(l,p,phase,index,channel):
    maximum=len(p['parts']) if phase=='part' else l['config']['term_buckets']
    if (phase not in ('part','bucket') or not isinstance(index,int) or isinstance(index,bool) or not 0<=index<maximum
        or (phase=='part' and channel is not None)
        or (phase=='bucket' and (not isinstance(channel,int) or isinstance(channel,bool) or not 0<=channel<4))):
        raise ValueError('Actual complete physical phase/channel required')


def worker(output,phase,index,channel=None):
    l,p=load(output);cfg=l['config'];checked_request(l,p,phase,index,channel);dest=directory(output,phase,index,channel)
    if any(f.name not in ('preflight.json','worker.log') for f in dest.iterdir()):raise ValueError('Immutable phase already started, no retry')
    budget=c=None
    try:
        gate=json.loads(verify(record(dest/'preflight.json')).read_text())
        if not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed'] or gate['policy_id']!=POLICY:raise ValueError('Explicit authorized pressure gate required')
        with Budget(cfg['resources']) as budget:
            c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET temp_directory=?',[str(dest/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            if phase=='part':
                spec=p['parts'][index];table=pq.ParquetFile(verify(spec['source'])).read_row_group(spec['row_group'],use_threads=False)
                table=table.slice(spec['group_slice_start'],spec['group_slice_stop']-spec['group_slice_start'])
                if len(table)!=spec['row_stop']-spec['row_start']:raise ValueError('Whole planned address slice required')
                stats=stage_addresses(c,table,dest,spec['alias_start'],spec['row_start'],cfg['term_buckets']);verify(spec['source'])
                files=[Path(s) for s in stats.pop('files')]
                counts={str(b):{str(ch):0 for ch in range(4)} for b in range(cfg['term_buckets'])}
                for path in files:
                    b=path.parent.name.removeprefix('term_bucket=')
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=131072,columns=['channel'],use_threads=False):
                        values=batch.column(0).to_numpy(zero_copy_only=False)
                        if np.any(values>3):raise ValueError('Every actual lexical channel required')
                        for ch in range(4):counts[b][str(ch)]+=int(np.count_nonzero(values==ch))
                if sum(sum(v.values()) for v in counts.values())!=stats['postings']:raise ValueError('Every channel posting required')
                stats['bucket_channel_counts']=counts
                files.append(dest/'features.parquet');inputs=dict(part=spec)
            else:
                s=complete_stage(output,l,p);items=s['buckets'][str(index)];files=[verify(i['artifact']) for i in items]
                expected=sum(i['channel_counts'][str(channel)] for i in items)
                stats=pack(c,files,dest,channel,index,cfg['term_buckets'],expected,p['aliases'])
                for item in items:verify(item['artifact'])
                files=list(dest.glob('*.npy'));inputs=dict(staging=record(output/'staging_manifest.json'),channel=channel,bucket=index)
            artifacts={str(f.relative_to(dest)):record(f) for f in files}
            c.close();c=None;load(output);size=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
            if size>cfg['max_phase_bytes']:raise ValueError('STOP_ADDRESS_ACCESS_PHASE_SIZE')
        write_json(dest/'manifest.json',dict(status='SEALED_V163_ADDRESS_'+phase.upper(),phase=phase,index=index,channel=channel,
            launch=record(output/'launch.json'),inputs=inputs,stats=stats,artifacts=artifacts,preflight=record(dest/'preflight.json'),
            output_bytes=size,resources=budget.report(),complete_retrieval_index=False,**{k:False for k in PROTECTED}))
    except BaseException as exc:
        if c is not None:c.close()
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**{k:False for k in PROTECTED}));raise


def execute(output,phase,index,channel=None):
    l,p=load(output);checked_request(l,p,phase,index,channel);cfg=l['config'];dest=directory(output,phase,index,channel)
    dest.mkdir(parents=True,exist_ok=False)
    try:
        write_json(dest/'preflight.json',quiet(cfg['phase_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds']))
        args=[sys.executable,'-u','-m','scripts.compile_retrieval_v163_address_access','worker','--output',str(output),'--phase',phase,'--index',str(index)]
        if channel is not None:args+=['--channel',str(channel)]
        with (dest/'worker.log').open('x') as log:
            supervision=supervise(args,log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        write_json(dest/'supervision.json',supervision)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def bounded_seal(function):
    @functools.wraps(function)
    def wrapped(output):
        l,p=load(output);cfg=l['config'];budget=None
        gate=quiet(cfg['phase_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
        try:
            with Budget(cfg['resources']) as budget:
                name,data=function(output,l,p)
        except RuntimeError as exc:
            write_json(output/('seal_'+function.__name__+'_stop.json'),dict(status='STOP',reason=str(exc),
                resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,**{k:False for k in PROTECTED}));raise
        write_json(output/(name+'.json'),dict(data,seal_resources=budget.report()))
    return wrapped


@bounded_seal
def seal_parts(output,l,p):
    if (output/'staging_manifest.json').exists():raise ValueError('Immutable staging already sealed')
    buckets={str(i):[] for i in range(l['config']['term_buckets'])};features=[];refs=[];aliases=posts=size=empty=0
    for spec in p['parts']:
        m=phase_record(output,'part',spec['part_id']);s=m['stats'];dest=directory(output,'part',spec['part_id'])
        if (m['inputs']!=dict(part=spec) or s['addresses']!=spec['row_stop']-spec['row_start']
            or not all(s[k] for k in ('all_scalar_feature_texts_exact','all_scalar_channel_sets_exact','all_exported_postings_exact','all_raw_addresses_retained'))
            or s['address_fact_attributes_or_scope_changed'] or s['alias_or_proof_cap'] is not None):
            raise ValueError('Every original whole-address channel set required')
        feature=m['artifacts']['features.parquet']
        if pq.ParquetFile(verify(feature)).metadata.num_rows!=s['addresses']:raise ValueError('Every raw alias feature required')
        features.append(dict(part=spec,artifact=feature));seen=0
        observed={str(i):{str(ch):0 for ch in range(4)} for i in range(l['config']['term_buckets'])}
        for relative,ref in m['artifacts'].items():
            if relative=='features.parquet':continue
            path=verify(ref);b=path.parent.name.removeprefix('term_bucket=')
            if b not in buckets or not path.resolve().is_relative_to((dest/'staging'/f'term_bucket={b}').resolve()):
                raise ValueError('Every actual physical posting path required')
            counts={str(ch):0 for ch in range(4)}
            for batch in pq.ParquetFile(path).iter_batches(batch_size=131072,columns=['channel'],use_threads=False):
                values=batch.column(0).to_numpy(zero_copy_only=False)
                if np.any(values>3):raise ValueError('Explicit known lexical channel required')
                for ch in range(4):counts[str(ch)]+=int(np.count_nonzero(values==ch))
            seen+=sum(counts.values());buckets[b].append(dict(part_id=spec['part_id'],artifact=ref,channel_counts=counts))
            for ch in range(4):observed[b][str(ch)]+=counts[str(ch)]
        if seen!=s['postings'] or observed!=s['bucket_channel_counts']:raise ValueError('Every whole-address posting required')
        aliases+=s['addresses'];posts+=seen;empty+=s['feature_empty'];size+=m['output_bytes'];refs.append(record(dest/'manifest.json'))
    if aliases!=p['aliases'] or size>l['config']['max_total_bytes']:raise ValueError('Whole address denominator/storage budget required')
    return 'staging_manifest',dict(status=STAGE,launch=record(output/'launch.json'),source=p['source_store'],
        aliases=aliases,postings=posts,feature_empty=empty,features=features,buckets=buckets,phase_authorities=refs,output_bytes=size,
        complete_retrieval_index=False,**{k:False for k in PROTECTED})


def complete_stage(output,l,p):
    path=output/'staging_manifest.json'
    if not path.is_file():raise ValueError('UNFINISHED_ALL_ADDRESS_PARTS')
    s=json.loads(path.read_text())
    validate_report(s['seal_resources'])
    if (s['status']!=STAGE or s['launch']!=record(output/'launch.json') or s['source']!=p['source_store']
        or s['aliases']!=p['aliases'] or any(s.get(k) is not False for k in PROTECTED)
        or s['complete_retrieval_index'] or len(s['features'])!=len(p['parts'])
        or len(s['phase_authorities'])!=len(p['parts'])
        or not s['seal_resources']['pass_resource_policy']
        or set(s['buckets'])!={str(i) for i in range(l['config']['term_buckets'])}):
        raise ValueError('Protected entire whole-address staging required')
    expected={str(i):[] for i in range(l['config']['term_buckets'])};aliases=posts=empty=size=0
    for spec,item,ref in zip(p['parts'],s['features'],s['phase_authorities']):
        m=phase_record(output,'part',spec['part_id'])
        if (item!=dict(part=spec,artifact=m['artifacts']['features.parquet']) or m['inputs']!=dict(part=spec)
            or ref!=record(directory(output,'part',spec['part_id'])/'manifest.json')):
            raise ValueError('Every exact original alias part required')
        st=m['stats']
        if st['addresses']!=spec['row_stop']-spec['row_start']:raise ValueError('Whole raw address part denominator required')
        for relative,artifact in m['artifacts'].items():
            if relative=='features.parquet':continue
            b=Path(artifact['path']).parent.name.removeprefix('term_bucket=')
            if b not in expected:raise ValueError('Actual physical address bucket required')
            # V131 emits one file per bucket for this bounded single stage.
            if any(x['part_id']==spec['part_id'] for x in expected[b]):
                raise ValueError('Each original physical posting file once required')
            expected[b].append(dict(part_id=spec['part_id'],artifact=artifact,channel_counts=st['bucket_channel_counts'][b]))
        aliases+=st['addresses'];posts+=st['postings'];empty+=st['feature_empty'];size+=m['output_bytes']
    if (expected!=s['buckets'] or aliases!=s['aliases'] or posts!=s['postings']
        or empty!=s['feature_empty'] or size!=s['output_bytes']):
        raise ValueError('Every original staged channel/artifact exactly once required')
    total=0
    for bucket,items in s['buckets'].items():
        for item in items:
            verify(item['artifact']);part=directory(output,'part',item['part_id'])
            if not Path(item['artifact']['path']).resolve().is_relative_to((part/'staging'/f'term_bucket={bucket}').resolve()):
                raise ValueError('Unchanged original posting path required')
            total+=sum(item['channel_counts'].values())
    if total!=s['postings']:raise ValueError('Whole staged posting denominator required')
    return s


def access_items(output,l,p,s):
    access=[];refs=[];posts=terms=0;size=s['output_bytes']
    for ch in range(4):
        for b in range(l['config']['term_buckets']):
            m=phase_record(output,'bucket',b,ch);st=m['stats'];dest=directory(output,'bucket',b,ch)
            if (m['inputs']!=dict(staging=record(output/'staging_manifest.json'),channel=ch,bucket=b)
                or st['channel']!=ch or st['bucket']!=b or st['postings']!=sum(i['channel_counts'][str(ch)] for i in s['buckets'][str(b)])
                or not all(st[k] for k in ('every_original_posting_and_token_reconstructed_exactly','all_postings_sorted_unique',
                    'native_unsigned_batched_csr_audit','source_semantics_unchanged','no_posting_or_owner_cap'))):
                raise ValueError('Every exact original address channel/posting required')
            specs={'post_ids':(np.uint32,st['postings']),'post_offsets':(np.uint64,st['terms']+1),
                'token_offsets':(np.uint64,st['terms']+1),'token_data':(np.uint8,st['token_bytes'])}
            if set(m['artifacts'])!={k+'.npy' for k in specs}:raise ValueError('Every typed whole-address access array required')
            for name,(dtype,length) in specs.items():
                a=np.load(verify(m['artifacts'][name+'.npy']),mmap_mode='r',allow_pickle=False)
                try:
                    if a.dtype!=dtype or a.shape!=(length,):raise ValueError('Exact original typed array required')
                    if name.endswith('offsets') and (a[0]!=0 or a[-1]!=(st['postings'] if name=='post_offsets' else st['token_bytes']) or np.any(a[1:]<=a[:-1])):
                        raise ValueError('Whole strictly increasing CSR required')
                finally:a._mmap.close()
            access.append(dict(channel=ch,bucket=b,postings=st['postings'],terms=st['terms'],artifacts=m['artifacts']))
            refs.append(record(dest/'manifest.json'));posts+=st['postings'];terms+=st['terms'];size+=m['output_bytes']
    if posts!=s['postings'] or size>l['config']['max_total_bytes']:raise ValueError('Complete packed posting/storage denominator required')
    return dict(access=access,phase_authorities=refs,postings=posts,terms=terms,output_bytes=size)


def validate_access(output,l,p,s,m):
    validate_report(m['seal_resources'])
    if (m['status']!=STATUS or m['launch']!=record(output/'launch.json') or m['staging']!=record(output/'staging_manifest.json')
        or m['source_store']!=p['source_store'] or m['catalog']!=p['catalog'] or m['aliases']!=p['aliases']
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] or m['ready_for_fold1']
        or not m['seal_resources']['pass_resource_policy']):
        raise ValueError('Protected complete whole-address access authority required')
    for kind in ('prefix','reverse'):
        ref,_=audit_record(output,'seal',kind)
        if (m['sealed_source_audits'][kind]!=ref or m['sealed_source_audit_supervisions'][kind]
            !=record(output/'source_audits'/'seal'/kind/'supervision.json')):
            raise ValueError('Complete final source rehash and actual exits required')
    data=access_items(output,l,p,s)
    if any(m[k]!=v for k,v in data.items()):raise ValueError('Every complete address access artifact required')


@bounded_seal
def seal_access(output,l,p):
    if (output/'manifest.json').exists():raise ValueError('Immutable address access already sealed')
    # Both original source audits rehashed in supervised children before this
    # bounded lexical seal. Validate both exact certificates and unchanged plan.
    for kind in ('prefix','reverse'):audit_record(output,'seal',kind)
    s=complete_stage(output,l,p);data=access_items(output,l,p,s);load(output)
    return 'manifest',dict(status=STATUS,launch=record(output/'launch.json'),staging=record(output/'staging_manifest.json'),
        source_store=p['source_store'],catalog=p['catalog'],aliases=p['aliases'],**data,
        sealed_source_audits={kind:audit_record(output,'seal',kind)[0] for kind in ('prefix','reverse')},
        sealed_source_audit_supervisions={kind:record(output/'source_audits'/'seal'/kind/'supervision.json') for kind in ('prefix','reverse')},
        lexical_address_access_built=True,complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED})

    
def seal(output):
    l,p=load(output)
    if (output/'manifest.json').exists():raise ValueError('Immutable address access already sealed')
    try:
        (_,pm),(_,rm)=run_audits(output,'seal')
        if any(rm[k]!=p[k] for k in ('dictionary','catalog','aliases','facts','sites',
                'original_observations','lexical_observations','parts','source_epochs')):
            raise ValueError('Same entire source plan at init and final seal required')
        seal_access(output)
    except BaseException as exc:
        write_json(output/'seal_source_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def remaining(output):
    l,p=load(output)
    for i in range(len(p['parts'])):
        dest=directory(output,'part',i)
        if dest.exists():phase_record(output,'part',i)
        else:execute(output,'part',i)
    if not (output/'staging_manifest.json').exists():seal_parts(output)
    s=complete_stage(output,l,p)
    for ch in range(4):
        for b in range(l['config']['term_buckets']):
            dest=directory(output,'bucket',b,ch)
            if dest.exists():phase_record(output,'bucket',b,ch)
            else:execute(output,'bucket',b,ch)
    if not (output/'manifest.json').exists():seal(output)
    else:validate_access(output,l,p,s,json.loads((output/'manifest.json').read_text()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['init','phase','worker','audit-worker','remaining','seal-parts','seal'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--source',type=Path);parser.add_argument('--sha256')
    parser.add_argument('--audit-tag',choices=['init','seal']);parser.add_argument('--audit-kind',choices=['prefix','reverse']);parser.add_argument('--phase',choices=['part','bucket']);parser.add_argument('--index',type=int);parser.add_argument('--channel',type=int);a=parser.parse_args()
    if a.mode=='init':
        if a.source is None or a.sha256 is None:parser.error('Complete sealed address store and exact SHA256 required')
        init(a.output,dict(path=str(a.source.resolve()),sha256=a.sha256))
    elif a.mode=='audit-worker':
        if a.audit_tag is None or a.audit_kind is None:parser.error('Explicit source audit tag/kind required')
        audit_worker(a.output,a.audit_tag,a.audit_kind)
    elif a.mode in ('phase','worker'):
        if a.phase is None or a.index is None:parser.error('Actual physical phase/index required')
        (worker if a.mode=='worker' else execute)(a.output,a.phase,a.index,a.channel)
    else:{'remaining':remaining,'seal-parts':seal_parts,'seal':seal}[a.mode](a.output)
