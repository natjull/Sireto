"""Full V154 store -> lexical access under authorized V153 pressure policy.

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

import scripts.compile_retrieval_v154_address_store as original
from src.xgb_matcher.retrieval_address_access_v155 import STATUS,physical_plan,stage_addresses
from src.xgb_matcher.retrieval_packed_names_v134 import pack
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report
from src.xgb_matcher.retrieval_supervisor_v157 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v155_address_access.json'
STAGE='SEALED_V155_ALL_WHOLE_ADDRESS_CHANNEL_POSTINGS'


def fingerprint(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def source_epochs(reference,m,d):
    """Guard already hashed source bytes by file epoch between bounded phases.

    Full source hashes at init/seal; the dictionary used by a worker also
    receives its full hash before/after THAT worker. No repeated multi-GB
    global source audit outside the worker Budget on every physical slice.
    Only explicit address authorities are traversed, never arbitrary JSON.
    """
    root=Path(reference['path']).parent;_,p=original.load(root)
    _,raw=original.projection_backend(p['source_projection']).load(Path(p['source_projection']['path']).parent)
    refs=[reference,p['source_projection'],m['catalog'],m['dictionary'],m['reverse_staged']]
    refs.extend(s['reference'] for s in raw['sources'].values())
    for part in p['parts']:refs.extend([part['source_manifest'],part['source']])
    for b in d['buckets']:refs.extend(b['artifacts'].values())
    for s in m['intervals']:refs.extend([*s['artifacts'].values(),s['numbers'],s['ordered_source'],*s['sources']])
    unique={r['path']:r for r in refs}
    return [dict(reference=ref,fingerprint=fingerprint(path)) for path,ref in sorted(unique.items())]


def config():
    cfg=json.loads(CONFIG.read_text())
    if (duckdb.__version__!=cfg['duckdb_version'] or np.__version__!=cfg['numpy_version']
        or any(not isinstance(cfg[k],int) or isinstance(cfg[k],bool) or cfg[k]<=0 for k in
            ('term_buckets','batch_rows','expected_sites','expected_original_observations','expected_source_rows'))
        or cfg['term_buckets']>254 or cfg['resources']!=LIMITS):
        raise ValueError('Pinned whole-source typed policy and authorized pressure/RSS limits required')
    return cfg


def store(reference,cfg):
    path=verify(reference);root=path.parent;m=json.loads(path.read_text())
    validate_report(m['seal_resources'])
    if (path.resolve()!=(root/'manifest.json').resolve() or m['status']!=original.GLOBAL
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] or m['ready_for_fold1']
        or m['lexical_address_access_built'] or m['sites']!=cfg['expected_sites']
        or m['original_observations']!=cfg['expected_original_observations']
        or not m['every_original_site_and_fact_preserved'] or not m['all_original_observations_and_provenance_retained']
        or not m['seal_resources']['pass_resource_policy']):
        raise ValueError('Complete protected original address store required, not partial successes')
    launch,plan=original.load(root)
    full=original.projection(plan['source_projection'],launch['config'])
    if (full!= {k:plan[k] for k in full} or full['source_rows']!=cfg['expected_source_rows']
        or full['observations']!=cfg['expected_original_observations'] or full['sites']!=cfg['expected_sites']
        or m['launch']!=record(root/'launch.json') or m['source_projection']!=plan['source_projection']
        or m['catalog']!=plan['catalog'] or m['lexical_observations']!=plan['lexical_observations']):
        raise ValueError('Entire unchanged original projection/namespace required')
    d=original.authority(root,'dictionary','SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY')
    if m['dictionary']!=record(root/'dictionary.json') or d['aliases']!=m['aliases'] or d['facts']!=m['facts']:
        raise ValueError('Same exact complete address dictionary required')
    original.authority(root,'staged','SEALED_V154_ALL_ORIGINAL_ADDRESS_STAGING')
    reverse=original.authority(root,'reverse_staged','SEALED_V154_ALL_ADDRESS_FACT_SITE_STAGING')
    if m['reverse_staged']!=record(root/'reverse_staged.json') or reverse['facts']!=d['facts']:
        raise ValueError('Same whole address reverse namespace required')
    for index,b in enumerate(d['buckets']):
        phase=original.phase_record(root,'dictionary',index);s=phase['stats']
        if (b['bucket']!=index or b['artifacts']!=phase['artifacts'] or b['aliases']!=s['addresses']
            or b['facts']!=s['facts'] or b['observations']!=s['observations']
            or not s['every_observation_linked_once'] or not s['every_site_address_and_attributes_reconstructed_exactly']):
            raise ValueError('Every original address dictionary authority required')
    maximum=original.maximum(root,'reverse-pack');cursor=facts=0
    if len(d['buckets'])!=launch['config']['dictionary_buckets'] or len(m['intervals'])!=maximum or len(m['phase_authorities'])!=maximum:
        raise ValueError('Complete dictionary and every physical site interval required')
    for index in range(maximum):
        phase=original.phase_record(root,'reverse-pack',index);s=phase['stats']
        if (m['phase_authorities'][index]!=record(original.directory(root,'reverse-pack',index)/'manifest.json')
            or m['intervals'][index]!=s or s['start']!=cursor or s['stop']<=cursor
            or s['sites']!=m['sites'] or s['aliases']!=d['aliases']
            or not s['every_original_field_and_site_csr_exact'] or not s['all_provenance_keys_preserved']):
            raise ValueError('Every original site/fact reverse proof required')
        for ref in s['sources']:verify(ref)
        cursor=s['stop'];facts+=s['facts']
    if cursor!=m['sites'] or facts!=d['facts']:raise ValueError('Whole site and fact denominator required')
    return m,d


def init(output,reference):
    cfg=config()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only output root required')
    if output.exists():raise FileExistsError('Immutable address access root already exists')
    if not Path(reference['path']).is_file():raise ValueError('UNFINISHED_COMPLETE_ADDRESS_STORE_NO_WORKER_LAUNCHED')
    gate=quiet(cfg['initial_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
    with Budget(cfg['resources']) as budget:
        m,d=store(reference,cfg);parts=physical_plan(d['buckets'],d['aliases'],cfg['batch_rows'])
        epochs=source_epochs(reference,m,d)
    output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
    write_json(output/'plan.json',dict(status='SEALED_V155_ALL_RAW_ADDRESS_PHYSICAL_PLAN',source_store=reference,
        dictionary=m['dictionary'],catalog=m['catalog'],aliases=d['aliases'],facts=d['facts'],sites=m['sites'],
        original_observations=m['original_observations'],lexical_observations=m['lexical_observations'],parts=parts,
        source_audit_resources=budget.report(),source_epochs=epochs,
        full_source_hashes_at_init_and_seal_not_each_phase=True,**{k:False for k in PROTECTED}))
    paths=[Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_address_access_v155.py',
        ROOT/'scripts/compile_retrieval_v154_address_store.py',ROOT/'scripts/compile_retrieval_v152_address_access.py',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v153.py',ROOT/'src/xgb_matcher/retrieval_supervisor_v157.py',
        ROOT/'src/xgb_matcher/retrieval_address_access_v152.py',*[ROOT/('src/xgb_matcher/'+s+'.py') for s in
        ('retrieval_name_access_v131','retrieval_packed_names_v134','retrieval_national_names_v138',
         'retrieval_coupled_blocks_v121','retrieval_variants_v118','retrieval_name_postings_v86',
         'retrieval_variant_factor_v119','retrieval_system_quiet_v143','retrieval_address_reverse_v147','retrieval_v20_workbench')]]
    inherited=json.loads(verify(m['launch']).read_text())['implementation']
    write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),preflight=record(output/'preflight.json'),
        implementation={**inherited,**{str(p):record(p) for p in paths}},**{k:False for k in PROTECTED}))


def load(output):
    if any(output.glob('seal_*_stop.json')):raise ValueError('STOP_SEAL_DO_NOT_RESTART')
    l=json.loads((output/'launch.json').read_text());cfg=l['config']
    if (cfg!=config() or any(l.get(k) is not False for k in PROTECTED) or duckdb.__version__!=cfg['duckdb_version']
        or np.__version__!=cfg['numpy_version'] or cfg['resources']!=LIMITS
        or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve())):
        raise ValueError('Protected immutable address access launch required')
    for ref in l['implementation'].values():verify(ref)
    p=json.loads(verify(l['plan']).read_text());m=json.loads(verify(p['source_store']).read_text())
    validate_report(p['source_audit_resources'])
    verify(p['dictionary']);verify(p['catalog'])
    for epoch in p['source_epochs']:
        if fingerprint(epoch['reference']['path'])!=epoch['fingerprint']:raise ValueError('Address source epoch drift, full rebuild forbidden')
    if (p['status']!='SEALED_V155_ALL_RAW_ADDRESS_PHYSICAL_PLAN' or any(p.get(k) is not False for k in PROTECTED)
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
    path=directory(output,phase,index,channel)/'manifest.json'
    if not path.is_file():raise ValueError('UNFINISHED_ALL_ADDRESS_'+phase.upper()+'_PHASES')
    if any((path.parent/name).exists() for name in ('worker_stop.json','status.json')):
        raise ValueError('STOP_PHASE_SUPERVISION_OR_WORKER_DO_NOT_RESTART')
    m=json.loads(path.read_text())
    validate_report(m['resources'])
    if (m['status']!='SEALED_V155_ADDRESS_'+phase.upper() or m['phase']!=phase or m['index']!=index
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
        write_json(dest/'manifest.json',dict(status='SEALED_V155_ADDRESS_'+phase.upper(),phase=phase,index=index,channel=channel,
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
        args=[sys.executable,'-u','-m','scripts.compile_retrieval_v155_address_access','worker','--output',str(output),'--phase',phase,'--index',str(index)]
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
    data=access_items(output,l,p,s)
    if any(m[k]!=v for k,v in data.items()):raise ValueError('Every complete address access artifact required')


@bounded_seal
def seal(output,l,p):
    if (output/'manifest.json').exists():raise ValueError('Immutable address access already sealed')
    s=complete_stage(output,l,p);data=access_items(output,l,p,s)
    # One final complete source rehash/audit INSIDE this seal's Budget.
    store(p['source_store'],l['config']);load(output)
    return 'manifest',dict(status=STATUS,launch=record(output/'launch.json'),staging=record(output/'staging_manifest.json'),
        source_store=p['source_store'],catalog=p['catalog'],aliases=p['aliases'],**data,
        lexical_address_access_built=True,complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED})


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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['init','phase','worker','remaining','seal-parts','seal'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--source',type=Path);parser.add_argument('--sha256')
    parser.add_argument('--phase',choices=['part','bucket']);parser.add_argument('--index',type=int);parser.add_argument('--channel',type=int);a=parser.parse_args()
    if a.mode=='init':
        if a.source is None or a.sha256 is None:parser.error('Complete sealed address store and exact SHA256 required')
        init(a.output,dict(path=str(a.source.resolve()),sha256=a.sha256))
    elif a.mode in ('phase','worker'):
        if a.phase is None or a.index is None:parser.error('Actual physical phase/index required')
        (worker if a.mode=='worker' else execute)(a.output,a.phase,a.index,a.channel)
    else:{'remaining':remaining,'seal-parts':seal_parts,'seal':seal}[a.mode](a.output)
