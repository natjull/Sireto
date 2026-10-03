"""Require full projection, then compile all address dictionaries/site CSR.

Explicit V153-authorized pressure/RSS budget. V148 implementation and
business kernels are unchanged historical authorities. No retry of partial
or failed phases, no labels, candidate pool, fit, recall or external expense.
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

import scripts.resume_retrieval_v153_addresses as original
from src.xgb_matcher.retrieval_address_dictionary_v145 import partition,dictionary
from src.xgb_matcher.retrieval_address_reverse_v147 import stage,pack
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report,supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v154_address_store.json'
PHASES=('part','dictionary','reverse-stage','reverse-pack')
GLOBAL='SEALED_V154_COMPLETE_OFFICIAL_ADDRESS_DICTIONARY_AND_SITE_REVERSE'


def config():
    cfg=json.loads(CONFIG.read_text())
    if (duckdb.__version__!=cfg['duckdb_version'] or np.__version__!=cfg['numpy_version']
        or not isinstance(cfg['dictionary_buckets'],int) or isinstance(cfg['dictionary_buckets'],bool)
        or not 1<=cfg['dictionary_buckets']<=254 or not isinstance(cfg['site_quantum'],int)
        or isinstance(cfg['site_quantum'],bool) or cfg['site_quantum']<=0
        or cfg['resources']!=LIMITS):
        raise ValueError('Pinned complete storage policy and authorized pressure/RSS limits required')
    return cfg


def projection(reference,cfg):
    """All original raw roles/parts, not a list of available successful lots."""
    m=json.loads(verify(reference).read_text())
    validate_report(m['seal_resources'])
    if (m['status']!='SEALED_V153_ALL_ORIGINAL_OFFICIAL_AND_RAW_CURRENT_ADDRESS_PROJECTION'
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] or m['ready_for_fold1']
        or m['observations']!=cfg['expected_observations'] or m['source_rows']!=cfg['expected_source_rows']
        or not m['every_original_source_full_hash_verified_before_and_after']
        or not m['every_source_row_and_role_retained'] or not m['every_exported_field_matches_second_source_pass']
        or not m['every_site_proof_scope_reconstructed'] or m['parent_addresses_spread_to_sites']
        or not m['seal_resources']['pass_resource_policy']):
        raise ValueError('Complete protected all-source projection required, never partial successes')
    l,p=original.load(Path(reference['path']).parent)
    if (m['launch']!=record(Path(reference['path']).parent/'launch.json') or m['plan']!=l['plan']
        or m['catalog']!=l['config']['catalog'] or p['role_order']!=['official_addresses','current_addresses']
        or l['config']['expected_namespace_sites']!=cfg['expected_sites']):
        raise ValueError('Whole unchanged original projection namespace and plan required')
    keys=AppendKeys(m['catalog']);parts=[];cursor=source_rows=lexical=size=0
    for role in p['role_order']:
        source=p['sources'][role];verify(source['reference']);row_cursor=0
        for spec in source['parts']:
            i=len(parts)
            if i>=len(m['parts']):raise ValueError('UNFINISHED_ALL_ORIGINAL_ADDRESS_PARTS')
            item=m['parts'][i];key=f'{role}/{spec["part_id"]}';imported=key in p['imported_parts']
            expected=p['imported_parts'][key]['manifest'] if imported else record(original.directory(Path(reference['path']).parent,role,spec['part_id'])/'manifest.json')
            if item!=dict(imported=imported,manifest=expected):raise ValueError('Every original physical part/order exactly once required')
            launch=p['imported_parts'][key]['launch'] if imported else m['launch']
            part=original.validate(expected,role,spec,source,launch,m['catalog'],l['config']['scalar_first_source_rows'])
            begin=source['observation_base']+spec['source_start']*source['factor']
            if begin!=cursor or spec['source_start']!=row_cursor:raise ValueError('Whole original observation and source intervals required')
            stop=begin+part['observations']
            parts.append(dict(index=i,role=role,source_manifest=expected,source=part['artifact'],
                observation_start=begin,observation_stop=stop,observations=part['observations'],
                lexical_observations=part['lexical_observations']))
            cursor=stop;row_cursor=spec['source_stop'];source_rows+=part['source_rows'];lexical+=part['lexical_observations'];size+=part['artifact']['size_bytes']
        if row_cursor!=source['source_rows']:raise ValueError('Complete original source role required')
    if (len(parts)!=len(m['parts']) or len(parts)!=p['total_parts'] or cursor!=m['observations']
        or source_rows!=m['source_rows'] or lexical!=m['lexical_observations'] or size!=m['output_bytes']
        or keys.sites!=cfg['expected_sites']):raise ValueError('Whole source/observation/lexical denominator required')
    return dict(source_projection=reference,catalog=m['catalog'],sites=keys.sites,parents=keys.parents,parts=parts,
        observations=cursor,lexical_observations=lexical,source_rows=source_rows)


def init(output,reference):
    cfg=config()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('CATNAT-only national output root required')
    if output.exists():raise FileExistsError('Immutable address store root already exists')
    # Missing full source fails BEFORE creating output or any Mac worker.
    if not Path(reference['path']).is_file():raise ValueError('UNFINISHED_COMPLETE_ADDRESS_PROJECTION_NO_WORKER_LAUNCHED')
    gate=quiet(cfg['initial_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
    with Budget(cfg['resources']) as budget:
        plan=projection(reference,cfg)
    output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
    write_json(output/'plan.json',dict(status='SEALED_V154_ALL_ADDRESS_SOURCE_STORAGE_PLAN',**plan,
        source_audit_resources=budget.report(),**{k:False for k in PROTECTED}))
    source=json.loads(verify(reference).read_text());old=json.loads(verify(source['launch']).read_text())
    implementation=dict(old['implementation'])
    paths=[Path(__file__),CONFIG,ROOT/'scripts/compile_retrieval_v148_address_store.py',ROOT/'docs/retrieval_v148_address_store.md',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v153.py',
        *[ROOT/('src/xgb_matcher/'+s+'.py') for s in ('retrieval_address_dictionary_v145','retrieval_address_reverse_v147',
            'retrieval_initial_collector_v146','retrieval_variant_factor_v119','retrieval_system_quiet_v143',
            'retrieval_append_catalog_v141','retrieval_proof_bridge_v137','retrieval_v20_workbench')],
        ROOT/'scripts/resume_retrieval_v153_addresses.py']
    implementation.update({str(p):record(p) for p in paths})
    write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),implementation=implementation,
        preflight=record(output/'preflight.json'),**{k:False for k in PROTECTED}))


def load(output):
    if any(output.glob('seal_*_stop.json')):raise ValueError('STOP_SEAL_DO_NOT_RESTART')
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if (not output.resolve().is_relative_to(Path(l['config']['output_root']).resolve())
        or any(l.get(k) is not False for k in PROTECTED) or duckdb.__version__!=l['config']['duckdb_version']
        or np.__version__!=l['config']['numpy_version'] or l['config']['resources']!=LIMITS):
        raise ValueError('Protected immutable storage launch required')
    p=json.loads(verify(l['plan']).read_text());verify(p['source_projection']);verify(p['catalog'])
    if p['status']!='SEALED_V154_ALL_ADDRESS_SOURCE_STORAGE_PLAN' or any(p.get(k) is not False for k in PROTECTED):
        raise ValueError('Complete source storage plan required')
    return l,p


def directory(output,phase,index):return output/phase/f'part{index:05d}'
def manifest(output,name):return json.loads((output/(name+'.json')).read_text())


def authority(output,name,status):
    m=manifest(output,name)
    validate_report(m['seal_resources'])
    if (m['status']!=status or m['launch']!=record(output/'launch.json')
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index']
        or not m['seal_resources']['pass_resource_policy']):
        raise ValueError('Protected complete '+name+' required')
    return m


def bounded_seal(function):
    """Publish a seal ONLY after its quiet gate and Budget have passed."""
    @functools.wraps(function)
    def wrapped(output,*args,**kwargs):
        l,p=load(output);cfg=l['config'];budget=None
        gate=quiet(cfg['phase_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
        try:
            with Budget(cfg['resources']) as budget:
                name,data=function(output,*args,**kwargs)
        except RuntimeError as exc:
            write_json(output/('seal_'+function.__name__+'_stop.json'),dict(status='STOP',reason=str(exc),
                resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
                **{k:False for k in PROTECTED}));raise
        preflight=output/('seal_'+name+'_preflight.json');write_json(preflight,gate)
        write_json(output/(name+'.json'),dict(data,seal_resources=budget.report(),preflight=record(preflight)))
    return wrapped


def maximum(output,phase):
    l,p=load(output)
    if phase=='part':return len(p['parts'])
    if phase in ('dictionary','reverse-stage'):return l['config']['dictionary_buckets']
    if phase=='reverse-pack':return (p['sites']+l['config']['site_quantum']-1)//l['config']['site_quantum']
    raise ValueError('Actual physical phase required')


def phase_record(output,phase,index):
    path=directory(output,phase,index)/'manifest.json'
    if not path.is_file():raise ValueError('UNFINISHED_ALL_'+phase.upper()+'_PARTS')
    m=json.loads(path.read_text())
    validate_report(m['resources'])
    if (m['status']!='SEALED_V154_PHYSICAL_'+phase.upper() or m['phase']!=phase or m['index']!=index
        or m['launch']!=record(output/'launch.json') or any(m.get(k) is not False for k in PROTECTED)
        or not m['resources']['pass_resource_policy']
        or m['complete_retrieval_index']):raise ValueError('Protected whole immutable physical phase required')
    gate=json.loads(verify(m['preflight']).read_text())
    if (verify(m['preflight']).resolve()!=(path.parent/'preflight.json').resolve()
        or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed'] or gate['policy_id']!=POLICY):
        raise ValueError('Explicit authorized pressure phase gate required')
    for name,ref in m['artifacts'].items():
        if verify(ref).resolve()!=(path.parent/name).resolve():raise ValueError('Actual immutable phase artifact path required')
    return m


def worker(output,phase,index):
    l,p=load(output);cfg=l['config'];dest=directory(output,phase,index);budget=c=None
    if phase not in PHASES or not isinstance(index,int) or isinstance(index,bool) or not 0<=index<maximum(output,phase):
        raise ValueError('Actual planned physical phase required')
    if any(k.name not in ('preflight.json','worker.log') for k in dest.iterdir()):raise ValueError('Immutable phase already started, no retry')
    try:
        gate=json.loads(verify(record(dest/'preflight.json')).read_text())
        if not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed'] or gate['policy_id']!=POLICY:raise ValueError('Explicit authorized pressure gate required')
        with Budget(cfg['resources']) as budget:
            c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET temp_directory=?',[str(dest/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            if phase=='part':
                spec=p['parts'][index];verify(spec['source_manifest']);source=verify(spec['source'])
                stats=partition(c,source,dest,spec['observation_start'],spec['observations'],spec['lexical_observations'],
                    cfg['dictionary_buckets'],p['sites'],p['parents']);verify(spec['source']);inputs=dict(part=spec)
            elif phase=='dictionary':
                s=authority(output,'staged','SEALED_V154_ALL_ORIGINAL_ADDRESS_STAGING');items=s['buckets'][str(index)]
                files=[verify(i['artifact']) for i in items]
                stats=dictionary(c,files,dest,index,cfg['dictionary_buckets'],sum(i['rows'] for i in items))
                for i in items:verify(i['artifact'])
                inputs=dict(staged=record(output/'staged.json'),bucket=index)
            elif phase=='reverse-stage':
                s=authority(output,'dictionary','SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY');b=s['buckets'][index]
                keys=AppendKeys(p['catalog'])
                stats=stage(c,verify(b['artifacts']['facts.parquet']),dest,keys,index,b['alias_start'],b['aliases'],b['facts'],cfg['site_quantum'])
                inputs=dict(dictionary=record(output/'dictionary.json'),bucket=b)
            else:
                s=authority(output,'reverse_staged','SEALED_V154_ALL_ADDRESS_FACT_SITE_STAGING');items=s['parts'][str(index)]
                files=[verify(i['artifact']) for i in items];start=index*cfg['site_quantum'];stop=min(start+cfg['site_quantum'],p['sites'])
                stats=pack(c,files,dest,start,stop,p['sites'],s['aliases'],sum(i['rows'] for i in items))
                inputs=dict(reverse_staged=record(output/'reverse_staged.json'),index=index,start=start,stop=stop)
            files=[Path(s) for s in stats.pop('files',[])];files.extend(dest.glob('*.parquet'));files.extend(dest.glob('*.npy'))
            artifacts={str(f.relative_to(dest)):record(f) for f in sorted(set(files))}
            c.close();c=None;load(output)
            size=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
            if size>cfg['max_phase_bytes']:raise ValueError('STOP_ADDRESS_PHASE_SIZE')
        write_json(dest/'manifest.json',dict(status='SEALED_V154_PHYSICAL_'+phase.upper(),phase=phase,index=index,
            launch=record(output/'launch.json'),preflight=record(dest/'preflight.json'),inputs=inputs,stats=stats,
            artifacts=artifacts,output_bytes=size,resources=budget.report(),complete_retrieval_index=False,
            **{k:False for k in PROTECTED}))
        print(json.dumps(dict(phase=phase,index=index,resources=budget.report())),flush=True)
    except BaseException as exc:
        if c is not None:c.close()
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
            **{k:False for k in PROTECTED}));raise


def execute(output,phase,index):
    if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<maximum(output,phase):raise ValueError('Planned phase index required')
    l,p=load(output);cfg=l['config'];dest=directory(output,phase,index);dest.mkdir(parents=True,exist_ok=False)
    try:
        gate=quiet(cfg['phase_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds']);write_json(dest/'preflight.json',gate)
        with (dest/'worker.log').open('x') as log:
            supervision=supervise([sys.executable,'-u','-m','scripts.compile_retrieval_v154_address_store','worker','--output',str(output),
                '--phase',phase,'--index',str(index)],log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        write_json(dest/'supervision.json',supervision)
        print((dest/'worker.log').read_text()[-2500:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


@bounded_seal
def seal_staging(output,reverse=False):
    l,p=load(output);cfg=l['config'];phase='reverse-stage' if reverse else 'part';name='reverse_staged' if reverse else 'staged'
    if (output/(name+'.json')).exists():raise ValueError('Immutable complete staging already sealed')
    source=authority(output,'dictionary','SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY') if reverse else p
    groups={str(i):[] for i in range(maximum(output,'reverse-pack') if reverse else cfg['dictionary_buckets'])}
    total=original_obs=size=0;refs=[]
    for i in range(maximum(output,phase)):
        m=phase_record(output,phase,i);s=m['stats']
        if reverse:
            b=source['buckets'][i]
            if m['inputs']!=dict(dictionary=record(output/'dictionary.json'),bucket=b) or s['facts']!=b['facts'] or not s['every_fact_staged_once']:
                raise ValueError('Every original dictionary fact staged exactly once required')
            expected=s['facts'];prefix='site_part='
        else:
            spec=p['parts'][i]
            if (m['inputs']!=dict(part=spec) or s['original_observations']!=spec['observations']
                or s['lexical_observations']!=spec['lexical_observations'] or not s['all_eligible_rows_unique_and_exact']):
                raise ValueError('Every original projection observation retained required')
            expected=s['lexical_observations'];original_obs+=s['original_observations'];prefix='bucket='
        seen=0
        for relative,ref in m['artifacts'].items():
            f=verify(ref);bucket=f.parent.name.removeprefix(prefix)
            if bucket not in groups or f.parent.name!=prefix+bucket or not f.resolve().is_relative_to((directory(output,phase,i)/'staging').resolve()):
                raise ValueError('Whole known physical storage artifact required')
            rows=pq.ParquetFile(f).metadata.num_rows;seen+=rows;groups[bucket].append(dict(index=i,artifact=ref,rows=rows))
        if seen!=expected:raise ValueError('Every staged source row exactly once required')
        total+=seen;size+=m['output_bytes'];refs.append(record(directory(output,phase,i)/'manifest.json'))
    wanted=source['facts'] if reverse else p['lexical_observations']
    if total!=wanted or (not reverse and original_obs!=p['observations']):raise ValueError('Complete staging denominator required')
    return name,dict(status='SEALED_V154_ALL_ADDRESS_FACT_SITE_STAGING' if reverse else 'SEALED_V154_ALL_ORIGINAL_ADDRESS_STAGING',
        launch=record(output/'launch.json'),source=record(output/'dictionary.json') if reverse else p['source_projection'],
        **({'parts':groups,'aliases':source['aliases'],'facts':total} if reverse else {'buckets':groups,'observations':original_obs,'lexical_observations':total}),
        phase_authorities=refs,output_bytes=size,complete_retrieval_index=False,**{k:False for k in PROTECTED})


@bounded_seal
def seal_dictionary(output):
    l,p=load(output);staged=authority(output,'staged','SEALED_V154_ALL_ORIGINAL_ADDRESS_STAGING')
    if (output/'dictionary.json').exists():raise ValueError('Immutable complete dictionary already sealed')
    aliases=facts=observations=size=0;buckets=[];refs=[]
    for i in range(maximum(output,'dictionary')):
        m=phase_record(output,'dictionary',i);s=m['stats']
        if (m['inputs']!=dict(staged=record(output/'staged.json'),bucket=i) or s['bucket']!=i
            or s['observations']!=sum(x['rows'] for x in staged['buckets'][str(i)])
            or not s['every_observation_linked_once'] or not s['every_site_address_and_attributes_reconstructed_exactly']):
            raise ValueError('Whole original dictionary observations/facts required')
        for name,n in (('addresses.parquet',s['addresses']),('facts.parquet',s['facts']),('observation_links.parquet',s['observations'])):
            if pq.ParquetFile(verify(m['artifacts'][name])).metadata.num_rows!=n:raise ValueError('Every whole dictionary export required')
        buckets.append(dict(bucket=i,alias_start=aliases,aliases=s['addresses'],facts=s['facts'],observations=s['observations'],artifacts=m['artifacts']))
        aliases+=s['addresses'];facts+=s['facts'];observations+=s['observations'];size+=m['output_bytes'];refs.append(record(directory(output,'dictionary',i)/'manifest.json'))
    if observations!=p['lexical_observations'] or max(aliases,facts)>=2**32:raise ValueError('Whole typed dictionary denominator required')
    return 'dictionary',dict(status='SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY',launch=record(output/'launch.json'),
        staged=record(output/'staged.json'),buckets=buckets,aliases=aliases,facts=facts,lexical_observations=observations,
        original_observations=p['observations'],output_bytes=size,phase_authorities=refs,complete_retrieval_index=False,
        **{k:False for k in PROTECTED})


@bounded_seal
def seal(output):
    l,p=load(output);cfg=l['config'];d=authority(output,'dictionary','SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY')
    staged=authority(output,'reverse_staged','SEALED_V154_ALL_ADDRESS_FACT_SITE_STAGING')
    if (output/'manifest.json').exists():raise ValueError('Immutable address store already sealed')
    intervals=[];refs=[];facts=0;cursor=0;size=d['output_bytes']+staged['output_bytes']+manifest(output,'staged')['output_bytes']
    for i in range(maximum(output,'reverse-pack')):
        m=phase_record(output,'reverse-pack',i);s=m['stats'];start=i*cfg['site_quantum'];stop=min(start+cfg['site_quantum'],p['sites'])
        if (m['inputs']!=dict(reverse_staged=record(output/'reverse_staged.json'),index=i,start=start,stop=stop)
            or s['start']!=cursor or s['stop']!=stop or s['sites']!=p['sites'] or s['aliases']!=d['aliases']
            or s['facts']!=sum(x['rows'] for x in staged['parts'][str(i)]) or not s['every_original_field_and_site_csr_exact']
            or not s['all_provenance_keys_preserved']):raise ValueError('Every whole source site interval/fact required')
        intervals.append(s);refs.append(record(directory(output,'reverse-pack',i)/'manifest.json'));facts+=s['facts'];cursor=stop;size+=m['output_bytes']
    if cursor!=p['sites'] or facts!=d['facts'] or size>cfg['max_total_bytes']:raise ValueError('Every site/fact/source and total storage budget required')
    projection(p['source_projection'],cfg);load(output)
    return 'manifest',dict(status=GLOBAL,launch=record(output/'launch.json'),source_projection=p['source_projection'],
        catalog=p['catalog'],dictionary=record(output/'dictionary.json'),reverse_staged=record(output/'reverse_staged.json'),
        intervals=intervals,phase_authorities=refs,sites=p['sites'],aliases=d['aliases'],facts=facts,
        original_observations=p['observations'],lexical_observations=p['lexical_observations'],output_bytes=size,
        every_original_site_and_fact_preserved=True,all_original_observations_and_provenance_retained=True,
        lexical_address_access_built=False,complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED})


def remaining(output):
    for phase,finish in (('part',lambda:seal_staging(output)),('dictionary',lambda:seal_dictionary(output)),
                         ('reverse-stage',lambda:seal_staging(output,True)),('reverse-pack',lambda:seal(output))):
        for i in range(maximum(output,phase)):
            dest=directory(output,phase,i)
            if dest.exists():
                if not (dest/'manifest.json').is_file():raise ValueError('STOP_OR_LIVE_PHASE_DO_NOT_RESTART')
                phase_record(output,phase,i)
            else:execute(output,phase,i)
        name={'part':'staged','dictionary':'dictionary','reverse-stage':'reverse_staged','reverse-pack':'manifest'}[phase]
        if not (output/(name+'.json')).exists():finish()
        else:
            status={'staged':'SEALED_V154_ALL_ORIGINAL_ADDRESS_STAGING','dictionary':'SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY',
                'reverse_staged':'SEALED_V154_ALL_ADDRESS_FACT_SITE_STAGING','manifest':GLOBAL}[name]
            authority(output,name,status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['init','phase','worker','remaining','seal-staging','seal-dictionary','seal-reverse-staging','seal'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--source',type=Path);parser.add_argument('--sha256')
    parser.add_argument('--phase',choices=PHASES);parser.add_argument('--index',type=int);a=parser.parse_args()
    if a.mode=='init':
        if a.source is None or a.sha256 is None:parser.error('Complete sealed source manifest and exact SHA256 required')
        init(a.output,dict(path=str(a.source.resolve()),sha256=a.sha256))
    elif a.mode in ('phase','worker'):
        if a.phase is None or a.index is None:parser.error('Actual physical phase and index required')
        (worker if a.mode=='worker' else execute)(a.output,a.phase,a.index)
    else:{'remaining':remaining,'seal-staging':lambda o:seal_staging(o),'seal-dictionary':seal_dictionary,
        'seal-reverse-staging':lambda o:seal_staging(o,True),'seal':seal}[a.mode](a.output)
