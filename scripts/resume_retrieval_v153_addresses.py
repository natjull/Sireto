"""V153 authorized pressure/RSS resume; unchanged V142 all-field projection.

Historical V142/V143 reports retain their own zero-global-swap policy.
New V153 artifacts explicitly use the approved pressure/RSS policy.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.probe_retrieval_v118_official_variants import fingerprint
from src.xgb_matcher.retrieval_address_projection_v142 import SCHEMA,required,project,audit_batch
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget,quiet,LIMITS,POLICY,validate_report,supervise
from scripts.resume_retrieval_v143_addresses import validate as legacy_validate
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v153_address_resume.json'
OLD='SEALED_V142_COMPLETE_ADDRESS_PHYSICAL_PART_ONLY'
NEW='SEALED_V153_COMPLETE_ADDRESS_PHYSICAL_PART_ONLY'


def validate(ref,role,spec,source,launch,catalog,first_rows):
    m=json.loads(verify(ref).read_text())
    if m['status']!=NEW:
        return legacy_validate(ref,role,spec,source,launch,catalog,first_rows)
    validate_report(m['resources']);artifact=verify(m['artifact'])
    if (m['status'] not in (OLD,NEW) or m['launch']!=launch or m['role']!=role or m['part']!=spec or m['catalog']!=catalog
        or m['source_rows']!=spec['source_stop']-spec['source_start'] or m['observations']!=m['source_rows']*source['factor']
        or not m['resources']['pass_resource_policy']
        or any(m.get(k) is not False for k in PROTECTED) or m['parent_addresses_spread_to_sites'] or m['context_or_masked_rows_removed']
        or not m['every_source_row_and_role_retained'] or not m['every_exported_field_matches_second_source_pass']
        or not m['every_exported_site_reconstructed_from_original_identity'] or m['complete_retrieval_index'] or m['ready_for_fold1']
        or sum(m['subject_counts'].values())!=m['observations'] or sum(r['observations'] for r in m['source_counts'])!=m['observations']
        or len({r['source'] for r in m['source_counts']})!=len(m['source_counts'])
        or m['lexical_observations']>m['subject_counts']['explicit_site']
        or artifact.resolve()!=(Path(ref['path']).parent/'observations.parquet').resolve()
        or pq.ParquetFile(artifact).metadata.num_rows!=m['observations'] or pq.ParquetFile(artifact).schema_arrow!=SCHEMA
        or (spec['part_id']==0 and m['scalar_first_source_rows_checked']!=min(m['source_rows'],first_rows))):
        raise ValueError('Every reused/new original proof part must be independently complete and protected')
    if m['status']==NEW:
        gate=json.loads(verify(m['preflight']).read_text())
        if not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed'] or gate['policy_id']!=POLICY:
            raise ValueError('Explicit authorized pressure gate required')
    return m


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    verify(l['resumption']['original_launch']);verify(l['resumption']['original_stop'])
    if l['config']['resources']!=LIMITS or l['resumption']['authorization']!='HUMAN_OUI_2026_10_04_1GIB_PRESSURE_SWAP_TELEMETRY':
        raise ValueError('Exact explicitly authorized memory policy required')
    if any(l.get(k) is not False for k in PROTECTED) or np.__version__!=l['config']['numpy_version']:
        raise ValueError('Protected unchanged source-only resumption launch required')
    p=json.loads(verify(l['plan']).read_text())
    if p['status']!='SEALED_V153_COMPLETE_ORIGINAL_ADDRESS_RESUMPTION_PLAN':raise ValueError('Whole original resumption plan required')
    verify(p['original_plan'])
    for s in p['sources'].values():
        if fingerprint(s['reference']['path'])!=s['fingerprint']:raise ValueError('Whole original source changed')
    return l,p


def directory(output,role,part):return output/role/f'part{part:05d}'


def init(output):
    resume=json.loads(CONFIG.read_text())
    if resume['authorization']!='HUMAN_OUI_2026_10_04_1GIB_PRESSURE_SWAP_TELEMETRY':
        raise ValueError('Explicit human memory-policy authorization required')
    resume['original_launch']=record(verify(resume['original_launch']))
    resume['original_stop']=record(verify(resume['original_stop']))
    old=json.loads(verify(resume['original_launch']).read_text())
    old_root=Path(resume['original_launch']['path']).parent
    for ref in old['implementation'].values():verify(ref)
    stop=json.loads(verify(resume['original_stop']).read_text())
    original=json.loads(verify(old['plan']).read_text())
    if (any(old.get(k) is not False for k in PROTECTED)
        or stop['status']!='STOP' or any(stop.get(k) is not False for k in PROTECTED)
        or stop['resources']['pass_resource_policy']
        or original['status']!='SEALED_V143_COMPLETE_ORIGINAL_ADDRESS_RESUMPTION_PLAN'
        or any(original.get(k) is not False for k in PROTECTED)):
        raise ValueError('Actual stopped complete protected historical source plan required')
    verify(original['original_plan'])
    cfg=dict(old['config']);cfg['resources']=dict(LIMITS)
    if cfg['batch_rows']!=resume['batch_rows'] or original['total_observations']!=cfg['expected_total_observations']:
        raise ValueError('Unchanged batches and every original observation required')
    output.mkdir(parents=True,exist_ok=False)
    print(json.dumps(dict(phase='initial_pressure_gate',required_seconds=resume['initial_quiet_seconds'])),flush=True)
    gate=quiet(resume['initial_quiet_seconds'],resume['min_system_free_percent'],resume['max_quiet_wait_seconds'])
    write_json(output/'preflight.json',gate);imports={}
    with Budget(cfg['resources']) as budget:
        catalog=AppendKeys(cfg['catalog'])
        if catalog.sites!=cfg['expected_namespace_sites']:raise ValueError('Whole unchanged namespace required')
        for role in original['role_order']:
            source=original['sources'][role];verify(source['reference'])
            if fingerprint(source['reference']['path'])!=source['fingerprint']:raise ValueError('Original source drift')
            for spec in source['parts']:
                key=f'{role}/{spec["part_id"]}'
                if key in original['imported_parts']:
                    ref=original['imported_parts'][key];expected=old['resumption']['original_launch']
                else:
                    path=directory(old_root,role,spec['part_id'])/'manifest.json'
                    if not path.exists():continue
                    ref=record(path);expected=resume['original_launch']
                legacy_validate(ref,role,spec,source,expected,cfg['catalog'],cfg['scalar_first_source_rows'])
                imports[key]=dict(manifest=ref,launch=expected)
        if len(imports)!=resume['expected_reused_parts'] or any(f'{role}/0' not in imports for role in original['role_order']):
            raise ValueError('All original successes and both scalar oracles required; never partial STOP')
    plan=dict(original,status='SEALED_V153_COMPLETE_ORIGINAL_ADDRESS_RESUMPTION_PLAN',
              previous_plan=old['plan'],imported_parts=imports,imported_part_count=len(imports),
              resumption_audit_resources=budget.report())
    write_json(output/'plan.json',plan)
    paths=[Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_memory_policy_v153.py',
           ROOT/'scripts/resume_retrieval_v143_addresses.py']
    implementation=dict(old['implementation']);implementation.update({str(p):record(p) for p in paths})
    write_json(output/'launch.json',dict(config=cfg,resumption=resume,plan=record(output/'plan.json'),
        implementation=implementation,preflight=record(output/'preflight.json'),**{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='plan',imported_parts=len(imports),total_parts=plan['total_parts'],
                         resources=budget.report())),flush=True)


def worker(output,role,part):
    l,p=load(output);cfg=l['config'];s=p['sources'][role];spec=s['parts'][part];dest=directory(output,role,part);budget=None
    if f'{role}/{part}' in p['imported_parts'] or any((dest/k).exists() for k in ('manifest.json','worker_stop.json','observations.parquet')):
        raise ValueError('No overwrite/retry of imported, successful, failed or partial address part')
    try:
        gate=json.loads(verify(record(dest/'preflight.json')).read_text())
        if not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed'] or gate['policy_id']!=POLICY:raise ValueError('Actual authorized pressure gate required')
        with Budget(cfg['resources']) as budget:
            catalog=AppendKeys(cfg['catalog']);f=pq.ParquetFile(s['reference']['path']);count=observed=eligible=0
            subjects=np.zeros(3,dtype=np.uint64);families={};cache={}
            def batches():return f.iter_batches(batch_size=cfg['batch_rows'],row_groups=list(range(spec['first_group'],spec['stop_group'])),
                columns=list(required(role)),use_threads=False)
            path=dest/'observations.parquet'
            with pq.ParquetWriter(path,SCHEMA,compression='zstd') as writer:
                for batch in batches():
                    original=pa.Table.from_batches([batch]);r=project(original,role,catalog,spec['source_start']+count,s['observation_base'],cache)
                    writer.write_table(r);count+=len(original);observed+=len(r)
                    eligible+=int(pc.sum(pc.cast(r['lexical_eligible'],pa.uint64())).as_py())
                    subjects+=np.bincount(r['subject_code'].to_numpy(),minlength=3).astype(np.uint64)
                    for row in pc.value_counts(r['source']).to_pylist():
                        name=row['values'];families[name]=families.get(name,0)+row['counts']
            if spec['source_start']+count!=spec['source_stop'] or observed!=count*s['factor']:raise ValueError('Whole address source denominator required')
            exported=pq.ParquetFile(path).iter_batches(batch_size=cfg['batch_rows']*s['factor'],use_threads=False);cursor=0
            for batch in batches():
                got=next(exported,None)
                if got is None:raise ValueError('Missing exported address proofs')
                original=pa.Table.from_batches([batch]);audit_batch(original,pa.Table.from_batches([got]),role,catalog,
                    spec['source_start']+cursor,s['observation_base']);cursor+=len(original)
            if next(exported,None) is not None or cursor!=count:raise ValueError('Every original exported proof audited once')
            load(output)
            for ref in catalog.references:verify(ref)
            artifact=record(path)
            if artifact['size_bytes']>cfg['max_part_output_bytes']:raise ValueError('STOP_ADDRESS_PART_OUTPUT_SIZE')
        write_json(dest/'manifest.json',dict(status=NEW,launch=record(output/'launch.json'),preflight=record(dest/'preflight.json'),
            role=role,part=spec,catalog=cfg['catalog'],source_rows=count,observations=observed,lexical_observations=eligible,
            subject_counts=dict(unverified_context=int(subjects[0]),parent_context=int(subjects[1]),explicit_site=int(subjects[2])),
            source_counts=[dict(source=k,observations=v) for k,v in families.items()],scalar_first_source_rows_checked=0,
            artifact=artifact,resources=budget.report(),every_source_row_and_role_retained=True,
            every_exported_field_matches_second_source_pass=True,every_exported_site_reconstructed_from_original_identity=True,
            all_rows_independent_scalar_oracle=False,parent_addresses_spread_to_sites=False,context_or_masked_rows_removed=False,
            complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(role=role,part=part,rows=count,observations=observed,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,role,part):
    l,p=load(output)
    if role not in p['sources'] or not isinstance(part,int) or isinstance(part,bool) or not 0<=part<len(p['sources'][role]['parts']):
        raise ValueError('Actual planned original address part required')
    if f'{role}/{part}' in p['imported_parts']:raise ValueError('Never rebuild imported successful original part')
    dest=directory(output,role,part);dest.mkdir(parents=True,exist_ok=False)
    try:
        cfg=l['resumption'];gate=quiet(cfg['phase_quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
        write_json(dest/'preflight.json',gate)
        with (dest/'worker.log').open('x') as log:
            supervision=supervise([sys.executable,'-u','-m','scripts.resume_retrieval_v153_addresses','worker','--output',str(output),
                '--role',role,'--part',str(part)],log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        write_json(dest/'supervision.json',supervision)
        print((dest/'worker.log').read_text()[-2500:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def seal(output):
    if (output/'manifest.json').exists():raise ValueError('Immutable whole resumption already sealed')
    l,p=load(output);refs=[];rows=observations=eligible=size=reused_size=0;subjects=dict(unverified_context=0,parent_context=0,explicit_site=0);families={}
    with Budget(l['config']['resources']) as budget:
        for role in p['role_order']:
            s=p['sources'][role];cursor=0
            for spec in s['parts']:
                key=f'{role}/{spec["part_id"]}';imported=key in p['imported_parts']
                path=directory(output,role,spec['part_id'])/'manifest.json'
                if not imported and not path.exists():raise ValueError('UNFINISHED_ALL_ORIGINAL_ADDRESS_PARTS')
                ref=p['imported_parts'][key]['manifest'] if imported else record(path)
                expected=p['imported_parts'][key]['launch'] if imported else record(output/'launch.json')
                m=validate(ref,role,spec,s,expected,l['config']['catalog'],l['config']['scalar_first_source_rows'])
                if spec['source_start']!=cursor:raise ValueError('Every original physical source part contiguous')
                refs.append(dict(imported=imported,manifest=ref));cursor=spec['source_stop'];rows+=m['source_rows'];observations+=m['observations']
                eligible+=m['lexical_observations'];size+=m['artifact']['size_bytes'];reused_size+=m['artifact']['size_bytes'] if imported else 0
                for key,value in m['subject_counts'].items():subjects[key]+=value
                for item in m['source_counts']:
                    key=item['source'];families[key]=families.get(key,0)+item['observations']
            if cursor!=s['source_rows']:raise ValueError('Complete original source denominator required')
            verify(s['reference'])
        if (len(refs)!=p['total_parts'] or rows!=p['total_source_rows'] or observations!=p['total_observations']
            or size>l['config']['max_total_output_bytes']):raise ValueError('Every original observation/source/part required')
    write_json(output/'manifest.json',dict(status='SEALED_V153_ALL_ORIGINAL_OFFICIAL_AND_RAW_CURRENT_ADDRESS_PROJECTION',
        launch=record(output/'launch.json'),plan=l['plan'],catalog=l['config']['catalog'],parts=refs,source_rows=rows,observations=observations,
        lexical_observations=eligible,subject_counts=subjects,source_counts=[dict(source=k,observations=v) for k,v in families.items()],
        output_bytes=size,reused_output_bytes=reused_size,new_output_bytes=size-reused_size,seal_resources=budget.report(),
        every_original_source_full_hash_verified_before_and_after=True,every_source_row_and_role_retained=True,
        every_exported_field_matches_second_source_pass=True,every_site_proof_scope_reconstructed=True,
        all_rows_independent_scalar_oracle=False,parent_addresses_spread_to_sites=False,
        complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='seal',source_rows=rows,observations=observations,lexical_observations=eligible,output_bytes=size)),flush=True)


def remaining(output):
    l,p=load(output)
    for role in p['role_order']:
        for spec in p['sources'][role]['parts']:
            part=spec['part_id'];key=f'{role}/{part}';dest=directory(output,role,part)
            if key in p['imported_parts']:continue
            if dest.exists():
                if not (dest/'manifest.json').exists():raise ValueError('STOP_OR_LIVE_PART_DO_NOT_RESTART')
                continue
            execute(output,role,part)
    seal(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['init','part','worker','seal','remaining'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--role');parser.add_argument('--part',type=int);a=parser.parse_args()
    if a.mode in ('part','worker'):
        if a.role is None or a.part is None:parser.error('role/part required')
        (execute if a.mode=='part' else worker)(a.output,a.role,a.part)
    else:{'init':init,'seal':seal,'remaining':remaining}[a.mode](a.output)
