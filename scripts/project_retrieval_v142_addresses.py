"""Whole-source official addresses, bounded immutable physical parts."""
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
from scripts.project_retrieval_v129_national_variants import planned_parts
from src.xgb_matcher.retrieval_address_projection_v142 import SCHEMA,required,project,audit_batch,scalar_text_oracle,LINE,FULL
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v142_address_projection.json'


def load(output):
    l=json.loads((output/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if any(l.get(k) is not False for k in PROTECTED) or np.__version__!=l['config']['numpy_version']:
        raise ValueError('Protected unchanged source-only launch required')
    plan=json.loads(verify(l['plan']).read_text())
    if plan['status']!='SEALED_V142_ALL_OFFICIAL_ADDRESS_PHYSICAL_PLAN':raise ValueError('Whole address plan required')
    for source in plan['sources'].values():
        if fingerprint(source['reference']['path'])!=source['fingerprint']:raise ValueError('Whole original address source changed')
    return l,plan


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text());sources={};base=0
    if set(cfg['sources'])!={'official_addresses','current_addresses'} or cfg['role_order']!=['official_addresses','current_addresses']:
        raise ValueError('Both whole original address sources required')
    if (not isinstance(cfg['batch_rows'],int) or isinstance(cfg['batch_rows'],bool) or cfg['batch_rows']<=0
        or not isinstance(cfg['scalar_first_source_rows'],int) or isinstance(cfg['scalar_first_source_rows'],bool)
        or not 1<=cfg['scalar_first_source_rows']<=cfg['batch_rows']):
        raise ValueError('Positive bounded first-source scalar oracle required')
    with Budget(cfg['resources']) as budget:
        catalog=AppendKeys(cfg['catalog'])
        if catalog.sites!=cfg['expected_namespace_sites']:raise ValueError('Whole unchanged address-admitting namespace required')
        for role in cfg['role_order']:
            ref=cfg['sources'][role];path=verify(ref);f=pq.ParquetFile(path)
            if f.metadata.num_rows!=ref['expected_rows'] or f.metadata.num_rows<=0:
                raise ValueError('Whole original address source count required')
            for key in required(role):
                if f.schema_arrow.field(key).type!=(pa.bool_() if key=='is_current' else pa.string()):
                    raise ValueError('Explicit original address field schema required')
            factor=1 if role=='official_addresses' else 4
            sources[role]=dict(reference=ref,fingerprint=fingerprint(path),source_rows=f.metadata.num_rows,
                source_fields=f.schema_arrow.names,projection_fields=list(required(role)),observation_base=base,factor=factor,
                parts=planned_parts(f,cfg['row_groups_per_part']))
            base+=factor*f.metadata.num_rows
        if base!=cfg['expected_total_observations'] or base>=2**32-1:raise ValueError('Every planned address observation required')
    write_json(output/'plan.json',dict(status='SEALED_V142_ALL_OFFICIAL_ADDRESS_PHYSICAL_PLAN',sources=sources,
        role_order=cfg['role_order'],total_source_rows=sum(s['source_rows'] for s in sources.values()),total_observations=base,
        total_parts=sum(len(s['parts']) for s in sources.values()),init_resources=budget.report(),
        projection_line_fields=list(LINE),projection_full_fields=list(FULL),
        primary_full_additional_fields=['dernierNumeroVoieEtablissement','indiceRepetitionDernierNumeroVoieEtablissement'],
        every_original_source_full_hash_verified=True,**{k:False for k in PROTECTED}))
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v142_address_projection.md',
        ROOT/'scripts/probe_retrieval_v118_official_variants.py',ROOT/'scripts/project_retrieval_v129_national_variants.py',
        *[ROOT/('src/xgb_matcher/'+k+'.py') for k in ('retrieval_address_projection_v142','retrieval_append_catalog_v141',
            'retrieval_proof_bridge_v137','retrieval_extended_catalog_v136','retrieval_canonical_inventory_v140',
            'retrieval_variant_batches_v128','retrieval_variants_v118','retrieval_site_attributes_v139',
            'retrieval_coupled_source_v121','retrieval_coupled_blocks_v121','retrieval_activity_joint_v114',
            'retrieval_official_context_v110','retrieval_activity_semantic_v113','retrieval_v20_workbench',
            'retrieval_parent_discovery_v18')]]
    write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),
        implementation={str(p):record(p) for p in paths},**{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='plan',source_rows=sum(s['source_rows'] for s in sources.values()),observations=base,
        parts=sum(len(s['parts']) for s in sources.values()),resources=budget.report())),flush=True)


def directory(output,role,part):return output/role/f'part{part:05d}'


def worker(output,role,part):
    l,plan=load(output);cfg=l['config'];s=plan['sources'][role];spec=s['parts'][part];dest=directory(output,role,part);budget=None
    if any((dest/k).exists() for k in ('manifest.json','worker_stop.json','observations.parquet')):
        raise ValueError('Immutable address projection part already started')
    try:
        with Budget(cfg['resources']) as budget:
            catalog=AppendKeys(cfg['catalog']);f=pq.ParquetFile(s['reference']['path']);count=0;observed=0;eligible=0
            subject_counts=np.zeros(3,dtype=np.uint64);families={};scalar=0;cache={}
            def batches():return f.iter_batches(batch_size=cfg['batch_rows'],row_groups=list(range(spec['first_group'],spec['stop_group'])),
                columns=list(required(role)),use_threads=False)
            path=dest/'observations.parquet'
            with pq.ParquetWriter(path,SCHEMA,compression='zstd') as writer:
                for batch in batches():
                    original=pa.Table.from_batches([batch]);result=project(original,role,catalog,spec['source_start']+count,s['observation_base'],cache)
                    if part==0 and count==0:
                        n=min(len(original),cfg['scalar_first_source_rows'])
                        scalar=scalar_text_oracle(original.slice(0,n),result.slice(0,n*s['factor']),role)
                    writer.write_table(result);count+=len(original);observed+=len(result)
                    eligible+=int(pc.sum(pc.cast(result['lexical_eligible'],pa.uint64())).as_py())
                    subject_counts+=np.bincount(result['subject_code'].to_numpy(),minlength=3).astype(np.uint64)
                    for row in pc.value_counts(result['source']).to_pylist():
                        name=row['values'];families[name]=families.get(name,0)+row['counts']
            if spec['source_start']+count!=spec['source_stop'] or observed!=count*s['factor']:
                raise ValueError('Every source row and original projection role must be present')
            # Fresh full source pass. Also every exported site is reconstructed
            # from the ORIGINAL identity, independently of the text projector.
            exported=pq.ParquetFile(path).iter_batches(batch_size=cfg['batch_rows']*s['factor'],use_threads=False)
            cursor=0
            for batch in batches():
                got=next(exported,None)
                if got is None:raise ValueError('Incomplete exported address observations')
                original=pa.Table.from_batches([batch]);audit_batch(original,pa.Table.from_batches([got]),role,catalog,
                    spec['source_start']+cursor,s['observation_base']);cursor+=len(original)
            if next(exported,None) is not None or cursor!=count:raise ValueError('Every exported address observation audited exactly once')
            load(output)
            for ref in catalog.references:verify(ref)
            artifact=record(path)
            if artifact['size_bytes']>cfg['max_part_output_bytes']:raise ValueError('STOP_ADDRESS_PART_OUTPUT_SIZE')
        write_json(dest/'manifest.json',dict(status='SEALED_V142_COMPLETE_ADDRESS_PHYSICAL_PART_ONLY',launch=record(output/'launch.json'),
            role=role,part=spec,catalog=cfg['catalog'],source_rows=count,observations=observed,lexical_observations=eligible,
            subject_counts=dict(unverified_context=int(subject_counts[0]),parent_context=int(subject_counts[1]),explicit_site=int(subject_counts[2])),
            source_counts=[dict(source=k,observations=v) for k,v in families.items()],
            scalar_first_source_rows_checked=scalar,artifact=artifact,resources=budget.report(),
            every_source_row_and_role_retained=True,every_exported_field_matches_second_source_pass=True,
            every_exported_site_reconstructed_from_original_identity=True,all_rows_independent_scalar_oracle=False,
            parent_addresses_spread_to_sites=False,context_or_masked_rows_removed=False,
            complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(role=role,part=part,rows=count,observations=observed,resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED},
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,role,part):
    _,plan=load(output)
    if role not in plan['sources'] or not isinstance(part,int) or isinstance(part,bool) or not 0<=part<len(plan['sources'][role]['parts']):
        raise ValueError('Whole-source planned physical address part required')
    dest=directory(output,role,part);dest.mkdir(parents=True,exist_ok=False)
    try:
        with (dest/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.project_retrieval_v142_addresses','worker','--output',str(output),
                '--role',role,'--part',str(part)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((dest/'worker.log').read_text()[-3000:],flush=True)
    except BaseException as exc:
        write_json(dest/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


def seal(output):
    if (output/'manifest.json').exists():raise ValueError('Immutable whole address projection already sealed')
    l,plan=load(output);parts=[];rows=0;observations=0;size=0;eligible=0;subjects=dict(unverified_context=0,parent_context=0,explicit_site=0);families={}
    for role in plan['role_order']:
        source=plan['sources'][role];cursor=0
        for spec in source['parts']:
            path=directory(output,role,spec['part_id'])/'manifest.json'
            if not path.exists():raise ValueError('UNFINISHED_ALL_OFFICIAL_ADDRESS_PARTS')
            m=json.loads(path.read_text());artifact=verify(m['artifact'])
            if (m['status']!='SEALED_V142_COMPLETE_ADDRESS_PHYSICAL_PART_ONLY' or m['launch']!=record(output/'launch.json')
                or m['role']!=role or m['part']!=spec or m['catalog']!=l['config']['catalog']
                or m['source_rows']!=spec['source_stop']-spec['source_start'] or spec['source_start']!=cursor
                or m['observations']!=m['source_rows']*source['factor'] or not m['resources']['pass_resource_policy']
                or any(m.get(k) is not False for k in PROTECTED) or m['parent_addresses_spread_to_sites'] or m['context_or_masked_rows_removed']
                or not m['every_source_row_and_role_retained'] or not m['every_exported_field_matches_second_source_pass']
                or not m['every_exported_site_reconstructed_from_original_identity'] or sum(m['subject_counts'].values())!=m['observations']
                or sum(r['observations'] for r in m['source_counts'])!=m['observations']
                or len({r['source'] for r in m['source_counts']})!=len(m['source_counts'])
                or m['lexical_observations']>m['subject_counts']['explicit_site']
                or artifact.resolve()!=(path.parent/'observations.parquet').resolve()
                or pq.ParquetFile(artifact).metadata.num_rows!=m['observations']
                or (spec['part_id']==0 and m['scalar_first_source_rows_checked']!=min(m['source_rows'],l['config']['scalar_first_source_rows']))):
                raise ValueError('Protected complete original address part authority required')
            parts.append(record(path));cursor=spec['source_stop'];rows+=m['source_rows'];observations+=m['observations'];size+=m['artifact']['size_bytes'];eligible+=m['lexical_observations']
            for key,value in m['subject_counts'].items():subjects[key]+=value
            for item in m['source_counts']:
                key=item['source'];families[key]=families.get(key,0)+item['observations']
        if cursor!=source['source_rows']:raise ValueError('Whole address input denominator incomplete')
        verify(source['reference'])
    if rows!=plan['total_source_rows'] or observations!=plan['total_observations'] or size>l['config']['max_total_output_bytes']:
        raise ValueError('Whole address projection denominator/size required')
    write_json(output/'manifest.json',dict(status='SEALED_V142_ALL_ORIGINAL_OFFICIAL_AND_RAW_CURRENT_ADDRESS_PROJECTION',
        launch=record(output/'launch.json'),plan=l['plan'],catalog=l['config']['catalog'],parts=parts,source_rows=rows,
        observations=observations,lexical_observations=eligible,subject_counts=subjects,
        source_counts=[dict(source=k,observations=v) for k,v in families.items()],output_bytes=size,
        every_original_source_full_hash_verified_before_and_after=True,every_source_row_and_role_retained=True,
        every_exported_field_matches_second_source_pass=True,every_site_proof_scope_reconstructed=True,
        all_rows_independent_scalar_oracle=False,parent_addresses_spread_to_sites=False,
        complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='seal',source_rows=rows,observations=observations,lexical_observations=eligible,output_bytes=size)),flush=True)


def remaining(output):
    _,plan=load(output)
    for role in plan['role_order']:
        for spec in plan['sources'][role]['parts']:
            dest=directory(output,role,spec['part_id'])
            if dest.exists():
                # Existing successful phases are immutable, but still checked
                # against their pinned implementation/source at final seal.
                if not (dest/'manifest.json').exists():raise ValueError('STOP_OR_LIVE_PART_DO_NOT_RESTART')
                continue
            execute(output,role,spec['part_id'])
    seal(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','part','worker','seal','remaining'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--role');p.add_argument('--part',type=int);a=p.parse_args()
    if a.mode in ('part','worker'):
        if a.role is None or a.part is None:p.error('role/part required')
        (execute if a.mode=='part' else worker)(a.output,a.role,a.part)
    else:{'init':init,'seal':seal,'remaining':remaining}[a.mode](a.output)
