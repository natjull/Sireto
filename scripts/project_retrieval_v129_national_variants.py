"""Resumable whole-source variant projection in immutable bounded physical parts.

Every planned row group must complete before seal. Partial parts are NOT a
national corpus or recall experiment. No query/label/source-name selection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.probe_retrieval_v118_official_variants import fingerprint,required_fields
from src.xgb_matcher.retrieval_variant_batches_v128 import project
from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA,observations,ROLES
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v129_national_variants.json'


def planned_parts(file,groups_per_part):
    if not isinstance(groups_per_part,int) or isinstance(groups_per_part,bool) or groups_per_part<=0:raise ValueError('Positive physical group quantum required')
    groups=file.metadata.num_row_groups;offsets=[0]
    for g in range(groups):offsets.append(offsets[-1]+file.metadata.row_group(g).num_rows)
    return [dict(part_id=i//groups_per_part,first_group=i,stop_group=min(groups,i+groups_per_part),
        source_start=offsets[i],source_stop=offsets[min(groups,i+groups_per_part)]) for i in range(0,groups,groups_per_part)]


def verify_plan(output):
    launch=json.loads((output/'launch.json').read_text())
    for ref in launch['implementation'].values():verify(ref)
    plan=json.loads(verify(launch['plan']).read_text())
    if plan['status']!='SEALED_WHOLE_SOURCE_PHYSICAL_PLAN_NO_QUERY_SELECTION':raise ValueError('Whole-source plan required')
    for role,source in plan['sources'].items():
        if fingerprint(source['path'])!=source['fingerprint']:raise ValueError('Whole official source changed: '+role)
    return launch,plan


def init(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text());sources={}
    if set(cfg['role_order'])!=set(ROLES) or set(cfg['sources'])!=set(ROLES) or len(cfg['role_order'])!=5:raise ValueError('All five official name roles required')
    for role in cfg['role_order']:
        ref=cfg['sources'][role];path=Path(ref['path']);f=pq.ParquetFile(path)
        if f.metadata.num_rows!=ref['expected_rows'] or f.metadata.num_rows<=0 or required_fields(role)-set(f.schema_arrow.names):raise ValueError('Complete official schema/count drift: '+role)
        sources[role]=dict(path=str(path),declared_full_sha256=ref['sha256'],whole_input_hash_verified=False,
            fingerprint=fingerprint(path),source_rows=f.metadata.num_rows,source_fields=f.schema_arrow.names,
            parts=planned_parts(f,cfg['row_groups_per_part']))
    plan=dict(status='SEALED_WHOLE_SOURCE_PHYSICAL_PLAN_NO_QUERY_SELECTION',sources=sources,role_order=cfg['role_order'],
        total_source_rows=sum(s['source_rows'] for s in sources.values()),total_parts=sum(len(s['parts']) for s in sources.values()),
        truth_opened=False,crm_opened=False,recall_measured=False,national_projection_complete=False)
    write_json(output/'plan.json',plan)
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v129_national_variants.md',ROOT/'src/xgb_matcher/retrieval_variant_batches_v128.py',
        ROOT/'src/xgb_matcher/retrieval_variants_v118.py',ROOT/'scripts/probe_retrieval_v118_official_variants.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,plan=record(output/'plan.json'),implementation={str(p):record(p) for p in paths},
        truth_opened=False,crm_opened=False))
    print(json.dumps(dict(planned=True,total_source_rows=plan['total_source_rows'],parts=plan['total_parts'])),flush=True)


def part_directory(output,role,part):return output/role/f'part{part:05d}'


def worker(output,role,part):
    launch,plan=verify_plan(output);cfg=launch['config'];source=plan['sources'][role];spec=source['parts'][part]
    destination=part_directory(output,role,part);budget=None
    try:
        with Budget(cfg['resources']) as budget:
            f=pq.ParquetFile(source['path']);count=0;observed=0;lexical=0;phase_seconds=0.;first_exact=False
            path=destination/'observations.parquet'
            with pq.ParquetWriter(path,OBS_SCHEMA,compression='zstd') as writer:
                for batch in f.iter_batches(batch_size=cfg['batch_rows'],row_groups=list(range(spec['first_group'],spec['stop_group'])),use_threads=False):
                    start=spec['source_start']+count;table=pa.Table.from_batches([batch]).append_column('source_row_ordinal',
                        pa.array(np.arange(start,start+len(batch),dtype=np.uint64)))
                    began=time.monotonic();result=project(table,role);phase_seconds+=time.monotonic()-began
                    original=pc.unique(result['source_row_ordinal']).to_numpy(zero_copy_only=False)
                    if not np.array_equal(original,np.arange(start,start+len(batch),dtype=np.uint64)):
                        raise ValueError('Every source row must retain at least one exact-ordinal observation')
                    if part==0 and count==0:
                        old=pa.Table.from_pylist([p for row in table.to_pylist() for p in observations(row,role,int(row['source_row_ordinal']))],schema=OBS_SCHEMA)
                        if not result.equals(old):raise ValueError('First whole-source batch27-field scalar oracle drift')
                        first_exact=True
                    writer.write_table(result);count+=len(batch);observed+=len(result)
                    lexical+=int(pc.sum(pc.cast(result['lexical_eligible'],pa.uint64())).as_py())
            if spec['source_start']+count!=spec['source_stop'] or count<=0:raise ValueError('Physical source-part denominator drift')
            verify_plan(output)
            size=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
            if size>cfg['max_part_output_bytes']:raise ValueError('STOP_PART_OUTPUT_SIZE')
        result=dict(status='SEALED_OFFICIAL_VARIANT_PHYSICAL_PART_ONLY',launch=record(output/'launch.json'),role=role,part=spec,
            source_rows=count,observations=observed,lexical_observations=lexical,projection_seconds=phase_seconds,
            every_source_row_retained=True,first_part_first_batch27_field_scalar_oracle=first_exact,
            artifact=record(path),resources=budget.report(),output_bytes=size,truth_opened=False,crm_opened=False,
            model_fit=False,recall_measured=False,fold0_opened=False,fold1_opened=False,
            full_national_projection_complete=False,complete_retrieval_index=False)
        write_json(destination/'manifest.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(destination/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def execute(output,role,part):
    _,plan=verify_plan(output)
    if role not in plan['role_order'] or not isinstance(part,int) or isinstance(part,bool) or not 0<=part<len(plan['sources'][role]['parts']):raise ValueError('Planned physical part required')
    destination=part_directory(output,role,part);destination.mkdir(parents=True,exist_ok=False)
    try:
        with (destination/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.project_retrieval_v129_national_variants','worker','--output',str(output),
                '--role',role,'--part',str(part)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((destination/'worker.log').read_text()[-5500:],flush=True)
    except BaseException as exc:
        write_json(destination/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False));raise


def seal(output):
    launch,plan=verify_plan(output);parts=[];total=0;observed=0;size=0
    for role in plan['role_order']:
        source=plan['sources'][role];cursor=0
        for spec in source['parts']:
            path=part_directory(output,role,spec['part_id'])/'manifest.json'
            if not path.exists():raise ValueError('UNFINISHED_SOURCE_PARTS: '+str(path))
            m=json.loads(path.read_text())
            if (m['status']!='SEALED_OFFICIAL_VARIANT_PHYSICAL_PART_ONLY' or m['launch']!=record(output/'launch.json')
                or m['part']!=spec or m['role']!=role or m['source_rows']!=spec['source_stop']-spec['source_start']
                or spec['source_start']!=cursor or not m['every_source_row_retained'] or not m['resources']['pass_resource_policy']
                or any(m.get(k) is not False for k in ('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened'))
                or (spec['part_id']==0 and not m['first_part_first_batch27_field_scalar_oracle'])):
                raise ValueError('Physical part authority/coverage drift')
            artifact=verify(m['artifact'])
            if artifact.resolve()!=(path.parent/'observations.parquet').resolve() or pq.ParquetFile(artifact).metadata.num_rows!=m['observations']:
                raise ValueError('Physical observation path/count drift')
            parts.append(record(path));cursor=spec['source_stop'];total+=m['source_rows'];observed+=m['observations'];size+=m['output_bytes']
        if cursor!=source['source_rows']:raise ValueError('Whole official role denominator incomplete')
    if total!=plan['total_source_rows'] or size>launch['config']['max_total_output_bytes']:raise ValueError('Whole projection completeness/size drift')
    if (output/'manifest.json').exists():raise ValueError('Immutable full projection already sealed')
    write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_ALL_OFFICIAL_VARIANT_SOURCE_PROJECTION',launch=record(output/'launch.json'),
        plan=record(output/'plan.json'),parts=parts,source_rows=total,observations=observed,output_bytes=size,
        every_source_role_and_row_complete=True,truth_opened=False,crm_opened=False,recall_measured=False,
        complete_retrieval_index=False,ready_for_fold1=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','part','worker','seal']);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--role');p.add_argument('--part',type=int);a=p.parse_args()
    if a.mode in ('part','worker'):
        if a.role is None or a.part is None:p.error('role/part required')
        (execute if a.mode=='part' else worker)(a.output,a.role,a.part)
    else:(init if a.mode=='init' else seal)(a.output)
