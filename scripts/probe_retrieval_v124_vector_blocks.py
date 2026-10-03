"""Official-source coupled-block cost/oracle probe, never recall on a sample."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.probe_retrieval_v118_official_variants import fingerprint
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query, weights
from src.xgb_matcher.retrieval_vector_blocks_v124 import VectorCoupledBlocks as CoupledBlocks
from src.xgb_matcher.retrieval_coupled_source_v122 import build_sites, family_batches
from src.xgb_matcher.retrieval_variants_v118 import observations
from src.xgb_matcher.retrieval_joint_v46 import geo_parameters
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v124_vector_blocks.json'


def inputs(cfg):
    vm=json.loads(verify(cfg['variant_manifest']).read_text());sm=json.loads(verify(cfg['sample_manifest']).read_text())
    if (vm['status']!='SEALED_SAME_VARIANT_HASHJOIN_COST_ONLY' or sm['status']!='SEALED_SOURCE_VARIANT_COST_PROTOTYPE_ONLY'
        or any(m['truth_opened'] or m['recall_measured'] for m in (vm,sm))):
        raise ValueError('Sealed official-only cost samples required')
    refs={key:vm['artifacts'][key] for key in ('names.parquet','facts.parquet','observations.parquet')}
    refs.update({key:sm['artifacts'][key] for key in ('sample_current_site.parquet','sample_successions.parquet')})
    for ref in refs.values():
        verify(ref)
    old=json.loads(verify(cfg['query_only_launch']).read_text())
    if old['truth_opened'] or old['old_candidates_opened'] or len(old['queries'])!=128:
        raise ValueError('Frozen query-only CRM population required')
    if any(q['fold'] not in (2,3,4) for q in old['queries']):
        raise ValueError('Closed/contaminated query fold')
    models=json.loads(verify(old['models']).read_text())
    ids={q['query_id'] for q in old['queries']}
    if set(models)!={'2','3','4'}:
        raise ValueError('Own-fold234 initial models required')
    for f,m in models.items():
        if (m['heldout_fold']!=int(f) or set(m['training_folds'])!={2,3,4}-{int(f)}
            or m['identifiers_as_features'] or ids & set(m['query_ids'])):
            raise ValueError('Unsafe initial-model transfer')
        weights(m['weights'])
    geo=json.loads(verify(old['geo_queries']).read_text())
    if {g['query_id'] for g in geo}!=ids:
        raise ValueError('CRM-only BAN geometry population drift')
    return refs,old,models,{g['query_id']:g for g in geo}


def extra_source(cfg,output,kind):
    ref=cfg[kind+'_source'];path=Path(ref['path']);before=fingerprint(path)
    data=[];groups=[];file=output/('family_'+kind+'.parquet');writer=None
    try:
        for family,group,start,batch in family_batches(path,cfg['rows_per_family'],tuple(cfg['source_families'])):
            table=pa.Table.from_batches([batch]).append_column('source_row_ordinal',pa.array(range(start,start+len(batch)),type=pa.uint64()))
            if writer is None:
                writer=pq.ParquetWriter(file,table.schema,compression='zstd')
            writer.write_table(table);data.extend(table.to_pylist())
            groups.append(dict(source=family,row_group=group,source_row_start=start,rows=len(batch)))
    finally:
        if writer is not None:
            writer.close()
    if fingerprint(path)!=before:
        raise ValueError('Source-family sample changed during read')
    return data,dict(path=str(path),declared_full_sha256=ref['sha256'],fingerprint=before,
                    sampled_groups=groups,record=record(file),whole_source_hash_verified=False)


def worker(output):
    launch=json.loads((output/'launch.json').read_text());budget=None
    try:
        for ref in launch['implementation'].values():verify(ref)
        cfg=launch['config']
        with Budget(cfg['resources']) as budget:
            refs,old,models,geography=inputs(cfg);began=time.monotonic()
            extra_names,name_source=extra_source(cfg,output,'name')
            extra_addresses,address_source=extra_source(cfg,output,'address')
            expanded=[p for row in extra_names for p in observations(row,'official_names',int(row['source_row_ordinal']))]
            def table(key,columns=None):return pq.read_table(verify(refs[key]),columns=columns,use_threads=False).to_pylist()
            sites,source_stats=build_sites(table('sample_current_site.parquet'),table('names.parquet'),table('facts.parquet'),
                table('observations.parquet',['fact_id','is_current']),expanded,extra_addresses,table('sample_successions.parquet'))
            source_seconds=time.monotonic()-began;began=time.monotonic();index=CoupledBlocks(sites,cfg['block_size'])
            build_seconds=time.monotonic()-began;rows=[]
            print(json.dumps(dict(phase='source_adapter_and_block_build',sites=len(sites),blocks=len(index.groups),
                source_seconds=source_seconds,build_seconds=build_seconds,known_xy=source_stats['with_geometry'])),flush=True)
            for raw in old['queries'][:cfg['queries']]:
                p,_,reliable=geo_parameters(geography[raw['query_id']])
                q=Query(raw['crm_name'],' '.join(raw[k] for k in ('crm_number','crm_suffix','crm_street')).strip(),
                    raw['crm_postcode'],raw['crm_insee'],raw['crm_number'],None if p is None else tuple(p),reliable)
                w=models[str(raw['fold'])]['weights']
                found,trace=index.search(q,w,limit=cfg['collector_cap'],max_seconds=cfg['max_query_seconds'])
                began=time.monotonic();oracle=index.exhaustive(q,w,limit=cfg['collector_cap']);oracle_seconds=time.monotonic()-began
                if found!=oracle:
                    raise ValueError('Initial100 coupled search/exhaustive oracle drift')
                rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],candidates=found,trace=trace,
                    oracle_seconds=oracle_seconds,oracle_exact=True,recall_measured=False))
                print(json.dumps(dict(query_id=raw['query_id'],trace=trace,oracle_seconds=oracle_seconds,oracle_exact=True)),flush=True)
            for source in (name_source,address_source):
                if fingerprint(source['path'])!=source['fingerprint']:raise ValueError('Official source changed after probe')
            inputs(cfg)
            write_json(output/'cost_outputs.json',dict(rows=rows,truth_opened=False,recall_measured=False,national_corpus=False))
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values():verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_VECTOR_COHERENT_BLOCK_SAMPLE_SCALAR_ORACLE_COST_ONLY',launch=record(output/'launch.json'),
            outputs=record(output/'cost_outputs.json'),source_stats=source_stats,extra_sources=[name_source,address_source],
            source_seconds=source_seconds,index_build_seconds=build_seconds,queries=len(rows),sites=len(sites),blocks=len(index.groups),
            resources=budget.report(),output_bytes=size,metric_family=cfg['metric_family'],truth_opened=False,crm_opened=True,
            fold0_opened=False,fold1_opened=False,model_fit=False,model_used=True,activity_grades_enabled=False,
            national_corpus=False,recall_measured=False,identifiers_as_features=False,downstream_models_changed=False,ready_for_fold1=False))
        print(json.dumps(dict(queries=len(rows),resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_coupled_blocks_v121.py',ROOT/'src/xgb_matcher/retrieval_tight_blocks_v123.py',ROOT/'src/xgb_matcher/retrieval_vector_blocks_v124.py',
        ROOT/'src/xgb_matcher/retrieval_coupled_source_v121.py',ROOT/'src/xgb_matcher/retrieval_coupled_source_v122.py',ROOT/'src/xgb_matcher/retrieval_variants_v118.py',
        ROOT/'src/xgb_matcher/retrieval_activity_joint_v114.py',ROOT/'src/xgb_matcher/retrieval_joint_v46.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',
        ROOT/'scripts/probe_retrieval_v118_official_variants.py',ROOT/'docs/retrieval_v124_vector_blocks.md']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v124_vector_blocks','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-10000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
