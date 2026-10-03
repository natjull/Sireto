"""Exact complete-observation projection and paired source-only cost."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_variant_batches_v128 import project
from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA,observations
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v128_variant_batches.json'


def scalar(table,role):
    rows=[p for row in table.to_pylist() for p in observations(row,role,int(row['source_row_ordinal']))]
    return pa.Table.from_pylist(rows,schema=OBS_SCHEMA)


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];budget=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            source=json.loads(verify(cfg['sample_manifest']).read_text())
            if source['truth_opened'] or source['crm_opened']:raise ValueError('Official-only snapshot required')
            before=pq.read_table(verify(source['artifacts']['raw_observations.parquet']),use_threads=False)
            tables=[];phases=[]
            for i,role in enumerate(cfg['role_order']):
                began=time.monotonic();table=pq.read_table(verify(source['artifacts']['sample_'+role+'.parquet']),use_threads=False)
                read_seconds=time.monotonic()-began;results={};times={}
                for implementation in (('batch','scalar') if i%2==0 else ('scalar','batch')):
                    began=time.monotonic();results[implementation]=(project(table,role) if implementation=='batch' else scalar(table,role))
                    times[implementation]=time.monotonic()-began
                if not results['batch'].equals(results['scalar']):raise ValueError('Complete ordered27-field scalar projection drift: '+role)
                tables.append(results['batch']);phases.append(dict(role=role,source_rows=len(table),observations=len(results['batch']),
                    read_seconds=read_seconds,batch_seconds=times['batch'],scalar_seconds=times['scalar'],ordered27_fields_exact=True))
                print(json.dumps(phases[-1]),flush=True)
            whole=pa.concat_tables(tables)
            if not whole.equals(before):raise ValueError('Historical sealed complete observation/order drift')
            pq.write_table(whole,output/'projected_observations.parquet',compression='zstd')
            family=pq.read_table(verify(cfg['family_names']),use_threads=False);began=time.monotonic();new=project(family,'official_names')
            batch_seconds=time.monotonic()-began;began=time.monotonic();old=scalar(family,'official_names');scalar_seconds=time.monotonic()-began
            if not new.equals(old):raise ValueError('All source-family ordered27-field drift')
            pq.write_table(new,output/'family_observations.parquet',compression='zstd')
            if not pq.read_table(output/'projected_observations.parquet',use_threads=False).equals(before):raise ValueError('Saved observation roundtrip drift')
            if not pq.read_table(output/'family_observations.parquet',use_threads=False).equals(old):raise ValueError('Saved family roundtrip drift')
            verify(cfg['sample_manifest']);verify(cfg['family_names'])
            for role in cfg['role_order']:verify(source['artifacts']['sample_'+role+'.parquet'])
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values():verify(ref)
        stats=dict(observations=len(whole),roles=phases,family_source_rows=len(family),family_observations=len(new),
            family_batch_seconds=batch_seconds,family_scalar_seconds=scalar_seconds,all_ordered27_fields_exact=True,
            reference='V118_COMPLETE_OBSERVATION_AUTHORITY_AND_FRESH_SCALAR',unicode_normalizer_unchanged=True,
            alias_or_period_or_site_caps=None,site_fanout_materialized=False)
        write_json(output/'manifest.json',dict(status='SEALED_EXACT_BATCH_VARIANT_PROJECTION_COST_ONLY',launch=record(output/'launch.json'),
            source=cfg['sample_manifest'],family=cfg['family_names'],stats=stats,artifacts={p.name:record(p) for p in output.glob('*.parquet')},
            resources=budget.report(),output_bytes=size,truth_opened=False,crm_opened=False,model_fit=False,recall_measured=False,
            fold0_opened=False,fold1_opened=False,identifiers_as_features=False,downstream_models_changed=False,
            national_corpus=False,complete_retrieval_index=False,ready_for_fold1=False))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v128_variant_batches.md',ROOT/'src/xgb_matcher/retrieval_variant_batches_v128.py',
        ROOT/'src/xgb_matcher/retrieval_variants_v118.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
        ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False,crm_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v128_variant_batches','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-7500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
