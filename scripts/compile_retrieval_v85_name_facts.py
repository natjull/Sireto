"""Compile all scoped name facts and their complete provenance references."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_facts_v85 import factorize,invert
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v85_name_facts.json'


def source(cfg):
    m=json.loads(verify(cfg['source_manifest']).read_text()); launch=json.loads(verify(m['launch']).read_text())
    for r in launch['implementation'].values(): verify(r)
    if (m['status']!='SEALED_FACTORIZED_SCOPED_OFFICIAL_SUCCESSION_NAMES' or m['truth_opened'] or m['crm_opened']
        or m['stats']['name_observations']!=cfg['expected_observations'] or m['stats']['relation_aliases_created']
        or m['stats']['sibling_site_names_transferred']):
        raise ValueError('Full official scoped source required')
    p=verify(m['artifacts']['scoped_name_observations.parquet'])
    if pq.ParquetFile(p).metadata.num_rows!=cfg['expected_observations']:
        raise ValueError('Source observation denominator drift')
    return p


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; c=None
    for r in launch['implementation'].values(): verify(r)
    try:
        with Budget(cfg['resources']) as budget:
            names=source(cfg); c=duckdb.connect(); c.execute('SET threads=2')
            c.execute('SET preserve_insertion_order=false'); c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET temp_directory=?',[str(output/'spill')]); c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            log=lambda value:print(json.dumps(value),flush=True)
            stats=factorize(c,names,output,log); c.close(); c=None
            stats['inverse']=invert(output,stats,log); source(cfg)
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for r in launch['implementation'].values(): verify(r)
        files={str(p.relative_to(output)):record(p) for p in output.rglob('*')
               if p.is_file() and p.suffix in ('.npy','.parquet')}
        write_json(output/'manifest.json',dict(status='SEALED_ALL_SCOPED_NAME_FACTS_AND_OBSERVATIONS',
            launch=record(output/'launch.json'),source_manifest=cfg['source_manifest'],stats=stats,
            artifacts=files,resources=budget.report(),output_bytes=size,
            truth_opened=False,crm_opened=False,recall_measured=False,lexical_index_built=False,
            model_fit=False,ready_for_fold1=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False,
            original_observation_metadata_in_source=True,observation_metadata_not_yet_in_mmap=True))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        if c is not None: c.close()
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_name_facts_v85.py',
           ROOT/'docs/retrieval_v85_name_facts.md',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v85_name_facts','worker','--output',str(output)],
                check=True,timeout=cfg['supervisor_seconds'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-9000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
