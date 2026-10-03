"""Compile exhaustive official one-hop access for the initial path collector."""
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

from src.xgb_matcher.retrieval_succession_csr_v84 import ARRAYS,build
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v84_succession_csr.json'


def sources(cfg):
    b=json.loads(verify(cfg['bindings_manifest']).read_text()); n=json.loads(verify(cfg['names_manifest']).read_text())
    if (b['status']!='SEALED_NATIONAL_OFFICIAL_SUCCESSION_SITE_BINDINGS' or b['truth_opened'] or b['crm_opened']
        or n['status']!='SEALED_FACTORIZED_SCOPED_OFFICIAL_SUCCESSION_NAMES' or n['truth_opened'] or n['crm_opened']
        or n['bindings_source']!=cfg['bindings_manifest'] or n['stats']['relation_aliases_created']):
        raise ValueError('Complete official scoped source required')
    for m in (b,n):
        launch=json.loads(verify(m['launch']).read_text())
        for r in launch['implementation'].values(): verify(r)
    return verify(b['artifacts']['succession_bindings.parquet']),verify(n['artifacts']['endpoint_scope.parquet'])


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']
    for r in launch['implementation'].values(): verify(r)
    try:
        with Budget(cfg['resources']) as budget:
            bindings,endpoints=sources(cfg)
            stats=build(bindings,output,lambda value:print(json.dumps(value),flush=True))
            if (stats['observations']!=cfg['expected_observations'] or stats['endpoint_sites']!=cfg['expected_endpoint_sites']
                or stats['legal_owners']!=cfg['expected_legal_owners']):
                raise ValueError('Exhaustive official CSR denominator drift')
            # Independent equality of ALL endpoint pairs with the scoped
            # name source, not a sampled check or derived parent prefix.
            table=pq.read_table(endpoints,columns=['siret','siren'],use_threads=False)
            keys=pc.cast(table['siret'],pa.uint64()).to_numpy(zero_copy_only=False)
            parents=pc.cast(table['siren'],pa.uint32()).to_numpy(zero_copy_only=False)
            order=np.argsort(keys,kind='stable')
            if (not np.array_equal(keys[order],np.load(output/'endpoint_keys.npy',mmap_mode='r'))
                or not np.array_equal(parents[order],np.load(output/'endpoint_parents.npy',mmap_mode='r'))):
                raise ValueError('CSR endpoint/name-scope equality drift')
            del table,keys,parents,order
            sources(cfg)
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for r in launch['implementation'].values(): verify(r)
        write_json(output/'manifest.json',dict(status='SEALED_EXHAUSTIVE_SUCCESSION_CSR',launch=record(output/'launch.json'),
            bindings_source=cfg['bindings_manifest'],names_source=cfg['names_manifest'],stats=stats,
            artifacts={k:record(output/(k+'.npy')) for k in ARRAYS},resources=budget.report(),output_bytes=size,
            every_endpoint_compared_to_scoped_name_source=True,truth_opened=False,crm_opened=False,
            recall_measured=False,model_fit=False,lexical_index_built=False,ready_for_fold1=False,
            fold0_opened=False,fold1_opened=False,downstream_models_changed=False,alias_transfer=False))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_succession_csr_v84.py',
           ROOT/'docs/retrieval_v84_succession_csr.md',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v84_succession_csr','worker','--output',str(output)],
                check=True,timeout=cfg['supervisor_seconds'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-7500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
