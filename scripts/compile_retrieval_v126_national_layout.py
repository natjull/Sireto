"""Bounded complete official-site physical layout, never a recall run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb
import numpy as np
import pyarrow.parquet as pq

from scripts.probe_retrieval_v118_official_variants import fingerprint
from src.xgb_matcher.retrieval_coupled_source_v121 import source_coordinate
from src.xgb_matcher.retrieval_national_layout_v126 import read_coordinates,layout
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v126_national_layout.json'


def membership(cfg):
    m=json.loads(verify(cfg['membership_manifest']).read_text())
    if (m['status']!='SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP' or m['truth_opened'] or m['crm_opened']
        or m['stats']['sites']!=cfg['expected_sites'] or m['stats']['closed_sites_removed']):
        raise ValueError('Every official site authority required')
    for name in ('endpoint_keys.npy','endpoint_states.npy'):verify(m['artifacts'][name])
    return m


def sample_oracle(cfg,keys,xy):
    m=json.loads(verify(cfg['sample_manifest']).read_text())
    if m['truth_opened'] or m['crm_opened']:raise ValueError('Official source-only oracle required')
    rows=pq.read_table(verify(m['artifacts']['sample_current_site.parquet']),use_threads=False).to_pylist()
    for row in rows:
        key=np.uint64(int(row['siret']));pos=int(np.searchsorted(keys,key));expected=source_coordinate(row)
        if pos==len(keys) or keys[pos]!=key:raise ValueError('Source sample lost from national site denominator')
        value=np.array(expected) if expected is not None else np.array([np.nan,np.nan])
        if not np.array_equal(xy[pos],value,equal_nan=True):raise ValueError('National/scalar source coordinate drift')
    return dict(source_sample_rows=len(rows),all_source_sample_coordinates_exact=True,selected_by_crm_or_gt=False)


def worker(output):
    launch=json.loads((output/'launch.json').read_text());cfg=launch['config'];budget=None;c=None
    for ref in launch['implementation'].values():verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            m=membership(cfg);keys=np.load(verify(m['artifacts']['endpoint_keys.npy']),mmap_mode='r',allow_pickle=False)
            source=Path(cfg['current_source']['path']);before=fingerprint(source)
            if pq.ParquetFile(source).metadata.num_rows!=cfg['expected_sites']:raise ValueError('Full current source denominator drift')
            c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET preserve_insertion_order=false');c.execute("SET max_temp_directory_size='0MB'")
            progress=lambda row:print(json.dumps(row),flush=True)
            xy=read_coordinates(c,source,keys,output,progress);c.close();c=None
            stats=layout(xy,output,cfg['block_size'],progress)
            oracle=sample_oracle(cfg,keys,xy)
            states=np.load(verify(m['artifacts']['endpoint_states.npy']),mmap_mode='r',allow_pickle=False)
            counts=np.bincount(states,minlength=4);state_counts={str(i):int(n) for i,n in enumerate(counts)}
            if sum(state_counts.values())!=len(keys):raise ValueError('National state coverage drift')
            if fingerprint(source)!=before:raise ValueError('Official raw source changed during layout')
            membership(cfg)
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']:raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values():verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_ALL_NATIONAL_SITE_PHYSICAL_LAYOUT_ONLY',
            launch=record(output/'launch.json'),membership=cfg['membership_manifest'],source=cfg['current_source'],
            source_fingerprint=before,whole_raw_source_hash_verified=False,stats=stats,source_coordinate_oracle=oracle,
            state_counts_v96=state_counts,artifacts={p.name:record(p) for p in output.glob('*.npy')},
            resources=budget.report(),output_bytes=size,truth_opened=False,crm_opened=False,model_fit=False,
            recall_measured=False,fold0_opened=False,fold1_opened=False,identifiers_as_features=False,
            complete_site_denominator=True,complete_retrieval_index=False,downstream_models_changed=False,ready_for_fold1=False))
        print(json.dumps(dict(stats=stats,source_coordinate_oracle=oracle,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        if c is not None:c.close()
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);cfg=json.loads(CONFIG.read_text())
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v126_national_layout.md',ROOT/'src/xgb_matcher/retrieval_national_layout_v126.py',
        ROOT/'src/xgb_matcher/retrieval_coupled_source_v121.py',ROOT/'scripts/probe_retrieval_v118_official_variants.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False,crm_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v126_national_layout','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-7000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();(run if a.mode=='run' else worker)(a.output)
