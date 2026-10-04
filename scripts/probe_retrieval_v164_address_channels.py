"""Exact one-part lexical cost probe; not a full index or recall certificate."""
import argparse
import json
import os
from pathlib import Path
import sys

import duckdb
import pyarrow.parquet as pq

import scripts.compile_retrieval_v163_address_access as original
from src.xgb_matcher.retrieval_address_channels_v164 import stage_addresses
from src.xgb_matcher.retrieval_address_authority_v163 import parent_exit
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, quiet, validate_report
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

OUTPUT_ROOT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v164_address_channels')


def source(reference,index):
    path=verify(reference)
    if path.name!='launch.json':raise ValueError('Original complete source audit launch required')
    l,p=original.load(path.parent)
    if type(index) is not int or not 0<=index<len(p['parts']):raise ValueError('Actual original physical part required')
    return l,p,p['parts'][index]


def worker(output):
    l=json.loads((output/'launch.json').read_text())
    if (l['resources']!=LIMITS or l['writer_group']!=8 or not l['only_single_original_physical_part']
        or l['complete_retrieval_index'] or any(l.get(k) is not False for k in PROTECTED)):
        raise ValueError('Protected one-part resource-only launch required')
    for ref in l['implementation'].values():verify(ref)
    budget=c=None
    try:
        with Budget(LIMITS) as budget:
            old,p,spec=source(l['source_launch'],l['index']);cfg=old['config']
            if spec!=l['part']:raise ValueError('Same entire original physical slice required')
            table=pq.ParquetFile(verify(spec['source'])).read_row_group(spec['row_group'],use_threads=False)
            table=table.slice(spec['group_slice_start'],spec['group_slice_stop']-spec['group_slice_start'])
            if len(table)!=spec['row_stop']-spec['row_start']:raise ValueError('Every original alias required')
            c=duckdb.connect();c.execute('SET threads=1');c.execute('SET memory_limit=?',[cfg['duckdb_memory']])
            c.execute('SET temp_directory=?',[str(output/'spill')]);c.execute('SET max_temp_directory_size=?',[cfg['spill_limit']])
            stats=stage_addresses(c,table,output,spec['alias_start'],spec['row_start'],cfg['term_buckets'],8)
            verify(spec['source']);files=[Path(s) for s in stats.pop('files')]+[output/'features.parquet']
            artifacts=[record(f) for f in files]
            c.close();c=None
            _,_,after=source(l['source_launch'],l['index'])
            if after!=spec:raise ValueError('Unchanged complete source plan required')
        validate_report(budget.report())
        write_json(output/'manifest.json',dict(status='SEALED_V164_SINGLE_ORIGINAL_ADDRESS_PART_RESOURCE_PROBE',
            launch=record(output/'launch.json'),source_launch=l['source_launch'],part=spec,stats=stats,artifacts=artifacts,
            resources=budget.report(),complete_retrieval_index=False,ready_for_fold1=False,**{k:False for k in PROTECTED}))
        print(json.dumps(dict(stats=stats,resources=budget.report())),flush=True)
    except BaseException as exc:
        if c is not None:c.close()
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None,
            **{k:False for k in PROTECTED}));raise


def run(output,reference,index):
    if output.parent.resolve()!=OUTPUT_ROOT.resolve() or output.exists():
        raise ValueError('Fresh immutable CATNAT resource-probe root required')
    old,_,spec=source(reference,index)
    gate=quiet(30,50,55);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'preflight.json',gate)
    paths=[Path(__file__),ROOT/'src/xgb_matcher/retrieval_address_channels_v164.py',
        ROOT/'src/xgb_matcher/retrieval_supervisor_v164.py']
    write_json(output/'launch.json',dict(source_launch=reference,index=index,part=spec,writer_group=8,resources=LIMITS,
        implementation={**old['implementation'],**{str(p):record(p) for p in paths}},
        only_single_original_physical_part=True,complete_retrieval_index=False,**{k:False for k in PROTECTED}))
    try:
        with (output/'worker.log').open('x') as log:
            report=supervise([sys.executable,'-u','-m','scripts.probe_retrieval_v164_address_channels',
                'worker','--output',str(output)],log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                output/'supervisor_stop.json')
        write_json(output/'supervision.json',report);parent_exit(output)
        m=json.loads((output/'manifest.json').read_text());validate_report(m['resources'])
        print(json.dumps(dict(manifest=record(output/'manifest.json'),resources=m['resources'],supervision=report)),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),**{k:False for k in PROTECTED}));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run','worker'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--source-launch',type=Path)
    parser.add_argument('--sha256');parser.add_argument('--index',type=int,default=0);a=parser.parse_args()
    if a.mode=='worker':worker(a.output)
    else:
        if a.source_launch is None or a.sha256 is None:parser.error('Original source audit launch and exact SHA required')
        run(a.output,dict(path=str(a.source_launch.resolve()),sha256=a.sha256),a.index)
