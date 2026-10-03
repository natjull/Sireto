"""Identical V81 joins in16 exhaustive disjoint physical buckets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import compile_retrieval_v81_succession_bindings as original
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,write_json

LOGICAL_COMPILE=original.compile_bindings


class PartitionedConnection:
    def __init__(self,connection,output,progress):
        self.c=connection; self.output=Path(output); self.progress=progress; self.intercepted=False

    def execute(self,sql,params=None):
        prefix='CREATE TEMP TABLE bindings AS '
        if not sql.startswith(prefix): return self.c.execute(sql,params or [])
        if params or self.intercepted: raise ValueError('Logical binding materialization drift')
        marker='FROM edges e LEFT JOIN sites p ON p.id=e.predecessor LEFT JOIN sites s ON s.id=e.successor'
        if sql.count(marker)!=1: raise ValueError('Expected exact logical join skeleton absent')
        fields=sql[len(prefix):].split(marker)[0]; files=[]
        for bucket in range(16):
            began=time.monotonic(); path=self.output/f'all_endpoint_observations_part{bucket:02d}.parquet'
            # ID modulo is storage routing ONLY, never a matching feature.
            edges=f'(SELECT * FROM edges WHERE predecessor%16={bucket})'
            predecessor_sites=f'(SELECT * FROM sites WHERE id%16={bucket})'
            successor_sites=f'''(SELECT sites.* FROM sites SEMI JOIN
                (SELECT DISTINCT successor id FROM edges WHERE predecessor%16={bucket}) ids USING(id))'''
            query=fields+f'FROM {edges} e LEFT JOIN {predecessor_sites} p ON p.id=e.predecessor LEFT JOIN {successor_sites} s ON s.id=e.successor'
            self.c.execute('COPY ('+query+') TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(path)])
            files.append(str(path))
            self.progress(dict(physical_bucket=bucket,seconds=time.monotonic()-began))
        paths=','.join("'"+p.replace("'","''")+"'" for p in files)
        self.intercepted=True
        return self.c.execute('CREATE TEMP VIEW bindings AS SELECT * FROM read_parquet(['+paths+'])')


def compile_partitioned(c,successions,sites,output,progress=lambda value:None):
    proxy=PartitionedConnection(c,output,progress)
    stats=LOGICAL_COMPILE(proxy,successions,sites,output,progress)
    if not proxy.intercepted: raise ValueError('Physical partitioning absent')
    return dict(stats,physical_plan='16_DISJOINT_EXHAUSTIVE_BUCKETS_SAME_LOGICAL_ENDPOINT_JOINS')


def worker(output):
    before=original.compile_bindings
    try:
        original.compile_bindings=compile_partitioned; original.worker(output)
    finally:
        original.compile_bindings=before


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(original.CONFIG.read_text())
    paths=(Path(__file__),Path(original.__file__),original.CONFIG,
           ROOT/'docs/retrieval_v81_succession_bindings.md',ROOT/'docs/retrieval_v81_partitioned_plan.md',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},
        physical_plan='16_DISJOINT_EXHAUSTIVE_BUCKETS_NOT_RETRIEVAL_POOL',logical_contract_unchanged=True,truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v81_partitioned','worker','--output',str(output)],
                check=True,timeout=cfg['supervisor_seconds'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-6500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
