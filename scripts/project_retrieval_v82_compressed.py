"""Identical scoped V82 observations, stronger lossless Parquet compression."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import project_retrieval_v82_scoped_succession_names as original
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,write_json

LOGICAL_PROJECT=original.project
COMPRESSION_LEVEL=6


class CompressedConnection:
    def __init__(self,connection):
        self.connection=connection; self.exports=0

    def execute(self,sql,params=None):
        marker='(FORMAT PARQUET,COMPRESSION ZSTD)'
        if sql.startswith('COPY '):
            if sql.count(marker)!=1 or self.exports>=2:
                raise ValueError('V82 logical COPY skeleton drift')
            sql=sql.replace(marker,f'(FORMAT PARQUET,COMPRESSION ZSTD,COMPRESSION_LEVEL {COMPRESSION_LEVEL})')
            self.exports+=1
        return self.connection.execute(sql,params or [])


def project_compressed(c,bindings,names,output,progress=lambda value:None):
    proxy=CompressedConnection(c)
    stats=LOGICAL_PROJECT(proxy,bindings,names,output,progress)
    if proxy.exports!=2: raise ValueError('Both complete V82 exports required')
    return dict(stats,physical_plan='IDENTICAL_SCOPED_OBSERVATIONS_ZSTD_LEVEL_6',compression_level=COMPRESSION_LEVEL)


def worker(output):
    before=original.project
    try:
        original.project=project_compressed; original.worker(output)
    finally:
        original.project=before


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(original.CONFIG.read_text())
    paths=(Path(__file__),Path(original.__file__),original.CONFIG,
           ROOT/'docs/retrieval_v82_scoped_succession_names.md',ROOT/'docs/retrieval_v82_compressed_plan.md',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},
        physical_plan='LOSSLESS_ZSTD_LEVEL_6_NO_OBSERVATION_REDUCTION',logical_contract_unchanged=True,truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.project_retrieval_v82_compressed','worker','--output',str(output)],
                check=True,timeout=cfg['supervisor_seconds'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-8500:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
