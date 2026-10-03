"""V81 unchanged logical compilation, stream the wide join instead of CTAS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import compile_retrieval_v81_succession_bindings as original
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, write_json

LOGICAL_COMPILE=original.compile_bindings


class StreamingConnection:
    def __init__(self,connection,output):
        self.connection=connection; self.output=Path(output); self.intercepted=False

    def execute(self,sql,params=None):
        prefix='CREATE TEMP TABLE bindings AS '
        if sql.startswith(prefix):
            if params or self.intercepted:
                raise ValueError('Unexpected logical binding materialization')
            path=self.output/'all_eligible_endpoint_observations.parquet'
            self.connection.execute('COPY ('+sql[len(prefix):]+') TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(path)])
            self.intercepted=True
            return self.connection.execute("CREATE TEMP VIEW bindings AS SELECT * FROM read_parquet('"+
                                           str(path).replace("'","''")+"')")
        return self.connection.execute(sql,params or [])


def compile_streaming(c,successions,sites,output,progress=lambda value:None):
    proxy=StreamingConnection(c,output)
    stats=LOGICAL_COMPILE(proxy,successions,sites,output,progress)
    if not proxy.intercepted:
        raise ValueError('Expected wide join interception absent')
    return dict(stats,physical_plan='STREAM_WIDE_JOIN_TO_PARQUET_IDENTICAL_LOGICAL_OBSERVATIONS')


def worker(output):
    # Process-local physical adapter; the historical source file stays intact.
    before=original.compile_bindings
    try:
        original.compile_bindings=compile_streaming
        original.worker(output)
    finally:
        original.compile_bindings=before


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(original.CONFIG.read_text())
    paths=(Path(__file__),Path(original.__file__),original.CONFIG,
           ROOT/'docs/retrieval_v81_succession_bindings.md',ROOT/'docs/retrieval_v81_streaming_plan.md',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},
        physical_plan='STREAM_WIDE_JOIN_TO_PARQUET_NOT_CTAS',logical_contract_unchanged=True,truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v81_streaming','worker','--output',str(output)],
                check=True,timeout=cfg['supervisor_seconds'],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-5000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
