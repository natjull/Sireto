"""Two CPU processes for independent queries, with an aggregate memory guard."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import retrieval_v33_full_replay as v33
from scripts.retrieval_v34_runtime import FastRuntime
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, write_json


def run_parallel(queries,output):
    """Split only the independent legacy stage; all other stages stay serial."""
    jobs=[];paths=[];started=time.monotonic();peak=0;before=Budget.swap()
    try:
        for i in range(2):
            selected=queries[i::2]
            if not selected:continue
            path=output/f'legacy_worker{i}';path.mkdir();paths.append(path)
            write_json(path/'queries.json',[dict(query_id=qid,fold=fold,query=vars(q)) for qid,fold,q in selected])
            log=(path/'worker.log').open('x')
            p=subprocess.Popen([sys.executable,'-m','scripts.retrieval_v34_parallel_legacy','--directory',str(path)],
              cwd=v33.ROOT,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((p,log))
        while any(p.poll() is None for p,_ in jobs):
            processes=[psutil.Process(os.getpid()),*psutil.Process(os.getpid()).children(recursive=True)]
            rss=0
            for p in processes:
                try:rss+=p.memory_info().rss
                except psutil.NoSuchProcess:pass
            peak=max(peak,rss)
            if peak>v33.RESOURCE['max_rss_bytes'] or Budget.swap()>before or time.monotonic()-started>1800:
                raise RuntimeError('STOP_PARALLEL_LEGACY_AGGREGATE_BUDGET')
            if any(p.poll() not in (None,0) for p,_ in jobs):raise RuntimeError('Legacy child failed')
            time.sleep(.25)
        if any(p.returncode!=0 for p,_ in jobs):raise RuntimeError('Legacy child failed')
        reports=[json.loads((p/'manifest.json').read_text()) for p in paths]
        if any(not r['resources']['pass_resource_policy'] for r in reports):raise RuntimeError('Legacy child budget')
        peak=max(peak,psutil.Process().memory_info().rss+sum(r['resources']['peak_rss_bytes'] for r in reports))
        if peak>v33.RESOURCE['max_rss_bytes'] or Budget.swap()>before:raise RuntimeError('STOP_PARALLEL_LEGACY_AGGREGATE_BUDGET')
        table=pq.read_table([p/'raw.parquet' for p in paths])
        # Restore caller query order and original union ranks without changing rows.
        positions={qid:i for i,(qid,_,_) in enumerate(queries)}
        rows=sorted(table.to_pylist(),key=lambda r:(positions[r['query_id']],r['union_rank']))
        pq.write_table(pa.Table.from_pylist(rows,schema=v33.RAW_SCHEMA),output/'raw.parquet',compression='zstd')
        timings=[r for p in paths for r in json.loads((p/'legacy_timing.json').read_text())]
        write_json(output/'legacy_timing.json',sorted(timings,key=lambda r:positions[r['query_id']]))
        write_json(output/'parallel_legacy.json',dict(workers=len(jobs),peak_aggregate_rss_bytes=peak,
          swapout_pages_delta=Budget.swap()-before,wall_seconds=time.monotonic()-started,
          workers_manifests=[record(p/'manifest.json') for p in paths]))
    finally:
        for p,log in jobs:
            if p.poll() is None:p.terminate()
        for p,log in jobs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()


def worker(directory):
    rows=json.loads((directory/'queries.json').read_text())
    queries=[(r['query_id'],r['fold'],RetrievalQuery(**r['query'])) for r in rows]
    runtime=FastRuntime()
    try:
        with Budget(v33.RESOURCE) as budget:runtime.legacy(queries,directory)
    finally:runtime.close()
    # These projections belong to this subprocess, never to a source dataset.
    for name in ('local_scope.parquet','geo_scope.parquet'):(directory/name).unlink()
    write_json(directory/'manifest.json',dict(status='SEALED_LEGACY_WORKER',queries=len(rows),
      resources=budget.report(),truth_columns_opened=False,inputs=record(directory/'queries.json'),raw=record(directory/'raw.parquet')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);worker(p.parse_args().directory)
