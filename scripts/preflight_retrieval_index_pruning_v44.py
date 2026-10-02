"""Sixteen-query label-blind output parity and physical cost probe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_index_pruning_v44 import search
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

SOURCE=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v42_indexed_metric/pilot_20261002_a/manifest.json',
    sha256='e9ffdfea70359f0945ffe4d7a76ea0c680d04f7e9c51204c9ef501000ced4b27')
MODELS=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v42_indexed_metric/training_20261002_a/models.json',
    sha256='0857206f4881e1cffa40af4eea8dd573f438941c1f80baa94b30064a33b7d32a')
RESOURCE=dict(max_seconds=80,max_rss_bytes=6*1024**3,max_swapout_pages_delta=0)


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(MODELS).read_text()); rows=[]
    with Budget(RESOURCE) as budget:
        b=TantivyBackend(v38.INDEX)
        if b.searcher.num_docs!=launch['index_documents']: raise ValueError('Index count drift')
        for i,raw in enumerate(launch['queries'][start:start+8],start):
            q=v42.as_query(raw); model=models[str(raw['fold'])]
            if raw['fold']!=model['heldout_fold'] or raw['fold'] in model['training_folds']:
                raise ValueError('OOF drift')
            results={}; seconds={}; trace={}
            for mode in (('original','optimized') if i%2==0 else ('optimized','original')):
                t=time.monotonic()
                if mode=='original': results[mode]=v42.search(b,q,model['weights'])
                else: results[mode],trace=search(b,q,model['weights'])
                seconds[mode]=time.monotonic()-t
            if results['original']!=results['optimized']: raise ValueError('STOP_EXACT_NATIVE_OUTPUT_PARITY')
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=results,seconds=seconds,trace=trace,exact_output_parity=True))
    v38.verify_launch(launch)
    write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        base=json.loads(verify(SOURCE).read_text())
        if base['status']!='SEALED_INITIAL_TOP100' or base['queries']!=128: raise ValueError('Source drift')
        source_launch=json.loads(verify(base['launch']).read_text())
        queries=source_launch['queries'][:16]
        if len(queries)!=16 or len({r['query_id'] for r in queries})!=16 or any(
            set(r)!=v38.QUERY_FIELDS or r['fold'] not in (2,3,4) for r in queries): raise ValueError('Query scope drift')
        paths=[Path(__file__),ROOT/'src/xgb_matcher/retrieval_index_pruning_v44.py',Path(v42.__file__),
            Path(v38.__file__),ROOT/'src/xgb_matcher/hierarchical_retrieval.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py']
        im=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
        launch=dict(queries=queries,source=SOURCE,source_launch=base['launch'],models=MODELS,
            implementation={str(p):record(p) for p in paths},index_manifest=record(v38.INDEX/'manifest.json'),
            index_meta=record(v38.INDEX/'tantivy/meta.json'),inventory=v38.inventory(),index_documents=im['num_documents'],
            initial_cap=100,truth_opened=False,global_and_local_weight_changes=False,fixed_reservoir=None)
        write_json(output/'launch.json',launch); refs=[]; rows=[]
        for start in (0,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.preflight_retrieval_index_pruning_v44','worker',
                    '--output',str(output),'--start',str(start)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90,
                    env={**os.environ,'RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            path=output/f'batch{start:03d}.json'; batch=json.loads(path.read_text())
            if len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy']: raise ValueError('Scope/resource drift')
            refs.append(record(path)); rows.extend(batch['rows']); print(json.dumps(dict(completed=len(rows))),flush=True)
        v38.verify_launch(launch)
        timing={mode:dict(zip(('p50','p95','p99'),np.quantile([r['seconds'][mode] for r in rows],[.5,.95,.99]).tolist()))
                for mode in ('original','optimized')}
        result=dict(status='SEALED_EXACT_OUTPUT_PARITY_COST_ONLY',queries=16,batches=refs,launch=record(output/'launch.json'),
            exact_output_parity=all(r['exact_output_parity'] for r in rows),pruned_queries=sum(r['trace']['pruned'] for r in rows),
            latency=timing,seconds=time.monotonic()-began,quality_measured=False,truth_opened=False,
            scope='Same-score physical query plan only, uncontrolled cache, no recall or production SLO certification',
            ready_for_fold1=False)
        write_json(output/'manifest.json',result); print(json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            ready_for_fold1=False,seconds=time.monotonic()-began)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='run': run(a.output)
    else:
        if a.start not in (0,8): p.error('Invalid worker start')
        worker(a.output,a.start)
