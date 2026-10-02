"""V43 geographic recipe with V44 same-score physical lexical pruning.

The original V43 STOP is retained. This preregistered new cost attempt changes
neither geographic metric nor learned weights. First16 cost gate remains4s.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v43_spatial as v43
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_index_pruning_v44 import search
from src.xgb_matcher.retrieval_spatial_v43 import nearest_sites
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

SPATIAL=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/retrieval_v43_spatial/index_20261002_a/manifest.json',
    sha256='4c1cd18c34b8dfe33c172c069d823193d23d15fb1d52ad84e39a3068653a0325')
PARITY=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v44_index_pruning/cost_20261002_a/manifest.json',
    sha256='fd34f72dea307105a02526a8576b5105ef15d3d26d8e50f5413a2eb40e6163e2')


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    geo={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
    models=json.loads(verify(launch['models']).read_text()); rows=[]
    with Budget(launch['resources']) as budget:
        paths=launch['spatial_files']
        for ref in paths.values(): verify(ref)
        with Path(paths['tree.pickle']['path']).open('rb') as handle: tree=pickle.load(handle)
        xy=tree.data; keys=np.load(paths['keys.npy']['path'],mmap_mode='r',allow_pickle=False)
        parents=np.load(paths['parents.npy']['path'],mmap_mode='r',allow_pickle=False)
        if len(xy)!=launch['spatial_sites']: raise ValueError('Spatial count drift')
        backend=TantivyBackend(v38.INDEX)
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('National doccount drift')
        for raw in launch['queries'][start:start+8]:
            g=geo[raw['query_id']]; model=models[str(raw['fold'])]
            if model['heldout_fold']!=raw['fold'] or raw['fold'] in model['training_folds']: raise ValueError('OOF drift')
            t=time.monotonic(); learned,pruning=search(backend,v42.as_query(raw),model['weights']); lexical_seconds=time.monotonic()-t
            t=time.monotonic()
            if g['spatial_crs_compatible']:
                spatial,trace=nearest_sites(tree,xy,keys,parents,[g['ban_x_l93'],g['ban_y_l93']])
                source='SPATIAL_DIRECT'; primary_seconds=time.monotonic()-t
            else:
                spatial=learned; trace={}; source='FROZEN_V42_NO_COMPATIBLE_QUERY_POSITION'; primary_seconds=lexical_seconds
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=spatial,comparator=learned),
                seconds=dict(primary=primary_seconds,comparator=lexical_seconds),primary_source=source,
                ban_resolved=g['ban_resolved'],ban_declared_weak=g['ban_declared_weak'],spatial_trace=trace,pruning_trace=pruning))
    v38.verify_launch(launch)
    write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        sm=verify(SPATIAL); parity=json.loads(verify(PARITY).read_text())
        if parity['status']!='SEALED_EXACT_OUTPUT_PARITY_COST_ONLY' or not parity['exact_output_parity'] or parity['queries']!=16:
            raise ValueError('Physical optimization parity prerequisite missing')
        v43.prepare(output,sm.parent)
        launch=json.loads((output/'launch.json').read_text())
        for p in (Path(__file__),ROOT/'src/xgb_matcher/retrieval_index_pruning_v44.py'):
            launch['implementation'][str(p)]=record(p)
        launch.update(physical_optimization='V44_PROVEN_ZERO_NONLOCAL_SCORE_UPPER_BOUND',
            parity_preflight=PARITY,spatial_authority=SPATIAL,changed_weights=False,changed_spatial_metric=False)
        write_json(output/'launch.json',launch); refs=[]; rows=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v45_spatial_pruned','worker',
                    '--output',str(output),'--start',str(start)],check=True,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(90,max(1,900-(time.monotonic()-began))),
                    env={**os.environ,'RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            path=output/f'batch{start:03d}.json'; batch=json.loads(path.read_text())
            if len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy']: raise ValueError('Scope/resource drift')
            rows.extend(batch['rows']); refs.append(record(path)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                cost=sum(sum(r['seconds'].values()) for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_pair_seconds=cost,passed=cost<=4.,quality_opened=False))
                if cost>4.: raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch)
        if len(rows)!=128 or {r['query_id'] for r in rows}!={r['query_id'] for r in launch['queries']}: raise ValueError('Denominator drift')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,launch=record(output/'launch.json'),
            batches=refs,seconds=time.monotonic()-began,truth_opened=False,model_used=True,full_corpus=False,
            full_prescribed_spatial_corpus=True,full_lexical_corpus=True,full_query_population=False,ready_for_fold1=False,
            spatial_eligible_corpus_not_full_catalog=True,spatial_rows=launch['spatial_sites'],
            spatial_queries=sum(r['primary_source']=='SPATIAL_DIRECT' for r in rows),
            physical_optimization=launch['physical_optimization'],pruned_queries=sum(r['pruning_trace']['pruned'] for r in rows)))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            truth_opened=False,ready_for_fold1=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='run': run(a.output)
    elif a.mode=='evaluate': v43.evaluate(a.output)
    else:
        if a.start not in range(0,128,8): p.error('Invalid start')
        worker(a.output,a.start)
