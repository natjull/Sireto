"""Blind cost-only native geography feasibility; no V63 quality claim."""
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
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_geo_postings_v64 import radial_memberships, query
from src.xgb_matcher.retrieval_sparse_flat_v61 import query as lexical_query
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

SOURCE = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v63_joint_contrastive/probe_20261002_a/manifest.json',
              sha256='3ebb92942d42e7d49fec61da6c6a44ef02c7435f2d64cc619cc017c3c5a5e658')
GEO = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v45_spatial_pruned/pilot_20261002_a/manifest.json',
           sha256='6960676ddb2c092d88788601d69288d33370c68bacebe01b6a671844f485f582')
SPATIAL = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/retrieval_v43_spatial/index_20261002_a/manifest.json',
               sha256='4c1cd18c34b8dfe33c172c069d823193d23d15fb1d52ad84e39a3068653a0325')
RESOURCES = dict(max_seconds=80,max_rss_bytes=6442450944,max_swapout_pages_delta=0)


def worker(output):
    launch = json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    model = json.loads(verify(launch['models']).read_text())
    geos = {r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
    budget = Budget(RESOURCES); timings = []
    try:
        with budget:
            for ref in launch['spatial_files'].values(): verify(ref)
            with verify(launch['spatial_files']['tree.pickle']).open('rb') as handle: tree = pickle.load(handle)
            keys = np.load(verify(launch['spatial_files']['keys.npy']),mmap_mode='r',allow_pickle=False)
            if tree.n != launch['spatial_sites']: raise ValueError('Full official spatial count drift')
            backend = TantivyBackend(v38.INDEX)
            if backend.searcher.num_docs != launch['index_documents']: raise ValueError('Full national count drift')
            for i, raw in enumerate(launch['queries']):
                m = model[str(raw['fold'])]
                if m['family'] != 'JOINT_STATIC' or m['heldout_fold']!=raw['fold'] or raw['fold'] in m['training_folds']:
                    raise ValueError('Source model/OOF drift')
                w = m['weights'][:19]; q = v42.as_query(raw); measured = {}
                # Alternate order; cache state is still uncontrolled, not a SLO.
                for mode in (('lexical','geo') if i%2==0 else ('geo','lexical')):
                    before = time.monotonic()
                    if mode=='geo':
                        bands, trace = radial_memberships(tree,keys,geos[raw['query_id']])
                        parsed = query(backend,q,w,bands,trace['reliable'])
                    else:
                        parsed = lexical_query(backend,q,w[:11],[],0.)
                    compiled = time.monotonic()
                    hits = backend.searcher.search(parsed,limit=100,count=False).hits
                    measured[mode] = dict(compile_seconds=compiled-before,
                        search_seconds=time.monotonic()-compiled,total_seconds=time.monotonic()-before,
                        returned_count=len(hits))
                    if len(hits)>100: raise ValueError('Native cap drift')
                    # IDs and scores are deliberately NOT persisted or evaluated.
                timings.append(dict(query_id=raw['query_id'],fold=raw['fold'],times=measured,geometry=trace))
                print(json.dumps(timings[-1]),flush=True)
        v38.verify_launch(launch)
        write_json(output/'manifest.json',dict(status='SEALED_BLIND_NATIVE_GEO_COST_ONLY',rows=timings,
            launch=record(output/'launch.json'),resources=budget.report(),
            recall_measured=False,no_labels_opened=True,no_candidates_written=True,
            metric_not_equivalent_to_v63=True,semantic_vectors_not_served=True,
            full_lexical_corpus=True,full_compatible_spatial_corpus=True,quality_recipe_trained=False,
            no_fixed_pool=True,initial_collector_cap=100,ready_for_fold1=False,fold0_opened=False,fold1_opened=False))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),completed_cost_rows=timings,
            resources=budget.report() if hasattr(budget,'elapsed') else None,recall_measured=False,no_labels_opened=True))
        raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        source=json.loads(verify(SOURCE).read_text()); geo=json.loads(verify(GEO).read_text())
        gl=json.loads(verify(geo['launch']).read_text()); sm=json.loads(verify(SPATIAL).read_text())
        if source['status']!='SEALED_TRAINING_ONLY_JOINT_CONTRASTIVE_PROBE' or geo['status']!='SEALED_INITIAL_TOP100':
            raise ValueError('Sealed sources required')
        if sm['truth_opened'] or sm['crm_opened']: raise ValueError('Official query-independent coordinates required')
        queries=gl['queries'][:8]
        if len(queries)!=8 or any(set(q)!=v38.QUERY_FIELDS or q['fold'] not in (2,3,4) for q in queries):
            raise ValueError('Blind prefix scope drift')
        launch=dict(**index_authority(),queries=queries,models=source['models'],geo_queries=gl['geo_queries'],
            spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],source=SOURCE,
            geo_source=GEO,spatial_source=SPATIAL,resources=RESOURCES,truth_scope='NONE_COST_ONLY')
        for p in (Path(__file__),ROOT/'src/xgb_matcher/retrieval_geo_postings_v64.py',
                  ROOT/'src/xgb_matcher/retrieval_sparse_flat_v61.py',ROOT/'docs/retrieval_v64_geo_postings_cost.md'):
            launch['implementation'][str(p)]=record(p)
        write_json(output/'launch.json',launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.profile_retrieval_v64_geo_postings','worker','--output',str(output)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-8000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            no_labels_opened=True,recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
