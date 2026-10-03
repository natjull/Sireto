"""Full-native predicate cost only; no recall, labels or candidate output."""
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
from scripts import retrieval_v46_joint as v46
from scripts.retrieval_v56_exact_fields import index_authority
from scripts.retrieval_v105_exact_intervals import validate_models
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_factored_native_v105 import query
from src.xgb_matcher.retrieval_v20_workbench import ROOT,Budget,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v106_native_cost.json'


def variants(weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(16,) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Cost-only frozen native weight drift')
    lexical=w.copy(); lexical[11:15]=0.
    alias=np.zeros(16); alias[11]=1.
    place=np.zeros(16); place[12]=1.
    return {'LEXICAL11_PLUS_WORDS':lexical,'CURRENT_INITIALISM_ONLY':alias,
            'CURRENT_INITIALISM_AT_PLACE':place}


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    source=json.loads(verify(launch['source_launch']).read_text()); v38.verify_launch(source)
    models=json.loads(verify(source['models']).read_text()); validate_models(models)
    expansions=json.loads(verify(source['expansions']).read_text())
    if expansions['truth_opened'] or expansions['old_candidates_opened']:
        raise ValueError('Query-only cost scope drift')
    if [(r['query_id'],r['fold']) for r in expansions['rows']]!=[(r['query_id'],r['fold']) for r in source['queries']]:
        raise ValueError('Cost expansion alignment drift')
    terms={r['query_id']:r['terms'] for r in expansions['rows']}; results=[]; budget=None
    try:
        with Budget(launch['config']['resources']) as budget:
            backend=TantivyBackend(v38.INDEX)
            if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Native national cardinality drift')
            for i,raw in enumerate(launch['queries']):
                if raw['fold'] not in (2,3,4): raise ValueError('Closed fold in cost worker')
                choices=list(variants(models[str(raw['fold'])]['weights']).items())
                if i%2: choices.reverse()
                for name,weights in choices:
                    before=time.monotonic()
                    parsed=query(backend,v42.as_query(raw),weights,terms[raw['query_id']])
                    build=time.monotonic()-before; before=time.monotonic()
                    hits=backend.searcher.search(parsed,limit=launch['config']['native_limit'],count=False).hits
                    elapsed=time.monotonic()-before
                    row=dict(query_id=raw['query_id'],fold=raw['fold'],variant=name,
                             construction_seconds=build,search_seconds=elapsed,returned_count=len(hits))
                    del hits
                    results.append(row); print(json.dumps(row),flush=True)
        v38.verify_launch(launch); v38.verify_launch(source)
        write_json(output/'metrics.json',dict(rows=results,resources=budget.report(),truth_opened=False,
                                             recall_measured=False,candidates_written=False))
        write_json(output/'manifest.json',dict(status='SEALED_NATIVE_COST_ABLATIONS_ONLY',
            launch=record(output/'launch.json'),metrics=record(output/'metrics.json'),
            no_label_or_sidecar_opened=True,no_retrieval_recipe_selected=True,
            no_candidates_written=True,full_national_native_index=True,full_query_population=False,
            recall_measured=False,truth_opened=False,ready_for_fold1=False,
            cache_state_uncontrolled=True,resources=budget.report()))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),completed_searches=len(results),
            resources=budget.report() if budget is not None else None,truth_opened=False,recall_measured=False))
        raise


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(CONFIG.read_text())
    try:
        source=json.loads(verify(cfg['query_only_launch']).read_text()); v38.verify_launch(source)
        v46.validate_queries(source['queries'])
        if (source['truth_opened'] or source['old_candidates_opened'] or cfg['queries']!=2 or cfg['native_limit']!=100):
            raise ValueError('Cost experiment scope drift')
        launch=dict(**index_authority(),config=cfg,source_launch=cfg['query_only_launch'],
                    queries=source['queries'][:2],truth_opened=False,recall_measured=False)
        for path in (Path(__file__),CONFIG,ROOT/'docs/retrieval_v106_native_cost.md',
                     ROOT/'src/xgb_matcher/retrieval_factored_native_v105.py',
                     ROOT/'src/xgb_matcher/retrieval_initialisms_v67.py',
                     ROOT/'src/xgb_matcher/retrieval_proof_v56.py',
                     ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py'):
            launch['implementation'][str(path)]=record(path)
        write_json(output/'launch.json',launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.profile_retrieval_v106_native_cost','worker','--output',str(output)],
                check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                     'RAYON_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
        print((output/'metrics.json').read_text(),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False,recall_measured=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    (run if args.mode=='run' else worker)(args.output)
