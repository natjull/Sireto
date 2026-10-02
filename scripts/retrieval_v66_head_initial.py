"""Actual direct national initial100, V65 selected head/safe-word recipe."""
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
from scripts import retrieval_v49_primary_probe as evaluator
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_initialisms_v65 import FEATURES
from src.xgb_matcher.retrieval_sparse_query_v59 import expand
from src.xgb_matcher.retrieval_sparse_flat_v61 import search
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v66_head_initial.json'
BASELINE=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v57_native_proof/pilot_20261002_a/manifest.json',
              sha256='11151f230e9616ee799dffbc47aa588bdbb6685ccebc357c6b326e61d3bd0384')


def reduce_model(m):
    w=np.asarray(m['weights'])
    if m['features']!=list(FEATURES) or m['family']!='HEAD_WORD' or w.shape!=(16,) or not np.isfinite(w).all() or np.any(w<0) or np.any(w[11:15]!=0):
        raise ValueError('Selected recipe is not the declared HEAD_WORD native metric')
    return dict(weights=w[:11].tolist(),alpha=float(w[15]),heldout_fold=m['heldout_fold'],
                training_folds=m['training_folds'],query_ids=m['query_ids'],family=m['family'])


def authority():
    launch=index_authority()
    paths=(Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_sparse_flat_v61.py',
        ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',ROOT/'src/xgb_matcher/retrieval_initialisms_v65.py',
        ROOT/'scripts/retrieval_v49_primary_probe.py',ROOT/'docs/retrieval_v66_head_initial_probe.md')
    launch['implementation'].update({str(p):record(p) for p in paths})
    return launch


def prepare(output,cfg):
    source=json.loads(verify(cfg['training_source']).read_text()); sl=json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
    if source['status']!='SEALED_TRAINING_ONLY_OFFICIAL_INITIALISM_NATIVE_PROBE' or source['recall_measured']:
        raise ValueError('Sealed training-only source required')
    models={f:reduce_model(m) for f,m in json.loads(verify(source['models']).read_text()).items()}
    queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4)
    if len(queries)!=128 or any(set(m['query_ids'])&{q['query_id'] for q in queries} for m in models.values()):
        raise ValueError('Training/evaluation query reuse')
    cached=json.loads(verify(cfg['query_cache_launch']).read_text()); v38.verify_launch(cached)
    encoded=json.loads(verify(cfg['query_cache_encoding']).read_text())
    bundle=json.loads(verify(cached['bundle']).read_text())
    vectors_source=json.loads(verify(sl['config']['vector_source']).read_text())
    vl=json.loads(verify(vectors_source['launch']).read_text()); v38.verify_launch(vl)
    if cached['queries']!=queries or encoded['truth_opened'] or not encoded['resources']['pass_resource_policy']:
        raise ValueError('Query-only cache authority mismatch')
    if bundle['model_refs']!=vl['model_refs'] or bundle['model_path']!=vl['config']['model_path'] or bundle['max_length']!=vl['config']['max_length'] or bundle['encoder_prefix']!='nom':
        raise ValueError('Query encoder/training representation drift')
    if [(r['query_id'],r['fold']) for r in encoded['rows']]!=[(r['query_id'],r['fold']) for r in queries]:
        raise ValueError('Query cache order drift')
    q=np.load(verify(encoded['vectors']),allow_pickle=False)
    if q.shape!=(128,384) or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q,axis=1),1.,atol=1e-6):
        raise ValueError('Query-only encoder shape/norm drift')
    expansions=[]
    for f,m in models.items():
        if int(f)!=m['heldout_fold'] or int(f) in m['training_folds']: raise ValueError('OOF model drift')
        vocab=json.loads(verify(source['vocabularies'][f]).read_text())
        p=np.load(verify(source['projections'][f]),allow_pickle=False)
        if p.shape!=(384,len(vocab)): raise ValueError('Safe word projection shape drift')
        for i,r in enumerate(queries):
            if str(r['fold'])==f:
                expansions.append(dict(query_id=r['query_id'],fold=r['fold'],terms=expand(q[i],p,vocab,r['crm_name'],8)))
    if len(expansions)!=128 or len({r['query_id'] for r in expansions})!=128: raise ValueError('Expansion denominator drift')
    write_json(output/'models.json',models); write_json(output/'expansions.json',expansions)
    return dict(**authority(),queries=queries,query_sources=qrefs,models=record(output/'models.json'),
        expansions=record(output/'expansions.json'),resources=cfg['resources'],training_source=cfg['training_source'],
        query_cache_launch=cfg['query_cache_launch'],query_cache_encoding=cfg['query_cache_encoding'],
        reused_query_vectors=encoded['vectors'],reuse_scope='QUERY_ONLY_ENCODER_NO_OLD_CANDIDATES_OR_OLD_EXPANSION_TERMS',
        max_query_search_seconds=cfg['max_query_search_seconds'],collector_cap=100,fixed_reservoir=None,
        truth_opened=False,old_candidates_opened=False,geometry_as_feature=False,development_panel_reused=True)


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text())
    terms={r['query_id']:r for r in json.loads(verify(launch['expansions']).read_text())}
    budget=Budget(launch['resources']); rows=[]
    try:
        with budget:
            backend=TantivyBackend(v38.INDEX)
            if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Full national corpus drift')
            for raw in launch['queries'][start:start+8]:
                m=models[str(raw['fold'])]
                if m['heldout_fold']!=raw['fold'] or raw['fold'] in m['training_folds']: raise ValueError('Serving OOF drift')
                before=time.monotonic()
                found,trace=search(backend,v42.as_query(raw),m['weights'],terms[raw['query_id']]['terms'],m['alpha'])
                elapsed=time.monotonic()-before
                if elapsed>launch['max_query_search_seconds']: raise RuntimeError('STOP_QUERY_COST')
                rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=found),
                    seconds=dict(primary=elapsed),primary_trace=trace))
        v38.verify_launch(launch)
        write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))
    except BaseException as exc:
        write_json(output/f'worker_stop{start:03d}.json',dict(status='STOP',reason=str(exc),completed=len(rows),
            resources=budget.report() if hasattr(budget,'elapsed') else None,truth_opened=False,recall_measured=False))
        raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        launch=prepare(output,cfg); write_json(output/'launch.json',launch); rows=[]; refs=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v66_head_initial','worker','--output',str(output),'--start',str(start)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,
                    timeout=min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))),
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
            path=output/f'batch{start:03d}.json'; part=json.loads(path.read_text())
            if len(part['rows'])!=8 or not part['resources']['pass_resource_policy']: raise ValueError('Worker scope/resource drift')
            rows.extend(part['rows']); refs.append(record(path)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                mean=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_primary_seconds=mean,passed=mean<=cfg['primary_mean16_seconds'],
                    quality_opened=False,not_production_slo=True))
                if mean>cfg['primary_mean16_seconds']: raise RuntimeError('STOP_PRIMARY_COST_PREFLIGHT')
        v38.verify_launch(launch)
        if len(rows)!=128 or len({r['query_id'] for r in rows})!=128: raise ValueError('Complete serving denominator drift')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=refs,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,truth_opened=False,model_used=True,full_query_population=False,ready_for_fold1=False,
            metric_family=cfg['metric_family'],full_lexical_corpus=True,full_spatial_corpus=False,geometry_as_feature=False,
            old_candidates_opened=False,comparator_replayed=False,query_vectors_reused_not_candidates=True))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            truth_opened=False,old_candidates_opened=False,ready_for_fold1=False)); raise


def evaluate(output):
    # Separate process, same audited all-human evaluator, last complete baseline
    # explicitly fixed here. No mutation of the historical evaluator on disk.
    original=evaluator.BASELINE
    try:
        evaluator.BASELINE=BASELINE; evaluator.evaluate(output)
    finally:
        evaluator.BASELINE=original


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='run': run(a.output)
    elif a.mode=='evaluate': evaluate(a.output)
    else:
        if a.start not in range(0,128,8): p.error('Invalid worker batch')
        worker(a.output,a.start)
