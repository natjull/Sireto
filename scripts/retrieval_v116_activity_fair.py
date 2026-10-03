"""Blind national activity/site initial100; separate sealed128 evaluator."""
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
from scripts import retrieval_v49_primary_probe as evaluator
from scripts.retrieval_v56_exact_fields import index_authority
from scripts.probe_retrieval_v114_activity_joint import inputs as joint_inputs
from src.xgb_matcher.retrieval_activity_semantic_v113 import normalized,class_lookup,scores
from src.xgb_matcher.retrieval_activity_decoder_v111 import query_design
from src.xgb_matcher.retrieval_activity_joint_v114 import FEATURES,linear_weights
from src.xgb_matcher.retrieval_activity_fair_v116 import search,OfficialActivityGrades
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import ROOT,Budget,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v116_activity_fair.json'


def trained(cfg):
    m=json.loads(verify(cfg['joint_manifest']).read_text())
    if (m['status']!='SEALED_TRAINING_ONLY_CROSSFIT_ACTIVITY_SITE_PROBE' or m['recall_measured'] or
        m['fold0_opened'] or m['fold1_opened'] or m['output_candidates'] or not m['outer_decoder_parity'] or
        not m['activity_head_training_cross_fitted'] or not m['resources']['pass_resource_policy']):
        raise ValueError('Safe frozen cross-fit initial score required')
    launch=json.loads(verify(m['launch']).read_text())
    for ref in launch['implementation'].values():verify(ref)
    refs=joint_inputs(launch['config'])
    if refs!=launch['inputs']:raise ValueError('Joint initial sources changed')
    models=json.loads(verify(m['models']).read_text())
    if set(models)!={'2','3','4'}:raise ValueError('Serving closed/missing fold')
    for f,model in models.items():
        if (model['heldout_fold']!=int(f) or set(model['training_folds'])!={2,3,4}-{int(f)} or
            model['features']!=list(FEATURES) or model['identifiers_as_features'] or model['raw_activity_without_site_proof'] or
            model['descriptions']!=refs['descriptions'] or model['official_catalog']!=refs['semantic_config']['catalog']):
            raise ValueError('Frozen own-fold official activity model drift')
        linear_weights(model['weights'],1.);verify(model['decoder_parameter'])
    return m,models,refs


def authority():
    a=index_authority()
    for path in (Path(__file__),CONFIG,ROOT/'docs/retrieval_v116_activity_fair.md',
        ROOT/'src/xgb_matcher/retrieval_activity_fair_v116.py',ROOT/'src/xgb_matcher/retrieval_activity_initial_v115.py',ROOT/'src/xgb_matcher/retrieval_activity_joint_v114.py',
        ROOT/'src/xgb_matcher/retrieval_activity_semantic_v113.py',ROOT/'src/xgb_matcher/retrieval_activity_decoder_v111.py',
        ROOT/'src/xgb_matcher/retrieval_ranked_path_v105.py',ROOT/'scripts/probe_retrieval_v114_activity_joint.py',
        ROOT/'scripts/probe_retrieval_v113_activity_semantic.py',ROOT/'scripts/probe_retrieval_v111_activity_decoder.py'):
        a['implementation'][str(path)]=record(path)
    return a


def query_vectors(old,refs):
    encoded=json.loads(verify(old['expansions']).read_text());bundle=json.loads(verify(old['bundle']).read_text())
    if (encoded['truth_opened'] or not encoded['resources']['pass_resource_policy'] or
        [(r['query_id'],r['fold']) for r in encoded['rows']]!=[(r['query_id'],r['fold']) for r in old['queries']] or
        bundle['model_refs']!=refs['semantic_refs']['model_refs'] or bundle['model_path']!=refs['semantic_refs']['model_path'] or
        bundle['max_length']!=refs['semantic_refs']['max_length'] or bundle['encoder_prefix']!='nom'):
        raise ValueError('Serving query-only encoder exact TRAIN compatibility drift')
    for ref in bundle['model_refs'].values():verify(ref)
    vectors=np.load(verify(encoded['vectors']),allow_pickle=False)
    if vectors.shape!=(128,384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-6):
        raise ValueError('Complete query-only representation required')
    return encoded['vectors']


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text());v38.verify_launch(launch);raw=None;rows=[];budget=None
    try:
        with Budget(launch['resources']) as budget:
            _,models,refs=trained(launch['config'])
            if refs!=launch['activity_inputs']:raise ValueError('Official serving alignment drift')
            sr=refs['semantic_refs'];descriptions=normalized(np.load(verify(refs['descriptions']),allow_pickle=False))
            labels=json.loads(verify(sr['labels']).read_text())['rows']
            if len(labels)!=launch['config']['expected_descriptions'] or len(descriptions)!=len(labels):raise ValueError('All official descriptions required')
            mapping=class_lookup(sr['nomenclatures'],sr['activities'],labels)
            keys=np.load(verify(sr['training']['keys']),mmap_mode='r',allow_pickle=False)
            nom=np.load(verify(sr['nomenclature']),mmap_mode='r',allow_pickle=False)
            activity=np.load(verify(sr['original_activity']),mmap_mode='r',allow_pickle=False)
            if len(keys)!=launch['config']['expected_sites']:raise ValueError('Full national activity corpus required')
            if np.any(nom>=mapping.shape[0]) or np.any(activity>=mapping.shape[1]):raise ValueError('Full official codebook alignment drift')
            vectors=np.load(verify(launch['query_vectors']),allow_pickle=False)
            geometry={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
            if set(geometry)!={q['query_id'] for q in launch['queries']}:raise ValueError('All query geometry required')
            tree,geo_keys,parents=v46.load_tree(launch);backend=TantivyBackend(v38.INDEX)
            if backend.searcher.num_docs!=launch['index_documents']:raise ValueError('Whole national native scope drift')
            parameters={f:np.load(verify(m['decoder_parameter']),allow_pickle=False) for f,m in models.items()}
            for i in range(start,start+8):
                raw=launch['queries'][i];model=models[str(raw['fold'])];began=time.monotonic()
                grades=scores(query_design(vectors[i:i+1]),descriptions,parameters[str(raw['fold'])],model['decoder']['family'])[0]
                store=OfficialActivityGrades(keys,nom,activity,mapping,grades)
                result,trace=search(backend,tree,geo_keys,parents,v42.as_query(raw),geometry[raw['query_id']],model['weights'],store,
                    max_seconds=launch['max_query_search_seconds'])
                elapsed=time.monotonic()-began
                if elapsed>launch['max_query_search_seconds']:raise ValueError('STOP_QUERY_COST')
                rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=result),seconds=dict(primary=elapsed),primary_trace=trace))
                print(json.dumps(dict(query_id=raw['query_id'],seconds=elapsed,trace=trace)),flush=True)
        v38.verify_launch(launch);trained(launch['config'])
        write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))
    except BaseException as exc:
        write_json(output/f'worker_stop{start:03d}.json',dict(status='STOP',reason=str(exc),query_id=raw['query_id'] if raw else None,
            certified_queries_completed=len(rows),trace=getattr(exc,'trace',None),resources=budget.report() if budget is not None else None,
            truth_opened=False,recall_measured=False));raise


def run(output):
    output.mkdir(parents=True,exist_ok=False);began=time.monotonic();cfg=json.loads(CONFIG.read_text())
    try:
        m,models,refs=trained(cfg);old=json.loads(verify(cfg['query_only_launch']).read_text());v38.verify_launch(old)
        queries=old['queries'];v46.validate_queries(queries)
        if old['truth_opened'] or old['old_candidates_opened'] or len(queries)!=128:raise ValueError('Safe complete query-only launch required')
        if any(set(model['query_ids'])&{q['query_id'] for q in queries} for model in models.values()):raise ValueError('TRAIN/serving query overlap')
        vectors=query_vectors(old,refs);spatial_cfg=json.loads(v46.CONFIG.read_text());sm=v46.spatial(spatial_cfg)
        gm=json.loads(verify(spatial_cfg['geo_evaluation_generation']).read_text());gl=json.loads(verify(gm['launch']).read_text())
        if gm['status']!='SEALED_INITIAL_TOP100' or gl['queries']!=queries:raise ValueError('Query-only BAN scope drift')
        verify(gl['geo_queries'])
        launch=dict(**authority(),config=cfg,queries=queries,query_sources=old['query_sources'],models=m['models'],model_manifest=cfg['joint_manifest'],
            activity_inputs=refs,query_vectors=vectors,spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],geo_queries=gl['geo_queries'],
            resources=cfg['resources'],max_query_search_seconds=cfg['max_query_search_seconds'],truth_opened=False,old_candidates_opened=False,
            collector_cap=100,fixed_reservoir=None,development_panel_reused=True,geometry_as_feature=True,relation_paths_included=False)
        write_json(output/'launch.json',launch);rows=[];batches=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v116_activity_fair','worker','--output',str(output),'--start',str(start)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))),
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','RAYON_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
            file=output/f'batch{start:03d}.json';batch=json.loads(file.read_text())
            if (len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy'] or
                [(r['query_id'],r['fold']) for r in batch['rows']]!=[(q['query_id'],q['fold']) for q in queries[start:start+8]] or
                any(not r['primary_trace']['certified'] for r in batch['rows'])):raise ValueError('Blind batch/certificate/resource drift')
            rows.extend(batch['rows']);batches.append(record(file));print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                mean=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(mean_seconds=mean,passed=mean<=cfg['primary_mean16_seconds'],truth_opened=False))
                if mean>cfg['primary_mean16_seconds']:raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch);trained(cfg)
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=batches,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,truth_opened=False,model_used=True,old_candidates_opened=False,full_query_population=False,
            full_lexical_corpus=True,full_activity_corpus=True,geometry_as_feature=True,relation_paths_included=False,
            ready_for_fold1=False,metric_family=cfg['metric_family']))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False,recall_measured=False));raise


def evaluate(output):
    prior=evaluator.BASELINE
    try:evaluator.BASELINE=json.loads(CONFIG.read_text())['comparator'];evaluator.evaluate(output)
    finally:evaluator.BASELINE=prior
    write_json(output/'evaluation_scope.json',dict(evaluation=record(output/'evaluation.json'),comparator='V101_AFTER_SEALED128',
        scope='V114_ACTIVITY_SITE30_INITIAL100_NOT_FULL_RELATION_PATHS',old_evaluator_narrative_overridden=True,
        latency_scope='Current loaded-index pilot; V101 historical timing not replayed/paired; no production SLO',full_query_population=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker','evaluate']);p.add_argument('--output',required=True,type=Path);p.add_argument('--start',type=int)
    a=p.parse_args()
    if a.mode=='worker':
        if a.start not in range(0,128,8):p.error('Invalid worker start')
        worker(a.output,a.start)
    else:(run if a.mode=='run' else evaluate)(a.output)
