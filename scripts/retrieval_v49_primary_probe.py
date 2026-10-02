"""Same frozen joint score, primary-only cost probe, separately sealed baseline.

No old candidates or labels enter generation. The evaluator alone opens the
previous sealed V42 initial outputs AFTER all128 new outputs are sealed.
Primary mean<=4s on16 remains a cost gate, NOT the production SLO.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v46_joint as v46
from scripts.retrieval_v47_joint_fair import TRAINED
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_joint_v46 import FEATURES
from src.xgb_matcher.retrieval_joint_v48 import search
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

BASELINE=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v42_indexed_metric/pilot_20261002_a/manifest.json',
    sha256='e9ffdfea70359f0945ffe4d7a76ea0c680d04f7e9c51204c9ef501000ced4b27')


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text())
    geo={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}; rows=[]
    with Budget(launch['resources']) as budget:
        tree,keys,parents=v46.load_tree(launch); backend=TantivyBackend(v38.INDEX)
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('National count drift')
        for raw in launch['queries'][start:start+8]:
            model=models[str(raw['fold'])]
            if model['features']!=list(FEATURES) or model['heldout_fold']!=raw['fold'] or raw['fold'] in model['training_folds']:
                raise ValueError('OOF metric drift')
            t=time.monotonic(); result,trace=search(backend,tree,keys,parents,v42.as_query(raw),geo[raw['query_id']],
                model['weights'],max_seconds=launch['max_query_search_seconds'])
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=result),
                seconds=dict(primary=time.monotonic()-t),primary_trace=trace))
    v38.verify_launch(launch); write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(v46.CONFIG.read_text())
    try:
        trained=json.loads(verify(TRAINED).read_text())
        if trained['status']!='SEALED_OOF_JOINT_INITIAL_METRICS': raise ValueError('Unsealed model')
        v38.verify_launch(trained)
        queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4); v46.validate_queries(queries)
        gm=json.loads(verify(cfg['geo_evaluation_generation']).read_text()); gl=json.loads(verify(gm['launch']).read_text())
        if gm['status']!='SEALED_INITIAL_TOP100' or queries!=gl['queries']: raise ValueError('Query geo authority drift')
        models=json.loads(verify(trained['models']).read_text())
        if any(set(m['query_ids'])&{r['query_id'] for r in queries} for m in models.values()): raise ValueError('Training query reuse')
        sm=v46.spatial(cfg); verify(gl['geo_queries']); launch=v46.index_launch()
        for p in (Path(__file__),ROOT/'scripts/retrieval_v47_joint_fair.py',
                  ROOT/'src/xgb_matcher/retrieval_joint_v47.py',ROOT/'src/xgb_matcher/retrieval_joint_v48.py'):
            launch['implementation'][str(p)]=record(p)
        launch.update(queries=queries,query_sources=qrefs,models=trained['models'],model_manifest=TRAINED,
            spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],spatial_manifest=cfg['spatial_manifest'],
            geo_queries=gl['geo_queries'],query_geo_generation=cfg['geo_evaluation_generation'],
            resources=cfg['resources_worker'],max_query_search_seconds=cfg['max_query_search_seconds'],
            collector_cap=100,fixed_reservoir=None,truth_opened=False,development_panel_reused=True,
            physical_plan='V48_UNCHANGED_PRIMARY_ONLY_NO_COMPARATOR_READ',model_parameters_changed=False,
            comparator_scope='SEPARATE_EVALUATOR_READS_SEALED_V42_AFTER128',cost_gate='PRIMARY_MEAN16_LE4S')
        write_json(output/'launch.json',launch); rows=[]; refs=[]
        for start in range(0,128,8):
            with (output/f'worker{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v49_primary_probe','worker','--output',str(output),'--start',str(start)],
                    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,max(1,900-(time.monotonic()-began))),
                    env={**os.environ,'RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
            path=output/f'batch{start:03d}.json'; batch=json.loads(path.read_text())
            if len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy']: raise ValueError('Scope/resource drift')
            rows.extend(batch['rows']); refs.append(record(path)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                cost=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_primary_seconds=cost,passed=cost<=4.,quality_opened=False,
                    comparator_replayed=False,not_a_production_slo=True))
                if cost>4.: raise ValueError('STOP_PRIMARY_COST_PREFLIGHT')
        v38.verify_launch(launch)
        if len(rows)!=128 or {r['query_id'] for r in rows}!={r['query_id'] for r in queries}: raise ValueError('Serving denominator drift')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=refs,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,model_used=True,truth_opened=False,full_lexical_corpus=True,full_spatial_corpus=False,
            full_query_population=False,ready_for_fold1=False,certified_initial_score=True,metric_family=cfg['metric_family'],
            comparator_replayed=False,old_candidates_opened=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
            truth_opened=False,old_candidates_opened=False,ready_for_fold1=False)); raise


def evaluate(output):
    dest=output/'evaluation.json'
    if dest.exists(): raise ValueError('Do not overwrite evaluation')
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']: raise ValueError('Complete sealed generation required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    rows=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids={r['query_id'] for r in launch['queries']}
    if len(rows)!=128 or {r['query_id'] for r in rows}!=ids or not all(r['primary_trace']['certified'] for r in rows): raise ValueError('Uncertified/incomplete generation')
    old=json.loads(verify(BASELINE).read_text()); ol=json.loads(verify(old['launch']).read_text())
    if old['status']!='SEALED_INITIAL_TOP100' or old['queries']!=128 or old['truth_opened'] or launch['queries']!=ol['queries']:
        raise ValueError('Sealed comparator population drift')
    if launch['index_manifest']!=ol['index_manifest'] or launch['index_meta']!=ol['index_meta'] or launch['inventory']!=ol['inventory']:
        raise ValueError('Comparator corpus changed')
    previous=[r for ref in old['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    reference={r['query_id']:r for r in previous}
    if len(previous)!=128 or set(reference)!=ids: raise ValueError('Comparator output coverage drift')
    ev=v38.load(v38.EVAL,v38.EVAL_SHA); side=json.loads(verify(ev['label_side']).read_text())
    for ref in side['files'].values(): verify(ref)
    labels={r['query_id']:r for r in pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    catalog={r['query_id']:r for r in pq.read_table(verify(ev['files']['official_catalog_sidecar.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    acceptable={}
    for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids: acceptable.setdefault(r['query_id'],set()).add(r['siret'])
    if set(labels)!=ids or set(catalog)!=ids or set(acceptable)!=ids: raise ValueError('Incomplete official label side')
    outcomes=[]
    for r in rows:
        label=labels[r['query_id']]
        if label['fold']!=r['fold'] or r['fold'] not in (2,3,4) or reference[r['query_id']]['fold']!=r['fold']: raise ValueError('Evaluation fold drift')
        outcome=dict(query_id=r['query_id'],**{c:label[c] for c in v38.SEGMENTS})
        for mode in v38.MODES:
            candidates=r['results']['primary'] if mode=='primary' else reference[r['query_id']]['results']['primary']
            found={n['siret'] for n in candidates}
            if len(found)!=len(candidates) or len(found)>100: raise ValueError('Initial cap drift')
            hit=label['human_ground_truth_siret'] in found
            outcome[mode+'_exact']=hit; outcome[mode+'_operational']=bool(found&acceptable[r['query_id']])
            outcome[mode+'_loss']='HIT' if hit else 'NOT_RETRIEVED_AT100' if catalog[r['query_id']]['official_catalog_present'] else 'OFFICIAL_CATALOG_ABSENT'
        outcomes.append(outcome)
    segments={}
    for column in v38.SEGMENTS:
        groups={}
        for r in outcomes:
            value=r[column]; groups.setdefault('UNKNOWN' if value is None or value=='' else str(value),[]).append(r)
        segments[column]={key:{mode:v38.summarize(group,mode) for mode in v38.MODES} for key,group in groups.items()}
    result=dict(status='MEASURED_LEARNED_JOINT_INITIAL_TOP100_PILOT',decision='PIVOT_GENERATION',
        primary=v38.summarize(outcomes,'primary'),comparator=v38.summarize(outcomes,'comparator'),segments=segments,
        gains=sum(r['primary_exact'] and not r['comparator_exact'] for r in outcomes),losses=sum(r['comparator_exact'] and not r['primary_exact'] for r in outcomes),
        latency_primary_current=dict(zip(('p50','p95','p99'),np.quantile([r['seconds']['primary'] for r in rows],[.5,.95,.99]).tolist())),
        latency_comparator_historical=dict(zip(('p50','p95','p99'),np.quantile([r['seconds']['primary'] for r in previous],[.5,.95,.99]).tolist())),
        latency_scope='Primary measured now, loaded indexes/no BAN; comparator from earlier V42 generation NOT replayed or paired timing; neither validates production SLO',
        loss_attribution={mode:dict(Counter(r[mode+'_loss'] for r in outcomes)) for mode in v38.MODES},
        traversal_quantiles=dict(zip(('p50','p95','p99'),np.quantile([r['primary_trace']['unique_documents_visited'] for r in rows],[.5,.95,.99]).tolist())),
        outcomes=outcomes,generation=record(output/'manifest.json'),baseline=BASELINE,ledger=record(v38.LEDGER),label_side=ev['label_side'],
        excluded_missing_human_labels=ev['excluded_missing_human_labels'],fine_loss_attribution_measured=False,
        model_used=True,full_query_population=False,ready_for_fold1=False,fold0_opened=False,fold1_opened=False,
        metric_family=manifest['metric_family'],implementation=record(Path(__file__)))
    if result['primary']['exact_recall']>=.992:
        result['decision']='STOP'; result['decision_reason']='PILOT_ONLY_NO_FULL_OOF_GENERALIZATION_SLO_GATE'
    write_json(dest,result); print(json.dumps({k:result[k] for k in ('primary','comparator','gains','losses','latency_primary_current','traversal_quantiles','decision')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='run': run(a.output)
    elif a.mode=='evaluate': evaluate(a.output)
    else:
        if a.start not in range(0,128,8): p.error('Invalid start')
        worker(a.output,a.start)
