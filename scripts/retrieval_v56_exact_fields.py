"""Train official proof primitives; then measure true initial top100 separately."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts.retrieval_v49_primary_probe import evaluate
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_proof_v56 import FEATURES,point_features,geo_features,fit,search
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v56_proof_metric.json'


def impl():
    paths=[Path(__file__),CONFIG,Path(v38.__file__),Path(v42.__file__),ROOT/'scripts/retrieval_v49_primary_probe.py',
        ROOT/'src/xgb_matcher/retrieval_proof_v56.py',ROOT/'src/xgb_matcher/retrieval_joint_v46.py',ROOT/'src/xgb_matcher/retrieval_joint_v47.py',
        ROOT/'src/xgb_matcher/retrieval_spatial_v43.py',ROOT/'src/xgb_matcher/hierarchical_retrieval.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
        ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    return {str(p):record(p) for p in paths}


def index_authority():
    index=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
    return dict(index_manifest=record(v38.INDEX/'manifest.json'),index_meta=record(v38.INDEX/'tantivy/meta.json'),
        index_documents=index['num_documents'],inventory=v38.inventory(),implementation=impl())


def spatial(cfg):
    manifest=json.loads(verify(cfg['spatial_manifest']).read_text())
    if manifest['status']!='SEALED_OFFICIAL_SPATIAL_INDEX' or manifest['truth_opened'] or manifest['crm_opened']: raise ValueError('Invalid spatial authority')
    for ref in manifest['files'].values(): verify(ref)
    return dict(spatial_files=manifest['files'],spatial_sites=manifest['compatible_coordinate_pairs'],spatial_manifest=cfg['spatial_manifest'])


def arrays(launch):
    for ref in launch['spatial_files'].values(): verify(ref)
    keys=np.load(launch['spatial_files']['keys.npy']['path'],mmap_mode='r',allow_pickle=False)
    xy=np.load(launch['spatial_files']['xy.npy']['path'],mmap_mode='r',allow_pickle=False)
    if len(keys)!=launch['spatial_sites'] or xy.shape!=(len(keys),2): raise ValueError('Spatial alignment drift')
    return keys,xy


def compatible(g,q):
    g=dict(g)
    g['spatial_crs_compatible']=bool(q['crm_insee'] and not q['crm_insee'].startswith(('97','98')) and g['ban_resolved']
        and 0<=g['ban_x_l93']<=1300000 and 6000000<=g['ban_y_l93']<=7300000)
    return g


def child(mode,output,start,began,cfg):
    with (output/f'{mode}{start:03d}.log').open('x') as log:
        subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v56_exact_fields',mode,'--output',str(output),'--start',str(start)],
            check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,max(1,cfg['controller_seconds']-(time.monotonic()-began))),
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})


def patch_scene(row,scene,exact_scores):
    """Preserve cached scene order and all17 unaffected coordinates."""
    if any(row[k]!=scene[k] for k in ('query_id','fold','positive_siren')) or len(row['negatives'])!=len(scene['negatives']):
        raise ValueError('Cached scene alignment drift')
    def vector(old,siret):
        values=np.asarray(old,dtype=np.float64)
        if values.shape!=(len(FEATURES),) or not np.isfinite(values).all() or values[2]!=0. or values[10]!=0.:
            raise ValueError('Inactive old exact-field evidence drift')
        score=float(exact_scores.get(siret,0.))
        if score not in (0.,1.): raise ValueError('Exact membership drift')
        result=values.copy(); result[2]=score; result[10]=float(bool(score and values[3] and values[4]))
        return result.tolist()
    negatives=[]
    for source,old in zip(row['negatives'],scene['negatives']):
        if source['siren']!=old['siren'] or not source['siret'].startswith(source['siren']):
            raise ValueError('Negative order/parent drift')
        negatives.append(dict(siren=old['siren'],features=vector(old['features'],source['siret'])))
    return dict(query_id=scene['query_id'],fold=scene['fold'],positive_siren=scene['positive_siren'],
        positive_features=vector(scene['positive_features'],row['positive_siret']),negatives=negatives)


def training_worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    cached=[r for ref in launch['cached_features'] for r in json.loads(verify(ref).read_text())['scenes']]
    if len(cached)!=512 or len(launch['training_rows'])!=512: raise ValueError('Cached training denominator drift')
    with Budget(launch['resources']) as budget:
        backend=TantivyBackend(v38.INDEX); scenes=[]; audit_queries=0; audit_vectors=0
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Official cardinality drift')
        tv=backend._tantivy
        from src.xgb_matcher.retrieval_proof_v56 import components
        for i,(row,scene) in enumerate(zip(launch['training_rows'],cached)):
            q=v42.as_query(row); ids=[row['positive_siret'],*[n['siret'] for n in row['negatives']]]
            if len(set(ids))!=len(ids) or not row['positive_siret'].startswith(row['positive_siren']): raise ValueError('Official training identity drift')
            gate=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',ids),0.)
            part=components(backend,q)[2]; scores={}
            if part is not None:
                parsed=tv.Query.boolean_query([(tv.Occur.Must,gate),(tv.Occur.Must,part)])
                scores={backend._first(backend.searcher.doc(loc).to_dict(),'siret'):float(score)
                    for score,loc in backend.searcher.search(parsed,limit=len(ids),count=False).hits}
            patched=patch_scene(row,scene,scores)
            if i<16:
                ordered,features=point_features(backend,q,ids)
                declared={row['positive_siret']:patched['positive_features'],**{n['siret']:p['features'] for n,p in zip(row['negatives'],patched['negatives'])}}
                if not np.array_equal(features,np.array([declared[s][:11] for s in ordered])):
                    raise ValueError('Cached/native lexical feature audit drift')
                audit_queries+=1; audit_vectors+=len(ordered)
            scenes.append(patched)
    v38.verify_launch(launch)
    for ref in launch['cached_features']: verify(ref)
    write_json(output/'features000.json',dict(scenes=scenes,resources=budget.report(),
        audit_queries=audit_queries,audit_vectors=audit_vectors,
        positive_exact_hits=sum(bool(s['positive_features'][2]) for s in scenes),
        negative_exact_hits=sum(bool(n['features'][2]) for s in scenes for n in s['negatives']),
        unaffected_feature_indices=[i for i in range(len(FEATURES)) if i not in (2,10)],
        old_exact_features_all_zero=True,exact_scope='CURRENT_LEGAL_TRADE_SITE_ONLY_NOT_HISTORICAL_SUPPORTING'))


def train(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        source=json.loads(verify(cfg['cached_proof_source']).read_text()); v38.verify_launch(source)
        sl=json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
        if source['status']!='SEALED_OOF_INITIAL_PROOF_MODELS' or source['training_queries']!=512: raise ValueError('Cached proof authority drift')
        for ref in source['features']: verify(ref)
        rows=sl['training_rows']
        if len(rows)!=512 or len({r['query_id'] for r in rows})!=512 or any(r['fold'] not in (2,3,4) for r in rows): raise ValueError('Training population drift')
        launch=dict(**index_authority(),training_rows=rows,registry=sl['registry'],cached_features=source['features'],
            cached_proof_source=cfg['cached_proof_source'],resources=cfg['resources'],truth_scope='TRAINER_ONLY',
            geometry_unchanged=True,exact_fields=['legal_names_exact','trade_names_exact','site_names_exact'])
        write_json(output/'launch.json',launch); child('training_worker',output,0,began,cfg)
        part=json.loads((output/'features000.json').read_text()); scenes=part['scenes']
        if len(scenes)!=512 or not part['resources']['pass_resource_policy']: raise ValueError('Repair training scope/resource drift')
        models={str(f):fit(scenes,f,launch['registry'],cfg['ridge']) for f in (2,3,4)}
        v38.verify_launch(launch); write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(status='SEALED_OOF_INITIAL_PROOF_MODELS',models=record(output/'models.json'),launch=record(output/'launch.json'),
            features=[record(output/'features000.json')],training_queries=512,seconds=time.monotonic()-began,fold0_opened=False,fold1_opened=False,
            identifiers_as_features=False,downstream_models_unchanged=True,ready_for_fold1=False,recall_measured=False,
            cached_proof_source=cfg['cached_proof_source'],exact_field_repair_before_any_v55_v56_recall=True,
            implementation=launch['implementation'],index_manifest=launch['index_manifest'],index_meta=launch['index_meta'],inventory=launch['inventory']))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,recall_measured=False)); raise


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text()); geos={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
    with Budget(launch['resources']) as budget:
        keys,xy=arrays(launch)
        with Path(launch['spatial_files']['tree.pickle']['path']).open('rb') as handle: tree=pickle.load(handle)
        parents=np.load(launch['spatial_files']['parents.npy']['path'],mmap_mode='r',allow_pickle=False)
        if tree.n!=len(keys) or not np.array_equal(tree.data,xy): raise ValueError('KD point alignment drift')
        backend=TantivyBackend(v38.INDEX); rows=[]
        for raw in launch['queries'][start:start+8]:
            m=models[str(raw['fold'])]
            if m['features']!=list(FEATURES) or m['heldout_fold']!=raw['fold'] or raw['fold'] in m['training_folds']: raise ValueError('Serving OOF model drift')
            t=time.monotonic(); result,trace=search(backend,tree,keys,parents,v42.as_query(raw),geos[raw['query_id']],m['weights'],launch['max_query_search_seconds'])
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],results=dict(primary=result),seconds=dict(primary=time.monotonic()-t),primary_trace=trace))
    v38.verify_launch(launch); write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output,model_manifest):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        trained=json.loads(model_manifest.read_text()); v38.verify_launch(trained)
        if trained['status']!='SEALED_OOF_INITIAL_PROOF_MODELS': raise ValueError('Models unsealed')
        models=json.loads(verify(trained['models']).read_text()); queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4)
        if any(set(m['query_ids'])&{r['query_id'] for r in queries} for m in models.values()): raise ValueError('Fit/evaluation query reuse')
        gm=json.loads(verify(cfg['query_geo_generation']).read_text()); gl=json.loads(verify(gm['launch']).read_text())
        if gm['status']!='SEALED_INITIAL_TOP100' or gl['queries']!=queries: raise ValueError('Sealed query-only geometry authority drift')
        launch=dict(**index_authority(),**spatial(cfg),queries=queries,query_sources=qrefs,models=trained['models'],model_manifest=record(model_manifest),
            geo_queries=gl['geo_queries'],resources=cfg['resources'],max_query_search_seconds=cfg['max_query_search_seconds'],collector_cap=100,
            fixed_reservoir=None,truth_opened=False,development_panel_reused=True,comparator_scope='SEPARATE_EVALUATOR_SEALED_V42_NO_GENERATOR_READ')
        write_json(output/'launch.json',launch); rows=[]; refs=[]
        for start in range(0,128,8):
            child('worker',output,start,began,cfg); path=output/f'batch{start:03d}.json'; part=json.loads(path.read_text())
            if len(part['rows'])!=8 or not part['resources']['pass_resource_policy']: raise ValueError('Serving scope/resource drift')
            rows.extend(part['rows']); refs.append(record(path)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                mean=sum(r['seconds']['primary'] for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_primary_seconds=mean,passed=mean<=cfg['primary_mean16_seconds'],quality_opened=False,not_production_slo=True))
                if mean>cfg['primary_mean16_seconds']: raise ValueError('STOP_PRIMARY_COST_PREFLIGHT')
        v38.verify_launch(launch)
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=refs,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,truth_opened=False,model_used=True,full_query_population=False,ready_for_fold1=False,
            metric_family=cfg['metric_family'],full_lexical_corpus=True,full_spatial_corpus=False,comparator_replayed=False,old_candidates_opened=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False,ready_for_fold1=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['train','training_worker','run','worker','evaluate']); p.add_argument('--output',required=True,type=Path)
    p.add_argument('--start',type=int); p.add_argument('--models',type=Path); a=p.parse_args()
    if a.mode=='train': train(a.output)
    elif a.mode=='evaluate': evaluate(a.output)
    elif a.mode=='run':
        if a.models is None: p.error('--models manifest required')
        run(a.output,a.models)
    elif a.mode=='training_worker':
        if a.start!=0: p.error('Invalid cached training repair batch')
        training_worker(a.output,a.start)
    else:
        if a.start not in range(0,128,8): p.error('Invalid serving batch')
        worker(a.output,a.start)

