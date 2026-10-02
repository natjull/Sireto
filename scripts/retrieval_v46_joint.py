"""Bounded OOF hard-negative joint-metric training and initial top100 pilot."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts.run_retrieval_geo_v16_workbench import _resolve_ban
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_geo_v16 import normalize_street
from src.xgb_matcher.retrieval_index_pruning_v44 import search as mining_search
from src.xgb_matcher.retrieval_joint_v46 import FEATURES,fit_metric,geo_features,search
from src.xgb_matcher.retrieval_spatial_v43 import nearest_sites
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v46_joint.json'


def implementation():
    paths=[Path(__file__),CONFIG,Path(v38.__file__),Path(v42.__file__),
        ROOT/'src/xgb_matcher/retrieval_joint_v46.py',ROOT/'src/xgb_matcher/retrieval_index_pruning_v44.py',
        ROOT/'src/xgb_matcher/retrieval_spatial_v43.py',ROOT/'src/xgb_matcher/retrieval_geo_v16.py',
        ROOT/'scripts/run_retrieval_geo_v16_workbench.py',ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
        ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    return {str(p):record(p) for p in paths}


def spatial(cfg):
    sm=json.loads(verify(cfg['spatial_manifest']).read_text())
    if sm['status']!='SEALED_OFFICIAL_SPATIAL_INDEX' or sm['truth_opened'] or sm['crm_opened']:
        raise ValueError('Unsealed spatial source')
    for ref in sm['files'].values(): verify(ref)
    return sm


def load_tree(launch):
    files=launch['spatial_files']
    for ref in files.values(): verify(ref)
    with Path(files['tree.pickle']['path']).open('rb') as handle: tree=pickle.load(handle)
    keys=np.load(files['keys.npy']['path'],mmap_mode='r',allow_pickle=False)
    parents=np.load(files['parents.npy']['path'],mmap_mode='r',allow_pickle=False)
    if tree.n!=launch['spatial_sites'] or len(keys)!=tree.n or len(parents)!=tree.n: raise ValueError('Spatial alignment drift')
    return tree,keys,parents


def index_launch():
    im=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
    return dict(index_manifest=record(v38.INDEX/'manifest.json'),index_meta=record(v38.INDEX/'tantivy/meta.json'),
        index_documents=im['num_documents'],index_sites=im['num_establishment_documents'],inventory=v38.inventory(),
        implementation=implementation())


def validate_queries(queries):
    if len(queries)!=128 or len({r['query_id'] for r in queries})!=128 or any(
        set(r)!=v38.QUERY_FIELDS or r['fold'] not in (2,3,4) for r in queries): raise ValueError('Query scope drift')
    if any(re.search(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)',r[k]) for r in queries for k in ('crm_name','crm_street')):
        raise ValueError('Identifier-shaped text cannot be a model feature')


def resolve_training_geo(output,queries,cfg):
    import duckdb
    surface=pa.Table.from_pylist([dict(query_id=r['query_id'],fold=r['fold'],crm_number=r['crm_number'],
        crm_street_norm=normalize_street(r['crm_street']),crm_postcode=r['crm_postcode'],crm_insee=r['crm_insee']) for r in queries])
    with Budget(cfg['resources_prepare']) as budget:
        ban=verify(cfg['ban']); conn=duckdb.connect(); conn.execute('SET threads=2')
        conn.execute("SET memory_limit='512MB'"); conn.execute("SET max_temp_directory_size='0B'")
        conn.execute('SET enable_progress_bar=false'); rows=_resolve_ban(conn,surface,ban).to_pylist(); conn.close()
    if len(rows)!=128: raise ValueError('BAN denominator drift')
    for g,q in zip(rows,queries):
        if (g['query_id'],g['fold'])!=(q['query_id'],q['fold']): raise ValueError('BAN order drift')
        g['spatial_crs_compatible']=bool(q['crm_insee'] and not q['crm_insee'].startswith(('97','98')) and g['ban_resolved']
            and 0<=g['ban_x_l93']<=1300000 and 6000000<=g['ban_y_l93']<=7300000)
    write_json(output/'training_geo.json',rows)
    return record(output/'training_geo.json'),budget.report()


def training_worker(output,held,start):
    launch=json.loads((output/'training_launch.json').read_text()); v38.verify_launch(launch)
    geo={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
    old=json.loads(verify(launch['mining_models']).read_text())[str(held)]
    if old['heldout_fold']!=held or held in old['training_folds']: raise ValueError('Mining model OOF drift')
    held_entities=set(launch['registry'][str(held)]); scenes=[]
    with Budget(launch['resources']) as budget:
        tree,keys,parents=load_tree(launch); backend=TantivyBackend(v38.INDEX)
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('National count drift')
        for raw in launch['rows_by_held'][str(held)][start:start+8]:
            if raw['fold']==held or raw['positive_siren'] in held_entities: raise ValueError('Heldout training leakage')
            q=v42.as_query(raw); g=geo[raw['query_id']]
            mined,_=mining_search(backend,q,old['weights'])
            geoneg=[]
            if g['spatial_crs_compatible']:
                geoneg,_=nearest_sites(tree,tree.data,keys,parents,[g['ban_x_l93'],g['ban_y_l93']])
            safe={n['siret']:n for n in [*mined,*geoneg,*raw['natural_negatives']]
                if n['siren'] not in held_entities and n['siret'] not in raw['acceptable']}
            ids,x=v42.training_features(backend,q,[raw['positive_siret'],*safe])
            combined=np.column_stack([x,geo_features(keys,tree.data,ids,g)])
            values=dict(zip(ids,combined.tolist())); positive_doc=backend.by_siret(raw['positive_siret'])
            if positive_doc is not None and positive_doc.siren!=raw['positive_siren']: raise ValueError('Positive source membership drift')
            p=values[raw['positive_siret']]
            scenes.append(dict(query_id=raw['query_id'],fold=raw['fold'],positive_siren=raw['positive_siren'],
                positive_features=p if positive_doc is not None else None,positive_catalog_present=positive_doc is not None,
                positive_text_eligible=bool(p[0] or p[1]),positive_geo_available=bool(p[8]),
                negatives=[dict(siren=n['siren'],features=values[s]) for s,n in safe.items()],
                lexical_hard100=len(mined),spatial_hard100=len(geoneg),safe_unique_negatives=len(safe)))
    v38.verify_launch(launch)
    write_json(output/f'train{held}_{start:03d}.json',dict(scenes=scenes,resources=budget.report()))


def subprocess_worker(mode,output,start,began,cfg,held=None):
    args=[sys.executable,'-u','-m','scripts.retrieval_v46_joint',mode,'--output',str(output),'--start',str(start)]
    if held is not None: args+=['--held',str(held)]
    with (output/f'{mode}{held or ""}_{start:03d}.log').open('x') as log:
        subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT,
            timeout=min(90,max(1,cfg['max_controller_seconds']-(time.monotonic()-began))),
            env={**os.environ,'RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})


def train(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),0); validate_queries(queries)
        baseline=json.loads(verify(cfg['natural_training_generation']).read_text())
        if baseline['status']!='SEALED_INITIAL_TOP100' or baseline['truth_opened']: raise ValueError('Natural negative source drift')
        if json.loads(verify(baseline['launch']).read_text())['queries']!=queries: raise ValueError('Training source mismatch')
        natural={r['query_id']:r for ref in baseline['batches'] for r in json.loads(verify(ref).read_text())['rows']}
        if set(natural)!={r['query_id'] for r in queries}: raise ValueError('Natural negatives incomplete')
        labels=pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),filters=[('fold','in',[2,3,4])]).to_pylist()
        labelmap={r['query_id']:r for r in labels}
        registry={str(f):sorted({r['human_ground_truth_siren'] for r in labels if r['fold']==f}) for f in (2,3,4)}
        ev=v38.load(v38.EVAL,v38.EVAL_SHA); side=json.loads(verify(ev['label_side']).read_text())
        for ref in side['files'].values(): verify(ref)
        acceptable={}
        for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
            acceptable.setdefault(r['query_id'],set()).add(r['siret'])
        sm=spatial(cfg); georef,georesources=resolve_training_geo(output,queries,cfg)
        rows=[]
        for raw in queries:
            label=labelmap[raw['query_id']]
            if label['fold']!=raw['fold'] or raw['query_id'] not in acceptable: raise ValueError('Training label-side drift')
            natural_neg=[n for mode in v38.MODES for n in natural[raw['query_id']]['results'][mode][:32]]
            rows.append(dict(raw,positive_siret=label['human_ground_truth_siret'],positive_siren=label['human_ground_truth_siren'],
                acceptable=sorted(acceptable[raw['query_id']]),natural_negatives=natural_neg))
        byheld={str(f):[r for r in rows if r['fold']!=f and r['positive_siren'] not in set(registry[str(f)])] for f in (2,3,4)}
        launch=dict(**index_launch(),rows_by_held=byheld,training_queries=128,registry=registry,query_sources=qrefs,
            spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],spatial_manifest=cfg['spatial_manifest'],
            geo_queries=georef,geo_resources=georesources,ban=cfg['ban'],mining_models=cfg['mining_models'],
            ledger=record(v38.LEDGER),label_side=ev['label_side'],resources=cfg['resources_worker'],truth_scope='TRAINER_ONLY')
        write_json(output/'training_launch.json',launch); models={}; features={}; counts={}
        for held in (2,3,4):
            scenes=[]; refs=[]
            for start in range(0,len(byheld[str(held)]),8):
                subprocess_worker('training_worker',output,start,began,cfg,held)
                path=output/f'train{held}_{start:03d}.json'; batch=json.loads(path.read_text())
                expected=min(8,len(byheld[str(held)])-start)
                if len(batch['scenes'])!=expected or not batch['resources']['pass_resource_policy']: raise ValueError('Training scope/resource drift')
                scenes.extend(batch['scenes']); refs.append(record(path)); print(json.dumps(dict(held=held,training_queries=len(scenes))),flush=True)
            if {r['query_id'] for r in scenes}!={r['query_id'] for r in byheld[str(held)]}: raise ValueError('Fit query coverage drift')
            models[str(held)]=fit_metric(scenes,held,registry,cfg['ridge']); features[str(held)]=refs
            counts[str(held)]=dict(queries=len(scenes),official_positive_absences=sum(not r['positive_catalog_present'] for r in scenes),
                positive_without_text_or_xy=sum(not r['positive_text_eligible'] and not r['positive_geo_available'] for r in scenes),
                safe_negative_pairs=sum(r['safe_unique_negatives'] for r in scenes))
        v38.verify_launch(launch); write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(status='SEALED_OOF_JOINT_INITIAL_METRICS',launch=record(output/'training_launch.json'),
            models=record(output/'models.json'),features=features,counts=counts,training_queries=128,seconds=time.monotonic()-began,
            fold0_opened=False,fold1_opened=False,identifiers_as_features=False,downstream_models_unchanged=True,
            implementation=launch['implementation'],index_manifest=launch['index_manifest'],index_meta=launch['index_meta'],inventory=launch['inventory']))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began)); raise


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text()); old=json.loads(verify(launch['mining_models']).read_text())
    geo={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}; rows=[]
    with Budget(launch['resources']) as budget:
        tree,keys,parents=load_tree(launch); backend=TantivyBackend(v38.INDEX)
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('National count drift')
        for i,raw in enumerate(launch['queries'][start:start+8],start):
            model=models[str(raw['fold'])]; previous=old[str(raw['fold'])]
            if model['features']!=list(FEATURES) or model['heldout_fold']!=raw['fold'] or raw['fold'] in model['training_folds']:
                raise ValueError('OOF metric drift')
            row=dict(query_id=raw['query_id'],fold=raw['fold'],results={},seconds={})
            for mode in (v38.MODES if i%2==0 else tuple(reversed(v38.MODES))):
                t=time.monotonic()
                if mode=='primary': result,trace=search(backend,tree,keys,parents,v42.as_query(raw),geo[raw['query_id']],
                    model['weights'],max_seconds=launch['max_query_search_seconds'])
                else: result,trace=mining_search(backend,v42.as_query(raw),previous['weights'])
                row['results'][mode]=result; row['seconds'][mode]=time.monotonic()-t; row[mode+'_trace']=trace
            rows.append(row)
    v38.verify_launch(launch); write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output,model_dir):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        trained=json.loads((model_dir/'manifest.json').read_text())
        if trained['status']!='SEALED_OOF_JOINT_INITIAL_METRICS': raise ValueError('Unsealed models')
        v38.verify_launch(trained)
        queries,qrefs=v42.input_rows(v38.load(v38.GEN,v38.GEN_SHA),4); validate_queries(queries)
        gm=json.loads(verify(cfg['geo_evaluation_generation']).read_text()); gl=json.loads(verify(gm['launch']).read_text())
        if gm['status']!='SEALED_INITIAL_TOP100' or queries!=gl['queries']: raise ValueError('Query-only geo authority drift')
        models=json.loads(verify(trained['models']).read_text())
        if any(set(m['query_ids'])&{r['query_id'] for r in queries} for m in models.values()): raise ValueError('Training query reuse')
        sm=spatial(cfg); verify(gl['geo_queries'])
        launch=dict(**index_launch(),queries=queries,query_sources=qrefs,models=trained['models'],model_manifest=record(model_dir/'manifest.json'),
            spatial_files=sm['files'],spatial_sites=sm['compatible_coordinate_pairs'],spatial_manifest=cfg['spatial_manifest'],
            geo_queries=gl['geo_queries'],query_geo_generation=cfg['geo_evaluation_generation'],mining_models=cfg['mining_models'],
            resources=cfg['resources_worker'],max_query_search_seconds=cfg['max_query_search_seconds'],
            collector_cap=100,fixed_reservoir=None,truth_opened=False,development_panel_reused=True)
        write_json(output/'launch.json',launch); rows=[]; refs=[]
        for start in range(0,128,8):
            subprocess_worker('worker',output,start,began,cfg)
            path=output/f'batch{start:03d}.json'; batch=json.loads(path.read_text())
            if len(batch['rows'])!=8 or not batch['resources']['pass_resource_policy']: raise ValueError('Worker scope/resource drift')
            rows.extend(batch['rows']); refs.append(record(path)); print(json.dumps(dict(generated=len(rows))),flush=True)
            if len(rows)==16:
                cost=sum(sum(r['seconds'].values()) for r in rows)/16
                write_json(output/'cost_gate.json',dict(queries=16,mean_pair_seconds=cost,passed=cost<=cfg['mean_pair_seconds_gate16'],quality_opened=False))
                if cost>cfg['mean_pair_seconds_gate16']: raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch)
        if len(rows)!=128 or {r['query_id'] for r in rows}!={r['query_id'] for r in queries}: raise ValueError('Serving denominator drift')
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,batches=refs,launch=record(output/'launch.json'),
            seconds=time.monotonic()-began,model_used=True,truth_opened=False,full_lexical_corpus=True,full_spatial_corpus=False,
            full_query_population=False,ready_for_fold1=False,certified_initial_score=True,metric_family=cfg['metric_family']))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False,ready_for_fold1=False)); raise


def evaluate(output):
    dest=output/'evaluation.json'
    if dest.exists(): raise ValueError('Do not overwrite evaluation')
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']: raise ValueError('Complete sealed generation required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    rows=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids={r['query_id'] for r in launch['queries']}
    if len(rows)!=128 or {r['query_id'] for r in rows}!=ids: raise ValueError('Evaluation denominator drift')
    if not all(r['primary_trace']['certified'] for r in rows): raise ValueError('Uncertified output')
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
        if label['fold']!=r['fold'] or r['fold'] not in (2,3,4): raise ValueError('Evaluation fold drift')
        outcome=dict(query_id=r['query_id'],**{c:label[c] for c in v38.SEGMENTS})
        for mode in v38.MODES:
            found={n['siret'] for n in r['results'][mode]}
            if len(found)!=len(r['results'][mode]) or len(found)>100: raise ValueError('Initial cap drift')
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
        latency={mode:dict(zip(('p50','p95','p99'),np.quantile([r['seconds'][mode] for r in rows],[.5,.95,.99]).tolist())) for mode in v38.MODES},
        latency_scope='Loaded joint index search and references, excludes BAN resolution and index loads; cache uncontrolled, not production SLO',
        loss_attribution={mode:dict(Counter(r[mode+'_loss'] for r in outcomes)) for mode in v38.MODES},
        traversal_quantiles=dict(zip(('p50','p95','p99'),np.quantile([r['primary_trace']['unique_documents_visited'] for r in rows],[.5,.95,.99]).tolist())),
        outcomes=outcomes,generation=record(output/'manifest.json'),ledger=record(v38.LEDGER),label_side=ev['label_side'],
        excluded_missing_human_labels=ev['excluded_missing_human_labels'],fine_loss_attribution_measured=False,
        model_used=True,full_query_population=False,ready_for_fold1=False,fold0_opened=False,fold1_opened=False,
        metric_family=manifest['metric_family'],implementation=record(Path(__file__)))
    if result['primary']['exact_recall']>=.992:
        result['decision']='STOP'; result['decision_reason']='PILOT_ONLY_NO_FULL_OOF_GENERALIZATION_SLO_GATE'
    write_json(dest,result); print(json.dumps({k:result[k] for k in ('primary','comparator','gains','losses','latency','traversal_quantiles','decision')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['train','training_worker','run','worker','evaluate'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--models',type=Path)
    p.add_argument('--held',type=int); p.add_argument('--start',type=int); a=p.parse_args()
    if a.mode=='train': train(a.output)
    elif a.mode=='run':
        if a.models is None: p.error('--models required')
        run(a.output,a.models)
    elif a.mode=='evaluate': evaluate(a.output)
    else:
        if a.start is None or a.start<0 or a.start%8 or a.start>=128: p.error('Invalid start')
        if a.mode=='training_worker':
            if a.held not in (2,3,4): p.error('Invalid heldout fold')
            training_worker(a.output,a.held,a.start)
        else: worker(a.output,a.start)
