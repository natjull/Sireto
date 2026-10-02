"""Learn an initial index score; collect <=100 directly on the full corpus.

The serving path has no candidate reservoir, pair feature extraction or
downstream reranker. A nonnegative metric is compiled into the native index
query. Training may join official truth documents, never serving/evaluation.
This tests learned structured sparse search, NOT a dense retriever/SC-Block.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

FEATURES = ('name', 'address', 'name_at_place', 'address_at_place',
            'name_and_address', 'name_and_address_at_place', 'place', 'number_at_place')
BASE = np.array([1., 1., 0., 0., 0., 0., 1., 0.])
CONFIG = ROOT/'config/retrieval_v42_indexed_metric.json'
RESOURCE = dict(max_seconds=80, max_rss_bytes=6*1024**3, max_swapout_pages_delta=0)


def as_query(raw):
    return RetrievalQuery(name=raw['crm_name'],
        address=' '.join(raw[k] for k in ('crm_number','crm_suffix','crm_street') if raw[k]),
        number=raw['crm_number'], insee=raw['crm_insee'], postcode=raw['crm_postcode'])


def components(backend, query):
    tv = backend._tantivy
    def conjunction(*parts):
        if any(p is None for p in parts): return None
        return tv.Query.boolean_query([(tv.Occur.Must, p) for p in parts])
    texts = []
    for fields, charfield, text in ((v38.NAMES, 'current_name_ngrams', query.name),
                                   (v38.ADDRESSES, 'address_ngrams', query.address)):
        views = [v38.mean_query(backend, f, v38.tokens(text)) for f in fields]
        views.append(v38.mean_query(backend, charfield, v38.tokens(text, True)))
        views = [q for q in views if q is not None]
        texts.append(tv.Query.disjunction_max_query(views, tie_breaker=0.) if views else None)
    name, address = texts
    geo = [tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, f, value, 'basic'), 1.)
           for f, value in (('insee',query.insee),('postcode',query.postcode)) if value]
    place = tv.Query.disjunction_max_query(geo, tie_breaker=0.) if geo else None
    gate = tv.Query.const_score_query(place, 0.) if place is not None else None
    number = tv.Query.term_query(backend.index.schema,'number',query.number,'basic') if query.number else None
    number_place = conjunction(number, gate)
    if number_place is not None: number_place = tv.Query.const_score_query(number_place, 1.)
    return [name, address, conjunction(name,gate), conjunction(address,gate),
            conjunction(name,address), conjunction(name,address,gate), place, number_place]


def index_query(backend, query, weights):
    w = np.asarray(weights,dtype=np.float64)
    if w.shape != (len(FEATURES),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Invalid index metric')
    tv = backend._tantivy; parts = components(backend, query)
    eligible = [q for q in parts[:2] if q is not None]
    if not eligible: return tv.Query.empty_query()
    doc_type = tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    must_text = tv.Query.const_score_query(tv.Query.boolean_query([(tv.Occur.Should,q) for q in eligible]),0.)
    clauses = [(tv.Occur.Must,doc_type),(tv.Occur.Must,must_text)]
    clauses += [(tv.Occur.Should,tv.Query.boost_query(q,float(value)))
                for q,value in zip(parts,w) if q is not None and value > 0]
    return tv.Query.boolean_query(clauses)


def search(backend, query, weights):
    hits = backend.searcher.search(index_query(backend,query,weights),limit=100,count=False).hits
    rows = []
    for score,loc in hits:
        doc = backend.searcher.doc(loc).to_dict()
        siren,siret = (backend._first(doc,k) for k in ('siren','siret'))
        if not (len(siren)==9 and siren.isdigit() and len(siret)==14 and siret.isdigit()
                and siret.startswith(siren) and math.isfinite(score)):
            raise ValueError('Official identity/score drift')
        rows.append(dict(siren=siren,siret=siret,score=float(score)))
    if len({r['siret'] for r in rows}) != len(rows): raise ValueError('Duplicate initial output')
    return sorted(rows,key=lambda r:(-r['score'],r['siret']))


def training_features(backend, query, ids):
    """TRAINER ONLY: IDs restrict point lookups, never model features.

    BM25 statistics remain those of the complete national index. This
    materialization is NOT part of serving and is never evaluated as recall.
    """
    tv = backend._tantivy
    ids = sorted(set(ids)); index = {s:i for i,s in enumerate(ids)}
    identity_gate = tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',ids),0.)
    result = np.zeros((len(ids),len(FEATURES)),dtype=np.float64)
    for j,part in enumerate(components(backend,query)):
        if part is None: continue
        parsed = tv.Query.boolean_query([(tv.Occur.Must,identity_gate),(tv.Occur.Must,part)])
        for score,loc in backend.searcher.search(parsed,limit=len(ids),count=False).hits:
            doc = backend.searcher.doc(loc).to_dict(); siret = backend._first(doc,'siret')
            if siret not in index: raise ValueError('Point lookup leaked outside training IDs')
            result[index[siret],j] = score
    return ids,result


def fit_metric(scenes, heldout_fold, registry, ridge=.0005):
    if heldout_fold not in (2,3,4) or set(registry) != {'2','3','4'}:
        raise ValueError('OOF registry/fold drift')
    heldout = set(registry[str(heldout_fold)])
    if not heldout: raise ValueError('Empty heldout registry')
    if any(r['fold'] not in (2,3,4) or r['positive_siren'] not in registry[str(r['fold'])] for r in scenes):
        raise ValueError('Training source fold/entity drift')
    differences,groups,kept_queries = [],[],[]
    for row in scenes:
        if row['fold']==heldout_fold or row['positive_siren'] in heldout or row['positive_features'] is None:
            continue
        negatives = [n for n in row['negatives'] if n['siren'] not in heldout]
        if not negatives: continue
        diff = np.asarray(row['positive_features'])-np.asarray([n['features'] for n in negatives])
        differences.extend(diff); groups.extend([1./len(negatives)]*len(negatives))
        kept_queries.append((row['query_id'],row['fold']))
    if len(kept_queries)<16: raise ValueError('Insufficient safe OOF fit queries')
    x = np.asarray(differences,dtype=np.float64); sample_weight = np.asarray(groups)/len(kept_queries)
    if x.shape[1]!=len(FEATURES) or not np.isfinite(x).all() or ridge<=0: raise ValueError('Invalid fit features')
    def objective(w):
        margin = x@w
        loss = float(sample_weight@np.logaddexp(0.,-margin)+ridge*np.sum((w-BASE)**2))
        gradient = x.T@(-sample_weight*expit(-margin))+2*ridge*(w-BASE)
        return loss,gradient
    fitted = minimize(objective,BASE,method='L-BFGS-B',jac=True,bounds=[(0.,64.)]*len(FEATURES),
                      options=dict(maxiter=500,ftol=1e-12,gtol=1e-8))
    if not fitted.success or not np.isfinite(fitted.x).all() or fitted.x.sum()<=0:
        raise ValueError('Metric fit failed: '+str(fitted.message))
    return dict(weights=fitted.x.tolist(),features=list(FEATURES),heldout_fold=heldout_fold,
        training_folds=sorted({f for _,f in kept_queries}),training_queries=len(kept_queries),
        training_pairs=len(x),query_ids=[q for q,_ in kept_queries],heldout_entity_overlap=0,
        objective_before=objective(BASE)[0],objective_after=float(fitted.fun),ridge=ridge,
        identifiers_as_features=False,trained_downstream_model=False)


def input_rows(generation,first_batch):
    rows,refs = [],[]
    for ref in generation['batches'][first_batch:first_batch+4]:
        batch = json.loads(verify(ref).read_text())
        if batch['count']!=32 or len(batch['shards'])!=1: raise ValueError('Query batch drift')
        shard = json.loads(verify(batch['shards'][0]).read_text())
        source = shard['files']['queries.parquet']; part = pq.read_table(verify(source)).to_pylist()
        if any(set(r)!=v38.QUERY_FIELDS or r['fold'] not in (2,3,4) for r in part):
            raise ValueError('Forbidden serving input')
        rows.extend(part); refs.append(source)
    if len(rows)!=128 or len({r['query_id'] for r in rows})!=128: raise ValueError('Population drift')
    return rows,refs


def training_worker(output,start):
    launch = json.loads((output/'training_launch.json').read_text()); v38.verify_launch(launch)
    scenes = []
    with Budget(RESOURCE) as budget:
        backend = TantivyBackend(v38.INDEX)
        for row in launch['training_rows'][start:start+8]:
            negative = row['negatives']; gt = row['positive_siret']
            ids,values = training_features(backend,as_query(row),[gt]+[n['siret'] for n in negative])
            mapped = dict(zip(ids,values.tolist())); positive = mapped[gt]
            positive_doc = backend.by_siret(gt)
            if positive_doc is not None and positive_doc.siren!=row['positive_siren']:
                raise ValueError('Official positive membership drift')
            scenes.append(dict(query_id=row['query_id'],fold=row['fold'],positive_siren=row['positive_siren'],
                positive_features=positive if positive_doc is not None else None,
                positive_catalog_present=positive_doc is not None,
                positive_text_eligible=bool(positive[0] or positive[1]),
                negatives=[dict(siren=n['siren'],features=mapped[n['siret']]) for n in negative]))
    write_json(output/f'train{start:03d}.json',dict(scenes=scenes,resources=budget.report()))


def train(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        cfg = json.loads(CONFIG.read_text()); generation = v38.load(v38.GEN,v38.GEN_SHA)
        queries,qrefs = input_rows(generation,0)
        baseline = json.loads(verify(cfg['training_generation']).read_text())
        if baseline['status']!='SEALED_INITIAL_TOP100' or baseline['truth_opened']:
            raise ValueError('Unsealed natural negative source')
        blaunch = json.loads(verify(baseline['launch']).read_text())
        if queries!=blaunch['queries']: raise ValueError('Training queries differ from sealed negative source')
        generated = [r for ref in baseline['batches'] for r in json.loads(verify(ref).read_text())['rows']]
        if {r['query_id'] for r in generated}!={r['query_id'] for r in queries}: raise ValueError('Negative coverage drift')
        natural = {r['query_id']:r for r in generated}
        labels = pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),
                              filters=[('fold','in',[2,3,4])]).to_pylist()
        labelmap = {r['query_id']:r for r in labels}
        registry = {str(f):sorted({r['human_ground_truth_siren'] for r in labels if r['fold']==f}) for f in (2,3,4)}
        ev = v38.load(v38.EVAL,v38.EVAL_SHA)
        label_side = json.loads(verify(ev['label_side']).read_text())
        for ref in label_side['files'].values(): verify(ref)
        acceptable = {}
        for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),
                               filters=[('fold','in',[2,3,4])]).to_pylist():
            acceptable.setdefault(r['query_id'],set()).add(r['siret'])
        training_rows=[]
        for raw in queries:
            label=labelmap[raw['query_id']]
            if raw['fold']!=label['fold'] or raw['query_id'] not in acceptable: raise ValueError('Training ledger drift')
            negatives=[]; seen=set()
            for mode in ('primary','comparator'):
                for n in natural[raw['query_id']]['results'][mode][:32]:
                    if n['siret'] not in acceptable[raw['query_id']] and n['siret'] not in seen:
                        negatives.append(n); seen.add(n['siret'])
            training_rows.append(dict(raw,positive_siret=label['human_ground_truth_siret'],
                positive_siren=label['human_ground_truth_siren'],negatives=negatives))
        im=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
        launch=dict(training_rows=training_rows,registry=registry,query_sources=qrefs,
            natural_generation=cfg['training_generation'],ledger=record(v38.LEDGER),label_side=ev['label_side'],
            index_manifest=record(v38.INDEX/'manifest.json'),index_meta=record(v38.INDEX/'tantivy/meta.json'),
            inventory=v38.inventory(),index_documents=im['num_documents'],implementation=implementation(),
            truth_join_scope='TRAINER_ONLY_NO_EVALUATION_INJECTION',resources=RESOURCE)
        write_json(output/'training_launch.json',launch); scenes=[]; refs=[]
        for start in range(0,128,8):
            run_worker('training_worker',output,start,began)
            p=output/f'train{start:03d}.json'; batch=json.loads(p.read_text())
            scenes.extend(batch['scenes']); refs.append(record(p))
            print(json.dumps(dict(training_features_completed=len(scenes))),flush=True)
        models={str(f):fit_metric(scenes,f,registry,cfg['ridge']) for f in (2,3,4)}
        if len(scenes)!=128 or {r['query_id'] for r in scenes}!={r['query_id'] for r in queries}:
            raise ValueError('Training denominator drift')
        write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(status='SEALED_OOF_INDEX_METRICS',launch=record(output/'training_launch.json'),
            features=refs,models=record(output/'models.json'),training_queries=128,
            official_positive_absences=sum(not r['positive_catalog_present'] for r in scenes),
            text_ineligible_positives=sum(not r['positive_text_eligible'] for r in scenes),
            identifiers_as_features=False,downstream_models_unchanged=True,seconds=time.monotonic()-began,
            implementation=launch['implementation'],index_manifest=launch['index_manifest'],
            index_meta=launch['index_meta'],inventory=launch['inventory'],
            fold0_opened=False,fold1_opened=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began)); raise


def implementation():
    files=[Path(__file__),CONFIG,Path(v38.__file__),ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
           ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
    return {str(p):record(p) for p in files}


def run_worker(mode,output,start,began):
    remaining=max(1,900-(time.monotonic()-began))
    with (output/f'{mode}{start:03d}.log').open('x') as log:
        subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v42_indexed_metric',mode,
                        '--output',str(output),'--start',str(start)],check=True,timeout=min(90,remaining),
                       stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'RAYON_NUM_THREADS':'2',
                           'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text()); rows=[]
    with Budget(RESOURCE) as budget:
        backend=TantivyBackend(v38.INDEX)
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Corpus count drift')
        for raw in launch['queries'][start:start+8]:
            model=models[str(raw['fold'])]
            if model['heldout_fold']!=raw['fold'] or raw['fold'] in model['training_folds']:
                raise ValueError('Serving OOF model selection drift')
            query=as_query(raw); row=dict(query_id=raw['query_id'],fold=raw['fold'],results={},seconds={})
            for mode in (('primary','comparator') if start%16==0 else ('comparator','primary')):
                t=time.monotonic()
                row['results'][mode]=search(backend,query,model['weights']) if mode=='primary' else v38.search(backend,query,'primary')
                row['seconds'][mode]=time.monotonic()-t
            rows.append(row)
    v38.verify_launch(launch)
    write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output,model_dir):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        cfg=json.loads(CONFIG.read_text()); generation=v38.load(v38.GEN,v38.GEN_SHA)
        queries,qrefs=input_rows(generation,4)
        trained=json.loads((model_dir/'manifest.json').read_text())
        if trained['status']!='SEALED_OOF_INDEX_METRICS': raise ValueError('Unsealed metrics')
        # Verify code/index lineage without loading trainer labels or registry.
        v38.verify_launch(trained)
        models=json.loads(verify(trained['models']).read_text())
        for raw in queries:
            if raw['query_id'] in models[str(raw['fold'])]['query_ids']:
                raise ValueError('Serving query was used for fit')
        im=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
        if im['build_scope']!='full' or im['contains_crm_labels']: raise ValueError('Invalid national corpus')
        launch=dict(queries=queries,query_sources=qrefs,generation=record(v38.GEN),
            training_manifest=record(model_dir/'manifest.json'),models=trained['models'],
            index_manifest=record(v38.INDEX/'manifest.json'),index_meta=record(v38.INDEX/'tantivy/meta.json'),
            inventory=v38.inventory(),index_documents=im['num_documents'],index_sites=im['num_establishment_documents'],
            implementation=implementation(),resources=RESOURCE,collector_limit=100,truth_opened=False,
            full_national_site_corpus=True,fixed_candidate_pool=None,score_extracted_after_discovery=False,
            corpus_full_bytes_rehashed=False,development_panel_reused=True)
        write_json(output/'launch.json',launch); refs=[]; generated=[]
        for start in range(0,128,8):
            run_worker('worker',output,start,began)
            p=output/f'batch{start:03d}.json'; batch=json.loads(p.read_text())
            if {(r['query_id'],r['fold']) for r in batch['rows']}!={(r['query_id'],r['fold']) for r in queries[start:start+8]}:
                raise ValueError('Serving denominator drift')
            generated.extend(batch['rows']); refs.append(record(p))
            print(json.dumps(dict(generated=len(generated))),flush=True)
            if len(generated)==16:
                cost=sum(sum(r['seconds'].values()) for r in generated)/16
                write_json(output/'cost_gate.json',dict(rows=16,mean_pair_seconds=cost,passed=cost<=cfg['mean_pair_seconds_gate'],truth_opened=False))
                if cost>cfg['mean_pair_seconds_gate']: raise ValueError('STOP_COST_PREFLIGHT')
        v38.verify_launch(launch)
        write_json(output/'manifest.json',dict(status='SEALED_INITIAL_TOP100',queries=128,
            launch=record(output/'launch.json'),batches=refs,seconds=time.monotonic()-began,
            truth_opened=False,model_used=True,full_corpus=True,full_query_population=False,ready_for_fold1=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,
                                           truth_opened=False,ready_for_fold1=False)); raise


def evaluate(output):
    dest=output/'evaluation.json'
    if dest.exists(): raise ValueError('Do not overwrite an evaluation')
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Complete sealed generation required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    generated=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids={r['query_id'] for r in launch['queries']}
    if len(generated)!=128 or {r['query_id'] for r in generated}!=ids: raise ValueError('Evaluation population drift')
    ev=v38.load(v38.EVAL,v38.EVAL_SHA); label_side=json.loads(verify(ev['label_side']).read_text())
    for ref in label_side['files'].values(): verify(ref)
    labels={r['query_id']:r for r in pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),
            filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    catalog={r['query_id']:r for r in pq.read_table(verify(ev['files']['official_catalog_sidecar.parquet']),
            filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    acceptable={}
    for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids: acceptable.setdefault(r['query_id'],set()).add(r['siret'])
    if set(labels)!=ids or set(catalog)!=ids or set(acceptable)!=ids: raise ValueError('Incomplete official label side')
    outcomes=[]
    for r in generated:
        label=labels[r['query_id']]
        if r['fold']!=label['fold'] or r['fold'] not in (2,3,4): raise ValueError('Evaluation fold drift')
        row=dict(query_id=r['query_id'],**{c:label[c] for c in v38.SEGMENTS})
        for mode in v38.MODES:
            candidates=r['results'][mode]; found={c['siret'] for c in candidates}
            if len(found)!=len(candidates) or len(found)>100: raise ValueError('Invalid initial cap')
            hit=label['human_ground_truth_siret'] in found
            row[mode+'_exact']=hit; row[mode+'_operational']=bool(found&acceptable[r['query_id']])
            row[mode+'_loss']='HIT' if hit else 'NOT_RETRIEVED_AT100' if catalog[r['query_id']]['official_catalog_present'] else 'OFFICIAL_CATALOG_ABSENT'
        outcomes.append(row)
    segments={}
    for column in v38.SEGMENTS:
        groups={}
        for r in outcomes:
            value=r[column]; key='UNKNOWN' if value is None or value=='' else str(value)
            groups.setdefault(key,[]).append(r)
        segments[column]={key:{mode:v38.summarize(group,mode) for mode in v38.MODES} for key,group in groups.items()}
    result=dict(status='MEASURED_LEARNED_INITIAL_INDEX_TOP100_PILOT',decision='PIVOT_GENERATION',
        primary=v38.summarize(outcomes,'primary'),comparator=v38.summarize(outcomes,'comparator'),segments=segments,
        gains=sum(r['primary_exact'] and not r['comparator_exact'] for r in outcomes),
        losses=sum(r['comparator_exact'] and not r['primary_exact'] for r in outcomes),
        loss_attribution={mode:dict(Counter(r[mode+'_loss'] for r in outcomes)) for mode in v38.MODES},
        latency={mode:dict(zip(('p50','p95','p99'),np.quantile([r['seconds'][mode] for r in generated],[.5,.95,.99]).tolist())) for mode in v38.MODES},
        latency_scope='Initial native search plus reference extraction; cache uncontrolled, not production SLO',
        excluded_missing_human_labels=ev['excluded_missing_human_labels'],generation=record(output/'manifest.json'),
        ledger=record(v38.LEDGER),label_side=ev['label_side'],outcomes=outcomes,fine_loss_attribution_measured=False,
        ready_for_fold1=False,fold0_opened=False,fold1_opened=False,model_used=True,full_query_population=False,
        metric_family='LEARNED_STRUCTURED_SPARSE_INITIAL_INDEX_SCORE',implementation=record(Path(__file__)))
    if result['primary']['exact_recall']>=.992:
        result['decision']='STOP'; result['decision_reason']='PILOT_ONLY_NO_FULL_OOF_GENERALIZATION_SLO_GATE'
    write_json(dest,result)
    print(json.dumps({k:result[k] for k in ('primary','comparator','gains','losses','latency','decision')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','training_worker','run','worker','evaluate'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--models',type=Path); p.add_argument('--start',type=int)
    args=p.parse_args()
    if args.mode=='train': train(args.output)
    elif args.mode=='run':
        if args.models is None: p.error('--models required')
        run(args.output,args.models)
    elif args.mode=='evaluate': evaluate(args.output)
    else:
        if args.start not in range(0,128,8): p.error('Invalid start')
        training_worker(args.output,args.start) if args.mode=='training_worker' else worker(args.output,args.start)
