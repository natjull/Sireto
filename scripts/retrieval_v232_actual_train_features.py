"""Actual V217 score decomposition on authorized TRAIN64 only; no recall."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v227_national_training_negatives as mining
from scripts.retrieval_v55_proof_metric import compatible
from src.xgb_matcher.retrieval_corroborated_initial_v217 import Evidence
from src.xgb_matcher.retrieval_training_score_v231 import native_features,fixed_scores,parity
from src.xgb_matcher.retrieval_sparse_query_v59 import expand
from src.xgb_matcher.retrieval_contrastive_v52 import surface
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles,identity
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v232_actual_train_features')
MINING=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v227_national_training_negatives/run_20261004_a')
STATUS='SEALED_V232_ACTUAL_INITIAL_SCORE_TRAIN64_FEATURES_NOT_RECALL'


def references(raw,source,mined):
    """Supervised TRAIN positives only; never a serving candidate list."""
    if raw['query_id']!=source['query_id'] or raw['query_id']!=mined['query_id'] or raw['fold']!=source['fold'] or raw['fold']!=mined['fold']:
        raise ValueError('Same selected TRAIN query required')
    result={source['positive_siret']:source['positive_siren']}
    for n in [*source['negatives'],*[c for values in mined['channels'].values() for c in values]]:
        s,p=identity(n)
        if s in result and result[s]!=p:raise ValueError('Official parent conflict')
        result[s]=p
    identity(dict(siret=source['positive_siret'],siren=source['positive_siren']))
    return dict(sorted(result.items()))


def implementation():
    refs=mining.implementation()
    for p in (Path(__file__),ROOT/'docs/retrieval_v232_actual_train_features_contract.md',
        ROOT/'src/xgb_matcher/retrieval_training_score_v231.py',
        ROOT/'src/xgb_matcher/retrieval_corroborated_initial_v217.py',
        ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',ROOT/'src/xgb_matcher/retrieval_contrastive_v52.py',
        ROOT/'scripts/retrieval_v55_proof_metric.py'):
        refs[str(p)]=record(p)
    return refs


def valid(rows,queries):
    if len(rows)!=len(queries):raise ValueError('Complete expected TRAIN observations required')
    for r,q in zip(rows,queries):
        if r['query_id']!=q['query_id'] or r['fold']!=q['fold'] or not 0<=r['seconds']<=20:
            raise ValueError('Original query and20s feature observation required')
        refs=r['references'];identity(dict(siret=r['positive_siret'],siren=r['positive_siren']))
        if refs.get(r['positive_siret'])!=r['positive_siren']:raise ValueError('Explicit supervised positive reference required')
        for s,p in refs.items():identity(dict(siret=s,siren=p))
        x=np.asarray(r['features'])
        if x.shape!=(len(refs),11) or not np.isfinite(x).all() or (x<0).any() or set(r['fixed_scores'])!={'2','3','4'}:
            raise ValueError('Actual full finite nonnegative feature matrix required')
        if set(r['maximum_native_parity_error'])!={'2','3','4'}:raise ValueError('Every fold head native parity required')
        for f,y in r['fixed_scores'].items():
            y=np.asarray(y)
            if y.shape!=(len(refs),) or not np.isfinite(y).all() or (y<0).any() or not np.isfinite(r['maximum_native_parity_error'][f]):
                raise ValueError('Actual full finite fixed contribution and parity required')


def load(root):
    if not root.resolve().is_relative_to(OUTPUT) or (root/'status.json').exists() or any(root.glob('batch*/supervisor_stop.json')):
        raise ValueError('Fresh non-STOP TRAIN feature namespace required')
    l=json.loads((root/'launch.json').read_text())
    for ref in l['implementation'].values():verify(ref)
    if l['truth_scope']!='TRAINER_ONLY' or l['resources']!=LIMITS or l['recall_measured'] is not False:
        raise ValueError('Supervised TRAIN only, no evaluation of injected points')
    if l['mining']!=record(MINING/'manifest.json'):raise ValueError('Same complete national TRAIN miner required')
    mining.sealed(MINING)
    return l


def worker(root,batch):
    l=load(root);dest=root/f'batch{batch:03d}';rows=[]
    if batch not in range(8):raise ValueError('Eight complete TRAIN batches only')
    with Budget(LIMITS) as budget:
        def guard():
            if budget.failure:raise RuntimeError(budget.failure)
        files=SourceFiles(guard)
        authority=json.loads(files.file(json.loads((MINING/'manifest.json').read_text())['training_authority']).read_text())
        proof=json.loads(files.file(authority['source_proof']).read_text());old=json.loads(files.file(authority['source_launch']).read_text())
        raw=json.loads(files.file(authority['raw_queries']).read_text())
        if mining.project(old['training_rows'],[])!=raw:raise ValueError('Same label-blind raw64 projection')
        sources={r['query_id']:r for r in old['training_rows']};mined={r['query_id']:r for r in mining.sealed(MINING)['rows']}
        cfg=json.loads(files.file(record(mining.SOURCE)).read_text())
        vp=json.loads(files.file(cfg['vector_source']).read_text());vl=json.loads(files.file(vp['launch']).read_text())
        vw=json.loads(files.file(vp['worker']).read_text());inputs=json.loads(files.file(vl['inputs']).read_text())
        if vw['query_ids']!=[r['query_id'] for r in old['training_rows']]:raise ValueError('All512 cached name-vector order required')
        for r in raw:
            if inputs['query_fields'][r['query_id']]['name']!=surface([r['crm_name']],'nom')[0]:raise ValueError('Cached encoder actual raw name mismatch')
        query_vectors=np.load(files.file(vw['query_vectors']),allow_pickle=False)
        if query_vectors.shape!=(512,768) or not np.isfinite(query_vectors).all():raise ValueError('Actual512 structured vectors required')
        gp=json.loads(files.file(proof['cached_proof_source']).read_text());gl=json.loads(files.file(gp['launch']).read_text())
        geos={};ids=[r['query_id'] for r in raw];rawbyid={r['query_id']:r for r in raw}
        for ref in gl['geometry_sources']:
            for g in pq.read_table(files.file(ref),filters=[('fold','in',[2,3,4]),('query_id','in',ids)],use_threads=False).to_pylist():
                q=rawbyid[g['query_id']]
                if g['fold']!=q['fold'] or g['query_id'] in geos:raise ValueError('Distinct matching TRAIN BAN source')
                value=compatible(g,q)
                if value!=gl['geo'][q['query_id']]:raise ValueError('Cached TRAIN geography must equal original query-only BAN')
                geos[q['query_id']]=value
        if set(geos)!=set(ids):raise ValueError('All64 natural CRM geography required, no target-inferred XY')
        ml=json.loads((MINING/'launch.json').read_text());models=json.loads(files.file(ml['models']).read_text());mining.base.validate_models(models)
        vocab={f:json.loads(files.file(m['vocabulary']).read_text()) for f,m in models.items()}
        projections={f:np.load(files.file(m['projection']),allow_pickle=False) for f,m in models.items()}
        _,source,_,_,_=mining.base.backend.inputs();evidence=Evidence(source,files)
        for pos in range(batch*8,batch*8+8):
            r=raw[pos];t=time.monotonic()
            def query_guard():
                guard()
                if time.monotonic()-t>20:raise TimeoutError('STOP_ACTUAL_TRAIN_FEATURE_QUERY_20S')
            files.guard=query_guard
            native=TantivyBackend(mining.base.backend.v38.INDEX)
            if native.searcher.num_docs!=old['index_documents']:raise ValueError('Actual national native cardinality')
            q=mining.as_query(r);refs=references(r,sources[r['query_id']],mined[r['query_id']]);ordered=list(refs)
            # Check existence separately: a zero-scoring point is not proof of presence.
            tv=native._tantivy
            check=tv.Query.boolean_query([(tv.Occur.Must,tv.Query.term_set_query(native.index.schema,'siret',ordered)),
                (tv.Occur.Must,tv.Query.term_query(native.index.schema,'document_type','siret','basic'))])
            present={}
            for _,loc in native.searcher.search(check,limit=len(ordered),count=False).hits:
                doc=native.searcher.doc(loc).to_dict();present[native._first(doc,'siret')]=native._first(doc,'siren')
            if present!=refs:raise ValueError('Missing/conflicting TRAIN reference: STOP, never silently omit')
            parents,sites,_=evidence.name_owners(q.name);related,_=evidence.related(parents,sites)
            rings,_=evidence.radial_proofs(geos[r['query_id']])
            ordered,x=native_features(native,q,ordered,query_guard);fixed={};errors={}
            for f,m in models.items():
                terms=expand(query_vectors[pos,:384],projections[f],vocab[f],r['crm_name'],8)
                y=fixed_scores(native,q,ordered,terms,m['alpha'],parents,sites,related,rings,query_guard)
                errors[f]=parity(native,q,ordered,m['weights'],terms,m['alpha'],parents,sites,related,rings,x,y,query_guard)
                fixed[f]=y.tolist()
            query_guard()
            rows.append(dict(query_id=r['query_id'],fold=r['fold'],positive_siret=sources[r['query_id']]['positive_siret'],
                positive_siren=sources[r['query_id']]['positive_siren'],references=refs,features=x.tolist(),fixed_scores=fixed,
                maximum_native_parity_error=errors,seconds=time.monotonic()-t))
            print(json.dumps(dict(query_id=r['query_id'],points=len(ordered),seconds=rows[-1]['seconds'],parity=errors)),flush=True)
            native=None;files.guard=guard;gc.collect();guard()
        evidence.close();files.verify_after();load(root);valid(rows,raw[batch*8:batch*8+8])
        write_json(dest/'payload.json',dict(rows=rows,source_epochs=list(files.entries.values()),truth_scope='TRAINER_ONLY',recall_measured=False))
    validate_report(budget.report())
    write_json(dest/'manifest.json',dict(resources=budget.report(),launch=record(root/'launch.json'),
        payload=record(dest/'payload.json'),preflight=record(dest/'preflight.json')))


def admission(root):
    l=load(root);rows=[];epochs=[];producers=[]
    raw=json.loads(verify(json.loads((MINING/'launch.json').read_text())['raw_queries']).read_text())
    for b in range(8):
        dest=root/f'batch{b:03d}';parent_exit(dest);m=json.loads((dest/'manifest.json').read_text());validate_report(m['resources'])
        gate=json.loads(verify(m['preflight']).read_text());p=json.loads(verify(m['payload']).read_text())
        if m['launch']!=record(root/'launch.json') or m['payload']!=record(dest/'payload.json') or m['preflight']!=record(dest/'preflight.json') or gate['pass_system_quiet_policy'] is not True or gate['quiet_seconds']<30:
            raise ValueError('Own successful supervised TRAIN feature batch')
        if [(r['query_id'],r['fold']) for r in p['rows']]!=[(r['query_id'],r['fold']) for r in raw[8*b:8*b+8]]:
            raise ValueError('Every original selected TRAIN control')
        valid(p['rows'],raw[8*b:8*b+8])
        if p['truth_scope']!='TRAINER_ONLY' or p['recall_measured'] is not False:raise ValueError('No product recall on TRAIN point lists')
        rows.extend(p['rows']);epochs.extend(p['source_epochs']);producers.append(record(dest/'manifest.json'))
    mining.base.check_epochs(epochs)
    return dict(rows=rows,source_epochs=epochs,producers=producers)


def sealed(root):
    value=admission(root);m=json.loads((root/'manifest.json').read_text())
    if m['status']!=STATUS or m['launch']!=record(root/'launch.json') or m['outputs']!=record(root/'outputs.json') or m['producers']!=value['producers'] or json.loads(verify(m['outputs']).read_text())!=value:
        raise ValueError('Whole successful64 actual TRAIN features required')
    return value


def run(root):
    mining.sealed(MINING)
    if root.exists() or not root.resolve().is_relative_to(OUTPUT):raise ValueError('Fresh dedicated232 TRAIN root')
    root.mkdir(parents=True)
    write_json(root/'launch.json',dict(implementation=implementation(),mining=record(MINING/'manifest.json'),resources=LIMITS,
        truth_scope='TRAINER_ONLY',recall_measured=False,fold0_opened=False,fold1_opened=False))
    try:
        for b in range(8):
            dest=root/f'batch{b:03d}';dest.mkdir();write_json(dest/'preflight.json',quiet(30,50))
            with (dest/'worker.log').open('x') as log:
                s=supervise([sys.executable,'-u','-m','scripts.retrieval_v232_actual_train_features','worker','--output',str(root),'--batch',str(b)],
                    log,{**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},dest/'supervisor_stop.json')
            write_json(dest/'supervision.json',s);parent_exit(dest)
            print(json.dumps(dict(completed=(b+1)*8,scope='ACTUAL_INITIAL_SCORE_TRAIN_ONLY')),flush=True)
        value=admission(root);write_json(root/'outputs.json',value)
        write_json(root/'manifest.json',dict(status=STATUS,launch=record(root/'launch.json'),outputs=record(root/'outputs.json'),
            producers=value['producers'],queries=64,recall_measured=False,ready_for_fold1=False))
        sealed(root)
    except BaseException as exc:
        write_json(root/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--batch',type=int);a=p.parse_args()
    run(a.output) if a.mode=='run' else worker(a.output,a.batch)
