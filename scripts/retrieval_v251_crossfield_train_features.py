"""Actual V217 score decomposition on authorized TRAIN64 only; no recall."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

from scripts import retrieval_v227_national_training_negatives as mining
from scripts import retrieval_v232_actual_train_features as previous_features
from src.xgb_matcher.retrieval_crossfield_proofs_v250 import train_points,conditional_parity
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles,identity
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget,LIMITS,quiet,validate_report,parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,verify,write_json

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v251_actual_train_features')
SOURCE=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v232_actual_train_features/run_20261004_a')
MINING=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v227_national_training_negatives/run_20261004_a')
STATUS='SEALED_V251_ACTUAL_INITIAL_SCORE_TRAIN64_FEATURES_NOT_RECALL'


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
    refs=previous_features.implementation()
    for p in (Path(__file__),ROOT/'docs/retrieval_v251_crossfield_train_contract.md',
        ROOT/'src/xgb_matcher/retrieval_crossfield_proofs_v250.py'):
        refs[str(p)]=record(p)
    return refs


def valid(rows,queries):
    for r in rows:
        error=conditional_parity(r['features'],r['extra_features'])
        if error!=r['conditional_parity_error']:raise ValueError('All native conditional parity recorded')
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
    previous_features.sealed(SOURCE)
    if l['source_features']!=record(SOURCE/'manifest.json'):raise ValueError('Same sealed complete232 features')
    return l


def worker(root,batch):
    from copy import deepcopy
    l=load(root);dest=root/f'batch{batch:03d}';rows=[]
    if batch not in range(8):raise ValueError('Eight complete TRAIN batches only')
    with Budget(LIMITS) as budget:
        def guard():
            if budget.failure:raise RuntimeError(budget.failure)
        files=SourceFiles(guard)
        sm=json.loads(files.file(l['source_features']).read_text())
        data=json.loads(files.file(sm['outputs']).read_text())
        oldrows={r['query_id']:r for r in data['rows']}
        ml=json.loads(files.file(record(MINING/'launch.json')).read_text())
        raw=json.loads(files.file(ml['raw_queries']).read_text())
        if len(raw)!=64 or set(oldrows)!={r['query_id'] for r in raw}:raise ValueError('Same full64 TRAIN scope')
        # Query-only sources and sealed232 supervised references, no128GT access.
        _,source,old,_,_=mining.base.backend.inputs()
        files.file(old['index_manifest']);files.file(old['index_meta'])
        for pos in range(batch*8,batch*8+8):
            r=raw[pos];t=time.monotonic();observed=deepcopy(oldrows[r['query_id']])
            def query_guard():
                guard()
                if time.monotonic()-t>20:raise TimeoutError('STOP_CROSSFIELD_TRAIN_QUERY_20S')
            files.guard=query_guard
            native=TantivyBackend(mining.base.backend.v38.INDEX)
            if native.searcher.num_docs!=old['index_documents']:raise ValueError('Same actual national native cardinality')
            q=mining.as_query(r);ordered=sorted(observed['references']);tv=native._tantivy
            check=tv.Query.boolean_query([(tv.Occur.Must,tv.Query.term_set_query(native.index.schema,'siret',ordered)),
                (tv.Occur.Must,tv.Query.term_query(native.index.schema,'document_type','siret','basic'))])
            present={}
            for _,loc in native.searcher.search(check,limit=len(ordered),count=False).hits:
                doc=native.searcher.doc(loc).to_dict();present[native._first(doc,'siret')]=native._first(doc,'siren')
            if present!=observed['references']:raise ValueError('Every natural supervised TRAIN reference must still exist')
            y=train_points(native,q,ordered)
            error=conditional_parity(observed['features'],y)
            query_guard()
            observed.update(extra_features=y.tolist(),conditional_parity_error=error,seconds=time.monotonic()-t)
            rows.append(observed)
            print(json.dumps(dict(query_id=r['query_id'],points=len(ordered),seconds=observed['seconds'],parity=error)),flush=True)
            native=None;files.guard=guard;gc.collect();guard()
        files.verify_after();load(root);valid(rows,raw[batch*8:batch*8+8])
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
    write_json(root/'launch.json',dict(implementation=implementation(),mining=record(MINING/'manifest.json'),source_features=record(SOURCE/'manifest.json'),resources=LIMITS,
        truth_scope='TRAINER_ONLY',recall_measured=False,fold0_opened=False,fold1_opened=False))
    try:
        for b in range(8):
            dest=root/f'batch{b:03d}';dest.mkdir();write_json(dest/'preflight.json',quiet(30,50))
            with (dest/'worker.log').open('x') as log:
                s=supervise([sys.executable,'-u','-m','scripts.retrieval_v251_crossfield_train_features','worker','--output',str(root),'--batch',str(b)],
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
