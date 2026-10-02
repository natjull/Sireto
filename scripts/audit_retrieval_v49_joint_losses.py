"""Post-seal official point-score audit; never an input to candidate search."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_joint_v46 import geo_features,geo_parameters
from src.xgb_matcher.retrieval_joint_v48 import index_query
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json

RESOURCE=dict(max_seconds=80,max_rss_bytes=6*1024**3,max_swapout_pages_delta=0)


def reason(present,text,geo,score,cutoff):
    if not present: return 'INDEX_DOCUMENT_ABSENT'
    if not text and not geo: return 'NO_TEXT_OR_COMPATIBLE_GEO_SIGNAL'
    if cutoff is None: raise ValueError('Missing complete100 cutoff')
    if score>cutoff+1e-4: raise ValueError('Truth above certified sealed cutoff')
    return 'TIE_AT_CAP' if abs(score-cutoff)<1e-5 else 'BELOW_JOINT_SCORE_CUTOFF'


def run(output):
    dest=output/'joint_loss_diagnostics.json'
    if dest.exists(): raise ValueError('Do not overwrite diagnostics')
    evref=record(output/'evaluation.json'); evaluation=json.loads(verify(evref).read_text())
    manifest=json.loads(verify(evaluation['generation']).read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Complete sealed generation/evaluation required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    rawrows=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    rows={r['query_id']:r for r in rawrows}; queries={r['query_id']:r for r in launch['queries']}
    outcomes={r['query_id']:r for r in evaluation['outcomes']}
    ids=set(queries)
    if len(rawrows)!=128 or set(rows)!=ids or set(outcomes)!=ids or not all(r['primary_trace']['certified'] for r in rows.values()):
        raise ValueError('Complete certified denominator required')
    models=json.loads(verify(launch['models']).read_text())
    geos={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
    labels={r['query_id']:r for r in pq.read_table(verify(evaluation['ledger']),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    if set(labels)!=ids or set(geos)!=ids: raise ValueError('Label-side denominator drift')
    files=launch['spatial_files']
    for ref in files.values(): verify(ref)
    keys=np.load(files['keys.npy']['path'],mmap_mode='r',allow_pickle=False)
    xy=np.load(files['xy.npy']['path'],mmap_mode='r',allow_pickle=False)
    diagnostics=[]; began=time.monotonic()
    with Budget(RESOURCE) as budget:
        backend=TantivyBackend(v38.INDEX); tv=backend._tantivy
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Index count drift')
        for qid,outcome in outcomes.items():
            if outcome['primary_exact']: continue
            raw=queries[qid]; label=labels[qid]; g=geos[qid]
            if raw['fold']!=label['fold'] or raw['fold'] not in (2,3,4): raise ValueError('Fold drift')
            q=v42.as_query(raw); gt=label['human_ground_truth_siret']; w=models[str(raw['fold'])]['weights']
            identity=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'siret',gt,'basic'),0.)
            lookup=backend.searcher.search(identity,limit=2,count=False).hits
            if len(lookup)>1: raise ValueError('Duplicate source identity')
            point_query=tv.Query.boolean_query([(tv.Occur.Must,identity),(tv.Occur.Must,index_query(backend,q,w[:8]))])
            native=backend.searcher.search(point_query,limit=2,count=False).hits
            if len(native)>1: raise ValueError('Duplicate point score')
            lexical=float(native[0][0]) if native else 0.
            _,rawfeatures=v42.training_features(backend,q,[gt])
            prox=float(geo_features(keys,xy,[gt],g)[0,0]); _,sigma,reliable=geo_parameters(g)
            geo_score=prox*(w[8]+w[9]*reliable); score=lexical+geo_score
            candidates=rows[qid]['results']['primary']; cutoff=candidates[-1]['score'] if len(candidates)==100 else None
            present=bool(lookup); text=bool(rawfeatures[0,0] or rawfeatures[0,1]); why=reason(present,text,prox>0,score,cutoff)
            official_names={}; official_addresses={}
            if lookup:
                doc=backend.searcher.doc(lookup[0][1]).to_dict()
                if backend._first(doc,'siren')!=label['human_ground_truth_siren']: raise ValueError('Official parent drift')
                official_names={k:doc.get(k,[]) for k in v38.NAMES}
                official_addresses={k:doc.get(k,[]) for k in ('addresses','historical_addresses','supporting_addresses')}
            same_parent=sum(n['siren']==label['human_ground_truth_siren'] for n in candidates)
            diagnostics.append(dict(query_id=qid,fold=raw['fold'],reason=why,index_present=present,text_eligible=text,
                geo_signal=bool(prox),geo_score=geo_score,lexical_score=lexical,joint_score=score,cutoff=cutoff,
                distance_m=sigma*(1/prox-1) if prox else None,ban_declared_weak=g['ban_declared_weak'],
                raw_lexical_components=dict(zip(v42.FEATURES,rawfeatures[0].tolist())),
                same_parent_candidates=same_parent,official_index_names=official_names,official_index_addresses=official_addresses))
    v38.verify_launch(launch); verify(evref)
    report=dict(status='SEALED_POST_EVALUATION_JOINT_LOSS_DIAGNOSTICS',queries=128,misses=len(diagnostics),
        reasons=dict(Counter(r['reason'] for r in diagnostics)),parent_present=sum(r['same_parent_candidates']>0 for r in diagnostics),
        no_text=sum(not r['text_eligible'] for r in diagnostics),no_geo_signal=sum(not r['geo_signal'] for r in diagnostics),
        diagnostics=diagnostics,seconds=time.monotonic()-began,resources=budget.report(),evaluation=evref,
        generation=evaluation['generation'],ledger=evaluation['ledger'],label_side=evaluation['label_side'],
        label_side_only=True,never_used_in_generation=True,ready_for_fold1=False,implementation=record(Path(__file__)))
    write_json(dest,report)
    print(json.dumps({k:report[k] for k in ('misses','reasons','parent_present','no_text','no_geo_signal','seconds')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output)
