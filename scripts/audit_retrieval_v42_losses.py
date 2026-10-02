"""Post-seal label-side diagnostics. Never called by the generator/trainer."""
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
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json


def run(output):
    destination=output/'loss_diagnostics.json'
    if destination.exists(): raise ValueError('Do not overwrite diagnostics')
    evaluation=json.loads((output/'evaluation.json').read_text())
    manifest=json.loads(verify(evaluation['generation']).read_text())
    if (manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128
            or not evaluation['model_used'] or evaluation['full_query_population']):
        raise ValueError('Complete sealed pilot and evaluation required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    rows=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    generated={r['query_id']:r for r in rows}
    queries={r['query_id']:r for r in launch['queries']}
    ids=set(queries)
    labels={r['query_id']:r for r in pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),
            filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    if len(generated)!=128 or set(generated)!=ids or set(labels)!=ids:
        raise ValueError('Diagnostic denominator drift')
    models=json.loads(verify(launch['models']).read_text())
    misses=[r for r in evaluation['outcomes'] if not r['primary_exact']]
    diagnostics=[]; began=time.monotonic()
    with Budget(v42.RESOURCE) as budget:
        backend=TantivyBackend(v38.INDEX)
        for outcome in misses:
            qid=outcome['query_id']; label=labels[qid]; raw=queries[qid]
            query=v42.as_query(raw); gt=label['human_ground_truth_siret']
            weights=np.asarray(models[str(raw['fold'])]['weights'])
            tv=backend._tantivy
            identity=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'siret',gt,'basic'),0.)
            lookup=backend.searcher.search(identity,limit=2,count=False).hits
            if len(lookup)>1: raise ValueError('Duplicate official index document')
            values=[]
            for part in v42.components(backend,query):
                if part is None: values.append(0.); continue
                parsed=tv.Query.boolean_query([(tv.Occur.Must,identity),(tv.Occur.Must,part)])
                hits=backend.searcher.search(parsed,limit=2,count=False).hits
                if len(hits)>1: raise ValueError('Duplicate truth component')
                values.append(float(hits[0][0]) if hits else 0.)
            leaf_score=backend.searcher.search(tv.Query.boolean_query([(tv.Occur.Must,identity),
                (tv.Occur.Must,v42.index_query(backend,query,weights))]),limit=2,count=False).hits
            if len(leaf_score)>1: raise ValueError('Duplicate initial-score truth')
            eligible=bool(leaf_score)
            positive_score=float(leaf_score[0][0]) if eligible else None
            found=generated[qid]['results']['primary']
            cutoff=found[-1]['score'] if len(found)==100 else None
            reason=('INDEX_DOCUMENT_ABSENT' if not lookup else 'TEXT_PREDICATE_INELIGIBLE' if not eligible
                    else 'TIE_AT_CAP' if cutoff is not None and abs(positive_score-cutoff)<1e-5
                    else 'BELOW_INITIAL_SCORE_CUTOFF')
            if eligible and not np.isclose(positive_score,np.asarray(values)@weights,rtol=1e-5,atol=1e-5):
                raise ValueError('Compiled score/point diagnostic mismatch')
            if eligible and cutoff is not None and positive_score>cutoff+1e-4:
                raise ValueError('Truth above sealed initial cutoff')
            same_parent=sum(r['siren']==label['human_ground_truth_siren'] for r in found)
            geography={}
            if lookup:
                doc=backend.searcher.doc(lookup[0][1]).to_dict()
                siren=backend._first(doc,'siren')
                if siren!=label['human_ground_truth_siren']: raise ValueError('Official membership drift')
                geography=dict(insee_match=bool(query.insee and backend._first(doc,'insee')==query.insee),
                               postcode_match=bool(query.postcode and backend._first(doc,'postcode')==query.postcode))
            diagnostics.append(dict(query_id=qid,fold=raw['fold'],reason=reason,official_index_present=bool(lookup),
                initial_predicate_eligible=eligible,features=dict(zip(v42.FEATURES,values)),
                official_place_matches=geography,truth_initial_score=positive_score,cutoff=cutoff,
                same_parent_candidates=same_parent,parent_present=bool(same_parent),
                address_matches_but_learned_weight_zero=bool(values[1] and weights[1]==0 and weights[3]==0 and weights[4]==0 and weights[5]==0)))
    write_json(destination,dict(status='SEALED_POST_EVALUATION_LOSS_DIAGNOSTICS',queries=128,misses=len(misses),
        evaluation=record(output/'evaluation.json'),generation=evaluation['generation'],ledger=record(v38.LEDGER),
        reasons=dict(Counter(r['reason'] for r in diagnostics)),
        parent_present=sum(r['parent_present'] for r in diagnostics),
        outside_declared_place=sum(not any(r['official_place_matches'].values()) for r in diagnostics),
        address_matches_but_learned_weight_zero=sum(r['address_matches_but_learned_weight_zero'] for r in diagnostics),
        diagnostics=diagnostics,resources=budget.report(),seconds=time.monotonic()-began,
        label_side_only=True,never_used_in_generation=True,implementation=record(Path(__file__)),ready_for_fold1=False))
    print(json.dumps({k:json.loads(destination.read_text())[k] for k in
                     ('misses','reasons','parent_present','outside_declared_place','address_matches_but_learned_weight_zero')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
