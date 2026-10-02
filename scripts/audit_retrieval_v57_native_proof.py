"""Post-seal native GT evidence audit only, never candidate generation."""
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
from src.xgb_matcher.retrieval_proof_v56 import point_features
from src.xgb_matcher.retrieval_native_v57 import LEXICAL,index_query
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json

def summarize_proof(features,score,cutoff,present,hit,same_parent):
    values=np.asarray(features)
    if values.shape!=(11,) or not np.isfinite(values).all(): raise ValueError('Native audit feature drift')
    violation=bool(present and not hit and cutoff is not None and score>cutoff+1e-5)
    return dict(index_present=present,exact_hit=hit,same_parent_in100=same_parent,
        name_coverage=float(values[0]),address_coverage=float(values[1]),
        current_exact_name=bool(values[2]),current_exact_address=bool(values[3]),
        place=bool(values[4]),number_at_place=bool(values[5]),
        no_name_proof=not bool(values[0] or values[2]),no_address_proof=not bool(values[1] or values[3]),
        no_text_proof=not bool(values[:4].any()),no_place_proof=not bool(values[4]),
        native_truth_score=score,output_cutoff=cutoff,native_top100_violation=violation,
        features=values.tolist())

def audit(output):
    dest=output/'native_proof_audit.json'
    if dest.exists(): raise ValueError('Do not overwrite audit')
    began=time.monotonic(); manifest=json.loads((output/'manifest.json').read_text())
    evaluation=json.loads((output/'evaluation.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Complete blind initial outputs required')
    if evaluation['generation']!=record(output/'manifest.json') or evaluation['metric_family']!=manifest['metric_family']:
        raise ValueError('Separate sealed evaluation authority drift')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    models=json.loads(verify(launch['models']).read_text())
    generated=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    byid={r['query_id']:r for r in generated}; ids={r['query_id'] for r in launch['queries']}
    labels={r['query_id']:r for r in pq.read_table(verify(evaluation['ledger']),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    if len(generated)!=128 or set(labels)!=ids or set(byid)!=ids: raise ValueError('Audit denominator drift')
    with Budget(launch['resources']) as budget:
        backend=TantivyBackend(v38.INDEX); rows=[]; tv=backend._tantivy
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Index cardinality drift')
        for raw in launch['queries']:
            query_id=raw['query_id']; label=labels[query_id]; generated_row=byid[query_id]
            if label['fold']!=raw['fold'] or raw['fold'] not in (2,3,4): raise ValueError('Audit fold drift')
            truth=label['human_ground_truth_siret']; q=v42.as_query(raw); m=models[str(raw['fold'])]
            doc=backend.by_siret(truth)
            if doc is not None and doc.siren!=truth[:9]: raise ValueError('Official parent drift')
            _,features=point_features(backend,q,[truth])
            gate=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',[truth]),0.)
            parsed=tv.Query.boolean_query([(tv.Occur.Must,index_query(backend,q,m['weights'])),(tv.Occur.Must,gate)])
            hits=backend.searcher.search(parsed,limit=1,count=False).hits; score=float(hits[0][0]) if hits else 0.
            found=generated_row['results']['primary']; present=doc is not None
            hit=truth in {r['siret'] for r in found}; same_parent=truth[:9] in {r['siren'] for r in found}
            proof=summarize_proof(features[0],score,found[-1]['score'] if found else None,present,hit,same_parent)
            rows.append(dict(query_id=query_id,fold=raw['fold'],**proof))
    v38.verify_launch(launch)
    misses=[r for r in rows if not r['exact_hit']]
    flags=('index_present','same_parent_in100','no_name_proof','no_address_proof','no_text_proof','no_place_proof','current_exact_name','current_exact_address','number_at_place','native_top100_violation')
    result=dict(status='SEALED_POST_EVALUATION_NATIVE_PROOF_AUDIT',queries=128,misses=len(misses),
        missing_flags={k:sum(r[k] for r in misses) for k in flags},
        all_query_flags={k:sum(r[k] for r in rows) for k in flags},
        native_top100_violations=sum(r['native_top100_violation'] for r in rows),rows=rows,
        resources=budget.report(),seconds=time.monotonic()-began,
        generation=record(output/'manifest.json'),evaluation=record(output/'evaluation.json'),
        ledger=evaluation['ledger'],implementation=record(Path(__file__)),
        fold0_opened=False,fold1_opened=False,model_parameters_changed=False,
        downstream_models_unchanged=True,ready_for_fold1=False,
        scope='POST_SEAL_POINT_EVIDENCE_NOT_EXHAUSTIVE_RAW_OFFICIAL_SOURCE_ATtribution')
    write_json(dest,result); print(json.dumps({k:v for k,v in result.items() if k not in ('rows','implementation','ledger','generation','evaluation')}))
    if result['native_top100_violations']: raise ValueError('Native score consistency failure')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    audit(p.parse_args().output)
