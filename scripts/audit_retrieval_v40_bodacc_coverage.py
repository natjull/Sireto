"""Label-side coverage attribution after a sealed complete v40 evaluation.

Never used by the generator. Measures whether BODACC compiled any official
path to the exact human site, including sites missed by the v38 comparator.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v40_initial_top100 as v40
from src.xgb_matcher.retrieval_v20_workbench import record,verify,write_json


def audit(output):
    dest=output/'bodacc_label_side_coverage.json'
    if dest.exists(): raise ValueError('Do not overwrite label-side audit')
    manifest=json.loads((output/'manifest.json').read_text())
    evaluation=json.loads((output/'evaluation.json').read_text())
    if manifest['status']!='SEALED_V40_INITIAL_TOP100' or evaluation['status']!='MEASURED_V40_INITIAL_TOP100_PILOT':
        raise ValueError('Complete generation and evaluation required')
    verify(evaluation['generation'])
    launch=json.loads(verify(manifest['launch']).read_text()); v40.verify_launch(launch)
    ids={r['query_id'] for r in launch['queries']}
    outcomes={r['query_id']:r for r in evaluation['outcomes']}
    labels={r['query_id']:r for r in pq.read_table(verify(evaluation['ledger']),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    if set(outcomes)!=ids or set(labels)!=ids or len(ids)!=128: raise ValueError('Diagnostic population drift')
    graph=json.loads(verify(launch['graph_manifest']).read_text())
    source=json.loads(verify(graph['source_manifest']).read_text())
    proofs=verify(source['artifacts']['proof_paths.parquet'])
    targets=sorted({r['human_ground_truth_siret'] for r in labels.values()})
    counts=Counter(r['siret'] for r in pq.read_table(proofs,columns=['siret'],filters=[('siret','in',targets)]).to_pylist())
    rows=[]
    for query_id in sorted(ids):
        label=labels[query_id]; outcome=outcomes[query_id]
        if label['fold']!=outcome['fold'] or label['fold'] not in (2,3,4): raise ValueError('Diagnostic fold drift')
        rows.append(dict(query_id=query_id,fold=label['fold'],official_paths_to_exact_truth=counts[label['human_ground_truth_siret']],
            truth_in_bodacc_component=bool(counts[label['human_ground_truth_siret']]),
            v38_exact=outcome['comparator_exact'],v40_exact=outcome['primary_exact']))
    result=dict(status='SEALED_LABEL_SIDE_BODACC_COMPONENT_COVERAGE',queries=128,
        truth_in_bodacc_component=sum(r['truth_in_bodacc_component'] for r in rows),
        v38_misses=sum(not r['v38_exact'] for r in rows),
        v38_misses_with_bodacc_truth=sum(not r['v38_exact'] and r['truth_in_bodacc_component'] for r in rows),
        v40_misses_with_bodacc_truth=sum(not r['v40_exact'] and r['truth_in_bodacc_component'] for r in rows),
        rows=rows,evaluation=record(output/'evaluation.json'),ledger=evaluation['ledger'],
        official_proofs=source['artifacts']['proof_paths.parquet'],implementation=record(Path(__file__)),
        diagnostic_only=True,used_by_generator=False,fold0_opened=False,fold1_opened=False,
        predicate_or_rank_eligibility_measured=False,ready_for_fold1=False)
    write_json(dest,result)
    print(json.dumps({k:result[k] for k in ('truth_in_bodacc_component','v38_misses','v38_misses_with_bodacc_truth','v40_misses_with_bodacc_truth')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
