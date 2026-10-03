"""Evaluation-only point audit after ALL128 outputs sealed; never generation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v46_joint as v46
from scripts.retrieval_v91_scoped_paths import official_sources
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings
from src.xgb_matcher.retrieval_scoped_paths_v90 import OwnerFacts,SuccessionCSR,typed_position
from src.xgb_matcher.retrieval_scoped_paths_v87 import ScopedPaths,literal_scores,NativeStream
from src.xgb_matcher.retrieval_sparse_native_v60 import query as native_query
from src.xgb_matcher.retrieval_joint_v46 import geo_features,geo_parameters
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json


def validate_population(manifest,evaluation,generation,queries,rows):
    """Fail closed before opening labels; duplicates cannot hide behind a dict."""
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Complete blind generation required before label-side audit')
    if evaluation['generation']!=record(generation/'manifest.json'):
        raise ValueError('Evaluated generation drift')
    v46.validate_queries(queries)
    if (len(rows)!=128 or [(r['query_id'],r['fold']) for r in rows]!=
            [(q['query_id'],q['fold']) for q in queries]):
        raise ValueError('Incomplete certified population')
    for row in rows:
        result=row['results']['primary']
        if (not row['primary_trace']['certified'] or len(result)>100 or
            len({r['siret'] for r in result})!=len(result) or
            any(not np.isfinite(r['score']) or r['score']<=0 for r in result) or
            any(a['score']<b['score'] for a,b in zip(result,result[1:]))):
            raise ValueError('Initial100 certificate/output drift')


def check_point(result,gt,full):
    """A post-seal point lookup must agree with the serving certificate."""
    if not np.isfinite(full) or full<0: raise ValueError('Nonfinite point metric')
    hit=next((r for r in result if r['siret']==gt),None)
    cutoff=result[-1]['score'] if len(result)==100 else 0.
    if hit is None and full>cutoff:
        raise ValueError('STOP_CERTIFICATE_POINT_AUDIT_CONTRADICTION')
    if hit is not None and full!=hit['score']:
        raise ValueError('STOP_EMITTED_POINT_METRIC_CONTRADICTION')
    return hit is not None,cutoff


def branch_reason(proof,query_geo_available,xy_available):
    """Describes only this added branch, not all-source absence or causality."""
    if proof['outside_graph_scope']: return 'OUTSIDE_V84_ENDPOINT_SCOPE'
    if not proof['score']: return 'NO_MATCHING_WORD_PROOF_IN_SCOPED_SOURCE'
    if not query_geo_available: return 'QUERY_BAN_UNAVAILABLE_OR_INCOMPATIBLE'
    if not xy_available: return 'TARGET_XY_ABSENT'
    return 'POSITIVE_SCOPED_WORD_SITE_PROOF'


def run(generation,output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    # Separate evaluator artifact, never read/exported by generation workers.
    manifest=json.loads((generation/'manifest.json').read_text())
    evaluation=json.loads((generation/'evaluation.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Complete blind generation required before label-side audit')
    if evaluation['generation']!=record(generation/'manifest.json'):
        raise ValueError('Evaluated generation drift')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    queries=launch['queries']
    generated_rows=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    validate_population(manifest,evaluation,generation,queries,generated_rows)
    emitted={r['query_id']:r for r in generated_rows}
    labels={r['query_id']:r for r in pq.read_table(verify(evaluation['ledger']),filters=[('fold','in',[2,3,4])]).to_pylist()
            if r['query_id'] in emitted}
    if set(labels)!=set(emitted): raise ValueError('All-human audit denominator drift')
    models=json.loads(verify(launch['models']).read_text())
    terms={r['query_id']:r['terms'] for r in json.loads(verify(launch['expansions']).read_text())['rows']}
    geos={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
    rows=[]
    try:
        with Budget(dict(max_seconds=80,max_rss_bytes=6442450944,max_swapout_pages_delta=0)) as budget:
            words,_=official_sources(launch); index=WordPostings(words['postings'])
            owners=OwnerFacts(Path(launch['word_manifest']['path']).parent)
            graph=SuccessionCSR(Path(launch['graph_manifest']['path']).parent)
            tree,keys,_=v46.load_tree(launch); base=TantivyBackend(v38.INDEX)
            for q in queries:
                label=labels[q['query_id']]; gt=label['human_ground_truth_siret']; parent=label['human_ground_truth_siren']
                if label['fold']!=q['fold'] or not gt.startswith(parent): raise ValueError('Official label membership/fold drift')
                scores,_=literal_scores(index,q['crm_name']); paths=ScopedPaths(owners,graph,scores)
                proof=paths.best(gt); g=geos[q['query_id']]; point,_,reliable=geo_parameters(g)
                factor=1. if reliable else .5
                pos=typed_position(keys,int(gt)); xy_available=pos<len(keys) and keys[pos]==int(gt)
                site=float(geo_features(keys,tree.data,[gt],g)[0,0]*factor)
                m=models[str(q['fold'])]; native=NativeStream(base,native_query(base,v42.as_query(q),m['weights'],terms[q['query_id']],m['alpha']))
                native_score=native.complete([gt])[gt]; graph_score=launch['path_weight']*proof['score']*site
                full=max(native_score,graph_score); result=emitted[q['query_id']]['results']['primary']
                hit,cutoff=check_point(result,gt,full)
                reason=branch_reason(proof,point is not None,xy_available)
                rows.append(dict(query_id=q['query_id'],fold=q['fold'],human_siret=gt,human_siren=parent,
                    exact_hit=hit,branch_zero_or_positive_reason=reason,native_score=native_score,
                    path_name_score=proof['score'],site_score=site,graph_score=graph_score,initial_score=full,
                    initial100_cutoff=cutoff,target_xy_available=bool(xy_available),query_geo_available=point is not None,
                    ban_reliability_factor=factor,scoped_name_proof=proof,
                    own_site_name_facts=len(owners.references('site',int(gt))),
                    scoped_parent_name_facts=len(owners.references('parent',int(parent))),
                    target_score_at_boundary=full==cutoff))
        v38.verify_launch(launch); official_sources(launch)
        if len(rows)!=128 or sum(r['exact_hit'] for r in rows)!=evaluation['primary']['exact_hits']:
            raise ValueError('Point audit population/hits drift')
        write_json(output/'report.json',dict(status='SEALED_EVALUATION_ONLY_POINT_PROOF_AUDIT',queries=128,
            misses=sum(not r['exact_hit'] for r in rows),rows=rows,
            miss_branch_reasons=dict(Counter(r['branch_zero_or_positive_reason'] for r in rows if not r['exact_hit'])),
            loss_attribution_scope='THIS_SCOPED_BRANCH_ONLY_NOT_ABSENCE_OF_ANY_OFFICIAL_SOURCE_EVIDENCE',
            generation=record(generation/'manifest.json'),evaluation=record(generation/'evaluation.json'),
            implementation=record(Path(__file__)),resources=budget.report(),seconds=time.monotonic()-began,
            truth_opened=True,truth_scope='EVALUATOR_ONLY_AFTER_SEALED128',fold0_opened=False,fold1_opened=False,
            generation_modified=False,model_fit=False,outputs_promoted_to_features=False,ready_for_fold1=False))
        print(json.dumps(dict(queries=len(rows),miss_branch_reasons=Counter(r['branch_zero_or_positive_reason'] for r in rows if not r['exact_hit']),resources=budget.report())),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_scope='EVALUATOR_ONLY',generation_modified=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--generation',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path); a=p.parse_args(); run(a.generation,a.output)
