"""Separate sealed-output parity and post-evaluation coordinate diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json

RESOURCE=dict(max_seconds=80,max_rss_bytes=6*1024**3,max_swapout_pages_delta=0)


def sealed(output):
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Complete sealed generation required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    generated=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    rows={r['query_id']:r for r in generated}
    if len(generated)!=128 or len(rows)!=128 or set(rows)!={r['query_id'] for r in launch['queries']}:
        raise ValueError('Denominator drift')
    return manifest,launch,rows


def parity(output):
    dest=output/'physical_parity128.json'
    if dest.exists(): raise ValueError('Do not overwrite audit')
    with Budget(RESOURCE) as budget:
        manifest,launch,rows=sealed(output)
        old=json.loads(verify(launch['query_generation']).read_text())
        if old['status']!='SEALED_INITIAL_TOP100' or old['queries']!=128 or old['truth_opened']: raise ValueError('Comparator generation drift')
        reference={r['query_id']:r for ref in old['batches'] for r in json.loads(verify(ref).read_text())['rows']}
        if set(reference)!=set(rows): raise ValueError('Comparator population drift')
        matches=sum(rows[q]['results']['comparator']==reference[q]['results']['primary'] for q in rows)
        if matches!=128: raise ValueError('STOP_EXACT_OUTPUT_PARITY128')
    write_json(dest,dict(status='SEALED_EXACT_OUTPUT_PARITY128',queries=128,exact_output_matches=matches,
        comparison='Identical ordered SIRET,SIREN and float scores to frozen V42 initial outputs',
        truth_opened=False,resources=budget.report(),generation=record(output/'manifest.json'),
        baseline=launch['query_generation'],implementation=record(Path(__file__)),ready_for_fold1=False))
    print(json.dumps(dict(queries=128,exact_output_matches=matches,truth_opened=False)))


def diagnostics(output):
    dest=output/'spatial_diagnostics.json'
    if dest.exists(): raise ValueError('Do not overwrite diagnostics')
    with Budget(RESOURCE) as budget:
        manifest,launch,rows=sealed(output)
        equality=json.loads((output/'physical_parity128.json').read_text())
        if equality['status']!='SEALED_EXACT_OUTPUT_PARITY128' or equality['exact_output_matches']!=128:
            raise ValueError('Full physical parity required')
        verify(equality['generation'])
        if equality['generation']!=record(output/'manifest.json'): raise ValueError('Parity generation drift')
        evaluation=json.loads((output/'evaluation.json').read_text())
        verify(evaluation['generation'])
        if evaluation['generation']!=record(output/'manifest.json'): raise ValueError('Evaluation generation drift')
        ids=set(rows)
        labels={r['query_id']:r for r in pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),
            filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
        geo={r['query_id']:r for r in json.loads(verify(launch['geo_queries']).read_text())}
        if set(labels)!=ids or set(geo)!=ids: raise ValueError('Diagnostic denominator drift')
        files=launch['spatial_files']
        for ref in files.values(): verify(ref)
        xy=np.load(files['xy.npy']['path'],mmap_mode='r',allow_pickle=False)
        keys=np.load(files['keys.npy']['path'],mmap_mode='r',allow_pickle=False)
        outcomes={r['query_id']:r for r in evaluation['outcomes']}
        if set(outcomes)!=ids: raise ValueError('Evaluation outcome drift')
        diagnostics=[]
        for qid,r in rows.items():
            label=labels[qid]; g=geo[qid]; outcome=outcomes[qid]
            if label['fold']!=r['fold'] or r['fold'] not in (2,3,4): raise ValueError('Fold drift')
            gt=int(label['human_ground_truth_siret']); position=int(np.searchsorted(keys,gt))
            present=position<len(keys) and int(keys[position])==gt
            distance=None
            if present and g['spatial_crs_compatible']:
                delta=xy[position]-np.array([g['ban_x_l93'],g['ban_y_l93']])
                distance=float(np.sqrt(np.sum(delta*delta)))
            if outcome['primary_exact']: reason='HIT'
            elif r['primary_source']!='SPATIAL_DIRECT': reason='FROZEN_LEXICAL_FALLBACK_MISS'
            elif not present: reason='TRUTH_WITHOUT_COMPATIBLE_SOURCE_COORDINATES'
            else:
                boundary=r['spatial_trace']['boundary_radius_m']; last=r['results']['primary'][-1]
                if distance<boundary-1e-7: raise ValueError('Truth strictly inside sealed spatial cutoff but absent')
                reason='SPATIAL_BOUNDARY_TIE' if abs(distance-boundary)<1e-7 else 'OUTSIDE_SPATIAL_NEAREST100'
                if reason=='SPATIAL_BOUNDARY_TIE' and gt<int(last['siret']):
                    raise ValueError('Canonical spatial tie drift')
            diagnostics.append(dict(query_id=qid,fold=r['fold'],primary_source=r['primary_source'],
                exact_hit=outcome['primary_exact'],comparator_exact_hit=outcome['comparator_exact'],
                reason=reason,official_truth_coordinate_present=bool(present),ban_declared_weak=g['ban_declared_weak'],
                ban_street_similarity=g['ban_street_similarity'],distance_to_official_truth_m=distance,
                spatial_boundary_radius_m=r['spatial_trace'].get('boundary_radius_m'),
                spatial_boundary_tie_overflow=r['spatial_trace'].get('boundary_tie_overflow')))
    result=dict(status='SEALED_POST_EVALUATION_SPATIAL_DIAGNOSTICS',queries=128,
        primary_misses=sum(not r['exact_hit'] for r in diagnostics),
        loss_reasons=dict(Counter(r['reason'] for r in diagnostics if not r['exact_hit'])),
        query_spatial_resolved=sum(r['primary_source']=='SPATIAL_DIRECT' for r in diagnostics),
        truth_coordinates_present=sum(r['official_truth_coordinate_present'] for r in diagnostics),
        gains=sum(r['exact_hit'] and not r['comparator_exact_hit'] for r in diagnostics),
        losses=sum(r['comparator_exact_hit'] and not r['exact_hit'] for r in diagnostics),
        diagnostics=diagnostics,resources=budget.report(),label_side_only=True,never_used_in_generation=True,
        evaluation=record(output/'evaluation.json'),generation=record(output/'manifest.json'),
        parity=record(output/'physical_parity128.json'),ledger=record(v38.LEDGER),
        implementation=record(Path(__file__)),ready_for_fold1=False,
        distance_disagreement_does_not_identify_ban_error_or_label_error=True)
    write_json(dest,result)
    print(json.dumps({k:result[k] for k in ('queries','primary_misses','loss_reasons','query_spatial_resolved','truth_coordinates_present','gains','losses')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['parity','diagnostics'])
    p.add_argument('--output',required=True,type=Path); a=p.parse_args()
    parity(a.output) if a.mode=='parity' else diagnostics(a.output)
