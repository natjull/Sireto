"""V76 immutable-output evaluator; V49 trace schema was not compatible.

No regeneration, candidate edit, score change, fit or new variant. Reference
visits are a diagnostic sum of decoded references, NOT distinct global sites.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v75_history_site import verify_history
from scripts.retrieval_v76_history_parent import CONFIG, verify_aliases
from src.xgb_matcher.retrieval_v20_workbench import record, verify, write_json


def reference_visits(row):
    counts = [s['decoded'] for s in row['primary_trace']['sources'].values()]
    if any(not isinstance(n,int) or n < 0 for n in counts):
        raise ValueError('Invalid diagnostic decode count')
    return sum(counts)


def evaluate(output):
    dest=output/'evaluation.json'
    if dest.exists():
        raise ValueError('Immutable evaluation')
    cfg=json.loads(CONFIG.read_text()); manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='SEALED_INITIAL_TOP100' or manifest['queries']!=128 or manifest['truth_opened']:
        raise ValueError('Full sealed population required')
    launch=json.loads(verify(manifest['launch']).read_text()); v38.verify_launch(launch)
    verify_aliases(launch['aliases']); verify_history(launch['history'])
    for ref in launch['query_sources']:
        verify(ref)
    verify(launch['models']); verify(launch['model_manifest'])
    rows=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids={r['query_id'] for r in launch['queries']}
    if (len(ids)!=128 or len(rows)!=128 or {r['query_id'] for r in rows}!=ids or
        [(r['query_id'],r['fold']) for r in rows]!=[(q['query_id'],q['fold']) for q in launch['queries']] or
        not all(r['primary_trace']['certified'] and r['primary_trace']['fixed_reservoir'] is None and
                r['primary_trace']['parent_limit'] is None for r in rows)):
        raise ValueError('Uncertified population drift')
    visits=[reference_visits(r) for r in rows]
    old=json.loads(verify(cfg['baseline']).read_text()); ol=json.loads(verify(old['launch']).read_text())
    v38.verify_launch(ol)
    if old['status']!='SEALED_INITIAL_TOP100' or old['queries']!=128 or old['truth_opened'] or ol['queries']!=launch['queries']:
        raise ValueError('Comparator population drift')
    if any(launch[k]!=ol[k] for k in ('index_manifest','index_meta','inventory')):
        raise ValueError('Comparator corpus drift')
    previous=[r for ref in old['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    baseline={r['query_id']:r for r in previous}
    if len(previous)!=128 or set(baseline)!=ids:
        raise ValueError('Comparator coverage drift')
    ev=v38.load(v38.EVAL,v38.EVAL_SHA); side=json.loads(verify(ev['label_side']).read_text())
    for ref in side['files'].values():
        verify(ref)
    labels={r['query_id']:r for r in pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),
        filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    catalog={r['query_id']:r for r in pq.read_table(verify(ev['files']['official_catalog_sidecar.parquet']),
        filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    acceptable={}
    for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids:
            acceptable.setdefault(r['query_id'],set()).add(r['siret'])
    if any(set(x)!=ids for x in (labels,catalog,acceptable)):
        raise ValueError('Incomplete all-human authority')
    outcomes=[]
    for r in rows:
        label=labels[r['query_id']]; prior=baseline[r['query_id']]
        if label['fold']!=r['fold'] or prior['fold']!=r['fold'] or r['fold'] not in (2,3,4):
            raise ValueError('Evaluation fold drift')
        item=dict(query_id=r['query_id'],**{k:label[k] for k in v38.SEGMENTS})
        for mode,candidates in (('primary',r['results']['primary']),('comparator',prior['results']['primary'])):
            found={c['siret'] for c in candidates}
            if len(found)!=len(candidates) or len(found)>100 or any(
                len(c['siret'])!=14 or len(c['siren'])!=9 or not c['siret'].isdecimal() or
                not c['siren'].isdecimal() or not c['siret'].startswith(c['siren']) for c in candidates):
                raise ValueError('Initial cap/identity drift')
            hit=label['human_ground_truth_siret'] in found
            item[mode+'_exact']=hit; item[mode+'_operational']=bool(found & acceptable[r['query_id']])
            item[mode+'_loss']='HIT' if hit else 'NOT_RETRIEVED_AT100' if catalog[r['query_id']]['official_catalog_present'] else 'OFFICIAL_CATALOG_ABSENT'
        outcomes.append(item)
    segments={}
    for column in v38.SEGMENTS:
        groups={}
        for r in outcomes:
            value=r[column]; groups.setdefault('UNKNOWN' if value is None or value=='' else str(value),[]).append(r)
        segments[column]={k:{m:v38.summarize(g,m) for m in v38.MODES} for k,g in groups.items()}
    quantiles=lambda x:dict(zip(('p50','p95','p99'),np.quantile(x,[.5,.95,.99]).tolist()))
    result=dict(status='MEASURED_V76_HISTORICAL_LEGAL_EXACT_JOIN_INITIAL100_PILOT',decision='PIVOT_GENERATION',
        primary=v38.summarize(outcomes,'primary'),comparator=v38.summarize(outcomes,'comparator'),segments=segments,
        gains=sum(r['primary_exact'] and not r['comparator_exact'] for r in outcomes),
        losses=sum(r['comparator_exact'] and not r['primary_exact'] for r in outcomes),
        latency_primary_current=quantiles([r['seconds']['primary'] for r in rows]),
        latency_comparator_historical=quantiles([r['seconds']['primary'] for r in previous]),
        latency_scope='Current V76 and historical V75 NOT replayed/paired; uncontrolled cache; not production SLO',
        loss_attribution={m:dict(Counter(r[m+'_loss'] for r in outcomes)) for m in v38.MODES},
        decoded_reference_visits_quantiles=quantiles(visits),
        traversal_scope='Sum decoded references across sources; cross-source duplicates included, not unique global sites',
        exact_historical_parent_predicate_nonempty=sum(r['primary_trace']['matching_parents']>0 for r in rows),
        max_matching_parents=max(r['primary_trace']['matching_parents'] for r in rows),
        source_winners=dict(Counter(c['winning_source'] for r in rows for c in r['results']['primary'])),
        changed_candidate_sets=sum({c['siret'] for c in r['results']['primary']}!={c['siret'] for c in baseline[r['query_id']]['results']['primary']} for r in rows),
        outcomes=outcomes,generation=record(output/'manifest.json'),baseline=cfg['baseline'],
        ledger=record(v38.LEDGER),label_side=ev['label_side'],excluded_missing_human_labels=ev['excluded_missing_human_labels'],
        fine_loss_attribution_measured=False,model_used=True,full_query_population=False,ready_for_fold1=False,
        fold0_opened=False,fold1_opened=False,metric_family=manifest['metric_family'],implementation=record(Path(__file__)),
        generic_evaluator_failure='V49_EXPECTED_MISSING_unique_documents_visited_NO_EVALUATION_WRITTEN',
        generation_changed=False,candidates_changed=False,score_changed=False)
    if result['primary']['exact_recall']>=.992:
        result['decision']='STOP'; result['decision_reason']='PILOT_ONLY_FULL_OOF_GENERALIZATION_SLO_GATE_NOT_VALIDATED'
    # Recheck all sealed authorities; no mutation of original generation.
    v38.verify_launch(launch); verify_aliases(launch['aliases']); verify_history(launch['history'])
    for ref in manifest['batches']:
        verify(ref)
    write_json(dest,result)
    print(json.dumps({k:result[k] for k in ('primary','comparator','gains','losses','latency_primary_current',
        'exact_historical_parent_predicate_nonempty','source_winners','decision')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    evaluate(p.parse_args().output)
