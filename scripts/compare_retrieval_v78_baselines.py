"""Post-seal paired all-human comparison, never used by generation."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v78_safe_sparse_initial import CONFIG
from src.xgb_matcher.retrieval_v20_workbench import record,verify,write_json


def candidate_set(candidates):
    found={c['siret'] for c in candidates}
    if len(candidates)>100 or len(found)!=len(candidates):
        raise ValueError('Candidate cap/duplicate drift')
    return found


def compare(output):
    dest=output/'baseline_comparisons.json'
    if dest.exists():
        raise ValueError('Immutable comparisons')
    cfg=json.loads(CONFIG.read_text()); gen=json.loads((output/'manifest.json').read_text())
    ev=json.loads((output/'evaluation.json').read_text())
    if gen['status']!='SEALED_INITIAL_TOP100' or gen['queries']!=128 or gen['truth_opened'] or ev['generation']!=record(output/'manifest.json'):
        raise ValueError('Complete sealed generation/evaluation required')
    launch=json.loads(verify(gen['launch']).read_text()); v38.verify_launch(launch)
    current=[r for ref in gen['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids={r['query_id'] for r in current}
    if len(current)!=128 or len(ids)!=128 or ids!={q['query_id'] for q in launch['queries']}:
        raise ValueError('All-human generation population drift')
    labels={r['query_id']:r for r in pq.read_table(verify(ev['ledger']),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    e=v38.load(v38.EVAL,v38.EVAL_SHA); side=json.loads(verify(e['label_side']).read_text())
    for ref in side['files'].values():
        verify(ref)
    acceptable={}
    for r in pq.read_table(verify(e['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids:
            acceptable.setdefault(r['query_id'],set()).add(r['siret'])
    if set(labels)!=ids or set(acceptable)!=ids:
        raise ValueError('All-human evaluation authority incomplete')
    comparisons={}
    for name,ref in cfg['comparators'].items():
        old=json.loads(verify(ref).read_text()); ol=json.loads(verify(old['launch']).read_text()); v38.verify_launch(ol)
        if old['status']!='SEALED_INITIAL_TOP100' or old['queries']!=128 or old['truth_opened'] or ol['queries']!=launch['queries']:
            raise ValueError('Comparator population drift')
        if any(ol[k]!=launch[k] for k in ('index_manifest','index_meta','inventory')):
            raise ValueError('Comparator corpus drift')
        previous=[r for b in old['batches'] for r in json.loads(verify(b).read_text())['rows']]
        prior={r['query_id']:r for r in previous}
        if len(previous)!=128 or set(prior)!=ids:
            raise ValueError('Comparator output drift')
        rows=[]
        for r in current:
            label=labels[r['query_id']]; p=prior[r['query_id']]
            if label['fold']!=r['fold'] or p['fold']!=r['fold'] or r['fold'] not in (2,3,4):
                raise ValueError('Comparison protected fold drift')
            item=dict(query_id=r['query_id'],**{k:label[k] for k in v38.SEGMENTS})
            for mode,values in (('primary',r['results']['primary']),('comparator',p['results']['primary'])):
                found=candidate_set(values)
                item[mode+'_exact']=label['human_ground_truth_siret'] in found
                item[mode+'_operational']=bool(found & acceptable[r['query_id']])
            rows.append(item)
        segments={}
        for column in v38.SEGMENTS:
            groups={}
            for r in rows:
                value=r[column]; groups.setdefault('UNKNOWN' if value is None or value=='' else str(value),[]).append(r)
            segments[column]={k:{m:v38.summarize(g,m) for m in v38.MODES} for k,g in groups.items()}
        comparisons[name]=dict(baseline=ref,primary=v38.summarize(rows,'primary'),comparator=v38.summarize(rows,'comparator'),
            gains=sum(r['primary_exact'] and not r['comparator_exact'] for r in rows),
            losses=sum(r['comparator_exact'] and not r['primary_exact'] for r in rows),segments=segments,outcomes=rows)
    v38.verify_launch(launch)
    write_json(dest,dict(status='SEALED_POST_EVALUATION_PAIRED_BASELINE_COMPARISONS',comparisons=comparisons,
        generation=record(output/'manifest.json'),evaluation=record(output/'evaluation.json'),implementation=record(Path(__file__)),
        fold0_opened=False,fold1_opened=False,ready_for_fold1=False,models_changed=False,
        comparator_timing_replayed=False,scope='SAME128_REUSED_DEVELOPMENT_PANEL_NOT_FULL_OOF'))
    print(json.dumps({k:{f:c[f] for f in ('primary','comparator','gains','losses')} for k,c in comparisons.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    compare(p.parse_args().output)
