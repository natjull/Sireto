"""Bounded post-seal mechanism audit; never predictions or retrieval inputs.

Only already-consumed development labels 2/3/4, public targets, frozen
numeric shards and sealed output100 are read. Numeric-only ranks are NOT
native ranks, recall or a candidate pool. No index scan, fitting or test data.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
BASE = HOME/'v115/streamed_activity_ablation_20261013_a'
LABELS = HOME/'v122/label_clarification_20261013_b'
OUTPUT = HOME/'v131/mechanism_20261013_night'


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def verify(ref):
    if record(ref['path']) != ref:
        raise ValueError('Immutable source binding changed')
    return Path(ref['path'])


def write(path, value):
    # Runtime outputs only, never source/config/code edits.
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main(out):
    if out.exists():
        raise ValueError('Fresh diagnostic namespace required')
    generation, owned = read(BASE/'generation.json'), read(BASE/'generate_exit.json')
    if generation['status'] != 'COMPLETE_SEALED_ALL526' or owned['exit_code'] or not owned['own_child_exit_confirmed']:
        raise ValueError('All526 sealed owned exit before label-side diagnosis')
    outputs = read(verify(generation['outputs']))['learned_native_ooc']
    outcomes = read(BASE/'outcomes.json')['learned_native_ooc']
    remaining = {r['query_id'] for r in outcomes if not r['exact']}
    facts = read(verify(read(LABELS/'report.json')['cases']))
    if len(outcomes) != 526 or len(remaining) != 22 or {r['query_id'] for r in facts} != remaining:
        raise ValueError('Exact full baseline and all22 misses')
    if any(r['crm']['fold'] not in (2, 3, 4) for r in facts):
        raise ValueError('Closed human fold forbidden')
    feature = read(BASE/'features_generation.json')
    calibration = read(BASE/'calibration/generation.json')
    assignment = read(verify(calibration['assignment']))
    models = {k: read(verify(ref)) for k, ref in calibration['models'].items()}
    out.mkdir(parents=True)
    concentration = []
    for r in outcomes:
        cs = outputs[r['query_id']]
        parents = Counter(c['siren'] for c in cs)
        concentration.append(dict(query_id=r['query_id'], exact=r['exact'], siren=r['siren'],
            distinct_parents=len(parents), max_parent_sites=max(parents.values(), default=0),
            parent_hhi=sum((n/len(cs))**2 for n in parents.values()) if cs else None,
            cut_score=cs[-1]['score'] if cs else None))
    by_qid = {r['query_id']: r for r in concentration}
    cases = []
    for fact in facts:
        qid = fact['query_id']
        row = dict(query_id=qid, query=fact['crm'], label_siret=fact['label_siret'],
            protected_not_reconstructed=fact['protected_no_reconstruction'],
            output_summary=by_qid[qid], numeric_only_not_native_rank=True,
            no_retrieval_or_label_change=True)
        if fact['protected_no_reconstruction']:
            row['numeric_target_not_read'] = True
            cases.append(row)
            continue
        model = models[str(assignment[qid])]
        if qid not in model['heldout_queries'] or qid in model['train_queries']:
            raise ValueError('Frozen heldout calibrator required')
        fields = model['feature_fields'][7:]
        weights = np.asarray(model['coefficients'][7:], dtype=np.float64)
        with np.load(verify(feature['features'][qid]), allow_pickle=False) as shard:
            keys, x = shard['keys'], shard['features']
            if x.shape != (len(keys), len(fields)):
                raise ValueError('Exact full numeric schema')
            target = int(fact['label_siret'])
            at = int(np.searchsorted(keys, target))
            inside = at < len(keys) and keys[at] == target
            row.update(full_numeric_references=len(keys), target_in_numeric_universe=bool(inside))
            if inside:
                value = x[at].astype(np.float64)
                contrib = value*weights
                numeric = x @ weights
                score = float(contrib.sum())
                row.update(target_numeric_score=score,
                    numeric_only_strict_rank_lower_bound=1+int(np.count_nonzero(numeric > score+1e-7)),
                    numerical_gap_to_native_cut=float(by_qid[qid]['cut_score']-score),
                    numeric_coverage_positive_fields=int(np.count_nonzero(value)),
                    top_numeric_contributions=[dict(field=fields[i],value=float(value[i]),
                        coefficient=float(weights[i]),contribution=float(contrib[i]))
                        for i in np.argsort(-contrib)[:10]],
                    target_numeric_features={name:float(v) for name,v in zip(fields,value)})
                cs = outputs[qid]
                refs = np.array([int(c['siret']) for c in cs], dtype=np.uint64)
                loc = np.searchsorted(keys, refs)
                valid = loc < len(keys)
                valid[valid] &= keys[loc[valid]] == refs[valid]
                rows = np.flatnonzero(valid)
                row['sealed100_in_numeric_universe'] = int(valid.sum())
                row['sealed100_numeric_and_non_numeric_scores'] = [dict(rank=int(j+1),
                    siret=cs[j]['siret'], native_score=cs[j]['score'],
                    numeric_score=float(numeric[loc[j]]),
                    native_minus_numeric=float(cs[j]['score']-numeric[loc[j]]))
                    for j in rows]
                del numeric
        cases.append(row)
        write(out/'state.json',dict(status='READONLY_POSTSEAL_MECHANISM_AUDIT',cases=len(cases),total=22,
            updated_unix=time.time(),no_index_scan=True))
        gc.collect()
    summary = {}
    for segment, hit in [('hits',True),('misses',False)]:
        group=[r for r in concentration if r['exact'] == hit]
        summary[segment]=dict(queries=len(group),
            distinct_parents_quantiles=np.quantile([r['distinct_parents'] for r in group],[0,.5,.9,1]).tolist(),
            max_parent_sites_quantiles=np.quantile([r['max_parent_sites'] for r in group],[0,.5,.9,1]).tolist(),
            cases_with_parent_at_least20=sum(r['max_parent_sites']>=20 for r in group))
    public=[r for r in cases if not r['protected_not_reconstructed']]
    write(out/'cases.json',cases)
    write(out/'concentration_all526.json',concentration)
    write(out/'report.json',dict(status='COMPLETE_READONLY_ALL22_MECHANISM_AUDIT',
        implementation=record(Path(__file__)),baseline_generation=record(BASE/'generation.json'),
        baseline_exit=record(BASE/'generate_exit.json'),label_clarification=record(LABELS/'report.json'),
        cases=record(out/'cases.json'),concentration=record(out/'concentration_all526.json'),
        summary=summary,public_targets=len(public),protected_skipped=22-len(public),
        public_targets_absent_numeric=sum(not r['target_in_numeric_universe'] for r in public),
        no_new_recall=True,no_index_scan=True,no_fitting=True,no_candidate_pool=True,
        source_score_not_fully_decomposed=True,numeric_only_ranks_not_native_ranks=True,
        closed_test_opened=False,no_positive_injected=True,observed_unix=time.time()))
    print(json.dumps(dict(summary=summary,public_targets_absent_numeric=sum(not r['target_in_numeric_universe'] for r in public))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    main(parser.parse_args().output)
