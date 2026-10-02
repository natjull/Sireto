"""National first-stage top100 search. No parent model, pool or reranker."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from src.xgb_matcher.hierarchical_retrieval import TantivyBackend, RetrievalQuery, character_ngrams
from src.xgb_matcher.retrieval_v20_workbench import ROOT, Budget, record, verify, write_json

DATA = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
INDEX = DATA/'indexes/hierarchical_retrieval_v3/4b332899d8698354'
INDEX_SHA = '9275f519508dca1c5a0738356faaa0e693b59bd80c9f8bacf2c214db6d95af9e'
GEN = DATA/'audits/retrieval_v34_fast_resume/full_job/generation/manifest.json'
GEN_SHA = '022b5c4e4436b335e78d217a81319c122390631e5698dd17c9d7db3f36c77e26'
EVAL = DATA/'audits/retrieval_v34_fast_resume/full_job/evaluation_resume_20261001/manifest.json'
EVAL_SHA = 'b6d7123d2fb33616c76dc14ed4277552a7696d2bdb17b56f540b7f76edfec3c9'
LEDGER = DATA/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'
LEDGER_SHA = '24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720'
NAMES = ('legal_names', 'trade_names', 'site_names', 'historical_names', 'supporting_names')
ADDRESSES = ('addresses', 'historical_addresses', 'supporting_addresses')
MODES = ('primary', 'comparator')
RESOURCE = dict(max_seconds=80, max_rss_bytes=6*1024**3, max_swapout_pages_delta=0)
SEGMENTS = ('fold', 'label_kind', 'qualification_v2', 'qualification_v3', 'data_origin',
            'ground_truth_state', 'unseen_siren', 'new_site_known_siren', 'mega_base_pool')
QUERY_FIELDS = {'query_id', 'fold', 'crm_name', 'crm_number', 'crm_suffix', 'crm_street', 'crm_insee', 'crm_postcode'}


def load(path, sha):
    return json.loads(verify(dict(path=path, sha256=sha)).read_text())


def inventory():
    return {str(p.relative_to(INDEX)): dict(size=p.stat().st_size, mtime=p.stat().st_mtime_ns)
            for p in sorted((INDEX/'tantivy').rglob('*')) if p.is_file() and not p.name.endswith('.lock')}


def tokens(text, chars=False):
    if chars:
        return sorted({x.split('_', 1)[1].lower() for x in character_ngrams(text) if '_' in x})
    return sorted(set(re.findall(r'[a-z0-9]+', text.lower())))


def mean_query(backend, field, values):
    if not values:
        return None
    tv = backend._tantivy
    clauses = [(tv.Occur.Should, tv.Query.term_query(backend.index.schema, field, value, 'basic')) for value in values]
    return tv.Query.boost_query(tv.Query.boolean_query(clauses), 1./len(values))


def query_for(backend, query, mode):
    if mode not in MODES:
        raise ValueError('Unknown recipe')
    tv = backend._tantivy
    if mode == 'comparator':
        parts = [mean_query(backend, field, tokens(text, True)) for field, text in
                 (('current_name_ngrams', query.name), ('address_ngrams', query.address))]
        if any(p is None for p in parts):
            return tv.Query.empty_query()
        text = tv.Query.boolean_query([(tv.Occur.Must, p) for p in parts])
        return backend._combined_query(query, text, document_type='siret')
    components = []
    for fields, charfield, text in ((NAMES, 'current_name_ngrams', query.name),
                                  (ADDRESSES, 'address_ngrams', query.address)):
        views = [mean_query(backend, field, tokens(text)) for field in fields]
        views.append(mean_query(backend, charfield, tokens(text, True)))
        views = [v for v in views if v is not None]
        if views:
            components.append(tv.Query.disjunction_max_query(views, tie_breaker=0.))
    if not components:
        return tv.Query.empty_query()
    text = tv.Query.boolean_query([(tv.Occur.Should, q) for q in components])
    doc_type = tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic')
    clauses = [(tv.Occur.Must, tv.Query.const_score_query(doc_type, 0.)), (tv.Occur.Must, text)]
    geo = [tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, field, value, 'basic'), 1.)
           for field, value in (('insee', query.insee), ('postcode', query.postcode)) if value]
    if geo:
        clauses.append((tv.Occur.Should, tv.Query.disjunction_max_query(geo, tie_breaker=0.)))
    return tv.Query.boolean_query(clauses)


def search(backend, query, mode):
    parsed = query_for(backend, query, mode)
    rows = []
    for score, loc in backend.searcher.search(parsed, limit=100, count=False).hits:
        doc = backend.searcher.doc(loc).to_dict()
        siren, siret = (backend._first(doc, k) for k in ('siren', 'siret'))
        if not (re.fullmatch(r'\d{9}', siren) and re.fullmatch(r'\d{14}', siret)
                and siret.startswith(siren) and math.isfinite(score)):
            raise ValueError('Invalid official identity/score')
        rows.append(dict(siren=siren, siret=siret, score=float(score)))
    if len(rows) > 100 or len({r['siret'] for r in rows}) != len(rows):
        raise ValueError('Top100 cap or uniqueness violated')
    # Only ties among the already returned documents are canonicalized.
    rows.sort(key=lambda r: (-r['score'], r['siret']))
    return rows


def input_rows(generation):
    rows, refs = [], []
    for i, ref in enumerate(generation['batches'][:4]):
        batch = json.loads(verify(ref).read_text())
        if batch['start'] != i*32 or batch['count'] != 32 or len(batch['shards']) != 1:
            raise ValueError('Scope batch drift')
        shard = json.loads(verify(batch['shards'][0]).read_text())
        source = shard['files']['queries.parquet']
        part = pq.read_table(verify(source)).to_pylist()
        if any(set(r) != QUERY_FIELDS or r['fold'] not in (2, 3, 4) for r in part):
            raise ValueError('Forbidden input columns or folds')
        rows.extend(part); refs.append(source)
    if len(rows) != 128 or len({r['query_id'] for r in rows}) != 128:
        raise ValueError('Population drift')
    return rows, refs


def verify_launch(launch):
    for ref in launch['implementation'].values():
        verify(ref)
    verify(launch['index_manifest']); verify(launch['index_meta'])
    if inventory() != launch['inventory']:
        raise ValueError('Index segment inventory drift')


def worker(output, start):
    launch = json.loads((output/'launch.json').read_text()); verify_launch(launch)
    budget = Budget(RESOURCE); rows = []
    with budget:
        backend = TantivyBackend(INDEX)
        if backend.searcher.num_docs != launch['index_documents']:
            raise ValueError('National corpus document count drift')
        for i, raw in enumerate(launch['queries'][start:start+8], start):
            address = ' '.join(raw[k] for k in ('crm_number', 'crm_suffix', 'crm_street') if raw[k])
            query = RetrievalQuery(name=raw['crm_name'], address=address, number=raw['crm_number'],
                                   insee=raw['crm_insee'], postcode=raw['crm_postcode'])
            row = dict(query_id=raw['query_id'], fold=raw['fold'], results={}, seconds={})
            for mode in (MODES if i % 2 == 0 else tuple(reversed(MODES))):
                t = time.monotonic(); row['results'][mode] = search(backend, query, mode)
                row['seconds'][mode] = time.monotonic()-t
            rows.append(row)
    verify_launch(launch)
    write_json(output/f'batch{start:03d}.json', dict(rows=rows, resources=budget.report()))


def run(output):
    output.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    try:
        generation = load(GEN, GEN_SHA)
        if generation['status'] != 'SEALED_FULL_PAIRED_GENERATION':
            raise ValueError('Unsealed query source')
        queries, refs = input_rows(generation)
        im = load(INDEX/'manifest.json', INDEX_SHA)
        if im['build_scope'] != 'full' or im['contains_crm_labels'] or not im['typed_name_portfolio']:
            raise ValueError('Invalid corpus')
        files = [Path(__file__), ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
                 ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
                 ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',
                 ROOT/'docs/retrieval_v38_initial_top100_contract.md']
        launch = dict(queries=queries, query_sources=refs, generation=record(GEN),
                      index_manifest=record(INDEX/'manifest.json'), index_meta=record(INDEX/'tantivy/meta.json'),
                      inventory=inventory(), index_documents=im['num_documents'], index_sites=im['num_establishment_documents'],
                      implementation={str(p): record(p) for p in files}, resources=RESOURCE,
                      collector_limit=100, truth_opened=False, index_full_bytes_rehashed=False)
        write_json(output/'launch.json', launch); batches = []; allrows = []
        for start in range(0, 128, 8):
            cmd = [sys.executable, '-u', '-m', 'scripts.retrieval_v38_initial_top100', 'worker',
                   '--output', str(output), '--start', str(start)]
            with (output/f'batch{start:03d}.log').open('x') as log:
                subprocess.run(cmd, check=True, timeout=min(90, max(1, 900-(time.monotonic()-began))),
                               stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'RAYON_NUM_THREADS': '2'})
            path = output/f'batch{start:03d}.json'; batch = json.loads(path.read_text())
            expected = {(r['query_id'], r['fold']) for r in queries[start:start+8]}
            if len(batch['rows']) != 8 or {(r['query_id'], r['fold']) for r in batch['rows']} != expected:
                raise ValueError('Worker denominator drift')
            if not batch['resources']['pass_resource_policy']:
                raise ValueError('Worker resource failure')
            allrows.extend(batch['rows']); batches.append(record(path))
            print(json.dumps(dict(completed=len(allrows), target=128, resource=batch['resources'])), flush=True)
            if len(allrows) == 16:
                cost = sum(sum(r['seconds'].values()) for r in allrows)/16
                write_json(output/'cost_gate.json', dict(queries=16, mean_pair_seconds=cost, passed=cost <= 4., quality_opened=False))
                if cost > 4.:
                    raise ValueError('STOP_COST_PREFLIGHT')
        verify_launch(launch)
        write_json(output/'manifest.json', dict(status='SEALED_INITIAL_TOP100', queries=128, launch=record(output/'launch.json'),
            batches=batches, seconds=time.monotonic()-began, truth_opened=False, model_used=False,
            full_corpus=True, full_query_population=False, ready_for_fold1=False))
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), seconds=time.monotonic()-began,
                                               quality_opened=False, ready_for_fold1=False))
        raise


def summarize(rows, mode):
    n = len(rows); hits = sum(r[mode+'_exact'] for r in rows); z = 1.959963984540054
    p = hits/n; denominator = 1+z*z/n
    return dict(queries=n, exact_hits=hits, exact_recall=p, operational_hits=sum(r[mode+'_operational'] for r in rows),
        operational_recall=sum(r[mode+'_operational'] for r in rows)/n,
        wilson95_lower=(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/denominator)


def evaluate(output):
    dest = output/'evaluation.json'
    if dest.exists():
        raise ValueError('Do not overwrite an evaluation')
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['status'] != 'SEALED_INITIAL_TOP100' or manifest['queries'] != 128 or manifest['truth_opened']:
        raise ValueError('Complete sealed generation required')
    launch = json.loads(verify(manifest['launch']).read_text()); verify_launch(launch)
    generated = [r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids = {r['query_id'] for r in launch['queries']}
    if len(generated) != 128 or {r['query_id'] for r in generated} != ids:
        raise ValueError('Evaluation population drift')
    ev = load(EVAL, EVAL_SHA); label_side = json.loads(verify(ev['label_side']).read_text())
    for ref in label_side['files'].values():
        verify(ref)
    labels = {r['query_id']: r for r in pq.read_table(verify(dict(path=LEDGER, sha256=LEDGER_SHA)),
              filters=[('fold', 'in', [2, 3, 4])]).to_pylist() if r['query_id'] in ids}
    catalog = {r['query_id']: r for r in pq.read_table(verify(ev['files']['official_catalog_sidecar.parquet']),
               filters=[('fold', 'in', [2, 3, 4])]).to_pylist() if r['query_id'] in ids}
    acceptable = {}
    for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']), filters=[('fold', 'in', [2, 3, 4])]).to_pylist():
        if r['query_id'] in ids:
            acceptable.setdefault(r['query_id'], set()).add(r['siret'])
    if set(labels) != ids or set(catalog) != ids or set(acceptable) != ids:
        raise ValueError('Incomplete label-side authority')
    outcomes = []
    for r in generated:
        label = labels[r['query_id']]
        if r['fold'] != label['fold'] or r['fold'] not in (2, 3, 4):
            raise ValueError('Evaluation fold drift')
        row = dict(query_id=r['query_id'], **{c: label[c] for c in SEGMENTS})
        for mode in MODES:
            candidates = r['results'][mode]; found = {c['siret'] for c in candidates}
            if len(found) != len(candidates) or len(found) > 100:
                raise ValueError('Invalid output')
            hit = label['human_ground_truth_siret'] in found
            row[mode+'_exact'] = hit; row[mode+'_operational'] = bool(found & acceptable[r['query_id']])
            row[mode+'_loss'] = 'HIT' if hit else 'NOT_RETRIEVED_AT100' if catalog[r['query_id']]['official_catalog_present'] else 'OFFICIAL_CATALOG_ABSENT'
        outcomes.append(row)
    segments = {}
    for column in SEGMENTS:
        groups = {}
        for r in outcomes:
            value = r[column]; key = 'UNKNOWN' if value is None or value == '' else str(value)
            groups.setdefault(key, []).append(r)
        segments[column] = {key: {mode: summarize(group, mode) for mode in MODES} for key, group in groups.items()}
    result = dict(status='MEASURED_INITIAL_SEARCH_TOP100_PILOT', decision='PIVOT_GENERATION',
        primary=summarize(outcomes, 'primary'), comparator=summarize(outcomes, 'comparator'), segments=segments,
        gains=sum(r['primary_exact'] and not r['comparator_exact'] for r in outcomes),
        losses=sum(r['comparator_exact'] and not r['primary_exact'] for r in outcomes),
        loss_attribution={mode: dict(Counter(r[mode+'_loss'] for r in outcomes)) for mode in MODES},
        latency={mode: dict(zip(('p50', 'p95', 'p99'), np.quantile([r['seconds'][mode] for r in generated], [.5, .95, .99]).tolist())) for mode in MODES},
        latency_scope='Search plus stored reference extraction; uncontrolled cache, not production SLO',
        excluded_missing_human_labels=ev['excluded_missing_human_labels'],
        generation=record(output/'manifest.json'), ledger=record(LEDGER), label_side=ev['label_side'], outcomes=outcomes,
        fine_loss_attribution_measured=False, ready_for_fold1=False, fold0_opened=False, fold1_opened=False,
        model_used=False, full_query_population=False, implementation=record(Path(__file__)))
    if result['primary']['exact_recall'] >= .992:
        result['decision'] = 'STOP'
        result['decision_reason'] = 'PILOT_ONLY_NO_FULL_OOF_CONFIDENCE_SLO_OR_FINE_ATTRIBUTION_GATE'
    write_json(dest, result)
    print(json.dumps({k: result[k] for k in ('primary', 'comparator', 'gains', 'losses', 'latency', 'decision')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run', 'worker', 'evaluate']); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start', type=int); a = p.parse_args()
    if a.mode == 'run': run(a.output)
    elif a.mode == 'worker':
        if a.start not in range(0, 128, 8): p.error('Invalid worker start')
        worker(a.output, a.start)
    else: evaluate(a.output)
