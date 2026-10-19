"""Query-only G0 -> external content ranker -> grouped pilot stacking.

All workers are sequential, local, read-only on sources, and separately sealed.
The V142 candidate lists and closed populations are never regenerated/opened.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from scripts import step2_features_v2 as v2
from scripts import train3000_frozen_v142_cohort as cohort
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend, RetrievalQuery, normalize_text
from src.xgb_matcher.retrieval_query_ban_v327 import compatible

b, d = v2.base, v2.diag
HOME = b.HOME
OWN = HOME/'step2_option_d_20261017'
DOC = ROOT/'docs/step2_option_d.md'
INDEX = cohort.io.INDEX
BATCH = 256
DROP = ['v142_rank', 'v142_reciprocal_rank', 'v142_score', 'v142_score_relative',
    'v142_delta_best', 'v142_delta_next', 'v142_top1_top2_gap',
    'same_siren_as_native_top1', 'native_top1_siren_count',
    'v2_distance_minus_native_log', 'v2_siren_first_native_rank']
CONTENT = [f for f in v2.FEATURES if f not in DROP]
CONTENT_COLS = [v2.FEATURES.index(f) for f in CONTENT]
NATIVE = b.FEATURES[32:39]
SCENE = b.FEATURES[39:46]
STACK = NATIVE+['content_'+f.removeprefix('v142_') for f in NATIVE]+SCENE
PARAMS = dict(b.PARAMS, objective='rank:ndcg')
UNION = b.VIEWS.index('union')
SEED = '20261017|content-curve|'


def write(path, value):
    b.write(path, value)


def inventory():
    return [b.fingerprint(p) for p in sorted((INDEX/'tantivy').iterdir()) if p.is_file()]


def competitors():
    found = {}
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if any(Path(arg).name == 'train3000_frozen_v142_supervisor.py' for arg in process.info['cmdline'] or []):
                for worker in [process]+process.children(recursive=True):
                    rss = worker.memory_info().rss
                    if rss > 4*1024**3:
                        found[worker.pid] = rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def wait():
    while active := competitors():
        print(json.dumps(dict(stage='WAIT_OTHER_GENERATION_RSS_GT4GIB', pids=active)), flush=True)
        time.sleep(60)


class Guard:
    def __init__(self, directory):
        self.directory, self.peak = directory, 0
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def monitor(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            if self.peak >= int(5.75*1024**3):
                write(self.directory/'resource_stop.json', dict(status='STOP_RSS', peak_rss_bytes=self.peak))
                os._exit(75)
            self.stop.wait(.1)


def load_plan(out):
    plan = b.read(out/'plan.json')
    for key in ('implementation', 'feature_implementation', 'base_implementation', 'custody_implementation', 'preregistration', 'index_manifest', 'index_meta'):
        b.verified(plan[key])
    if plan['params'] != PARAMS or plan['content_features'] != CONTENT or plan['stack_features'] != STACK:
        raise ValueError('Preregistered implementation/recipe changed')
    return plan


def prepare(out, smoke):
    sample = b.read(ROOT/'docs/train3000_frozen_v142_sample.json')
    pinned = subprocess.check_output(['git', 'show', 'd54cec2c:docs/train3000_frozen_v142_sample.json'], cwd=ROOT)
    if pinned != (ROOT/'docs/train3000_frozen_v142_sample.json').read_bytes():
        raise ValueError('Sealed3000 draw changed')
    rows, pilot, _, audit = cohort.population()
    if (len(rows), audit['eligible_companies'], audit['excluded_pilot_companies']) != (17427, 11483, 405):
        raise ValueError('Custody population drift')
    labels = b.index(rows)
    selected = sample['selected_query_ids']
    if cohort.digest_ids(selected) != 'f2a359afb834b02d5ca30d905808b62ea2d613eb038acbb9eaa1c82e8df2deb3' or not set(selected) <= labels.keys():
        raise ValueError('Sealed draw not reproduced')
    order = sorted(selected, key=lambda q: hashlib.sha256((SEED+q).encode()).digest())
    order += sorted(set(labels)-set(selected), key=lambda q: hashlib.sha256((SEED+q).encode()).digest())
    if smoke:
        order = sorted(sample['smoke_query_ids'], key=lambda q: hashlib.sha256((SEED+q).encode()).digest())
        pilot = [q for q in pilot if q['query_id'] in set(qid for fold in (2, 3, 4) for qid in sorted(
            [p['query_id'] for p in pilot if p['fold'] == fold], key=lambda q: hashlib.sha256(q.encode()).hexdigest())[:12])]
    pilot = sorted(pilot, key=lambda q: q['query_id'])
    ids = set(order)
    queries, geometries, source_refs = {}, {}, []
    for batch in b.read(cohort.io.GEN)['batches']:
        for shard in b.read(batch['path'])['shards']:
            manifest = b.read(shard['path'])
            qref = manifest['files']['queries.parquet']
            filtered = pq.read_table(qref['path'], columns=list(cohort.old.CRM),
                filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', order)], use_threads=False).to_pylist()
            if not filtered:
                continue
            b.verified(qref)
            source_refs.append(qref)
            bref = manifest['files']['ban.parquet']
            b.verified(bref)
            source_refs.append(bref)
            cached = b.index(pq.read_table(bref['path'], filters=[('fold', 'in', [2, 3, 4]),
                ('query_id', 'in', [q['query_id'] for q in filtered])], use_threads=False).to_pylist())
            for q in filtered:
                qid = q['query_id']
                if qid in queries:
                    raise ValueError('Duplicate query-only source')
                queries[qid] = q
                geometries[qid] = dict(query_id=qid, geometry=compatible(q, cached[qid]))
    if set(queries) != ids:
        raise ValueError('Missing permitted CRM input')
    source = b.read(ROOT/'docs/step2_features_v2.json')['source_refs']
    original_geo = b.index(b.read(source['geometry']['path']))
    for q in pilot:
        geometries[q['query_id']] = original_geo[q['query_id']]
    pilot_companies = set(pq.read_table(source['ledger']['path'], columns=['human_ground_truth_siren'],
        filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', [q['query_id'] for q in pilot])], use_threads=False).column(0).to_pylist())
    if set(labels[q]['human_ground_truth_siren'] for q in order) & pilot_companies:
        raise ValueError('External content/pilot company overlap')
    write(out/'training_queries.json', [queries[q] for q in order])
    write(out/'pilot_queries.json', pilot)
    write(out/'geometry.json', geometries)
    write(out/'label_side.json', {q: labels[q] for q in order})
    write(out/'cohort.json', dict(population=audit, selected_id_set_sha256=cohort.digest_ids(order[:min(3000, len(order))]),
        training_queries=len(order), training_companies=len({labels[q]['human_ground_truth_siren'] for q in order}),
        training_order_sha256=d.digest(order), pilot_queries=len(pilot), external_pilot_company_overlap=0,
        original_pilot_companies_excluded=405, query_sources=source_refs, sample=b.record(ROOT/'docs/train3000_frozen_v142_sample.json'),
        artifacts={name: b.record(out/name) for name in ('training_queries.json', 'pilot_queries.json', 'geometry.json', 'label_side.json')}))
    print(json.dumps(dict(stage='CUSTODY_SEALED', train=len(order), pilot=len(pilot), closed_sets_opened=False)), flush=True)


class G0:
    def __init__(self, plan):
        self.backend = TantivyBackend(INDEX)
        self.tv = self.backend._tantivy
        layout = plan['layout']
        self.arrays = {k: np.load(ref['path'], mmap_mode='r') for k, ref in layout.items()}
        if self.backend.searcher.num_docs != 82772691:
            raise ValueError('National index scope changed')

    def documents(self, query, fields, text, local=True):
        backend, tv = self.backend, self.tv
        if local:
            values = backend._query(query, fields, text, 100) if text else []
        else:
            text_query = backend.index.parse_query_lenient(text, default_field_names=list(fields), conjunction_by_default=False)[0] if text else tv.Query.all_query()
            parsed = tv.Query.boolean_query([(tv.Occur.Must, text_query),
                (tv.Occur.Must, tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'))])
            values = [(float(score), backend.searcher.doc(loc).to_dict()) for score, loc in backend.searcher.search(parsed, limit=100).hits]
        result = {backend._first(doc, 'siret'): backend._first(doc, 'siren') for _, doc in sorted(values, key=lambda t: (-t[0], backend._first(t[1], 'siret')))}
        return list(result.items())

    def geographic(self, geometry):
        point = b.crm_point(geometry)['point_l93']
        if point is None:
            return []
        a, radius = self.arrays, 5000.
        x, y = point
        lo, hi = a['block_lo.npy'], a['block_hi.npy']
        dx = np.maximum(np.maximum(lo[:, 0]-x, x-hi[:, 0]), 0.)
        dy = np.maximum(np.maximum(lo[:, 1]-y, y-hi[:, 1]), 0.)
        lower = dx**2+dy**2
        blocks = np.flatnonzero(lower <= radius**2)
        blocks = blocks[np.argsort(lower[blocks], kind='stable')]
        offsets, sites = a['block_offsets.npy'], a['block_sites.npy']
        best_keys, best_squared = np.array([], dtype=np.uint64), np.array([], dtype=np.float64)
        cutoff = radius**2
        for start in range(0, len(blocks), 32):
            selected = blocks[start:start+32]
            selected = selected[lower[selected] <= cutoff]
            if not len(selected):
                break
            locations = np.concatenate([sites[int(offsets[k]):int(offsets[k+1])] for k in selected])
            xy = a['site_xy.npy'][locations]
            squared = (xy[:, 0]-x)**2+(xy[:, 1]-y)**2
            keep = np.flatnonzero((squared <= cutoff) & a['public'][locations])
            keys = np.r_[best_keys, a['keys'][locations[keep]]]
            distances = np.r_[best_squared, squared[keep]]
            chosen = np.lexsort((keys, distances))[:100]
            best_keys, best_squared = keys[chosen], distances[chosen]
            if len(chosen) == 100:
                cutoff = float(best_squared[-1])
        return [(f'{int(key):014d}', f'{int(key):014d}'[:9]) for key in best_keys]

    def retrieve(self, q, geometry):
        text = normalize_text(b.text(q['crm_name']))
        address = normalize_text(' '.join(b.text(q[k]) for k in ('crm_number', 'crm_suffix', 'crm_street')))
        query = RetrievalQuery.from_mapping(dict(crm_name=text, crm_addr=address, crm_postcode=q['crm_postcode'], crm_insee=q['crm_insee']))
        channels = [self.documents(query, ['names'], text), self.documents(query, ['addresses'], address),
            self.documents(query, ['names', 'addresses'], (text+' '+address).strip()), self.geographic(geometry)]
        scores, parents = {}, {}
        for channel in channels:
            for rank, (siret, siren) in enumerate(channel, 1):
                scores[siret] = scores.get(siret, 0.)+1/(60+rank)
                parents[siret] = siren
        chosen = sorted(scores, key=lambda s: (-scores[s], s))[:100]
        national = 0
        if len(chosen) < 100:
            for siret, siren in self.documents(query, ['names', 'addresses'], (text+' '+address).strip(), local=False):
                if siret not in chosen:
                    chosen.append(siret)
                    parents[siret], scores[siret] = siren, 0.
                    national += 1
                    if len(chosen) == 100:
                        break
        if len(chosen) < 100:
            for siret, siren in self.documents(query, ['names'], '', local=False):
                if siret not in chosen:
                    chosen.append(siret)
                    parents[siret], scores[siret] = siren, 0.
                    national += 1
                    if len(chosen) == 100:
                        break
        if len(chosen) != 100 or len(set(chosen)) != 100:
            raise ValueError('Cannot emit100 natural official sites')
        return [dict(siret=s, siren=parents[s], score=scores[s]) for s in chosen], dict(geo_count=len(channels[-1]), national_fill=national, blank_text=not bool(text or address))


def generate(out, scope, start, end, directory):
    plan = load_plan(out)
    if inventory() != plan['index_inventory']:
        raise ValueError('Index fingerprint changed')
    queries = b.read(out/(scope+'_queries.json'))[start:end]
    geometry = b.read(out/'geometry.json')
    engine = G0(plan)
    rows = []
    for q in queries:
        wait()
        began = time.monotonic()
        candidates, meta = engine.retrieve(q, geometry[q['query_id']]['geometry'])
        rows.append(dict(query_id=q['query_id'], candidates=candidates, seconds=time.monotonic()-began, **meta))
    if inventory() != plan['index_inventory']:
        raise ValueError('Index changed during G0')
    write(directory/'candidates.json', rows)
    return dict(queries=len(rows), search_seconds=sum(r['seconds'] for r in rows), candidates=b.record(directory/'candidates.json'))


def feature_batch(out, parts, directory):
    plan = load_plan(out)
    rows = [r for part in parts for r in b.read(b.verified(part['candidates']))]
    ids = [r['query_id'] for r in rows]
    queries = b.index(b.read(out/'training_queries.json'))
    panel = {q: queries[q] for q in ids}
    outputs = {r['query_id']: r['candidates'] for r in rows}
    geometry = b.read(out/'geometry.json')
    refs = plan['official']
    sites, units, points = b.hydrate(refs, outputs)
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    con.register('requested', pa.table({'siret': sorted({c['siret'] for cs in outputs.values() for c in cs})}))
    closures = {r['siret']: r['closure_date'] for r in con.execute("""SELECT siret, MAX(try_cast(dateDebut AS DATE)) closure_date
        FROM read_parquet(?) h SEMI JOIN requested r USING(siret)
        WHERE etatAdministratifEtablissement='F' AND try_cast(changementEtatAdministratifEtablissement AS BOOLEAN)=true
        AND try_cast(dateDebut AS DATE)<=DATE '2026-08-01' GROUP BY siret""", [refs['history']['path']]).fetch_arrow_table().to_pylist()}
    pairs = sorted({(c['siren'], panel[q]['crm_insee']) for q, cs in outputs.items() for c in cs if panel[q]['crm_insee']})
    con.register('requested_pairs', pa.table({'siren': pa.array([p[0] for p in pairs], type=pa.string()), 'commune': pa.array([p[1] for p in pairs], type=pa.string())}))
    counts = {(r['siren'], r['commune']): r['active_count'] for r in con.execute("""SELECT s.siren, s.codeCommuneEtablissement commune, count(*) active_count
        FROM read_parquet(?) s JOIN requested_pairs r ON s.siren=r.siren AND s.codeCommuneEtablissement=r.commune
        WHERE s.etatAdministratifEtablissement='A' AND s.statutDiffusionEtablissement='O' GROUP BY s.siren,s.codeCommuneEtablissement""", [refs['stock']['path']]).fetch_arrow_table().to_pylist()}
    con.close()
    old = b.feature_cube(panel, outputs, geometry, sites, units, points)
    new = v2.feature_cube(old, panel, outputs, geometry, sites, points, closures, counts)
    X = np.concatenate([old, new], axis=2)[:, :, CONTENT_COLS]
    if X.shape != (len(rows), 100, 85) or np.isinf(X).any():
        raise ValueError('Content feature shape/values')
    np.save(directory/'features.npy', X)
    np.save(directory/'candidate_keys.npy', np.array([[c['siret'] for c in r['candidates']] for r in rows], dtype='S14'))
    return dict(queries=len(rows), query_ids=ids, features=b.record(directory/'features.npy'), keys=b.record(directory/'candidate_keys.npy'))


def content_fit(out, n, directory):
    plan = load_plan(out)
    matrix = b.read(out/'matrix.json')
    X = np.load(b.verified(matrix['features']), mmap_mode='r')[:n]
    keys = np.load(b.verified(matrix['keys']), mmap_mode='r')[:n]
    custody = b.read(out/'cohort.json')['artifacts']
    queries = b.read(b.verified(custody['training_queries.json']))[:n]
    labels = b.read(b.verified(custody['label_side.json']))
    truth = np.array([labels[q['query_id']]['human_ground_truth_siret'] for q in queries], dtype='S14')
    target = (keys == truth[:, None]).astype(np.float32)
    if target.sum(axis=1).max() > 1:
        raise ValueError('Duplicate positive')
    model = xgb.XGBRanker(**PARAMS)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(X.reshape(-1, 85), target.reshape(-1), group=np.full(n, 100))
    model.save_model(directory/'model.json')
    pilot_ref = b.read(ROOT/'docs/step2_features_v2.json')['matrix']
    pilot_pack = np.load(b.verified(pilot_ref))
    pilot_ids = [q['query_id'] for q in b.read(out/'pilot_queries.json')]
    full_ids = pilot_pack['query_ids'].tolist()
    pilot_X = pilot_pack['features'][[full_ids.index(q) for q in pilot_ids]][:, :, CONTENT_COLS]
    predictions = model.predict(pilot_X.reshape(-1, 85)).reshape(len(pilot_ids), 100)
    np.save(directory/'pilot_scores.npy', predictions)
    return dict(n=n, features=CONTENT, params=PARAMS, model=b.record(directory/'model.json'), scores=b.record(directory/'pilot_scores.npy'),
        pilot_ids=pilot_ids, target_sha256=hashlib.sha256(target.tobytes()).hexdigest(), positive_queries=int(target.sum()),
        absent_positive_queries=int((target.sum(axis=1) == 0).sum()),
        train_id_set_sha256=cohort.digest_ids([q['query_id'] for q in queries]),
        train_companies=len({labels[q['query_id']]['human_ground_truth_siren'] for q in queries}),
        warnings=[str(w.message) for w in captured], pilot_company_overlap=0, matrix=matrix)


def stack_cube(native_X, pred):
    order = np.argsort(-pred, axis=1, kind='stable')
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, 101)[None, :], axis=1)
    sorted_scores = np.take_along_axis(pred, order, axis=1)
    differences = sorted_scores-np.c_[sorted_scores[:, 1:], sorted_scores[:, -1:]]
    gaps = np.empty_like(pred)
    np.put_along_axis(gaps, order, differences, axis=1)
    low, high = pred.min(axis=1, keepdims=True), pred.max(axis=1, keepdims=True)
    derived = np.stack([ranks, 1/ranks, pred, (pred-low)/np.maximum(high-low, 1e-6), pred-high,
        gaps, np.broadcast_to(sorted_scores[:, :1]-sorted_scores[:, 1:2], pred.shape)], axis=2)
    return np.concatenate([native_X[:, :, 32:39], derived, native_X[:, :, 39:46]], axis=2).astype(np.float32)


class StackRunner:
    def __init__(self, X, y, groups, ids, directory, fixed=None):
        self.X, self.y, self.groups, self.ids, self.directory, self.fixed = X, y, groups, ids, directory, fixed
        self.models, self.receipts, self.fixed_uses = {}, [], []
        directory.mkdir()

    def score(self, train, test, columns, purpose):
        train, test = np.sort(train), np.asarray(test)
        if set(self.groups[train]) & set(self.groups[test]):
            raise ValueError('Pilot company overlap')
        use = dict(purpose=purpose, train_queries=self.ids[train].tolist(), scored_queries=self.ids[test].tolist(), company_overlap=0)
        if self.fixed is not None:
            self.fixed_uses.append(use)
            return self.fixed[test]
        key = d.digest(train.tolist())
        if key not in self.models:
            model = xgb.XGBRanker(**PARAMS)
            target = self.y[train].reshape(-1)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(self.X[train].reshape(-1, len(STACK)), target, group=np.full(len(train), 100))
            path = self.directory/(key+'.json')
            model.save_model(path)
            receipt = dict(key=key, model=b.record(path), params=PARAMS, features=STACK,
                train_queries=self.ids[train].tolist(), target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                zero_positive_queries=int((self.y[train].sum(axis=1) == 0).sum()), warnings=[str(w.message) for w in captured], uses=[])
            self.models[key] = model, receipt
            self.receipts.append(receipt)
        model, receipt = self.models[key]
        receipt['uses'].append(use)
        return model.predict(self.X[test].reshape(-1, len(STACK))).reshape(len(test), 100)


def evaluate(out, directory, g0_only=False):
    plan = load_plan(out)
    Xold, y, views, groups, folds, ids, native, refs, outer, _, _ = d.load(plan['smoke'])
    pqueries = b.index(b.read(out/'pilot_queries.json'))
    if ids.tolist() != sorted(pqueries):
        raise ValueError('Pilot row mapping changed')
    old_v2 = b.read(ROOT/'docs/step2_features_v2.json')
    pack = np.load(b.verified(old_v2['matrix']))
    X = pack['features'][[pack['query_ids'].tolist().index(q) for q in ids]]
    geometry = b.read(out/'geometry.json')
    pilot_parts = b.read(out/'pilot_parts.json')
    G0rows = [r for part in pilot_parts for r in b.read(b.verified(part['candidates']))]
    G0outputs = {r['query_id']: r['candidates'] for r in G0rows}
    sites, _, points = b.hydrate(refs, G0outputs)
    labels = b.index(pq.read_table(refs['ledger']['path'], columns=['query_id', 'human_ground_truth_siret', 'human_ground_truth_siren'],
        filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', ids.tolist())], use_threads=False).to_pylist())
    accepted = {q: {labels[q]['human_ground_truth_siret']} for q in ids}
    for r in pq.read_table(refs['acceptable_operational']['path'], filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', ids.tolist())], use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    _, gviews = b.targets_and_views(pqueries, G0outputs, labels, accepted, geometry, sites, points)
    # pqueries must be kept in the same deterministic sorted order as ids.
    if list(pqueries) != ids.tolist():
        raise ValueError('G0 view row mapping')
    r3pred = np.load(b.verified(old_v2['predictions']))['v1_plus_v2'][[pack['query_ids'].tolist().index(q) for q in ids]]
    comparisons = dict(V142=views[:, 0], R3_v2=views[np.arange(len(ids)), b.select_top(r3pred)[0]])
    curves, predictions = {}, {}
    for fit in ([] if g0_only else b.read(out/'content_parts.json')):
        n = fit['n']
        pred = np.load(b.verified(fit['scores']))
        if fit['pilot_ids'] != ids.tolist() or not np.isfinite(pred).all():
            raise ValueError('External prediction mapping')
        SX = stack_cube(Xold, pred)
        runner = StackRunner(SX, y, groups, ids, directory/f'stack_{n}')
        stacked = np.full((len(ids), 100), np.nan)
        for fold, train, test in outer:
            stacked[test] = runner.score(train, test, list(range(len(STACK))), f'outer_{fold}')
        fixed = StackRunner(SX, y, groups, ids, directory/f'content_{n}', fixed=pred)
        confidence = {}
        for name, scorer in [('content', fixed), ('stack', runner)]:
            conf, common = v2.confidence(scorer, outer, X, views, groups, folds, ids, list(range(len(STACK))), directory, f'{name}_{n}')
            if name == 'content':
                conf['protocol'] = 'Fixed external content model saw NO pilot company;60% scene fitting only,20% isotonic,20% threshold; no pilot ranker fit.'
                if not np.array_equal(common, pred):
                    raise ValueError('External ranker changed in calibration')
            confidence[name] = conf
            predictions[f'{name}_{n}_confidence'] = common
        predictions[f'content_{n}'], predictions[f'stack_{n}'] = pred, stacked
        curves[str(n)] = dict(n=n, fit=fit,
            content=d.summarize(pred, views, folds, ids, comparisons)[0],
            stack=d.summarize(stacked, views, folds, ids, comparisons)[0], confidence=confidence,
            stack_fit_receipts=runner.receipts, external_score_uses=fixed.fixed_uses)
        print(json.dumps(dict(stage='CURVE', n=n, content=curves[str(n)]['content']['totals']['top1'], stack=curves[str(n)]['stack']['totals']['top1'])), flush=True)
    tests = [c[m]['totals']['paired'][ref][view] for c in curves.values() for m in ('content', 'stack') for ref in ('V142', 'R3_v2') for view in ('exact', 'union')]
    previous = 0.
    for i, test in enumerate(sorted(tests, key=lambda t: t['exact_two_sided_p'])):
        previous = max(previous, min(1., (len(tests)-i)*test['exact_two_sided_p']))
        test['holm_p'] = previous
    times = [r['seconds'] for r in G0rows]
    g0 = dict(presence={v: b.metric(int(gviews[:, :, j].any(axis=1).sum()), len(ids)) for j, v in enumerate(b.VIEWS)},
        v142_presence={v: b.metric(int(views[:, :, j].any(axis=1).sum()), len(ids)) for j, v in enumerate(b.VIEWS)},
        overlap_mean=float(np.mean([len({c['siret'] for c in G0outputs[q]} & {c['siret'] for c in native[q]}) for q in ids])),
        search_seconds_per_query=float(np.mean(times)), search_p95_seconds=float(np.quantile(times, .95)),
        worker_seconds_per_query=sum(p['parent_wall_seconds'] for p in pilot_parts)/len(ids),
        peak_rss_bytes=max(p['peak_rss_bytes'] for p in pilot_parts),
        geo_queries=sum(bool(r['geo_count']) for r in G0rows), national_fill_queries=sum(bool(r['national_fill']) for r in G0rows),
        blank_queries=sum(r['blank_text'] for r in G0rows), per_query=[dict(query_id=str(q), overlap=len({c['siret'] for c in G0outputs[q]} & {c['siret'] for c in native[q]}),
            presence={v: bool(gviews[i, :, j].any()) for j, v in enumerate(b.VIEWS)}) for i, q in enumerate(ids)])
    if g0_only:
        write(directory/'result.json', dict(status='G0_PILOT_DIAGNOSTIC_ONLY', queries=len(ids), G0=g0,
            source_plan=b.record(out/'plan.json'), candidate_sources=pilot_parts, ranking_fit=False))
        print(json.dumps(dict(stage='G0_DIAGNOSTIC', queries=len(ids), presence=g0['presence'],
            overlap=g0['overlap_mean'], seconds_per_query=g0['search_seconds_per_query'])), flush=True)
        return dict(result=b.record(directory/'result.json'))
    np.savez_compressed(directory/'predictions.npz', **predictions)
    result = dict(status='MEASURED_DEVELOPMENT_NOT_PRODUCTION', smoke=plan['smoke'], queries=len(ids), source_plan=b.record(out/'plan.json'),
        cohort=b.read(out/'cohort.json'), generation=b.read(out/'generation.json'), G0=g0, curves=curves,
        references={name: {v: b.metric(int(hits[:, j].sum()), len(ids)) for j, v in enumerate(b.VIEWS)} for name, hits in comparisons.items()},
        predictions=b.record(directory/'predictions.npz'), pilot_ids=ids.tolist(), pilot_folds=folds.tolist(),
        content_features=CONTENT, stack_features=STACK, params=PARAMS, ranking_target='exact', confidence_target='union',
        labels_changed=False, positive_injection=False, retrieval_v142_changed=False, closed_sets_opened=False,
        upstream_v142_full_oof=False, production_promotion_allowed=False)
    for ref in plan['official'].values():
        b.verified(ref)
    for ref in plan['layout'].values():
        b.verified(ref)
    if inventory() != plan['index_inventory']:
        raise ValueError('Index changed during experiment')
    if subprocess.check_output(['git', 'show', '36c7a476:docs/retrieval_frozen_v142.json'], cwd=ROOT) != (ROOT/'docs/retrieval_frozen_v142.json').read_bytes():
        raise ValueError('Frozen V142 seal changed')
    write(directory/'result.json', result)
    return dict(result=b.record(directory/'result.json'), predictions=result['predictions'])


def execute(out, mode, directory, extra=()):
    wait()
    directory.mkdir()
    command = [sys.executable, '-u', str(Path(__file__)), mode, '--output-dir', str(out), '--directory', str(directory), *map(str, extra)]
    began = time.monotonic()
    budget = b.read(out/'g0_budget.json')['seconds'] if (out/'g0_budget.json').exists() else 0.
    try:
        with (directory/'worker.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                timeout=max(.1, 6*3600-budget) if mode == '_g0' else None)
    finally:
        if mode == '_g0':
            write(out/'g0_budget.json', dict(seconds=budget+time.monotonic()-began))
    result = b.read(directory/'receipt.json')
    result['parent_wall_seconds'] = time.monotonic()-began
    print(json.dumps(dict(stage=mode, directory=directory.name, seconds=result['parent_wall_seconds'], peak_mib=result['peak_rss_bytes']/1024**2)), flush=True)
    return result


def run(out, smoke):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = b.read(ROOT/'docs/step2_features_v2.json')
    official = {k: original['source_refs'][k] for k in ('stock', 'legal_units', 'study', 'history')}
    launch = b.read(HOME/'v142/bano_native_20261013_night/launch.json')
    layout_root = Path(launch['xy']['path']).parent
    lm = b.read(layout_root/'manifest.json')
    layout = {k: lm['artifacts'][k] for k in ('block_lo.npy', 'block_hi.npy', 'block_offsets.npy', 'block_sites.npy', 'site_xy.npy')}
    layout.update(keys=launch['keys'], public=launch['public_mask'])
    for ref in [*official.values(), *layout.values()]:
        b.verified(ref)
    prereg = DOC.read_text().split('\n## Résultats\n')[0]+'\n'
    (out/'preregistered.md').write_text(prereg)
    plan = dict(status='BEFORE_GENERATION_AND_FITS', smoke=smoke, implementation=b.record(Path(__file__)),
        params=PARAMS, content_features=CONTENT, dropped_features=DROP, stack_features=STACK,
        official=official, layout=layout, index_inventory=inventory(), index_manifest=b.record(INDEX/'manifest.json'),
        index_meta=b.record(INDEX/'tantivy/meta.json'), preregistration=b.record(out/'preregistered.md'),
        feature_implementation=b.record(Path(v2.__file__)), base_implementation=b.record(Path(b.__file__)),
        custody_implementation=b.record(Path(cohort.__file__)), batch_size=BATCH, ranking_target='exact', confidence_target='union')
    write(out/'plan.json', plan)
    execute(out, '_prepare', out/'custody', ['--smoke'] if smoke else [])
    pilot_count = len(b.read(out/'pilot_queries.json'))
    pilot_parts = [execute(out, '_g0', out/f'pilot_{start:05d}', ['--scope', 'pilot', '--start', start, '--end', min(start+BATCH, pilot_count)]) for start in range(0, pilot_count, BATCH)]
    write(out/'pilot_parts.json', pilot_parts)
    execute(out, '_diagnose', out/'pilot_diagnostic')
    available = len(b.read(out/'training_queries.json'))
    initial = min(3000, available)
    train_parts = [execute(out, '_g0', out/f'train_{start:05d}', ['--scope', 'training', '--start', start, '--end', min(start+BATCH, initial)]) for start in range(0, initial, BATCH)]
    spent = sum(p['parent_wall_seconds'] for p in train_parts)
    estimate = b.read(out/'g0_budget.json')['seconds']+1.25*(spent/initial)*(available-initial)
    extend = estimate <= 6*3600
    extension_stop = None
    if extend:
        for start in range(initial, available, BATCH):
            if b.read(out/'g0_budget.json')['seconds'] >= 6*3600:
                extension_stop = 'G0_6H_BUDGET'
                break
            try:
                train_parts.append(execute(out, '_g0', out/f'train_{start:05d}', ['--scope', 'training', '--start', start, '--end', min(start+BATCH, available)]))
            except subprocess.TimeoutExpired:
                extension_stop = 'G0_6H_BUDGET_PARTIAL_WORKER_PRESERVED_NOT_USED'
                break
    n = sum(p['queries'] for p in train_parts)
    write(out/'train_parts.json', train_parts)
    generation = dict(initial=initial, available=available, generated=n, extension_allowed_by_cost=extend,
        initial_worker_seconds=spent, projected_full_seconds_with25pct_margin=estimate,
        total_worker_seconds=sum(p['parent_wall_seconds'] for p in train_parts),
        pilot_and_training_budget_seconds=b.read(out/'g0_budget.json')['seconds'], extension_stop=extension_stop,
        search_seconds=sum(p['search_seconds'] for p in train_parts), peak_rss_bytes=max(p['peak_rss_bytes'] for p in train_parts))
    write(out/'generation.json', generation)
    feature_parts = [execute(out, '_features', out/f'features_{i:04d}', ['--part', i]) for i in range(0, len(train_parts), 4)]
    X = np.lib.format.open_memmap(out/'features.npy', mode='w+', dtype=np.float32, shape=(n, 100, 85))
    keys = np.lib.format.open_memmap(out/'candidate_keys.npy', mode='w+', dtype='S14', shape=(n, 100))
    cursor, qids = 0, []
    for part in feature_parts:
        count = part['queries']
        X[cursor:cursor+count] = np.load(b.verified(part['features']), mmap_mode='r')
        keys[cursor:cursor+count] = np.load(b.verified(part['keys']), mmap_mode='r')
        qids += part['query_ids']
        cursor += count
    X.flush(); keys.flush()
    del X, keys
    gc.collect()
    if qids != [q['query_id'] for q in b.read(out/'training_queries.json')[:n]]:
        raise ValueError('Feature batch order changed')
    write(out/'matrix.json', dict(status='SEALED_BEFORE_FIT_TARGET_READ', features=b.record(out/'features.npy'), keys=b.record(out/'candidate_keys.npy'),
        shape=[n, 100, 85], query_ids_sha256=d.digest(qids), feature_names=CONTENT, parts=feature_parts))
    volumes = sorted({min(10, n), n}) if smoke else sorted({500, 1000, 3000, n})
    fits = [execute(out, '_content', out/f'fit_{size}', ['--n', size]) for size in volumes]
    write(out/'content_parts.json', fits)
    receipt = execute(out, '_evaluate', out/'evaluation')
    result = b.read(b.verified(receipt['result']))
    result['elapsed_seconds'] = time.monotonic()-started
    result['peak_rss_bytes'] = max([p['peak_rss_bytes'] for p in [*pilot_parts, *train_parts, *feature_parts, *fits, receipt]])
    write(out/'result.json', result)
    (out/'result.md').write_text(prereg+'\n'+markdown(result))
    if not smoke:
        write(ROOT/'docs/step2_option_d.json', result)
        DOC.write_text(prereg+'\n'+markdown(result))
    print(json.dumps(dict(stage='COMPLETE', seconds=result['elapsed_seconds'], peak_mib=result['peak_rss_bytes']/1024**2, result=str(out/'result.json'))), flush=True)


def markdown(r):
    g, gen = r['G0'], r['generation']
    lines = ['## Résultats', '', '**Développement, pas production ; aucune modification du retrieval V142.**', '',
        f'- G0 sur{r["queries"]} pilotes : '+', '.join(f'{v} {g["presence"][v]["hits"]}' for v in b.VIEWS)+'.',
        f'- Intersection moyenne avec V142 :{g["overlap_mean"]:.2f}/100 ; recherche{g["search_seconds_per_query"]:.4f}s/q, p95{g["search_p95_seconds"]:.4f}s ; workers complets{g["worker_seconds_per_query"]:.4f}s/q ; pic{g["peak_rss_bytes"]/1024**2:.1f}Mio.',
        f'- Apprentissage :{gen["generated"]}/{gen["available"]} dossiers générés ; G0 workers{gen["total_worker_seconds"]:.1f}s, projection préalable avec marge25%{gen["projected_full_seconds_with25pct_margin"]:.1f}s ; extension={gen["extension_allowed_by_cost"]}.', '',
        '| Volume | Méthode | Exact | SIREN | Opérationnel | Site | Union |', '|---:|---|---:|---:|---:|---:|---:|']
    for name, metrics in r['references'].items():
        lines.append('| — | '+name+' | '+' | '.join(str(metrics[v]['hits']) for v in b.VIEWS)+' |')
    for size, c in r['curves'].items():
        for method in ('content', 'stack'):
            lines.append('| '+size+' | '+method+' | '+' | '.join(str(c[method]['totals']['top1'][v]['hits']) for v in b.VIEWS)+' |')
    lines += ['', '### Positifs réellement disponibles', '']
    for size, c in r['curves'].items():
        lines.append(f'- {size} :{c["fit"]["positive_queries"]} positifs présents ;{c["fit"]["absent_positive_queries"]} groupes sans positif, conservés dans les mesures/fit, sans injection.')
    lines += ['', '### Confiance, cible union98% en calibration seulement', '',
        '| Volume/méthode | Politique | SUR | Union, Wilson95 | Exact, Wilson95 | Couverture, Wilson95 |', '|---|---|---:|---|---|---|']
    for size, c in r['curves'].items():
        for method, conf in c['confidence'].items():
            for policy, value in conf['methods'].items():
                t = value['totals']
                lines.append(f'| {size}/{method} | {policy} | {t["count"]} | {b.md_metric(t["precision"]["union"])} | {b.md_metric(t["precision"]["exact"])} | {b.md_metric(t["rate"])} |')
    last = r['curves'][str(max(map(int, r['curves']))) ]
    lines += ['', '### Paires finales et limites', '']
    for method in ('content', 'stack'):
        for ref in ('V142', 'R3_v2'):
            p = last[method]['totals']['paired'][ref]
            lines.append('- '+method+' vs '+ref+' : '+', '.join(f'{v} +{len(p[v]["gains"])}/−{len(p[v]["losses"])} p={p[v]["exact_two_sided_p"]:.5g}, Holm={p[v]["holm_p"]:.5g}' for v in ('exact', 'union'))+'.')
    p = last['stack']['totals']['paired']['R3_v2']['union']
    recommendation = ('Confirmer l’empilement sur une validation indépendante autorisée, sans modifier cette recette.' if p['holm_p'] < .05 and len(p['gains']) > len(p['losses']) else
        'Ne pas promouvoir cet empilement ; privilégier ensuite un signal sémantique propre au site plutôt qu’augmenter encore ce même apprentissage de contenu.')
    lines += ['', '### Recommandation unique', '', recommendation, '',
        '- Cinq vues, folds, Wilson, McNemar et tous volumes publiés dans le JSON. Holm ne corrige pas la dépendance intra-entreprise ni la consommation du pilote. Pas de sous-ensemble choisi après résultats.',
        '- Content reste externe à toutes les405 entreprises ; empilement et calibration sans entreprise partagée entre fit/calibration/test. Risque de décalage G0→V142 conservé, pas masqué.',
        '- BAN ancien query-only réutilisé, aucun géocodage nouveau ; v17 pilote conservé. Dates et agrégats officiels identiques à v2. Les deux scores BAN probabilistes sont NaN.',
        '- Pas de Gemma/BANO/OSM/activité dans G0, aucune écriture v174, VM/web/installation/positif injecté/jeu fermé. Index segments contrôlés par empreintes avant/après (pas nouveau SHA intégral98Gio), manifestes et sources de features/layout par SHA.',
        f'- Total{r["elapsed_seconds"]:.1f}s ; pic maximum des workers{r["peak_rss_bytes"]/1024**2:.1f}Mio ; sources et gel vérifiés avant/après. Poids, matrices, cibles et rôles scellés dans `H/step2_option_d_20261017/`.',
        '- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_option_d.py run --output-dir <dossier_neuf_sous_H/step2_option_d_20261017>` ; ajouter `--smoke` pour20 nouveaux/36 pilotes.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', '_prepare', '_g0', '_features', '_content', '_diagnose', '_evaluate'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--scope', choices=['pilot', 'training'])
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--part', type=int)
    parser.add_argument('--n', type=int)
    args = parser.parse_args()
    out = args.output_dir or OWN/('smoke' if args.smoke else 'full')
    if not out.resolve().is_relative_to(OWN.resolve()) or out.resolve() == OWN.resolve():
        raise ValueError('Only a fresh owned step2_option_d namespace is writable')
    if args.directory and not args.directory.resolve().is_relative_to(out.resolve()):
        raise ValueError('Worker directory outside owned namespace')
    if args.mode == 'run':
        wait()
        run(out, args.smoke)
        return
    wait()
    started = time.monotonic()
    guard = Guard(args.directory)
    if args.mode == '_prepare':
        load_plan(out)
        prepare(out, args.smoke)
        receipt = dict(cohort=b.record(out/'cohort.json'))
    elif args.mode == '_g0':
        receipt = generate(out, args.scope, args.start, args.end, args.directory)
    elif args.mode == '_features':
        receipt = feature_batch(out, b.read(out/'train_parts.json')[args.part:args.part+4], args.directory)
    elif args.mode == '_content':
        receipt = content_fit(out, args.n, args.directory)
    else:
        receipt = evaluate(out, args.directory, g0_only=args.mode == '_diagnose')
    guard.stop.set()
    receipt.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=guard.peak)
    write(args.directory/'receipt.json', receipt)


if __name__ == '__main__':
    main()
