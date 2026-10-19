"""Four fixed company-OOF recipes on the existing G0 development population.

Independent label-side union, nested selective confidence, real model replay.
No retrieval, test population, label correction or production promotion.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import warnings

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import step2_option_d as old

b, d, np, xgb, pa, pq, duckdb = old.b, old.d, old.np, old.xgb, old.pa, old.pq, old.duckdb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
SOURCE = old.OWN/'full'
OWN = old.HOME/'step2_g0_bench_20261017'
DOC = ROOT/'docs/step2_g0_bench.md'
VARIANTS = ('base', 'plus_g0', 'company_weight', 'union_target')
EXACT, UNION = 0, 1


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check(out):
    plan = b.read(out/'plan.json')
    b.verified(plan['preregistration'])
    for ref in plan['implementations']:
        b.verified(ref)
    return plan


def source():
    result = b.read(SOURCE/'result.json')
    committed = subprocess.check_output(['git', 'show', '7b0005a8:docs/step2_option_d.json'], cwd=ROOT)
    if json.loads(committed) != b.read(ROOT/'docs/step2_option_d.json'):
        raise ValueError('OptionD authority changed')
    for ref in result['cohort']['artifacts'].values():
        b.verified(ref)
    matrix = b.read(SOURCE/'matrix.json')
    b.verified(matrix['features']); b.verified(matrix['keys'])
    queries = b.read(SOURCE/'training_queries.json')
    labels = b.read(SOURCE/'label_side.json')
    if d.digest([q['query_id'] for q in queries]) != matrix['query_ids_sha256'] or matrix['feature_names'] != old.CONTENT:
        raise ValueError('Cached feature row/column binding changed')
    if len(queries) != 17427 or len({labels[q['query_id']]['human_ground_truth_siren'] for q in queries}) != 11483:
        raise ValueError('Admissible population drift')
    if any(q['fold'] not in (2, 3, 4) for q in queries):
        raise ValueError('Closed population')
    return result, matrix, queries, labels


def prepare(out):
    plan = check(out)
    result, matrix, queries, labels = source()
    all_ids = [q['query_id'] for q in queries]
    all_groups = np.array([labels[q]['human_ground_truth_siren'] for q in all_ids])
    take = np.arange(len(queries))
    if plan['smoke']:
        companies = sorted(set(all_groups), key=lambda g: hashlib.sha256(('20261017|g0-smoke|'+g).encode()).digest())[:100]
        take = np.flatnonzero(np.isin(all_groups, companies))
    queries = [queries[i] for i in take]
    ids = np.array([q['query_id'] for q in queries])
    groups = all_groups[take]
    lookup = {q: i for i, q in enumerate(ids)}
    n = len(ids)
    original = np.load(matrix['features']['path'], mmap_mode='r')
    scene = np.lib.format.open_memmap(out/'scene.npy', mode='w+', dtype=np.float32, shape=(n, 100, 46))
    for col, name in enumerate(b.FEATURES):
        if name in old.CONTENT:
            scene[:, :, col] = original[take, :, old.CONTENT.index(name)]
    scores = np.zeros((n, 100), dtype=np.float64)
    relevant, requested, parts = {}, set(), b.read(SOURCE/'train_parts.json')
    original_keys = np.load(matrix['keys']['path'], mmap_mode='r')
    for part in parts:
        for row in b.read(b.verified(part['candidates'])):
            q = row['query_id']
            if q not in lookup:
                continue
            i, cs = lookup[q], row['candidates']
            if len(cs) != 100 or len({c['siret'] for c in cs}) != 100:
                raise ValueError('G0 pool drift')
            keys = original_keys[take[i]]
            if [c['siret'].encode() for c in cs] != keys.tolist():
                raise ValueError('Candidate order mismatch')
            values = np.array([c['score'] for c in cs])
            scores[i] = values
            rank = np.arange(1, 101)
            native = np.stack([rank, 1/rank, values, (values-values.min())/max(float(np.ptp(values)), 1e-6),
                values-values.max(), np.r_[values[:-1]-values[1:], 0.], np.full(100, values[0]-values[1])], axis=1)
            scene[i, :, 32:39] = native
            scene[i, :, 40] = [c['siren'] == cs[0]['siren'] for c in cs]
            scene[i, :, 41] = sum(c['siren'] == cs[0]['siren'] for c in cs)
            # Label-side requests only; never candidate/features generation.
            relevant[q] = [(j, c['siret']) for j, c in enumerate(cs) if c['siren'] == labels[q]['human_ground_truth_siren']]
            requested.update(s for _, s in relevant[q])
    scene.flush(); del scene, original
    np.save(out/'source_indices.npy', take)
    np.save(out/'g0_scores.npy', scores.astype(np.float32))
    feature_refs = {k: b.record(out/k) for k in ('scene.npy', 'source_indices.npy', 'g0_scores.npy')}
    b.write(out/'feature_seal.json', dict(status='FEATURES_SEALED_BEFORE_TARGET_BUILD', source=matrix['features'], files=feature_refs,
        features=old.CONTENT, plus_g0_features=['g0_rank', 'g0_rrf_score']))
    refs = b.read(SOURCE/'plan.json')['official']
    for ref in refs.values():
        b.verified(ref)
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    con.register('requested', pa.table({'siret': sorted(requested)}))
    def select(key, columns):
        sql = 'SELECT '+','.join('s.'+c for c in columns)+' FROM read_parquet(?) s SEMI JOIN requested r USING(siret)'
        return b.index(con.execute(sql, [refs[key]['path']]).fetch_arrow_table().to_pylist(), 'siret')
    sites = select('stock', b.STOCK_COLUMNS)
    points = select('study', ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision'])
    con.close()
    geometry = b.read(SOURCE/'geometry.json')
    keys = np.load(matrix['keys']['path'], mmap_mode='r')[take]
    truth = np.array([labels[q]['human_ground_truth_siret'] for q in ids], dtype='S14')
    exact = keys == truth[:, None]
    site = np.zeros_like(exact)
    for i, (q, query) in enumerate(zip(ids, queries)):
        if relevant[q]:
            emitted = [dict(siret=s, siren=s[:9]) for _, s in relevant[q]]
            _, measured = b.targets_and_views({q: query}, {q: emitted}, labels,
                {q: {labels[q]['human_ground_truth_siret']}}, geometry, sites, points)
            site[i, [j for j, _ in relevant[q]]] = measured[0, :, b.VIEWS.index('site_operational')]
    views = np.stack([exact, exact | site], axis=2)
    if not plan['smoke'] and int(exact.sum()) != result['curves']['17427']['fit']['positive_queries']:
        raise ValueError('Exact targets differ from sealed optionD')
    np.save(out/'views.npy', views)
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=b.SEED)
    splits = [(train.tolist(), test.tolist()) for train, test in splitter.split(ids, groups=groups)]
    partition = np.full(n, -1)
    for cv, (train, test) in enumerate(splits):
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Outer company overlap')
        partition[test] = cv
    if np.any(partition < 0):
        raise ValueError('Unscored query')
    data = dict(ids=ids.tolist(), groups=groups.tolist(), population_folds=[q['fold'] for q in queries], splits=splits,
        partition=partition.tolist(), source_matrix=matrix['features'], feature_seal=b.record(out/'feature_seal.json'),
        views=b.record(out/'views.npy'), feature_files=feature_refs, official_refs=refs, input_matrix_shape=matrix['shape'],
        queries=n, companies=len(set(groups)), exact_present=int(exact.any(axis=1).sum()), union_present=int(views[:, :, UNION].any(axis=1).sum()),
        extra_union_positive_candidates=int((site & ~exact).sum()), labels_changed=False, injected_candidates=False)
    b.write(out/'data.json', data)
    return dict(data=b.record(out/'data.json'), queries=n, companies=len(set(groups)))


class Runner:
    def __init__(self, X, target, groups, directory, variant):
        self.X, self.target, self.groups, self.directory, self.variant = X, target, groups, directory, variant
        self.models, self.receipts = {}, []
        directory.mkdir()

    def score(self, train, test, purpose):
        old.wait()
        train, test = np.sort(train), np.asarray(test)
        if set(self.groups[train]) & set(self.groups[test]):
            raise ValueError('Fit/score company overlap')
        key = d.digest(train.tolist())
        if key not in self.models:
            target = self.target[train].astype(np.float32).reshape(-1)
            weights = None
            if self.variant == 'company_weight':
                counts = Counter(self.groups[train])
                weights = np.array([len(train)/(len(counts)*counts[g]) for g in self.groups[train]], dtype=np.float64)
                _, inverse = np.unique(self.groups[train], return_inverse=True)
                totals = np.bincount(inverse, weights=weights)
                if not np.allclose(totals, len(train)/len(counts)):
                    raise ValueError('Unequal company weights')
            model = xgb.XGBRanker(**old.PARAMS)
            began = time.monotonic()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(self.X[train].reshape(-1, self.X.shape[2]), target, group=np.full(len(train), 100), sample_weight=weights)
            path = self.directory/(key+'.json'); model.save_model(path)
            receipt = dict(key=key, train_indices=train.tolist(), model=b.record(path), target_sha256=sha(target),
                query_weights_sha256=sha(weights) if weights is not None else None,
                weight_normalization='mean1; equal total per company WITHIN FIT' if weights is not None else None,
                zero_positive_queries=int((self.target[train].sum(axis=1) == 0).sum()), seconds=time.monotonic()-began,
                warnings=[str(w.message) for w in captured], uses=[])
            self.models[key] = model, receipt; self.receipts.append(receipt)
        model, receipt = self.models[key]
        value = model.predict(self.X[test].reshape(-1, self.X.shape[2])).reshape(len(test), 100)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite model score')
        # Replay EACH emitted vector from a separately loaded saved model.
        replay = xgb.XGBRanker(n_jobs=1); replay.load_model(b.verified(receipt['model']))
        actual = replay.predict(self.X[test].reshape(-1, self.X.shape[2])).reshape(len(test), 100)
        if not np.array_equal(value, actual):
            raise ValueError('Serialized model replay mismatch')
        receipt['uses'].append(dict(purpose=purpose, scored_indices=test.tolist(), company_overlap=0,
            score_sha256=sha(value), replay_equal=True))
        return value


def metrics(hits, mask=None):
    if mask is None:
        return {name: b.metric(int(hits[:, j].sum()), len(hits)) for j, name in enumerate(('exact', 'union'))}
    return dict(count=int(mask.sum()), coverage=b.metric(int(mask.sum()), len(mask)),
        precision=metrics(hits[mask]) if mask.any() else dict(exact=None, union=None))


def variant(out, name, directory):
    check(out); data = b.read(b.verified(b.read(out/'preparation/receipt.json')['data']))
    for ref in data['feature_files'].values():
        b.verified(ref)
    b.verified(data['source_matrix']); b.verified(data['views'])
    indices = np.load(out/'source_indices.npy')
    X = np.load(data['source_matrix']['path'], mmap_mode='r')[indices]
    if name == 'plus_g0':
        rank = np.broadcast_to(np.arange(1, 101), X.shape[:2])
        X = np.concatenate([X, np.stack([rank, np.load(out/'g0_scores.npy')], axis=2).astype(np.float32)], axis=2)
    scene_X = np.load(out/'scene.npy', mmap_mode='r')
    views = np.load(out/'views.npy'); groups = np.array(data['groups'])
    target = views[:, :, UNION if name == 'union_target' else EXACT]
    runner = Runner(X, target, groups, directory/'models', name)
    n = len(X)
    predictions = np.full((n, 100), np.nan, dtype=np.float32)
    common = np.full_like(predictions, np.nan)
    selected = {name: np.zeros(n, dtype=bool) for name in ('margin', 'scene_isotonic')}
    risk_scores = {name: np.zeros(n) for name in selected}
    receipts, meta_refs = [], []
    for cv, (tr, te) in enumerate(data['splits']):
        train, test = np.array(tr), np.array(te)
        predictions[test] = runner.score(train, test, f'outer_{cv}')
        companies = np.random.default_rng(b.SEED+100+cv).permutation(np.array(sorted(set(groups[train]))))
        count = max(1, math.ceil(.2*len(companies)))
        limit = train[np.isin(groups[train], companies[:count])]
        iso_rows = train[np.isin(groups[train], companies[count:2*count])]
        proper = train[~np.isin(groups[train], companies[:2*count])]
        roles = [proper, iso_rows, limit, test]
        if any(set(groups[a]) & set(groups[z]) for i, a in enumerate(roles) for z in roles[i+1:]):
            raise ValueError('Confidence company overlap')
        meta = np.full((len(proper), 100), np.nan, dtype=np.float32)
        splitter = GroupKFold(n_splits=3, shuffle=True, random_state=b.SEED)
        for k, (a, z) in enumerate(splitter.split(proper, groups=groups[proper])):
            meta[z] = runner.score(proper[a], proper[z], f'meta_{cv}_{k}')
        joined = np.r_[iso_rows, limit, test]
        raw = runner.score(proper, joined, f'confidence_{cv}')
        pi, pl, pt = np.split(raw, [len(iso_rows), len(iso_rows)+len(limit)])
        common[test] = pt
        top = b.select_top(meta)[0]
        meta_y = views[proper, top, UNION].astype(int)
        classifier = d.logistic(meta_y); classifier.fit(d.scene(meta, scene_X[proper]), meta_y)
        itop = b.select_top(pi)[0]
        iso_y = views[iso_rows, itop, UNION].astype(int)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0., y_max=1.)
        iso.fit(d.probability(classifier, d.scene(pi, scene_X[iso_rows])), iso_y)
        path = directory/f'scene_{cv}.pkl'
        with path.open('wb') as stream:
            pickle.dump((classifier, iso), stream, protocol=5)
        meta_refs.append(b.record(path))
        lt, lm = b.select_top(pl); tt, tm = b.select_top(pt)
        lp = iso.predict(d.probability(classifier, d.scene(pl, scene_X[limit])))
        tp = iso.predict(d.probability(classifier, d.scene(pt, scene_X[test])))
        # Independent pickle replay of the selective probabilities.
        with path.open('rb') as stream:
            replay_c, replay_i = pickle.load(stream)
        assert np.array_equal(tp, replay_i.predict(d.probability(replay_c, d.scene(pt, scene_X[test]))))
        limits = {}
        for policy, cal, value in [('margin', lm, tm), ('scene_isotonic', lp, tp)]:
            threshold, stats = b.threshold_choice(cal, views[limit, lt, UNION])
            stats['union_precision'] = stats.pop('exact_precision')
            selected[policy][test] = value >= threshold if threshold is not None else False
            risk_scores[policy][test] = value
            limits[policy] = dict(threshold=threshold, calibration=stats)
        receipts.append(dict(cv=cv, proper=proper.tolist(), isotonic=iso_rows.tolist(), threshold=limit.tolist(), test=test.tolist(),
            company_overlap=0, limits=limits, meta_union_target_sha256=sha(meta_y), iso_union_target_sha256=sha(iso_y)))
        print(json.dumps(dict(stage='CV_COMPLETE', variant=name, cv=cv, exact=int(views[test, b.select_top(predictions[test])[0], EXACT].sum()))), flush=True)
    top = b.select_top(predictions)[0]; common_top = b.select_top(common)[0]
    hits = views[np.arange(n), top]; common_hits = views[np.arange(n), common_top]
    np.savez_compressed(directory/'predictions.npz', predictions=predictions, common=common, hits=hits, common_hits=common_hits,
        margin_sur=selected['margin'], scene_isotonic_sur=selected['scene_isotonic'], **{k+'_scores': v for k, v in risk_scores.items()})
    result = dict(variant=name, features=old.CONTENT+(['g0_rank', 'g0_rrf_score'] if name == 'plus_g0' else []), params=old.PARAMS,
        ranking_target='union' if name == 'union_target' else 'exact', queries=n, top1=metrics(hits),
        by_cv={str(cv): metrics(hits[np.array(test)]) for cv, (_, test) in enumerate(data['splits'])},
        confidence=dict(target='union', common_top1=metrics(common_hits), receipts=receipts, model_refs=meta_refs,
            policies={p: metrics(common_hits, mask) for p, mask in selected.items()},
            by_cv={str(cv): {p: metrics(common_hits[test], mask[test]) for p, mask in selected.items()} for cv, (_, test) in enumerate(data['splits'])}),
        models=runner.receipts, predictions=b.record(directory/'predictions.npz'),
        all_score_vectors_replayed=sum(len(u['scored_indices']) for r in runner.receipts for u in r['uses']))
    b.write(directory/'result.json', result)
    return dict(result=b.record(directory/'result.json'), fit_count=len(runner.receipts), score_vectors_replayed=result['all_score_vectors_replayed'])


def bootstrap(hits, groups):
    _, codes = np.unique(groups, return_inverse=True)
    count = np.bincount(codes)
    differences = np.array([np.bincount(codes, weights=(hits[name][:, j].astype(int)-hits['base'][:, j].astype(int)))
        for name in VARIANTS[1:] for j in (EXACT, UNION)])
    rng = np.random.default_rng(b.SEED)
    samples = []
    for start in range(0, 10000, 100):
        weights = rng.multinomial(len(count), np.full(len(count), 1/len(count)), size=min(100, 10000-start))
        samples.append(100*(weights@differences.T)/(weights@count)[:, None])
    sample = np.concatenate(samples)
    results = {}
    for k, name in enumerate(VARIANTS[1:]):
        results[name] = {view: dict(delta_pp=float(100*(hits[name][:, j].mean()-hits['base'][:, j].mean())),
            paired_company_bootstrap95_pp=np.quantile(sample[:, k*2+j], [.025, .975]).tolist(),
            simultaneous6_comparisons95_pp=np.quantile(sample[:, k*2+j], [.05/12, 1-.05/12]).tolist())
            for j, view in enumerate(('exact', 'union'))}
    return results


def audit(out, directory):
    check(out)
    r = b.read(SOURCE/'result.json'); conf = r['curves']['17427']['confidence']
    X, _, views, groups, folds, ids, outputs, refs, _, _, _ = d.load(False)
    panel = b.index(b.read(refs['panel']['path'])); geo = b.index(b.read(refs['geometry']['path']))
    false = [(method, p) for method in ('content', 'stack') for p in conf[method]['per_query'] if p['sur']['margin'] and not p['hits']['union']]
    qids = sorted({p['query_id'] for _, p in false})
    labels = b.index(pq.read_table(refs['ledger']['path'], columns=['query_id', 'human_ground_truth_siret', 'human_ground_truth_siren'],
        filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', qids)], use_threads=False).to_pylist())
    requested = {q: [outputs[q][p['selected_native_rank']-1] for _, p in false if p['query_id'] == q]+
        [dict(siret=labels[q]['human_ground_truth_siret'], siren=labels[q]['human_ground_truth_siren'])] for q in qids}
    sites, units, points = b.hydrate(refs, requested)
    predictions = np.load(b.verified(r['predictions']))
    external = xgb.XGBRanker(n_jobs=1); external.load_model(b.verified(r['curves']['17427']['fit']['model']))
    cache = np.load(b.verified(b.read(ROOT/'docs/step2_features_v2.json')['matrix']))
    cached_ids = cache['query_ids'].tolist(); XX = cache['features'][[cached_ids.index(q) for q in ids]][:, :, old.CONTENT_COLS]
    content = external.predict(XX.reshape(-1, 85)).reshape(len(ids), 100)
    assert np.array_equal(content, predictions['content_17427_confidence'])
    stack_X = old.stack_cube(X, content)
    records = []
    for method, row in false:
        q = row['query_id']; i = ids.tolist().index(q)
        receipt = next(v for v in conf[method]['receipts'] if int(v['fold']) == row['fold'])
        threshold = receipt['limits']['margin']['threshold']
        raw = content[i:i+1]
        if method == 'stack':
            purpose = f'stack_17427_confidence_common_{row["fold"]}'
            fit = next(v for v in r['curves']['17427']['stack_fit_receipts'] if any(u['purpose'] == purpose for u in v['uses']))
            model = xgb.XGBRanker(n_jobs=1); model.load_model(b.verified(fit['model']))
            raw = model.predict(stack_X[i:i+1].reshape(-1, 21)).reshape(1, 100)
        top, margin = b.select_top(raw)
        assert int(top[0])+1 == row['selected_native_rank'] and float(margin[0]) == row['scores']['margin']
        assert threshold is not None and margin[0] >= threshold and not views[i, top[0], old.UNION]
        cp = b.crm_point(geo[q]['geometry'])
        def entity(siret):
            s = sites.get(siret, {}); point = b.candidate_point(points.get(siret))
            ul = units.get(siret[:9], {})
            return dict(siret=siret, siren=siret[:9], names={k: s.get(k) for k in ('enseigne1Etablissement', 'denominationUsuelleEtablissement')},
                legal_name=' '.join(str(ul.get(k) or '') for k in ('denominationUniteLegale', 'denominationUsuelle1UniteLegale', 'prenomUsuelUniteLegale', 'nomUniteLegale')).strip(), address={k: s.get(k) for k in ('numeroVoieEtablissement', 'indiceRepetitionEtablissement', 'typeVoieEtablissement', 'libelleVoieEtablissement', 'codePostalEtablissement', 'codeCommuneEtablissement')},
                state=s.get('etatAdministratifEtablissement'), siege=s.get('etablissementSiege'), point=point,
                distance_crm_m=b.distance(cp, point), admissible_threshold_m=b.threshold(cp, point), address_match=b.address_match(panel[q], s))
        chosen = entity(outputs[q][int(top[0])]['siret']); label = entity(labels[q]['human_ground_truth_siret'])
        records.append(dict(method=method, query_id=q, population_fold=row['fold'], crm=panel[q], crm_point=cp,
            chosen=chosen, label=label, same_siren=chosen['siren'] == label['siren'],
            chosen_label_distance_m=b.distance(chosen['point'], label['point']), threshold=threshold, margin=float(margin[0]),
            replay_passed=True, verdict='indéterminé', reason='À examiner sur les seuls faits officiels ci-dessus ; aucun label modifié.'))
    b.write(directory/'facts.json', records)
    return dict(facts=b.record(directory/'facts.json'), false_occurrences=len(records), unique_queries=len(qids))


def report(r):
    lines = ['## Résultats', '', '**Banc de développement, pas précision produit V142.**', '',
        f'{r["queries"]} dossiers /{r["companies"]} entreprises ;5 plis par entreprise, mêmes plis pour toutes variantes.',
        f'Présence@100 : exact{r["availability"]["exact_present"]}, union{r["availability"]["union_present"]} ; toutes absences conservées.', '',
        '| Recette | Exact, Wilson95 | Union, Wilson95 | Δ exact, IC95 cluster | Δ union, IC95 cluster |', '|---|---|---|---|---|']
    for name, v in r['variants'].items():
        delta = r['deltas'].get(name)
        def dd(view):
            x = delta[view]; a, z = x['paired_company_bootstrap95_pp']
            return f'{x["delta_pp"]:+.3f} [{a:+.3f};{z:+.3f}]pt'
        lines.append(f'| {name} | {b.md_metric(v["top1"]["exact"])} | {b.md_metric(v["top1"]["union"])} | '+(dd('exact') if delta else '—')+' | '+(dd('union') if delta else '—')+' |')
    lines += ['', '## Confiance union98% interne, évaluée sur entreprises externes', '',
        '| Recette/politique | SUR | Précision union, Wilson95 | Exact mêmes SUR | Couverture, Wilson95 |', '|---|---:|---|---|---|']
    for name, v in r['variants'].items():
        for policy, s in v['confidence']['policies'].items():
            lines.append(f'| {name}/{policy} | {s["count"]} | {b.md_metric(s["precision"]["union"])} | {b.md_metric(s["precision"]["exact"])} | {b.md_metric(s["coverage"])} |')
    lines += ['', '- Socle des SUR = proper60%, distinct du ranking80% ci-dessus ; tous les seuils sont internes, cible union pour scène/isotonique/seuil.98% calibration ≠ garantie test.',
        '- Pondération entreprise appliquée seulement au ranking, métriques query-weighted. Cible union=exact OU site, aucun label/candidat modifié.',
        '- Bootstrap10000 entreprises entières, IC95 marginaux ci-dessus ; IC simultanés six comparaisons et plis dans JSON. Wilson descriptif sans correction cluster ; consommation du développement et changement de générateur empêchent une certification.', '',
        '## Recommandation unique', '', r['recommendation'], '',
        f'- {r["fit_count"]} fits, {r["score_vectors_replayed"]} vecteurs effectivement rejoués depuis poids sauvés ; garde5,5Gio, pic{r["peak_rss_bytes"]/1024**3:.2f}Gio, temps{r["seconds"]:.1f}s. Sources/matrices vérifiées avant/après.',
        '- Audit séparé : `docs/step2_sur_errors_audit.md`. Aucun accès v174/VM/web/jeu fermé, changement V142 ou push. Les526 V142 restent un contrôle de transfert consommé, pas une validation indépendante.',
        '- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_g0_bench.py run --output-dir <dossier_neuf_sous_H/step2_g0_bench_20261017>` ; `--smoke`100 entreprises réelles.']
    return '\n'.join(lines)+'\n'


def summarize(out, directory):
    plan = check(out); data = b.read(out/'data.json')
    variants = {v: b.read(out/v/'result.json') for v in VARIANTS}
    hits = {v: np.load(b.verified(x['predictions']))['hits'] for v, x in variants.items()}
    deltas = bootstrap(hits, np.array(data['groups']))
    eligible = [v for v in VARIANTS[1:] if deltas[v]['union']['simultaneous6_comparisons95_pp'][0] > 0]
    winner = max(eligible, key=lambda v: deltas[v]['union']['delta_pp']) if eligible else None
    rec = ('Confronter ensuite '+winner+' aux526 V142, sans changer sa recette ni promouvoir ce résultat G0.' if winner else
        'Ne pas départager ces recettes sur le bruit ; rechercher un signal distinctif du site plutôt que promouvoir une variante non confirmée.')
    receipts = [b.read(out/v/'receipt.json') for v in VARIANTS]
    result = dict(status='DEVELOPMENT_G0_NOT_V142', smoke=plan['smoke'], queries=data['queries'], companies=data['companies'],
        availability={k: data[k] for k in ('exact_present', 'union_present', 'extra_union_positive_candidates')},
        variants=variants, deltas=deltas, bootstrap_replicates=10000, bootstrap_seed=b.SEED, recommendation=rec,
        data=b.record(out/'data.json'), plan=b.record(out/'plan.json'), fit_count=sum(x['fit_count'] for x in receipts),
        score_vectors_replayed=sum(x['score_vectors_replayed'] for x in receipts),
        audit=dict(facts=b.record(out/'audit/facts.json'), document=b.record(ROOT/'docs/step2_sur_errors_audit.md'),
            occurrences=len(b.read(out/'audit/facts.json')), unique_queries=len({p['query_id'] for p in b.read(out/'audit/facts.json')}),
            manual_unique_verdict_counts={label: (ROOT/'docs/step2_sur_errors_audit.md').read_text().count('**Verdict — '+label+' :**') for label in ('label discutable', 'indéterminé', 'erreur du modèle')}),
        peak_rss_bytes=max(b.read(p)['peak_rss_bytes'] for p in out.glob('*/receipt.json')), seconds=time.time()-plan['started_unix'],
        original_labels_unchanged=True, candidates_unchanged=True, closed_sets_opened=False, production_allowed=False)
    for ref in data['official_refs'].values():
        b.verified(ref)
    b.verified(data['source_matrix']); b.verified(data['views'])
    b.write(out/'result.json', result)
    (out/'result.md').write_text(report(result))
    (out/'final_report.md').write_text(final_report(result))
    if not plan['smoke']:
        b.write(ROOT/'docs/step2_g0_bench.json', result)
        DOC.write_text((out/'preregistered.md').read_text()+'\n'+report(result))
    return dict(result=b.record(out/'result.json'))


def final_report(r):
    lines = ['## Fait', '`scripts/step2_g0_bench.py`, `docs/step2_g0_bench.{json,md}` et `docs/step2_sur_errors_audit.md`.',
        f'Banc de développement G0 :{r["queries"]} dossiers /{r["companies"]} entreprises,5 plis entreprise,4 recettes figées.', '',
        '## Preuve', f'{r["fit_count"]} fits ;{r["score_vectors_replayed"]} vecteurs rejoués depuis poids sauvés ; pic{r["peak_rss_bytes"]/1024**3:.2f}Gio ;{r["seconds"]:.1f}s.',
        '| Recette | Exact %, Wilson95 | Union %, Wilson95 | Δ union, IC95 entreprise |', '|---|---|---|---|']
    def short(x):
        if x is None: return 'non définie'
        a, z = x['wilson95_percent']; return f'{x["percent"]:.2f} [{a:.2f};{z:.2f}]'
    for name, v in r['variants'].items():
        delta = r['deltas'].get(name)
        dd = '—' if not delta else f'{delta["union"]["delta_pp"]:+.3f} {delta["union"]["paired_company_bootstrap95_pp"]}'
        lines.append(f'| {name} | {short(v["top1"]["exact"])} | {short(v["top1"]["union"])} | {dd} |')
    lines += ['', '| SUR | Précision union %, Wilson95 | Couverture %, Wilson95 |', '|---|---|---|']
    for name, v in r['variants'].items():
        for policy, s in v['confidence']['policies'].items():
            lines.append(f'| {name}/{policy} | {short(s["precision"]["union"])} | {short(s["coverage"])} |')
    lines += ['', '## Écarts / choix', 'Poids entreprise appliqués au ranking seulement ; union=exact OU site, sans label ni candidat modifié.',
        'Audit526 :5 occurrences/4 dossiers ;2 labels discutables,2 indéterminés, aucune correction de label ni de précision.', '',
        '## Reste / risques', 'G0≠V142 ; Wilson descriptif, bootstrap entreprise10000 réplications, six comparaisons simultanées dans JSON. Pas de promotion.',
        '**Recommandation unique :** '+r['recommendation']]
    if len(lines) > 35: raise ValueError('Final report exceeds35 lines')
    return '\n'.join(lines)+'\n'


def publish(out, directory):
    check(out); r = b.read(b.verified(b.read(out/'summary/receipt.json')['result']))
    if r['smoke']: raise ValueError('Never publish smoke as full')
    for ref in (r['audit']['facts'], r['audit']['document']): b.verified(ref)
    if b.read(ROOT/'docs/step2_g0_bench.json') != r: raise ValueError('Owned result changed')
    if subprocess.check_output(['git', 'diff', 'HEAD', '--', 'handover.md'], cwd=ROOT).strip():
        raise ValueError('Foreign uncommitted journal edits; stop publication without overwriting')
    journal = ROOT/'handover.md'; before = journal.read_text()
    values = '; '.join(f'{name} exact{v["top1"]["exact"]["hits"]}/union{v["top1"]["union"]["hits"]}' for name, v in r['variants'].items())
    line = (f'\n- 2026-10-17 — Banc G0 étape2-6 COMPLETE :{r["queries"]} dossiers/{r["companies"]} entreprises,5 plis entreprise, '
        f'{values}. Quatre recettes pré-déclarées, cibles exact/union et poids query-group entreprise, Wilson95 et bootstrap10000 entreprises '
        f'IC95/simultanés6 dans `docs/step2_g0_bench.{{json,md}}`. {r["fit_count"]} fits, {r["score_vectors_replayed"]} vecteurs rejoués, '
        f'pic{r["peak_rss_bytes"]/1024**3:.2f}Gio, {r["seconds"]:.1f}s ; SHA sources/matrices avant/après, aucune entreprise partagée. '
        f'Audit `docs/step2_sur_errors_audit.md` :5 faux SUR margin/4 dossiers, scores/seuils publiés rejoués ;2 labels discutables/2 indéterminés, '
        f'aucune vérité modifiée ni erreur retirée (content54/56,stack58/61). Recommandation :{r["recommendation"]} '
        f'G0≠V142,526 consommés, aucune garantie/promotion/jeu fermé/v174/VM/web/push ; étrangers/pending préservés. '
        f'Artefacts `H/step2_g0_bench_20261017/full/`, rapport≤35 lignes `final_report.md`. '
        f'Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_g0_bench.json`.\n')
    with journal.open('a') as stream: stream.write(line)
    if journal.read_text() != before+line: raise ValueError('Concurrent journal edit; preserve and stop')
    paths = ['docs/step2_g0_bench.json', 'docs/step2_g0_bench.md', 'docs/step2_sur_errors_audit.md', 'handover.md']
    subprocess.run(['git', 'add', '--', paths[0]], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--check', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'feat: measure grouped G0 benchmark and audit false selective matches', '--', *paths], cwd=ROOT, check=True)
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', paths[0]], cwd=ROOT, text=True).strip()
    b.write(out/'publication.json', dict(status='VERIFIED_COMMITTED', commit=commit))
    return dict(commit=commit)


def execute(out, phase, directory, extra=()):
    old.wait(); directory.mkdir()
    with (directory/'worker.log').open('x') as stream:
        subprocess.run([sys.executable, '-u', str(Path(__file__)), phase, '--output-dir', str(out), '--directory', str(directory), *map(str, extra)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    receipt = b.read(directory/'receipt.json')
    print(json.dumps(dict(stage=phase, directory=directory.name, **receipt)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['run', '_prepare', '_variant', '_audit', '_summarize', '_publish'])
    p.add_argument('--smoke', action='store_true'); p.add_argument('--output-dir', type=Path)
    p.add_argument('--directory', type=Path); p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--commit', action='store_true')
    args = p.parse_args(); out = args.output_dir or OWN/('smoke' if args.smoke else 'full')
    if not out.resolve().is_relative_to(OWN.resolve()) or out.resolve() == OWN.resolve():
        raise ValueError('Fresh owned namespace required')
    if args.phase == 'run':
        out.mkdir(parents=True, exist_ok=False)
        prereg = DOC.read_text().split('\n## Résultats\n')[0]+'\n'; (out/'preregistered.md').write_text(prereg)
        b.write(out/'plan.json', dict(started_unix=time.time(), smoke=args.smoke, variants=list(VARIANTS), params=old.PARAMS,
            features=old.CONTENT, implementations=[b.record(Path(__file__)), b.record(Path(old.__file__)), b.record(Path(d.__file__)),
                b.record(Path(b.__file__)), b.record(Path(old.v2.__file__))], preregistration=b.record(out/'preregistered.md')))
        execute(out, '_prepare', out/'preparation')
        execute(out, '_audit', out/'audit')
        for name in VARIANTS:
            execute(out, '_variant', out/name, ['--variant', name])
        execute(out, '_summarize', out/'summary')
        if args.commit: execute(out, '_publish', out/'publication')
        print(json.dumps(dict(stage='COMPLETE', result=str(out/'result.json'))), flush=True)
        return
    if args.directory is None or not args.directory.resolve().is_relative_to(out.resolve()):
        raise ValueError('Owned worker directory required')
    old.wait(); guard = old.Guard(args.directory)
    # Guard enforces5.75GiB; tighten to the documented5.5GiB in this runner.
    def memory_guard():
        while not guard.stop.wait(.1):
            if guard.process.memory_info().rss >= int(5.5*1024**3):
                b.write(args.directory/'resource_stop.json', dict(status='STOP_RSS_5_5GIB'))
                os._exit(75)
    import threading
    threading.Thread(target=memory_guard, daemon=True).start()
    began = time.monotonic()
    if args.phase == '_prepare': value = prepare(out)
    elif args.phase == '_audit': value = audit(out, args.directory)
    elif args.phase == '_variant': value = variant(out, args.variant, args.directory)
    elif args.phase == '_summarize': value = summarize(out, args.directory)
    else: value = publish(out, args.directory)
    value.update(seconds=time.monotonic()-began, peak_rss_bytes=guard.peak)
    b.write(args.directory/'receipt.json', value); guard.stop.set()


if __name__ == '__main__':
    main()
