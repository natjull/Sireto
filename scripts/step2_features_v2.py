"""Fixed R3 with preregistered site features; union-calibrated confidence.

Immutable V142 candidates, exact ranking targets, grouped folds2/3/4 only.
New candidate features are sealed before the evaluation ledger is opened.
"""
import argparse
from collections import Counter
from datetime import date
import hashlib
import math
import os
from pathlib import Path
import pickle
import sys
import threading
import time
import warnings

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import psutil
import pyarrow as pa
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from scripts import step2_diagnostic_v1 as diag

base = diag.base
HOME = base.HOME
DEFAULT_RUN = HOME/'step2_features_v2_20261017'
DOC = ROOT/'docs/step2_features_v2.md'
HISTORY = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockEtablissementHistorique.parquet')
SNAPSHOT = date(2026, 8, 1)
FAMILIES = dict(
    address=['v2_number_exact', 'v2_number_within2', 'v2_number_both_missing',
        'v2_number_one_missing', 'v2_number_log_gap', 'v2_suffix_equal', 'v2_candidate_range',
        'v2_street_exact', 'v2_street_jaro', 'v2_street_token_overlap', 'v2_commune_match',
        'v2_distance_log', 'v2_distance_minus_native_log', 'v2_crm_ban_score', 'v2_candidate_ban_score',
        'v2_crm_street_similarity', 'v2_crm_log_uncertainty', 'v2_candidate_log_distance_precision',
        'v2_crm_house', 'v2_crm_street', 'v2_crm_commune', 'v2_candidate_house',
        'v2_candidate_street', 'v2_candidate_commune', 'v2_crm_declared_weak',
        'v2_geo_missing', 'v2_geo_admissible'],
    site_name=['v2_enseigne_jaro_max', 'v2_usual_jaro', 'v2_site_best_jaro',
        'v2_site_name_present', 'v2_site_name_exact', 'v2_enseigne_token_overlap',
        'v2_usual_token_overlap', 'v2_enseigne_other_count', 'v2_site_name_gap_same_siren',
        'v2_site_vs_ul_jaro_gap'],
    status_dates=['v2_is_active', 'v2_is_closed', 'v2_is_siege', 'v2_closure_age_log_days',
        'v2_closure_recency_vs_siren_days', 'v2_closure_missing', 'v2_active_same_siren_pool_count',
        'v2_active_same_siren_crm_commune_count', 'v2_state_missing'],
    siren_scene=['v2_siren_first_native_rank', 'v2_siren_candidate_count',
        'v2_siren_is_nearest', 'v2_siren_distance_excess_log'])
NEW_FEATURES = [f for names in FAMILIES.values() for f in names]
FEATURES = base.FEATURES+NEW_FEATURES
UNION = base.VIEWS.index('union')


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (ValueError, TypeError):
        return math.nan


def log(value):
    value = numeric(value)
    return math.log1p(value) if value >= 0 else math.nan


def feature_inputs(smoke, refs):
    """Only CRM, frozen lists, official sources and the cached label-free V1 cube."""
    panel = base.index(base.read(refs['panel']['path']))
    ids_all = sorted(panel)
    ids = ids_all
    if smoke:
        ids = sorted(q for fold in (2, 3, 4) for q in sorted(
            [q for q in ids_all if panel[q]['fold'] == fold],
            key=lambda q: hashlib.sha256(q.encode()).hexdigest())[:12])
    if len(ids_all) != 526 or any(panel[q]['fold'] not in (2, 3, 4) for q in ids):
        raise ValueError('Only sealed526 and folds2/3/4')
    panel = {q: panel[q] for q in ids}
    outputs = base.read(refs['outputs']['path'])['learned_native_ooc']
    outputs = {q: outputs[q] for q in ids}
    if any(len(cs) != 100 or len({c['siret'] for c in cs}) != 100 for cs in outputs.values()):
        raise ValueError('Frozen pool changed')
    geometry = base.index(base.read(refs['geometry']['path']))
    cached = np.load(refs['feature_matrix']['path'])['features']
    if cached.shape != (526, 100, 46):
        raise ValueError('V1 cache changed')
    X = cached[[ids_all.index(q) for q in ids]]
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    requested = sorted({c['siret'] for cs in outputs.values() for c in cs})
    con.register('requested', pa.table({'siret': requested}))

    def select(key, columns):
        sql = 'SELECT '+','.join('s.'+c for c in columns)+' FROM read_parquet(?) s SEMI JOIN requested r USING(siret)'
        return base.index(con.execute(sql, [refs[key]['path']]).fetch_arrow_table().to_pylist(), 'siret')

    sites = select('stock', base.STOCK_COLUMNS)
    points = select('study', ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision'])
    history_sql = """SELECT siret, MAX(try_cast(dateDebut AS DATE)) AS closure_date
        FROM read_parquet(?) h SEMI JOIN requested r USING(siret)
        WHERE etatAdministratifEtablissement='F'
        AND try_cast(changementEtatAdministratifEtablissement AS BOOLEAN)=true
        AND try_cast(dateDebut AS DATE)<=DATE '2026-08-01' GROUP BY siret"""
    closures = {r['siret']: r['closure_date'] for r in con.execute(history_sql, [refs['history']['path']]).fetch_arrow_table().to_pylist()}
    pairs = sorted({(c['siren'], panel[q]['crm_insee']) for q, cs in outputs.items() for c in cs if panel[q]['crm_insee']})
    con.register('requested_pairs', pa.table({'siren': [p[0] for p in pairs], 'commune': [p[1] for p in pairs]}))
    count_sql = """SELECT s.siren, s.codeCommuneEtablissement AS commune, count(*) AS active_count
        FROM read_parquet(?) s JOIN requested_pairs r
        ON s.siren=r.siren AND s.codeCommuneEtablissement=r.commune
        WHERE s.etatAdministratifEtablissement='A' AND s.statutDiffusionEtablissement='O'
        GROUP BY s.siren, s.codeCommuneEtablissement"""
    counts = {(r['siren'], r['commune']): r['active_count'] for r in con.execute(count_sql, [refs['stock']['path']]).fetch_arrow_table().to_pylist()}
    con.close()
    return X, panel, outputs, geometry, sites, points, closures, counts


def feature_cube(X, panel, outputs, geometry, sites, points, closures, counts):
    """No ledger, human SIREN, acceptable sidecar, labels or metric flags here."""
    cube = []
    for qi, (qid, q) in enumerate(panel.items()):
        cs = outputs[qid]
        crm = base.crm_point(geometry[qid]['geometry'])
        query_name = base.normalize_name(base.text(q['crm_name']))
        qstreet = base.normalize(base.text(q['crm_street']), street=True)
        qnum = base.number(q['crm_number'], q['crm_suffix'])
        public = [sites.get(c['siret'], {}) if sites.get(c['siret'], {}).get('statutDiffusionEtablissement') == 'O' else {} for c in cs]
        pts = [base.candidate_point(points.get(c['siret'])) if s else base.candidate_point(None) for c, s in zip(cs, public)]
        distances = [base.distance(crm, p) for p in pts]
        eligible = [d if d is not None and base.threshold(crm, p) is not None else None for d, p in zip(distances, pts)]
        enseignes = [{base.normalize_name(base.text(s.get(k))) for k in ('enseigne1Etablissement', 'enseigne2Etablissement', 'enseigne3Etablissement')} - {''} for s in public]
        usual = [base.normalize_name(base.text(s.get('denominationUsuelleEtablissement'))) for s in public]
        site_names = [max([base.jaro(query_name, n) for n in en]+[base.jaro(query_name, us)]) for en, us in zip(enseignes, usual)]
        closure = [closures.get(c['siret']) if s.get('etatAdministratifEtablissement') == 'F' else None for c, s in zip(cs, public)]
        group_indices = {g: [i for i, c in enumerate(cs) if c['siren'] == g] for g in {c['siren'] for c in cs}}
        active = Counter(c['siren'] for c, s in zip(cs, public) if s.get('etatAdministratifEtablissement') == 'A')
        rows = []
        for i, (c, s, p, d) in enumerate(zip(cs, public, pts, distances)):
            family = group_indices[c['siren']]
            ds = [eligible[j] for j in family if eligible[j] is not None]
            dates = [closure[j] for j in family if closure[j] is not None]
            others = [site_names[j] for j in family if j != i]
            candidate_range = bool(s.get('dernierNumeroVoieEtablissement') or s.get('indiceRepetitionDernierNumeroVoieEtablissement'))
            snum = base.number(s.get('numeroVoieEtablissement'), s.get('indiceRepetitionEtablissement')) if not candidate_range else None
            street = base.normalize(' '.join(base.text(s.get(k)) for k in ('typeVoieEtablissement', 'libelleVoieEtablissement')), street=True)
            gap = abs(qnum[0]-snum[0]) if qnum and snum else None
            values = [bool(gap == 0), bool(gap is not None and gap <= 2), not qnum and not snum,
                bool(qnum) != bool(snum), log(gap), bool(qnum and snum and qnum[1] == snum[1]), candidate_range,
                bool(qstreet and street == qstreet), base.jaro(qstreet, street), base.overlap(qstreet, street),
                bool(q['crm_insee'] and q['crm_insee'] == s.get('codeCommuneEtablissement')),
                log(d), log(d)-log(distances[0]), math.nan, math.nan,
                numeric(geometry[qid]['geometry'].get('ban_street_similarity')), log(crm['uncertainty_m']), log(p['distance_precision_m']),
                crm['precision'] == 'housenumber_equivalent', crm['precision'] == 'street_imprecise', crm['precision'] == 'commune',
                p['precision'] == 'housenumber_equivalent', p['precision'] == 'street_imprecise', p['precision'] == 'commune',
                bool(crm['declared_weak']), d is None, base.threshold(crm, p) is not None,
                max((base.jaro(query_name, n) for n in enseignes[i]), default=0.), base.jaro(query_name, usual[i]), site_names[i],
                bool(enseignes[i] or usual[i]), bool(query_name and query_name in enseignes[i] | {usual[i]}),
                max((base.overlap(query_name, n) for n in enseignes[i]), default=0.), base.overlap(query_name, usual[i]),
                sum(bool(enseignes[i] & en) for j, en in enumerate(enseignes) if j != i),
                site_names[i]-max(others) if others else math.nan, site_names[i]-X[qi, i, base.FEATURES.index('name_ul_jaro')],
                s.get('etatAdministratifEtablissement') == 'A', s.get('etatAdministratifEtablissement') == 'F',
                str(s.get('etablissementSiege')).lower() in ('true', '1'),
                log((SNAPSHOT-closure[i]).days) if closure[i] else math.nan,
                (closure[i]-max(dates)).days if closure[i] and dates else math.nan,
                s.get('etatAdministratifEtablissement') == 'F' and closure[i] is None,
                active[c['siren']], counts.get((c['siren'], q['crm_insee']), 0) if q['crm_insee'] else math.nan,
                s.get('etatAdministratifEtablissement') not in ('A', 'F'),
                family[0]+1, len(family), eligible[i] == min(ds) if eligible[i] is not None else math.nan,
                log(eligible[i])-log(min(ds)) if eligible[i] is not None else math.nan]
            if len(values) != len(NEW_FEATURES):
                raise ValueError('Preregistered feature count changed')
            rows.append(values)
        cube.append(rows)
    result = np.asarray(cube, dtype=np.float32)
    if result.shape != (len(panel), 100, 50) or np.isinf(result).any():
        raise ValueError('Invalid label-free v2 cube')
    return result


def gemma_inventory(outputs, refs):
    """Read-only coverage upper bound: no model loading, tokenizing or encoding."""
    started = time.monotonic()
    root = HOME/'v14/representative_20261008_b/zones'
    requested = {c['siret'] for cs in outputs.values() for c in cs}
    covered, inventories, zones = set(), [], 0
    for zone in sorted(root.iterdir()):
        if not zone.is_dir() or '.pending' in zone.name:
            continue
        docs_path = zone/'documents.json'
        binding = zone/'v3_official512/texts_binding.json'
        if not binding.exists() and (zone/'v3_official512.seal.json').exists():
            binding = Path(base.read(zone/'v3_official512.seal.json')['path']).parent/'v3_official512/texts_binding.json'
        if not docs_path.exists() or not binding.exists():
            continue
        before = base.fingerprint(docs_path)
        docs = base.read(docs_path)
        covered.update(d['key'] for d in docs if d['key'] in requested)
        if base.fingerprint(docs_path) != before:
            raise ValueError('Site cache inventory changed during read')
        inventories.append(before)
        zones += 1
        del docs
    missing = requested-covered
    if not missing:
        raise ValueError('STOP: full potential site coverage requires verified tensor-row/query mapping; do not silently skip Gemma')
    source = HOME/'v33/global_cached_parent_dense_20261012_b'
    refs['gemma_query_binding'] = base.record(source/'query_vector_binding.json')
    refs['gemma_encoding_start'] = base.record(source/'encode_queries_process.json')
    refs['gemma_encoding_exit'] = base.record(source/'encode_queries_exit.json')
    elapsed = base.read(refs['gemma_encoding_exit']['path'])['finished_unix']-base.read(refs['gemma_encoding_start']['path'])['started_unix']
    return dict(status='SKIPPED_INCOMPLETE_SITE_VECTORS', feature='v2_gemma_cosine',
        unique_sites_requested=len(requested), potential_covered_sites=len(covered), definitely_missing_sites=len(missing),
        covered_is_upper_bound=True, source_zones=zones, inventory_fingerprints=inventories,
        audit_seconds=time.monotonic()-started, new_encodings=0,
        reason='Official site-vector cache is incomplete even before validating tensor-row mapping. Persisted names-only query/SIREN identity vectors are not complete site vectors.',
        historical_names_query_encoding=dict(queries=526, elapsed_seconds=elapsed,
            seconds_per_query_including_load=elapsed/526),
        conditional_missing_sites_names_rate_seconds=elapsed/526*len(missing),
        conditional_full_queries_and_sites_names_rate_seconds=elapsed/526*(526+len(requested)),
        cost_warning='Historical names-only throughput incl model load; arithmetic scenario, NOT a measured or guaranteed512-token site encoding budget. No fresh encoding or runtime/model installation attempted.')


class Runner:
    def __init__(self, X, y, groups, ids, out):
        self.X, self.y, self.groups, self.ids, self.out = X, y, groups, ids, out
        out.mkdir()
        self.models, self.receipts = {}, []

    def score(self, train, test, columns, purpose):
        train, test = np.sort(train), np.asarray(test)
        if set(self.groups[train]) & set(self.groups[test]):
            raise ValueError('Company leakage '+purpose)
        key = diag.digest([train.tolist(), list(columns)])
        if key not in self.models:
            params = dict(base.PARAMS, objective='rank:ndcg')
            model = xgb.XGBRanker(**params)
            target = self.y[train].reshape(-1)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(self.X[train][:, :, columns].reshape(-1, len(columns)), target, group=np.full(len(train), 100))
            path = self.out/(key+'.json')
            model.save_model(path)
            receipt = dict(key=key, params=params, features=[FEATURES[i] for i in columns],
                train_queries=self.ids[train].tolist(), train_companies=len(set(self.groups[train])),
                zero_positive_queries=int((self.y[train].sum(axis=1) == 0).sum()),
                target_sha256=hashlib.sha256(target.tobytes()).hexdigest(), model=base.record(path),
                warnings=[str(w.message) for w in captured], uses=[])
            self.receipts.append(receipt)
            self.models[key] = model, receipt
        model, receipt = self.models[key]
        receipt['uses'].append(dict(purpose=purpose, scored_queries=self.ids[test].tolist(), company_overlap=0))
        result = model.predict(self.X[test][:, :, columns].reshape(-1, len(columns))).reshape(len(test), 100)
        if not np.isfinite(result).all():
            raise ValueError('Invalid score')
        return result


def confidence(runner, outer, X, views, groups, folds, ids, columns, out, variant):
    selected = {m: np.zeros(len(ids), dtype=bool) for m in ('margin', 'scene_isotonic')}
    scores = {m: np.zeros(len(ids)) for m in selected}
    common = np.full((len(ids), 100), np.nan)
    receipts, models = [], []
    for fold, train, test in outer:
        companies = np.random.default_rng(base.SEED+100+int(fold)).permutation(np.array(sorted(set(groups[train]))))
        count = max(1, math.ceil(.2*len(companies)))
        limit = train[np.isin(groups[train], companies[:count])]
        iso_rows = train[np.isin(groups[train], companies[count:2*count])]
        proper = train[~np.isin(groups[train], companies[:2*count])]
        roles = [proper, iso_rows, limit, test]
        if any(set(groups[a]) & set(groups[b]) for i, a in enumerate(roles) for b in roles[i+1:]):
            raise ValueError('Confidence roles overlap')
        meta = np.full((len(proper), 100), np.nan)
        splitter = GroupKFold(n_splits=3, shuffle=True, random_state=base.SEED)
        for k, (a, b) in enumerate(splitter.split(proper, groups=groups[proper])):
            meta[b] = runner.score(proper[a], proper[b], columns, f'{variant}_scene_meta_{fold}_{k}')
        joined = np.r_[iso_rows, limit, test]
        pred = runner.score(proper, joined, columns, f'{variant}_confidence_common_{fold}')
        pi, pl, pt = np.split(pred, [len(iso_rows), len(iso_rows)+len(limit)])
        common[test] = pt
        top, _ = base.select_top(meta)
        scene_target = views[proper, top, UNION].astype(int)
        classifier = diag.logistic(scene_target)
        classifier.fit(diag.scene(meta, X[proper, :, :46]), scene_target)
        itop, _ = base.select_top(pi)
        iso_target = views[iso_rows, itop, UNION].astype(int)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0., y_max=1.)
        iso.fit(diag.probability(classifier, diag.scene(pi, X[iso_rows, :, :46])), iso_target)
        path = out/f'{variant}_scene_union_fold{fold}.pkl'
        with path.open('wb') as file:
            pickle.dump((classifier, iso), file, protocol=5)
        models.append(base.record(path))
        ltop, lm = base.select_top(pl)
        ttop, tm = base.select_top(pt)
        limit_hits = views[limit, ltop]
        lp = iso.predict(diag.probability(classifier, diag.scene(pl, X[limit, :, :46])))
        tp = iso.predict(diag.probability(classifier, diag.scene(pt, X[test, :, :46])))
        limits = {}
        for name, cal, value in [('margin', lm, tm), ('scene_isotonic', lp, tp)]:
            threshold, stats = base.threshold_choice(cal, limit_hits[:, UNION])
            stats['union_precision'] = stats.pop('exact_precision')
            mask = cal >= threshold if threshold is not None else np.zeros(len(cal), dtype=bool)
            selected[name][test] = value >= threshold if threshold is not None else False
            scores[name][test] = value
            limits[name] = dict(threshold=threshold, calibration=stats,
                all_calibration_views=base.sur_metrics(limit_hits, mask))
        receipts.append(dict(fold=fold, proper_fit_queries=ids[proper].tolist(),
            scene_fit_queries=ids[proper].tolist(), isotonic_queries=ids[iso_rows].tolist(),
            threshold_queries=ids[limit].tolist(), test_queries=ids[test].tolist(), all_role_company_overlaps=0,
            scene_target='union', isotonic_target='union', threshold_target='union',
            scene_meta_target_sha256=hashlib.sha256(scene_target.tobytes()).hexdigest(),
            isotonic_target_sha256=hashlib.sha256(iso_target.tobytes()).hexdigest(), limits=limits))
    top, _ = base.select_top(common)
    hits = views[np.arange(len(ids)), top]
    result = dict(threshold_target='union', target_precision=.98,
        protocol='Same diagnostic60/20/20 companies; scene3-fold companyOOF ONLY within proper60%; isotonic20%, threshold20%, outer test untouched. Exact ranker labels unchanged.',
        scene_features=diag.SCENE_FEATURES, scene_C=1., scene_max_iter=1000,
        common_ranker_top1={v: base.metric(int(hits[:, j].sum()), len(ids)) for j, v in enumerate(base.VIEWS)},
        receipts=receipts, model_refs=models,
        methods={name: dict(totals=base.sur_metrics(hits, mask),
            by_fold={str(f): base.sur_metrics(hits[folds == f], mask[folds == f]) for f in (2, 3, 4)},
            union_point_target_met=bool(mask.sum() and hits[mask, UNION].mean() >= .98)) for name, mask in selected.items()},
        per_query=[dict(query_id=str(q), fold=int(folds[i]), selected_native_rank=int(top[i])+1,
            hits={v: bool(hits[i, j]) for j, v in enumerate(base.VIEWS)},
            scores={m: float(value[i]) for m, value in scores.items()},
            sur={m: bool(mask[i]) for m, mask in selected.items()}) for i, q in enumerate(ids)])
    return result, common


def ceiling(ids, y, X, refs):
    historic = base.read(refs['site_metric_result']['path'])
    rows = base.index(historic['per_query'])
    chosen = [rows[str(q)] for q in ids]
    far = [i for i, r in enumerate(chosen) if r['exact'] and r['label_site_diagnostic']['label_far_from_crm']]
    far_all = sum(bool(r['label_site_diagnostic']['label_far_from_crm']) for r in chosen)
    comparable_all = sum(bool(r['label_site_diagnostic']['comparable']) for r in chosen)
    comparable = sum(r['label_site_diagnostic']['comparable'] for r in chosen if r['exact'])
    present = int(y.any(axis=1).sum())
    seats = sum(bool(X[i, int(np.argmax(y[i])), base.FEATURES.index('is_siege')]) for i in far)
    if len(ids) == 526 and (len(far), comparable, present) != (86, 440, 507):
        raise ValueError('Existing label/CRM diagnostic changed')
    return dict(contract_commit=historic['contract_commit'], far_labels=len(far), far_labels_sieges=seats,
        far_labels_all=far_all, comparable_all=comparable_all, not_comparable_all=len(ids)-comparable_all,
        far_among_all=base.metric(far_all, len(ids)), far_among_present=base.metric(len(far), present),
        comparable_present=comparable, not_comparable_present=present-comparable,
        exact_global_oracle=base.metric(present, len(ids)),
        conclusion='Global exact98% impossible on this fixed pool (507/526). Exact98% on a selected SUR subset is NOT excluded. Far labels indicate CRM/exact-target tension, not incorrect labels and not an additional exact oracle ceiling.')


def report(r):
    lines = ['## Résultats', '', '**Mesures de développement, aucune promotion production.**', '',
        f'{r["queries"]}×100 candidats,46+50 features ; Gemma sautée. Cinq vues et folds, Wilson et paires détaillés dans le JSON.', '',
        '| Variante | Exact | SIREN | Opérationnel | Site | Union | Gains/pertes exact vsV1 | p exact / union vsV1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, a in r['variants'].items():
        t = a['totals'];p = t['paired']['v1']['exact']
        lines.append('| '+name+' | '+' | '.join(str(t['top1'][v]['hits']) for v in base.VIEWS)+
            f' | +{len(p["gains"])}/−{len(p["losses"])} | {p["exact_two_sided_p"]:.4g} / {t["paired"]["v1"]["union"]["exact_two_sided_p"]:.4g} |')
    lines += ['', '- Ajouts seuls = V1+UNE famille ; pas une recherche du meilleur sous-ensemble. Toutes les variantes annoncées sont publiées, aucun réglage après mesure.',
        '- McNemar par requête exploratoire, Holm sur exact/union des variantes ajoutées ; ni correction des corrélations intra-entreprise ni validation indépendante d’un pilote déjà consommé.', '',
        '### Confiance calibrée sur UNION', '',
        '| Ranker / confiance | SUR | Précision union, Wilson95 | Précision exacte des mêmes SUR, Wilson95 | Couverture, Wilson95 |',
        '|---|---:|---|---|---|']
    for variant, c in r['confidence'].items():
        for method, m in c['methods'].items():
            t = m['totals']
            lines.append(f'| {variant}/{method} | {t["count"]} | {base.md_metric(t["precision"]["union"])} | {base.md_metric(t["precision"]["exact"])} | {base.md_metric(t["rate"])} |')
    lines += ['', '- Cible union utilisée pour le classifieur scène, l’isotonique ET le seuil. Les modèles ranking restent entraînés sur EXACT ; aucun label transformé/retiré.',
        '- Les socles de confiance60% ne sont pas les full-fits du tableau top1 : '+', '.join(f'{k} exact{v["common_ranker_top1"]["exact"]["hits"]}, union{v["common_ranker_top1"]["union"]["hits"]}' for k, v in r['confidence'].items())+'.',
        '- La précision empirique≥98% en calibration ne garantit pas98% sur test ; petits effectifs, Wilson descriptif par requête et pilote consommé. Aucun seuil appris sur les tests.', '',
        '### Disponibilité réelle des informations', '',
        '- Scores BAN probabilistes CRM/candidat : absents, deux colonnes intégralement NaN. Incertitude CRM et qualité/distance_precision SIRENE restent distinctes. Pas de nouveau géocodage, seuils union200/500m inchangés.',
        '- Fermetures uniquement via transitions historiques officielles ; couverture parmi les candidats fermés : '+str(r['availability']['closed_with_date'])+'/'+str(r['availability']['closed_candidate_rows'])+'. DateDebut du stock non utilisée comme date de fermeture.',
        f'- Audit Gemma : {r["gemma"]["audit_seconds"]:.2f}s, {r["gemma"]["definitely_missing_sites"]}/{r["gemma"]["unique_sites_requested"]} sites définitivement absents de la banque site inspectée ; couverture restante seulement une borne haute avant vérification du mapping tenseur. **Famille sautée**, zéro réencodage.',
        f'- Référence historique mesurée :526 vecteurs requête noms seuls en{r["gemma"]["historical_names_query_encoding"]["elapsed_seconds"]:.2f}s, chargement inclus. Extrapolation arithmétique conditionnelle aux sites manquants :{r["gemma"]["conditional_missing_sites_names_rate_seconds"]:.1f}s AU MÊME débit noms seuls ; **pas un coût mesuré/prédit pour les textes512 tokens**, coût réel non établi et aucun encodage massif lancé.', '',
        '### Plafond exact et désaccord adresse', '',
        f'- Contrat `fb604665` reproduit : {r["ceiling"]["far_labels"]}/507 labels présents loin du CRM, dont {r["ceiling"]["far_labels_sieges"]} sièges ; {r["ceiling"]["comparable_present"]} comparables, {r["ceiling"]["not_comparable_present"]} non mesurables parmi les vérités présentes. En incluant les absences du pool : {r["ceiling"]["far_labels_all"]}/{r["queries"]} labels loin ({r["ceiling"]["far_among_all"]["percent"]:.2f}%), {r["ceiling"]["comparable_all"]} comparables et {r["ceiling"]["not_comparable_all"]} non mesurables.',
        f'- Plafond exact global du pool : {base.md_metric(r["ceiling"]["exact_global_oracle"])}. **98% global impossible avec ces100**, indépendamment du ranker.98% exact sur un sous-ensemble SUR reste possible en principe ; distance au CRM ≠ label erroné ou plafond exact supplémentaire.', '',
        '### Recommandation unique', '', r['recommendation'], '',
        '### Vérifications et limites', '',
        f'- Smoke réel puis run{r["queries"]} ; {r["fit_count"]} fits ranking, {r["elapsed_seconds"]:.2f}s, pic RSS{r["peak_rss_bytes"]/1024**2:.1f}Mio (<7,5Gio). Sources SHA avant/après ; inventaire Gemma par empreintes taille/mtime/inode.',
        '- V1 R3 reproduit à l’identique par requête ; nouveaux champs scellés avant lecture du ledger/sidecar. SHA des cibles et des modèles, matrices et rôles de calibration disponibles dans les reçus. Rappel100 et19 absences conservés, cinq vues inchangées.',
        '- Amont supervisé consommé/non full-OOF et routage humain inchangés ; aucun nouveau candidat, alias, ID comme feature, positif injecté, jeu fermé, VM, installation, web ou push.',
        '- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_features_v2.py --output-dir <dossier_CATNAT_neuf>` ; `--smoke`36 vrais dossiers, ne publie pas les docs.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir or DEFAULT_RUN/('smoke' if args.smoke else 'full')
    if out.exists():
        raise ValueError('Require a NEW owned directory')
    out.mkdir(parents=True)
    started, stop, resources = time.monotonic(), threading.Event(), dict(peak_rss_bytes=0)

    def monitor():
        process = psutil.Process()
        while not stop.is_set():
            resources['peak_rss_bytes'] = max(resources['peak_rss_bytes'], process.memory_info().rss)
            if resources['peak_rss_bytes'] >= int(7.5*1024**3):
                base.write(out/'resource_stop.json', resources)
                os._exit(75)
            stop.wait(.1)

    threading.Thread(target=monitor, daemon=True).start()
    prereg = DOC.read_text().split('\n## Résultats\n')[0]+'\n'
    if any('`'+f+'`' not in prereg for f in NEW_FEATURES) or len(NEW_FEATURES) != 50:
        raise ValueError('Feature preregistration mismatch')
    (out/'preregistered.md').write_text(prereg)
    v1 = base.read(ROOT/'docs/step2_xgb_ranker_v1.json')
    refs = dict(v1['source_refs'])
    refs['v1_implementation'] = refs.pop('implementation')
    refs.update(implementation=base.record(Path(__file__)), diagnostic_implementation=base.record(Path(diag.__file__)),
        diagnostic_result=base.record(ROOT/'docs/step2_diagnostic_v1.json'), history=base.record(HISTORY),
        feature_matrix=v1['feature_matrix'], site_metric_result=base.record(ROOT/'docs/retrieval_site_operational_metric_result_20261014.json'))
    for ref in refs.values():
        base.verified(ref)
    frozen = base.subprocess.check_output(['git', 'show', '36c7a476:docs/retrieval_frozen_v142.json'], cwd=ROOT)
    if frozen != Path(refs['frozen']['path']).read_bytes():
        raise ValueError('Permanent retrieval seal changed')
    inputs = feature_inputs(args.smoke, refs)
    Xv1, panel, outputs = inputs[:3]
    extra = feature_cube(*inputs)
    X = np.concatenate([Xv1, extra], axis=2)
    np.savez_compressed(out/'feature_matrix.npz', features=X, query_ids=np.array(list(panel)), feature_names=np.array(FEATURES))
    matrix_ref = base.record(out/'feature_matrix.npz')
    gemma = gemma_inventory(outputs, refs)
    variants = dict(v1=list(range(46)))
    for family, names in FAMILIES.items():
        variants['plus_'+family] = list(range(46))+[FEATURES.index(f) for f in names]
    variants['v1_plus_v2'] = list(range(len(FEATURES)))
    plan = dict(status='BEFORE_ANY_FIT_AND_NEW_LEDGER_READ', preregistration=base.record(out/'preregistered.md'),
        source_refs=refs, matrix=matrix_ref, features=FEATURES, families=FAMILIES, gemma_status=gemma['status'],
        variants={k: [FEATURES[i] for i in v] for k, v in variants.items()},
        params=dict(base.PARAMS, objective='rank:ndcg'), ranking_target='exact', confidence_target='union',
        confidence='diagnostic60/20/20 companies; same20 scene features; scene3foldOOF within proper60; target union at meta/iso/threshold',
        smoke=args.smoke)
    base.write(out/'plan.json', plan)
    print(base.json.dumps(dict(stage='FEATURES_SEALED_BEFORE_LEDGER', queries=len(X), shape=list(X.shape),
        matrix_sha256=matrix_ref['sha256'], gemma_missing=gemma['definitely_missing_sites'])), flush=True)
    Xcheck, y, views, groups, folds, ids, outcheck, _, outer, _, _ = diag.load(args.smoke)
    if list(panel) != ids.tolist() or outcheck != outputs or not np.array_equal(Xv1, Xcheck, equal_nan=True):
        raise ValueError('Label-free row/candidate mapping mismatch')
    runner = Runner(X, y, groups, ids, out/'models')
    predictions = {}
    for name, cols in variants.items():
        pred = np.full((len(ids), 100), np.nan)
        for f, train, test in outer:
            pred[test] = runner.score(train, test, cols, f'full_{name}_{f}')
        predictions[name] = pred
        print(base.json.dumps(dict(stage='variant', name=name, exact=int(y[np.arange(len(ids)), base.select_top(pred)[0]].sum()),
            union=int(views[np.arange(len(ids)), base.select_top(pred)[0], UNION].sum()))), flush=True)
    old = base.read(refs['diagnostic_result']['path'])
    if not args.smoke and (base.select_top(predictions['v1'])[0]+1).tolist() != old['algorithms']['R3']['selected_native_ranks']:
        raise ValueError('R3 V1 did not reproduce exactly')
    comparison = dict(v1=views[np.arange(len(ids)), base.select_top(predictions['v1'])[0]], R0=views[:, 0])
    summaries = {name: diag.summarize(pred, views, folds, ids, comparison)[0] for name, pred in predictions.items()}
    tests = [summaries[name]['totals']['paired']['v1'][v] for name in variants if name != 'v1' for v in ('exact', 'union')]
    previous = 0.
    for i, t in enumerate(sorted(tests, key=lambda t: t['exact_two_sided_p'])):
        previous = max(previous, min(1., (len(tests)-i)*t['exact_two_sided_p']))
        t['holm_p'] = previous
    confidences = {}
    for name in ('v1', 'v1_plus_v2'):
        confidences[name], predictions[name+'_confidence'] = confidence(runner, outer, X, views, groups, folds, ids, variants[name], out, name)
        print(base.json.dumps(dict(stage='confidence_union', variant=name, methods={m: v['totals'] for m, v in confidences[name]['methods'].items()})), flush=True)
    np.savez_compressed(out/'predictions.npz', **predictions)
    near_ceiling = ceiling(ids, y, X, refs)
    idx = {f: i for i, f in enumerate(FEATURES)}
    closed = X[:, :, idx['v2_is_closed']] == 1
    availability = dict(closed_candidate_rows=int(closed.sum()),
        closed_with_date=int((closed & np.isfinite(X[:, :, idx['v2_closure_age_log_days']])).sum()),
        all_missing_columns=[f for f, j in idx.items() if np.isnan(X[:, :, j]).all()],
        finite_cells={f: int(np.isfinite(X[:, :, j]).sum()) for f, j in idx.items()})
    for ref in refs.values():
        base.verified(ref)
    for ref in gemma['inventory_fingerprints']:
        if base.fingerprint(ref['path']) != ref:
            raise ValueError('Gemma inventory source changed')
    base.verified(matrix_ref)
    stop.set()
    result = dict(status='MEASURED_DEVELOPMENT_NOT_PRODUCTION', queries=len(ids), candidates=len(ids)*100,
        source_refs=refs, plan=base.record(out/'plan.json'), matrix=matrix_ref,
        predictions=base.record(out/'predictions.npz'), query_ids=ids.tolist(), folds=folds.tolist(),
        features=FEATURES, families=FAMILIES, variants=summaries, confidence=confidences,
        ceiling=near_ceiling, gemma=gemma, availability=availability, fit_receipts=runner.receipts,
        fit_count=len(runner.receipts), elapsed_seconds=time.monotonic()-started, **resources,
        recall100={v: base.metric(int(views[:, :, j].any(axis=1).sum()), len(ids)) for j, v in enumerate(base.VIEWS)},
        ranking_labels_unchanged=True, retrieval_changed=False, closed_sets_opened=False,
        production_promotion_allowed=False, smoke=args.smoke)
    full = result['variants']['v1_plus_v2']['totals']['paired']['v1']
    confirmed = any(full[v]['holm_p'] < .05 and len(full[v]['gains']) > len(full[v]['losses']) for v in ('exact', 'union'))
    result['recommendation'] = ('Confirmer la recette v2 complète sur un protocole indépendant autorisé avant toute promotion ; ne pas la retuner sur ce pilote.' if confirmed else
        'Priorité à un signal sémantique de site réellement distinct (activité/usage et nom propre officiels), avec un petit benchmark local autorisé avant tout encodage massif ; aucun gain v2 confirmé justifiant une promotion.')
    base.write(out/'result.json', result)
    (out/'result.md').write_text(prereg+'\n'+report(result))
    if not args.smoke:
        base.write(ROOT/'docs/step2_features_v2.json', result)
        DOC.write_text(prereg+'\n'+report(result))
    print(base.json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'],
        peak_rss_bytes=resources['peak_rss_bytes'], fits=result['fit_count'], recommendation=result['recommendation'])), flush=True)


if __name__ == '__main__':
    main()
