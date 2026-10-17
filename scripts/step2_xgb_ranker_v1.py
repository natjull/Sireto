"""One preregistered small XGB ranker on the immutable V142 ALL526 lists.

No retrieval import/execution, HPO, injected positive or legacy model reuse.
Nested group-OOF margins select SUR thresholds without the outer test labels.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from rapidfuzz.distance import JaroWinkler, Levenshtein

from scripts.retrieval_site_operational_metric import (
    read, record, verified, fingerprint, index, normalize, number,
    address_match, crm_point, candidate_point, distance, threshold,
)
from scripts.step2_baseline_v142 import VIEWS, NAMES, metric, rank_bin, BINS
from src.xgb_matcher.naming import build_candidate_names, normalize_name, NameSource

HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
DEFAULT_RUN = HOME/'step2_v142_20261017/ranker_v1'
SEED = 20261017
# Fixed BEFORE the first fit, no dev/test-driven parameter selection.
PARAMS = dict(objective='rank:pairwise', n_estimators=200, max_depth=3,
    learning_rate=0.05, min_child_weight=10, reg_lambda=10, reg_alpha=1,
    subsample=1.0, colsample_bytree=1.0, tree_method='hist', max_bin=128,
    lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=8,
    random_state=SEED, n_jobs=1, verbosity=0)
CURVE_THRESHOLDS = (0., 0.1, 0.25, 0.5, 1., 2.)
TARGET_PRECISION = 0.98
FEATURES = [
    'name_jaro_max', 'name_jaro_second', 'name_jaro_gap', 'name_levenshtein_max',
    'name_token_overlap', 'name_exact', 'name_etab_jaro', 'name_ul_jaro',
    'name_sigle_jaro', 'name_count', 'name_missing', 'name_first_token',
    'street_jaro', 'street_tokens', 'street_exact', 'number_exact', 'number_log_diff',
    'number_missing', 'suffix_exact', 'postcode_match', 'commune_match',
    'full_address_match', 'address_missing', 'name_address_interaction',
    'geo_log_distance', 'geo_missing', 'geo_both_house', 'geo_any_street',
    'is_active', 'state_missing', 'is_siege', 'site_missing',
    'v142_rank', 'v142_reciprocal_rank', 'v142_score', 'v142_score_relative',
    'v142_delta_best', 'v142_delta_next', 'v142_top1_top2_gap',
    'candidate_same_siren_count', 'same_siren_as_native_top1',
    'native_top1_siren_count', 'unique_siren_count', 'same_address_count',
    'name_gap_best_other', 'street_gap_best_other',
]
STOCK_COLUMNS = ['siret', 'statutDiffusionEtablissement', 'etatAdministratifEtablissement',
    'etablissementSiege', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
    'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
    'typeVoieEtablissement', 'libelleVoieEtablissement', 'codePostalEtablissement',
    'codeCommuneEtablissement', 'enseigne1Etablissement', 'enseigne2Etablissement',
    'enseigne3Etablissement', 'denominationUsuelleEtablissement']
UL_COLUMNS = ['siren', 'statutDiffusionUniteLegale', 'sigleUniteLegale',
    'denominationUniteLegale', 'denominationUsuelle1UniteLegale',
    'nomUniteLegale', 'prenomUsuelUniteLegale', 'categorieJuridiqueUniteLegale']


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def text(value):
    value = str(value or '')
    if value.strip().upper() in ('[ND]', 'ND', 'NAN', 'NONE'):
        return ''
    # Never turn a literal SIREN/SIRET embedded in a name into a match feature.
    return re.sub(r'\b\d{9}(?:\d{5})?\b', ' ', value)


def jaro(a, b):
    return float(JaroWinkler.normalized_similarity(a, b)) if a and b else 0.


def overlap(a, b):
    sa, sb = set(re.findall(r'\w+', a)), set(re.findall(r'\w+', b))
    return len(sa & sb)/len(sa | sb) if sa and sb else 0.


def hydrate(refs, outputs):
    """Join ALL emitted candidates, independently of every human label."""
    cs = [c for candidates in outputs.values() for c in candidates]
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB',
        'max_temp_directory_size': '0B'})
    con.register('requested', pa.table({'siret': sorted({c['siret'] for c in cs})}))
    con.register('requested_ul', pa.table({'siren': sorted({c['siren'] for c in cs})}))

    def select(key, columns, ident, requested):
        sql = 'SELECT '+','.join('s.'+c for c in columns)+f' FROM read_parquet(?) s SEMI JOIN {requested} r USING({ident})'
        return index(con.execute(sql, [refs[key]['path']]).fetch_arrow_table().to_pylist(), ident)

    sites = select('stock', STOCK_COLUMNS, 'siret', 'requested')
    units = select('legal_units', UL_COLUMNS, 'siren', 'requested_ul')
    points = select('study', ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision'], 'siret', 'requested')
    con.close()
    return sites, units, points


def feature_cube(panel, outputs, geometry, sites, units, points):
    """Explicit label-free interface; no ledger, acceptance sidecar or targets."""
    cube = []
    name_cache = {}
    for qid, q in panel.items():
        query_name = normalize_name(text(q['crm_name']))
        query_street = normalize(text(q['crm_street']), street=True)
        query_num = number(q['crm_number'], q['crm_suffix'])
        cp = crm_point(geometry[qid]['geometry'])
        cs = outputs[qid]
        group_counts = Counter(c['siren'] for c in cs)
        scores = np.array([c['score'] for c in cs])
        scale = max(float(scores.max()-scores.min()), 1e-6)
        top_gap = float(scores[0]-scores[1])
        address_keys, rows = [], []
        for i, c in enumerate(cs):
            raw_site, raw_unit = sites.get(c['siret'], {}), units.get(c['siren'], {})
            site = raw_site if raw_site.get('statutDiffusionEtablissement') == 'O' else {}
            unit = raw_unit if raw_unit.get('statutDiffusionUniteLegale') == 'O' else {}
            if c['siret'] not in name_cache:
                mapping = dict(enseigne1=site.get('enseigne1Etablissement'),
                    enseigne2=site.get('enseigne2Etablissement'), enseigne3=site.get('enseigne3Etablissement'),
                    denomination=site.get('denominationUsuelleEtablissement'), sigle_ul=unit.get('sigleUniteLegale'),
                    denomination_ul=unit.get('denominationUniteLegale'),
                    denomination_usuelle_ul=unit.get('denominationUsuelle1UniteLegale'),
                    nom_ul=unit.get('nomUniteLegale'), prenom_usuel_ul=unit.get('prenomUsuelUniteLegale'),
                    cj_ul=unit.get('categorieJuridiqueUniteLegale'))
                mapping = {k: text(v) for k, v in mapping.items()}
                name_cache[c['siret']] = build_candidate_names(mapping)
            names = name_cache[c['siret']]
            similarities = sorted([jaro(query_name, n.text) for n in names], reverse=True)
            best = similarities[0] if similarities else 0.
            second = similarities[1] if len(similarities) > 1 else 0.
            street = normalize(' '.join(text(site.get(k)) for k in ('typeVoieEtablissement', 'libelleVoieEtablissement')), street=True)
            snum = number(site.get('numeroVoieEtablissement'), site.get('indiceRepetitionEtablissement'))
            st_jaro = jaro(query_street, street)
            commune = bool(q['crm_insee'] and q['crm_insee'] == site.get('codeCommuneEtablissement'))
            postcode = bool(q['crm_postcode'] and q['crm_postcode'] == site.get('codePostalEtablissement'))
            number_equal = bool(query_num and snum and query_num[0] == snum[0])
            street_equal = bool(query_street and query_street == street)
            point = candidate_point(points.get(c['siret'])) if site else candidate_point(None)
            d = distance(cp, point) if threshold(cp, point) is not None else None
            address_keys.append((site.get('codeCommuneEtablissement'), street, snum) if street and snum and commune else None)
            vals = [best, second, best-second,
                max((float(Levenshtein.normalized_similarity(query_name, n.text)) for n in names if query_name), default=0.),
                max((overlap(query_name, n.text) for n in names), default=0.),
                float(any(query_name and query_name == n.text for n in names)),
                max((jaro(query_name, n.text) for n in names if not n.is_ul_name), default=0.),
                max((jaro(query_name, n.text) for n in names if n.is_ul_name), default=0.),
                max((jaro(query_name, n.text) for n in names if n.source == NameSource.UL_SIGLE), default=0.),
                len(names), float(not names),
                float(any(query_name and query_name.split()[0] == n.text.split()[0] for n in names)),
                st_jaro, overlap(query_street, street), float(street_equal), float(number_equal),
                math.log1p(abs(query_num[0]-snum[0])) if query_num and snum else float('nan'),
                float(not (query_num and snum)),
                float(bool(query_num and snum and query_num[1] == snum[1])), float(postcode), float(commune),
                float(address_match(q, site)), float(not street), best*st_jaro,
                math.log1p(d) if d is not None else float('nan'), float(d is None),
                float(cp['precision'] == point['precision'] == 'housenumber_equivalent'),
                float('street_imprecise' in (cp['precision'], point['precision'])),
                float(site.get('etatAdministratifEtablissement') == 'A'),
                float(site.get('etatAdministratifEtablissement') not in ('A', 'F')),
                float(str(site.get('etablissementSiege')).lower() in ('true', '1')), float(not site),
                i+1, 1/(i+1), float(scores[i]), float((scores[i]-scores.min())/scale),
                float(scores[i]-scores[0]), float(scores[i]-scores[i+1]) if i < 99 else 0., top_gap,
                group_counts[c['siren']], float(c['siren'] == cs[0]['siren']),
                group_counts[cs[0]['siren']], len(group_counts), 0., 0., 0.]
            if len(vals) != len(FEATURES):
                raise ValueError('Feature order mismatch')
            rows.append(vals)
        matrix = np.asarray(rows, dtype=np.float32)
        counts = Counter(k for k in address_keys if k is not None)
        matrix[:, FEATURES.index('same_address_count')] = [counts[k] if k else 0 for k in address_keys]
        for feature, gap in [('name_jaro_max', 'name_gap_best_other'), ('street_jaro', 'street_gap_best_other')]:
            v = matrix[:, FEATURES.index(feature)]
            matrix[:, FEATURES.index(gap)] = [v[i]-np.max(np.delete(v, i)) for i in range(100)]
        cube.append(matrix)
    result = np.asarray(cube, dtype=np.float32)
    if result.shape != (len(panel), 100, len(FEATURES)) or np.isinf(result).any():
        raise ValueError('Invalid feature matrix')
    return result


def targets_and_views(panel, outputs, labels, accepted, geometry, sites, points):
    truth, views = [], []
    for qid, q in panel.items():
        label = labels[qid]
        cp = crm_point(geometry[qid]['geometry'])
        rows = []
        for c in outputs[qid]:
            exact = c['siret'] == label['human_ground_truth_siret']
            same = c['siren'] == label['human_ground_truth_siren']
            point = candidate_point(points.get(c['siret']))
            d, t = distance(cp, point), threshold(cp, point)
            site = bool(same and (address_match(q, sites.get(c['siret'])) or (d is not None and t is not None and d <= t)))
            rows.append([exact, same, c['siret'] in accepted[qid], site, exact or site])
        rows = np.asarray(rows, dtype=bool)
        truth.append(rows[:, 0].astype(np.float32))
        views.append(rows)
    return np.asarray(truth), np.asarray(views)


def splits(folds, groups):
    disjoint = all(not(set(groups[folds == a]) & set(groups[folds == b])) for a, b in ((2, 3), (2, 4), (3, 4)))
    if disjoint:
        return [(str(f), np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in (2, 3, 4)], True
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return [(f'gkf_{i}', a, b) for i, (a, b) in enumerate(splitter.split(folds, groups=groups))], False


def select_top(predictions):
    order = np.argsort(-predictions, axis=1, kind='stable')
    top = order[:, 0]
    margins = predictions[np.arange(len(top)), top]-predictions[np.arange(len(top)), order[:, 1]]
    return top, margins


def sur_metrics(hits, selected):
    n = len(hits)
    chosen = hits[selected]
    k = len(chosen)
    return dict(count=k, rate=metric(k, n),
        precision={v: metric(int(chosen[:, j].sum()), k) if k else None for j, v in enumerate(VIEWS)})


def threshold_choice(margins, correct):
    order = np.argsort(-margins, kind='stable')
    m, c = margins[order], correct[order]
    counts = np.arange(1, len(m)+1)
    good = np.cumsum(c)/counts >= TARGET_PRECISION
    ends = np.r_[m[:-1] != m[1:], True]
    valid = np.flatnonzero(good & ends)
    if not len(valid):
        return None, dict(count=0, exact_precision=None, queries=len(m))
    stop = int(valid[-1])
    return float(m[stop]), dict(count=stop+1, queries=len(m),
        exact_precision=metric(int(c[:stop+1].sum()), stop+1))


def validate(X, y, groups, folds, qids, out, permutation=False):
    outer, existing_disjoint = splits(folds, groups)
    predictions = np.full((len(X), 100), np.nan, dtype=np.float32)
    decisions = np.zeros(len(X), dtype=bool)
    thresholds, partition, receipts, model_refs, importances = {}, {}, [], [], []
    for outer_id, train, test in outer:
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Outer company leakage')
        if existing_disjoint:
            inner = [(train[folds[train] != f], train[folds[train] == f]) for f in sorted(set(folds[train]))]
        else:
            kfold = GroupKFold(n_splits=3, shuffle=True, random_state=SEED)
            inner = [(train[a], train[b]) for a, b in kfold.split(train, groups=groups[train])]
        cal_predictions = np.full((len(X), 100), np.nan, dtype=np.float32)

        def fit(a, b, tag):
            if set(groups[a]) & set(groups[b]) or set(a) & set(b):
                raise ValueError('Company or query overlap')
            target = y[a].copy()
            if permutation:
                # Transfer full 100-position relevance vectors BETWEEN train
                # queries. Preserve positives/rank prior, never borrow test labels.
                seed = SEED+int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], 'big')
                target = target[np.random.default_rng(seed).permutation(len(a))]
            model = xgb.XGBRanker(**PARAMS)
            model.fit(X[a].reshape(-1, len(FEATURES)), target.reshape(-1), group=np.full(len(a), 100))
            path = out/f'{tag}.json'
            model.save_model(path)
            model_refs.append(record(path))
            receipts.append(dict(tag=tag, train_queries=[str(qids[i]) for i in a],
                scored_queries=[str(qids[i]) for i in b], train_companies=len(set(groups[a])),
                scored_companies=len(set(groups[b])), company_overlap=0,
                positive_queries=int(target.sum()), zero_positive_queries=int((target.sum(axis=1) == 0).sum()),
                target_matrix_sha256=hashlib.sha256(target.tobytes()).hexdigest()))
            return model, model.predict(X[b].reshape(-1, len(FEATURES))).reshape(len(b), 100)

        for inner_id, (a, b) in enumerate(inner):
            _, cal_predictions[b] = fit(a, b, f'outer_{outer_id}_inner_{inner_id}')
        cal_top, cal_margins = select_top(cal_predictions[train])
        correct = y[train, cal_top].astype(bool)
        if not np.isfinite(cal_predictions[train]).all():
            raise ValueError('Incomplete inner OOF calibration')
        limit, calibration = threshold_choice(cal_margins, correct)
        model, pred = fit(train, test, f'outer_{outer_id}')
        predictions[test] = pred
        _, margins = select_top(pred)
        decisions[test] = margins >= limit if limit is not None else False
        importances.append(model.feature_importances_.tolist())
        thresholds[outer_id] = dict(threshold=limit, calibration=calibration,
            calibration_origin='inner_company_OOF_on_outer_training_only', train_queries=len(train),
            test_queries=len(test), train_companies=len(set(groups[train])), test_companies=len(set(groups[test])),
            calibration_curve=[dict(threshold=t, count=int((cal_margins >= t).sum()),
                exact_precision=metric(int(correct[cal_margins >= t].sum()), int((cal_margins >= t).sum()))
                if (cal_margins >= t).any() else None) for t in CURVE_THRESHOLDS])
        for i in test:
            partition[str(qids[i])] = outer_id
        print(json.dumps(dict(stage='permutation' if permutation else 'real', fold=outer_id,
            calibration=calibration, threshold=limit, scored=len(test))), flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError('Every query must have exactly one OOF prediction')
    importance = np.mean(importances, axis=0)
    return dict(predictions=predictions, decisions=decisions, thresholds=thresholds,
        partition=partition, model_refs=model_refs, fit_receipts=receipts,
        existing_folds_company_disjoint=existing_disjoint,
        feature_importance=sorted([dict(feature=f, mean_outer_gain=float(v)) for f, v in zip(FEATURES, importance)],
            key=lambda r: -r['mean_outer_gain']))


def paired_summary(before, after, qids):
    return {v: dict(before=metric(int(before[:, j].sum()), len(qids)),
        after=metric(int(after[:, j].sum()), len(qids)),
        gains=[str(qids[i]) for i in np.flatnonzero(~before[:, j] & after[:, j])],
        losses=[str(qids[i]) for i in np.flatnonzero(before[:, j] & ~after[:, j])]) for j, v in enumerate(VIEWS)}


def evaluate(run, view_cube, y, qids, folds, outputs):
    top, margins = select_top(run['predictions'])
    before, after = view_cube[:, 0], view_cube[np.arange(len(top)), top]
    per_query = []
    for i, qid in enumerate(qids):
        outer_id = run['partition'][str(qid)]
        per_query.append(dict(query_id=str(qid), fold=int(folds[i]), oof_partition=outer_id,
            before_siret=outputs[str(qid)][0]['siret'], after_siret=outputs[str(qid)][int(top[i])]['siret'],
            selected_native_rank=int(top[i])+1, margin=float(margins[i]),
            threshold=run['thresholds'][outer_id]['threshold'], status='SUR' if run['decisions'][i] else 'A_VERIFIER',
            before={v: bool(before[i, j]) for j, v in enumerate(VIEWS)},
            after={v: bool(after[i, j]) for j, v in enumerate(VIEWS)}))

    def scope(idx):
        ranks = []
        order = np.argsort(-run['predictions'][idx], axis=1, kind='stable')
        for at, original in enumerate(idx):
            matches = np.flatnonzero(y[original, order[at]])
            ranks.append(int(matches[0])+1 if len(matches) else None)
        return dict(queries=len(idx), paired=paired_summary(before[idx], after[idx], qids[idx]),
            sur=sur_metrics(after[idx], run['decisions'][idx]),
            exact_rank_distribution={b: sum(rank_bin(r) == b for r in ranks) for b in BINS},
            recall100={v: metric(int(view_cube[idx, :, j].any(axis=1).sum()), len(idx)) for j, v in enumerate(VIEWS)},
            margin_curve=[dict(threshold=t, **sur_metrics(after[idx], margins[idx] >= t)) for t in CURVE_THRESHOLDS])

    return dict(totals=scope(np.arange(len(qids))), by_fold={str(f): scope(np.flatnonzero(folds == f)) for f in (2, 3, 4)},
        thresholds=run['thresholds'], feature_importance=run['feature_importance'],
        per_query=per_query, model_refs=run['model_refs'], fit_receipts=run['fit_receipts'])


def additional_examples(refs, qids):
    # This is the ONLY read beyond the pilot: projected label fields, folds2/3/4.
    rows = pq.read_table(refs['ledger']['path'], columns=['query_id', 'fold', 'human_ground_truth_siret',
        'human_ground_truth_siren'], filters=[('fold', 'in', [2, 3, 4])], use_threads=False).to_pylist()
    valid = [r for r in rows if re.fullmatch(r'\d{14}', str(r['human_ground_truth_siret']))
        and r['human_ground_truth_siren'] == r['human_ground_truth_siret'][:9]]
    if len({r['query_id'] for r in valid}) != len(valid):
        raise ValueError('Duplicate census queries')
    pilot = set(qids)
    extra = sum(r['query_id'] not in pilot for r in valid)
    timings = read(refs['native_timings']['path'])
    fresh = [r['total_seconds'] for r in timings.values() if not r['unchanged_outputs_reused']]
    comparable = read(refs['cost_reference']['path'])
    mean_process = comparable['generation']['search_process_seconds']/20
    return dict(labelled_queries_234=len(valid), additional_queries=extra,
        labelled_queries_by_fold={str(f): sum(r['fold'] == f for r in valid) for f in (2, 3, 4)},
        fresh_v142_kernel_queries=len(fresh), fresh_v142_mean_kernel_seconds=float(np.mean(fresh)),
        additional_native_hours_estimate=[extra*np.mean(fresh)/3600, extra*mean_process/3600],
        estimated_rss_gib=[comparable['peak_rss_bytes']/1024**3, 8],
        estimate_not_confidence_interval=True, generation_not_run=True,
        assumptions='107 actual V142 kernels, and20 comparable V173 process receipts; same Mac, index/caches, similar query mix. Preparation, source scans and waits excluded; memory <8GiB not guaranteed outside sample.',
        production_blocker='Frozen assignment covers526 only; unique label-free production settings absent. No extension or unique fit authorized/executed here.')


def md_metric(m):
    if m is None:
        return 'non définie'
    lo, hi = m['wilson95_percent']
    return f'{m["hits"]}/{m["queries"]} ({m["percent"]:.2f} %, IC95 [{lo:.2f};{hi:.2f}])'


def markdown(result):
    real, control = result['real'], result['permutation']
    lines = ['# Premier reclasseur XGBoost V142 — paramètres fixes', '',
        f'**{result["queries"]} requêtes**,100 candidats inchangés chacune ; aucun modèle de recherche recalculé.',
        'Statut : développement sur pilote consommé, **pas certification production**.',
        f'**Conclusion : {result["verdict"]}.** Les statuts `SUR` ci-dessous sont expérimentaux, non promus.', '',
        '## Protocole fixé avant les fits', '',
        '- `rank:pairwise`,200 arbres, profondeur3, taux0,05, min_child10, L2=10, L1=1, max_bin128, un thread ; graine20261017. Aucune HPO/early stopping.',
        '- Audit SIREN humain : folds2/3/4 disjoints par entreprise. Chaque test externe exclu de tous ses fits et seuils.',
        '- Pour chaque test externe, prédictions OOF sur les deux folds d’apprentissage (deux fits internes). Seuil de marge brut maximisant la couverture à précision exacte empirique≥98 % sur ces prédictions internes uniquement.',
        '- Si aucun seuil admissible : tous `A_VERIFIER`. Aucun minimum de couverture ni borne Wilson imposé ;98 % est une cible de calibration, **pas une garantie test**.',
        '- Ex æquo : ordre V142 conservé. Requêtes sans positif gardées au fit et à toute évaluation ; aucun positif ajouté.',
        '- Familles reprises des anciens XGB : noms officiels propres UL/établissement/sigle, Jaro/Levenshtein/tokens, adresse/numéro, distance admissible et précision, état/siège, rang/score, concurrence sur les100.',
        '- Aucun nom tiers dirigeant, alias CRM, identifiant brut, variable de label ou vérité de site en entrée. IDs utilisés seulement pour jointures/groupes et comptages SIREN demandés.',
        '- Features construites par interface sans labels sur TOUS les100 ; ledger/sidecar lus séparément pour cibles et métriques. Pas de TF-IDF global appris, d’encodeur ni de poids historiques réutilisés.', '',
        '## Top1 avant / après (Wilson95, dénominateur intégral)', '',
        '| Vue | Avant | Après | Gains / pertes |', '|---|---|---|---:|']
    for v in VIEWS:
        p = real['totals']['paired'][v]
        lines.append(f'| {NAMES[v]} | {md_metric(p["before"])} | {md_metric(p["after"])} | +{len(p["gains"])}/−{len(p["losses"])} |')
    lines += ['', '## Par fold', '', '| Fold | N | Top1 exact avant | Après | Gains/pertes | Taux SUR | Précision exacte SUR |',
        '|---|---:|---|---|---:|---|---|']
    for f, s in real['by_fold'].items():
        p = s['paired']['exact']
        lines.append(f'| {f} | {s["queries"]} | {md_metric(p["before"])} | {md_metric(p["after"])} | '
            f'+{len(p["gains"])}/−{len(p["losses"])} | {md_metric(s["sur"]["rate"])} | {md_metric(s["sur"]["precision"]["exact"])} |')
    sur = real['totals']['sur']
    lines += ['', '## SUR / A_VERIFIER', '', f'- Taux SUR : **{md_metric(sur["rate"])}**.']
    for v in VIEWS:
        lines.append(f'- Précision SUR {NAMES[v]} : {md_metric(sur["precision"][v])}.')
    for f, t in real['thresholds'].items():
        lines.append(f'- Test fold {f} : seuil={t["threshold"]}, calibration interne {t["calibration"]["count"]}/{t["train_queries"]} acceptés ; précision exacte {md_metric(t["calibration"]["exact_precision"])}.')
    lines += ['', '### Courbe descriptive sur prédictions externes (jamais utilisée pour choisir le seuil)', '',
        '| Marge ≥ | SUR | Taux SUR | Précision exacte SUR |', '|---:|---:|---|---|']
    for r in real['totals']['margin_curve']:
        lines.append(f'| {r["threshold"]} | {r["count"]} | {md_metric(r["rate"])} | {md_metric(r["precision"]["exact"])} |')
    p = real['totals']['paired']['exact']
    lines += ['', '## Appariement exact (query_id)', '', 'Gains : `'+', '.join(p['gains'])+'`.',
        'Pertes : `'+', '.join(p['losses'])+'`.', '', '## Top10 importance (gain moyen normalisé des trois modèles externes)', '']
    for r in real['feature_importance'][:10]:
        lines.append(f'- `{r["feature"]}` : {r["mean_outer_gain"]:.6f}.')
    p = control['totals']['paired']['exact']
    lines += ['', '## Contrôle permutation', '',
        '- Même matrice, paramètres et validation. Dans CHAQUE fit, permutation des vecteurs pertinence100 entre requêtes d’apprentissage, sans échantillon externe. Les labels sources restent immuables ; les positifs par rang sont conservés en distribution.',
        '- Ce contrôle transfère les positions pertinentes, pas les SIRET entre pools (ce qui annulerait presque tous les positifs et rendrait le fit trivial). Seuils internes et test évalués sur les vérités originales.',
        f'- Top1 exact contrôle : **{md_metric(p["after"])}**, contre {p["before"]["hits"]} avant ; gains/pertes +{len(p["gains"])}/−{len(p["losses"])}.',
        '- Le rang natif reste informatif car amont supervisé gelé : le contrôle ne prouve pas une indépendance end-to-end. Les autres vues, seuils et reçus du contrôle sont dans le JSON.']
    c = result['additional_examples']
    lines += ['', '## Plus d’exemples — comptage seulement', '',
        f'- Ledger folds2/3/4 : **{c["labelled_queries_234"]}** labels humains exploitables, dont **{c["additional_queries"]}** hors pilote ; aucun statut ne filtre ce compte.',
        f'- Temps natif supplémentaire conditionnel : **{c["additional_native_hours_estimate"][0]:.1f}–{c["additional_native_hours_estimate"][1]:.1f} h**, hors préparation/sources/attentes ;107 noyaux V142 réellement recalculés et20 processus V173 comparables (pas des coûts mesurés sur les nouvelles requêtes).',
        f'- Pic indicatif {c["estimated_rss_gib"][0]:.2f}–8 Gio, pas garanti sous8 hors échantillon ; affectation V142 limitée526 et modèle production unique absent. Aucune génération lancée.', '',
        '## Intégrité / limites', '',
        '- Rappels@100 inchangés : exact507, SIREN511, opérationnel508, site404, union511 ; résultat vérifié requête par requête contre la baseline.',
        '- Vue site : même SIREN ET (adresse concordante OU distance admissible200/500 m). Exact ne vaut pas automatiquement site, union=exact OU site. Aucune vérité modifiée.',
        '- OOF strict concerne CE ranker ; amont consommé, non full-OOF et sélection de coefficients par SIREN humain inchangés. Ne pas présenter ces scores comme une validation indépendante de toute la chaîne.',
        '- Wilson95 brut par requête est descriptif ; il ne corrige ni corrélation intra-entreprise ni consommation du pilote. Les seuils calibrés sur de très petits effectifs acceptés restent fragiles et les marges de fits internes/externes peuvent changer d’échelle.',
        '- Le JSON contient métriques/IC des cinq vues par fold, paires gains/pertes, décisions526, ordre des features, hashes matrice/entrées/modèles et reçus de disjonction de chaque fit.',
        f'- Temps total : {result["elapsed_seconds"]:.2f}s ; pic RSS échantillonné : {result["peak_rss_bytes"]/1024**2:.1f} Mio, garde7,5 Gio. Aucun jeu fermé, web, VM ou push.',
        '- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_xgb_ranker_v1.py --output-dir <nouveau_dossier_CATNAT>` ; `--smoke` mesure6 vraies requêtes et ne remplace pas les résultats docs.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir or (DEFAULT_RUN/'smoke' if args.smoke else DEFAULT_RUN/'full')
    if out.exists():
        raise ValueError('Fresh owned output directory required; no overwrite of prior runs')
    out.mkdir(parents=True)
    started, stop = time.monotonic(), threading.Event()
    resource = dict(peak_rss_bytes=0)

    def monitor():
        process = psutil.Process()
        while not stop.is_set():
            resource['peak_rss_bytes'] = max(resource['peak_rss_bytes'], process.memory_info().rss)
            if resource['peak_rss_bytes'] >= int(7.5*1024**3):
                write(out/'resource_stop.json', resource)
                os._exit(75)
            stop.wait(.1)

    threading.Thread(target=monitor, daemon=True).start()
    baseline_path = ROOT/'docs/step2_baseline_v142.json'
    baseline = read(baseline_path)
    refs = dict(baseline=record(baseline_path), frozen=record(ROOT/'docs/retrieval_frozen_v142.json'),
        implementation=record(Path(__file__)), baseline_implementation=baseline['source_refs']['implementation'],
        naming=record(ROOT/'src/xgb_matcher/naming.py'),
        old_features_reference=record(ROOT/'src/xgb_matcher/features.py'))
    source = baseline['source_refs']
    for k in ('outputs', 'generation', 'panel', 'geometry', 'ledger', 'acceptable_operational',
              'evaluation', 'label_side', 'operational_policy', 'metric_rule'):
        refs[k] = source[k]
        verified(refs[k])
    seal = read(refs['frozen']['path'])
    if seal['status'] != 'FROZEN_BY_NATHAN_2026_10_17':
        raise ValueError('Commit A seal required')
    frozen_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', 'docs/retrieval_frozen_v142.json'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'show', f'{frozen_commit}:docs/retrieval_frozen_v142.json'], cwd=ROOT) != Path(refs['frozen']['path']).read_bytes():
        raise ValueError('Frozen authority must already be committed and unchanged')
    sealed_artifacts = {r['path']: r for r in seal['artifacts']}
    if sealed_artifacts[refs['outputs']['path']]['sha256'] != refs['outputs']['sha256']:
        raise ValueError('Commit A and baseline outputs disagree')
    for key in ('stock', 'study'):
        previous = source[key]
        if any(previous[k] != v for k, v in fingerprint(previous['path']).items()):
            raise ValueError('Official source changed')
        expected_sha = sealed_artifacts[previous['path']]['sha256'] if key == 'stock' else previous['inherited_sha256']
        refs[key] = record(previous['path'], expected_sha)
    ul_path = '/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet'
    ul = next(r for r in seal['artifacts'] if r['path'] == ul_path)
    refs['legal_units'] = record(ul_path, ul['sha256'])
    for key, path in [('native_timings', HOME/'v142/bano_native_20261013_night/all_new_sources_max_bano/native_timings.json'),
                      ('cost_reference', ROOT/'docs/retrieval_site_target_timing_20261015.json')]:
        refs[key] = record(path)
    generation = read(refs['generation']['path'])
    if generation['outputs']['sha256'] != refs['outputs']['sha256'] or generation['status'] != 'COMPLETE_SEALED_ALL526':
        raise ValueError('Incomplete or mismatched V142 seal')
    full_panel = index(read(refs['panel']['path']))
    outputs_all = read(refs['outputs']['path'])['learned_native_ooc']
    if len(full_panel) != 526 or set(full_panel) != set(outputs_all) or any(q['fold'] not in (2, 3, 4) for q in full_panel.values()):
        raise ValueError('Only the sealed ALL526 folds2/3/4 scope is allowed')
    ids = sorted(full_panel)
    if args.smoke:
        ids = sorted(q for f in (2, 3, 4) for q in sorted(
            [q for q in ids if full_panel[q]['fold'] == f], key=lambda q: hashlib.sha256(q.encode()).hexdigest())[:2])
    panel = {q: full_panel[q] for q in ids}
    outputs = {q: outputs_all[q] for q in ids}
    if any(len(cs) != 100 or len({c['siret'] for c in cs}) != 100 for cs in outputs.values()):
        raise ValueError('Exactly100 unique candidates required, without injection')
    geometry = index(read(refs['geometry']['path']))
    write(out/'plan.json', dict(params=PARAMS, features=FEATURES, seed=SEED, target_precision=TARGET_PRECISION,
        thresholds=CURVE_THRESHOLDS, source_refs=refs, frozen_commit=frozen_commit, smoke=args.smoke,
        permutation='Within each fit: permute full 100-position target vectors among TRAIN queries only',
        threshold_selection='Nested company-OOF on outer training only, maximum coverage at empirical exact precision>=98%'))
    sites, units, points = hydrate(refs, outputs)
    # Finish the entire feature matrix BEFORE labels or sidecar are read.
    X = feature_cube(panel, outputs, geometry, sites, units, points)
    np.savez_compressed(out/'features.npz', features=X)
    matrix_ref = record(out/'features.npz')
    filters = [('fold', 'in', [2, 3, 4]), ('query_id', 'in', ids)]
    labels = index(pq.read_table(refs['ledger']['path'], columns=['query_id', 'fold',
        'human_ground_truth_siret', 'human_ground_truth_siren'], filters=filters, use_threads=False).to_pylist())
    if set(labels) != set(panel) or any(labels[q]['fold'] != panel[q]['fold'] or
            not re.fullmatch(r'\d{14}', labels[q]['human_ground_truth_siret']) or
            labels[q]['human_ground_truth_siret'][:9] != labels[q]['human_ground_truth_siren'] for q in panel):
        raise ValueError('Human label scope or identity mismatch')
    accepted = {q: {labels[q]['human_ground_truth_siret']} for q in ids}
    for r in pq.read_table(refs['acceptable_operational']['path'], filters=filters, use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    y, views = targets_and_views(panel, outputs, labels, accepted, geometry, sites, points)
    expected = index(baseline['per_query'])
    for i, q in enumerate(ids):
        for j, v in enumerate(VIEWS):
            ranks = np.flatnonzero(views[i, :, j])
            if (int(ranks[0])+1 if len(ranks) else None) != expected[q]['first_rank'][v]:
                raise ValueError('Baseline metric disagreement')
    qids = np.asarray(ids)
    groups = np.array([labels[q]['human_ground_truth_siren'] for q in ids])
    folds = np.array([panel[q]['fold'] for q in ids])
    real_dir, perm_dir = out/'real', out/'permutation'
    real_dir.mkdir(); perm_dir.mkdir()
    real_run = validate(X, y, groups, folds, qids, real_dir)
    perm_run = validate(X, y, groups, folds, qids, perm_dir, permutation=True)
    real, perm = evaluate(real_run, views, y, qids, folds, outputs), evaluate(perm_run, views, y, qids, folds, outputs)
    more = additional_examples(refs, list(full_panel))
    if not args.smoke and (real['totals']['paired']['exact']['before']['hits'] != 353 or
            [real['totals']['recall100'][v]['hits'] for v in VIEWS] != [507, 511, 508, 404, 511]):
        raise ValueError('ALL526 baseline/recall gate mismatch')
    for ref in refs.values():
        if 'sha256' in ref:
            verified(ref)
        elif any(ref[k] != v for k, v in fingerprint(ref['path']).items()):
            raise ValueError('Input changed during experiment')
    stop.set()
    precision = real['totals']['sur']['precision']['exact']
    target_met = precision is not None and precision['percent'] >= 100*TARGET_PRECISION
    result = dict(status='MEASURED_DEVELOPMENT_NOT_PRODUCTION',
        sur_target_met_on_outer_oof=target_met,
        verdict='CIBLE_SUR_ATTEINTE_EMPIRIQUEMENT_NON_CERTIFIEE' if target_met else 'CIBLE_SUR_NON_ATTEINTE_POLITIQUE_NON_VALIDEE',
        queries=len(ids), candidates=len(ids)*100,
        source_refs=refs, frozen_commit=frozen_commit, params=PARAMS, seed=SEED,
        target_precision=TARGET_PRECISION, curve_thresholds=CURVE_THRESHOLDS,
        features=FEATURES, feature_matrix=matrix_ref, feature_matrix_shape=list(X.shape),
        existing_folds_company_disjoint=real_run['existing_folds_company_disjoint'],
        groups_by_fold={str(f): len(set(groups[folds == f])) for f in (2, 3, 4)},
        no_positive_injected=True, no_retrieval_executed=True, source_labels_modified=False,
        closed_sets_opened=False, production_promotion_allowed=False, smoke=args.smoke,
        versions=dict(python=sys.version, xgboost=xgb.__version__, sklearn=sklearn.__version__),
        real=real, permutation=perm, permutation_at_or_below_native=perm['totals']['paired']['exact']['after']['hits'] <= real['totals']['paired']['exact']['before']['hits'],
        additional_examples=more, elapsed_seconds=time.monotonic()-started, **resource)
    write(out/'result.json', result)
    (out/'result.md').write_text(markdown(result))
    if not args.smoke:
        write(ROOT/'docs/step2_xgb_ranker_v1.json', result)
        (ROOT/'docs/step2_xgb_ranker_v1.md').write_text(markdown(result))
    print(json.dumps(dict(status=result['status'], queries=len(ids),
        before=real['totals']['paired']['exact']['before'], after=real['totals']['paired']['exact']['after'],
        sur=real['totals']['sur'], permutation=perm['totals']['paired']['exact']['after'],
        more=more, elapsed_seconds=result['elapsed_seconds'], peak_rss_bytes=resource['peak_rss_bytes'])), flush=True)


if __name__ == '__main__':
    main()
