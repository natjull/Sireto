"""Preregistered data/algorithm/features diagnostic on the sealed V142 matrix.

No candidate generation, new embedding, upstream fit, HPO or closed fold access.
Confidence uses a common held-calibration R3, never a model fit on calibration labels.
"""
import argparse
from collections import Counter
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

import numpy as np
import psutil
import pyarrow.parquet as pq
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from scripts import step2_xgb_ranker_v1 as base

SEEDS = (20261017, 20261018, 20261019, 20261020, 20261021)
FRACTIONS = (.25, .5, .75, 1.)
ALGORITHMS = ('R0', 'R1', 'R2', 'R3', 'R4', 'R5')
GROUPS = dict(name=base.FEATURES[:12], address=base.FEATURES[12:24],
    geography=base.FEATURES[24:28], state=base.FEATURES[28:32],
    native=base.FEATURES[32:39], scene=base.FEATURES[39:])
# Remove cross-family proxies too; an ablation must not leave a renamed copy.
DEPENDENCIES = dict(name=['name_address_interaction', 'name_gap_best_other'],
    address=['same_address_count', 'street_gap_best_other'],
    native=['same_siren_as_native_top1', 'native_top1_siren_count'])
SCENE_FEATURES = ['margin', 'score_top1', 'score_top2', 'softmax_top1', 'score_std',
    'entropy', 'native_agreement', 'selected_native_rank', 'same_siren_count',
    'unique_siren_count', 'name_top1', 'name_gap_top2', 'street_top1', 'street_gap_top2',
    'number_exact', 'full_address_match', 'geo_log_distance', 'geo_missing', 'is_active', 'is_siege']
FAMILIES = ('ferme_vs_actif', 'siege_vs_site', 'meme_siren_autre_site', 'homonyme', 'adresse_proche', 'autre')
DEFAULT_RUN = base.HOME/'step2_diagnostic_20261017'


def digest(value):
    return hashlib.sha256(base.json.dumps(value, sort_keys=True).encode()).hexdigest()


def mcnemar(before, after, ids):
    gains = np.flatnonzero(~before & after)
    losses = np.flatnonzero(before & ~after)
    n, k = len(gains)+len(losses), min(len(gains), len(losses))
    p = min(1., 2*sum(math.comb(n, i) for i in range(k+1))/2**n) if n else 1.
    return dict(gains=[str(ids[i]) for i in gains], losses=[str(ids[i]) for i in losses],
        discordant=n, exact_two_sided_p=p)


def probability(model, X):
    p = model.predict_proba(X)
    positive = np.flatnonzero(model.classes_ == 1)
    return p[:, positive[0]] if len(positive) else np.zeros(len(X))


def logistic(y):
    classifier = LogisticRegression(C=1., max_iter=1000, random_state=base.SEED)
    if len(np.unique(y)) < 2:
        classifier = DummyClassifier(strategy='prior')
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
        StandardScaler(), classifier)


class Runner:
    def __init__(self, X, y, groups, ids, out):
        self.X, self.y, self.groups, self.ids, self.out = X, y, groups, ids, out
        self.models, self.receipts = {}, []
        self.out.mkdir()

    def score(self, algorithm, train, test, columns, purpose):
        train, test = np.sort(train), np.asarray(test)
        if set(self.groups[train]) & set(self.groups[test]):
            raise ValueError('Company leakage: '+purpose)
        key = digest([algorithm, train.tolist(), list(columns)])
        if key not in self.models:
            a = self.X[train][:, :, columns].reshape(-1, len(columns))
            target = self.y[train].reshape(-1)
            if algorithm == 'R1':
                model = logistic(target)
            elif algorithm == 'R5':
                # Structural/default settings unchanged; seed is for reproducibility.
                model = HistGradientBoostingClassifier(random_state=base.SEED)
            else:
                params = dict(base.PARAMS)
                if algorithm == 'R3':
                    params['objective'] = 'rank:ndcg'
                if algorithm == 'R4':
                    params['objective'] = 'binary:logistic'
                    params.pop('lambdarank_pair_method')
                    params.pop('lambdarank_num_pair_per_sample')
                    model = xgb.XGBClassifier(**params)
                else:
                    model = xgb.XGBRanker(**params)
            start = time.monotonic()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                if algorithm in ('R2', 'R3'):
                    model.fit(a, target, group=np.full(len(train), 100))
                else:
                    model.fit(a, target)
            path = self.out/(key+('.json' if algorithm in ('R2', 'R3', 'R4') else '.pkl'))
            if algorithm in ('R2', 'R3', 'R4'):
                model.save_model(path)
            else:
                with path.open('wb') as file:
                    pickle.dump(model, file, protocol=5)
            receipt = dict(key=key, algorithm=algorithm, model=base.record(path),
                parameters={k: ('NaN' if isinstance(v, float) and math.isnan(v) else v) for k, v in model.get_params().items()} if algorithm in ('R2', 'R3', 'R4', 'R5') else dict(C=1., max_iter=1000, imputation='train_median', scaling='train_standard'),
                train_queries=[str(self.ids[i]) for i in train], train_companies=len(set(self.groups[train])),
                zero_positive_queries=int((self.y[train].sum(axis=1) == 0).sum()),
                columns=[base.FEATURES[i] for i in columns], target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                elapsed_seconds=time.monotonic()-start, warnings=[str(w.message) for w in captured], uses=[])
            self.receipts.append(receipt)
            self.models[key] = (model, receipt)
        model, receipt = self.models[key]
        receipt['uses'].append(dict(purpose=purpose, company_overlap=0,
            scored_queries=[str(self.ids[i]) for i in test]))
        a = self.X[test][:, :, columns].reshape(-1, len(columns))
        pred = model.predict(a) if algorithm in ('R2', 'R3') else probability(model, a)
        result = pred.reshape(len(test), 100)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite score')
        return result


def softmax(pred):
    z = np.exp(pred-pred.max(axis=1, keepdims=True))
    return z/z.sum(axis=1, keepdims=True)


def scene(pred, X):
    order = np.argsort(-pred, axis=1, kind='stable')
    i, first, second = np.arange(len(pred)), order[:, 0], order[:, 1]
    a, b = X[i, first], X[i, second]
    f = {name: j for j, name in enumerate(base.FEATURES)}
    p = softmax(pred)
    return np.column_stack([pred[i, first]-pred[i, second], pred[i, first], pred[i, second],
        p[i, first], pred.std(axis=1), -(p*np.log(np.maximum(p, 1e-30))).sum(axis=1),
        first == 0, a[:, f['v142_rank']], a[:, f['candidate_same_siren_count']],
        a[:, f['unique_siren_count']], a[:, f['name_jaro_max']],
        a[:, f['name_jaro_max']]-b[:, f['name_jaro_max']], a[:, f['street_jaro']],
        a[:, f['street_jaro']]-b[:, f['street_jaro']], a[:, f['number_exact']],
        a[:, f['full_address_match']], a[:, f['geo_log_distance']], a[:, f['geo_missing']],
        a[:, f['is_active']], a[:, f['is_siege']]])


def load(smoke):
    v1path = ROOT/'docs/step2_xgb_ranker_v1.json'
    v1 = base.read(v1path)
    refs = dict(v1['source_refs'])
    refs['v1_result'] = base.record(v1path)
    # The diagnostic implementation is distinct from its immutable V1 dependency.
    refs['v1_implementation'] = refs.pop('implementation')
    refs['implementation'] = base.record(Path(__file__))
    refs['feature_matrix'] = v1['feature_matrix']
    for ref in refs.values():
        base.verified(ref)
    frozen = base.subprocess.check_output(['git', 'rev-parse', '36c7a476'], cwd=ROOT, text=True).strip()
    if base.subprocess.check_output(['git', 'show', frozen+':docs/retrieval_frozen_v142.json'], cwd=ROOT) != Path(refs['frozen']['path']).read_bytes():
        raise ValueError('The permanent retrieval seal changed')
    if v1['features'] != base.FEATURES or v1['params'] != base.PARAMS:
        raise ValueError('V1 recipe or46 features changed')
    panel_all = base.index(base.read(refs['panel']['path']))
    ids_all = sorted(panel_all)
    if ids_all != [r['query_id'] for r in v1['real']['per_query']] or len(ids_all) != 526:
        raise ValueError('Cached matrix row mapping mismatch')
    Xall = np.load(refs['feature_matrix']['path'])['features']
    if Xall.shape != (526, 100, 46):
        raise ValueError('Immutable matrix shape mismatch')
    ids = ids_all
    if smoke:
        ids = sorted(q for f in (2, 3, 4) for q in sorted(
            [q for q in ids_all if panel_all[q]['fold'] == f], key=lambda q: hashlib.sha256(q.encode()).hexdigest())[:12])
    panel = {q: panel_all[q] for q in ids}
    if any(q['fold'] not in (2, 3, 4) for q in panel.values()):
        raise ValueError('Closed fold')
    X = Xall[[ids_all.index(q) for q in ids]]
    outputs_all = base.read(refs['outputs']['path'])['learned_native_ooc']
    outputs = {q: outputs_all[q] for q in ids}
    if any(len(cs) != 100 or len({c['siret'] for c in cs}) != 100 for cs in outputs.values()):
        raise ValueError('Frozen candidate list changed')
    filters = [('fold', 'in', [2, 3, 4]), ('query_id', 'in', ids)]
    labels = base.index(pq.read_table(refs['ledger']['path'], columns=['query_id', 'fold',
        'human_ground_truth_siret', 'human_ground_truth_siren'], filters=filters, use_threads=False).to_pylist())
    if set(labels) != set(panel):
        raise ValueError('Label scope mismatch')
    geometry = base.index(base.read(refs['geometry']['path']))
    sites, _, points = base.hydrate(refs, outputs)
    accepted = {q: {labels[q]['human_ground_truth_siret']} for q in ids}
    for r in pq.read_table(refs['acceptable_operational']['path'], filters=filters, use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    y, views = base.targets_and_views(panel, outputs, labels, accepted, geometry, sites, points)
    expected = base.index(base.read(refs['baseline']['path'])['per_query'])
    for i, q in enumerate(ids):
        if labels[q]['fold'] != panel[q]['fold']:
            raise ValueError('Label fold changed')
        for j, v in enumerate(base.VIEWS):
            hits = np.flatnonzero(views[i, :, j])
            if (int(hits[0])+1 if len(hits) else None) != expected[q]['first_rank'][v]:
                raise ValueError('Evaluation differs from immutable baseline')
    groups = np.array([labels[q]['human_ground_truth_siren'] for q in ids])
    folds = np.array([panel[q]['fold'] for q in ids])
    outer, disjoint = base.splits(folds, groups)
    if not disjoint:
        raise ValueError('Expected existing company-disjoint folds2/3/4')
    return X, y, views, groups, folds, np.array(ids), outputs, refs, outer, frozen, v1


def summarize(pred, views, folds, ids, comparisons):
    top, _ = base.select_top(pred)
    hits = views[np.arange(len(top)), top]

    def scope(idx):
        return dict(queries=len(idx), top1={v: base.metric(int(hits[idx, j].sum()), len(idx)) for j, v in enumerate(base.VIEWS)},
            paired={name: {v: mcnemar(other[idx, j], hits[idx, j], ids[idx]) for j, v in enumerate(base.VIEWS)} for name, other in comparisons.items()})

    return dict(totals=scope(np.arange(len(ids))), by_fold={str(f): scope(np.flatnonzero(folds == f)) for f in (2, 3, 4)},
        selected_native_ranks=(top+1).tolist()), hits


def learning_curve(runner, outer, folds, ids, y, columns):
    records, pooled = [], {}
    for fraction in FRACTIONS:
        pooled[fraction] = []
        for seed in SEEDS:
            correct = np.zeros(len(ids), dtype=bool)
            for f, train, test in outer:
                companies = np.array(sorted(set(runner.groups[train])))
                rng = np.random.default_rng(seed+int(f))
                chosen = set(rng.permutation(companies)[:max(1, math.ceil(fraction*len(companies)))])
                subset = train[np.isin(runner.groups[train], list(chosen))]
                pred = runner.score('R2', subset, test, columns, f'learning_{fraction}_{seed}_fold{f}')
                top, _ = base.select_top(pred)
                correct[test] = y[test, top].astype(bool)
                records.append(dict(fold=f, fraction=fraction, seed=seed, train_queries=len(subset),
                    train_queries_available=len(train), train_companies=len(chosen),
                    train_companies_available=len(companies), exact=base.metric(int(correct[test].sum()), len(test))))
            pooled[fraction].append(int(correct.sum()))
        print(base.json.dumps(dict(stage='learning', fraction=fraction, exact_by_seed=pooled[fraction])), flush=True)
    summary = []
    for fraction in FRACTIONS:
        scores = np.array(pooled[fraction], dtype=float)
        selected = [r for r in records if r['fraction'] == fraction]
        per_fold = {}
        for f in ('2', '3', '4'):
            rows = [r for r in selected if r['fold'] == f]
            vals = [r['exact']['percent'] for r in rows]
            per_fold[f] = dict(mean_percent=float(np.mean(vals)), std_percent=float(np.std(vals, ddof=1)),
                train_queries=[r['train_queries'] for r in rows], train_companies=[r['train_companies'] for r in rows])
        summary.append(dict(fraction=fraction, exact_by_seed=scores.astype(int).tolist(),
            mean_hits=float(scores.mean()), std_hits=float(scores.std(ddof=1)),
            mean_percent=float(scores.mean()/len(ids)*100), std_percent=float(scores.std(ddof=1)/len(ids)*100), by_fold=per_fold))
    delta = np.array(pooled[1.])-np.array(pooled[.75])
    return dict(model='R2', fractions=list(FRACTIONS), seeds=list(SEEDS),
        sampling='Nested company permutations within each outer train; ceil(fraction*companies), whole companies, no label stratification. Model seed fixed.',
        std='Sample standard deviation across5 sampling seeds, not a confidence interval;100% is the same deterministic fit.',
        runs=records, summary=summary, last_step_delta_mean_pp=float(delta.mean()/len(ids)*100),
        last_step_delta_std_pp=float(delta.std(ddof=1)/len(ids)*100),
        last_step_positive_seeds=int((delta > 0).sum()),
        rising_observed_at_full=bool(delta.mean() > 0))


def confidence(runner, outer, X, y, views, groups, folds, ids, columns, out):
    n = len(X)
    choices = {m: np.zeros(n, dtype=bool) for m in ('margin', 'scene_isotonic', 'split_conformal')}
    common = np.full((n, 100), np.nan)
    scores = {m: np.zeros(n) for m in ('margin', 'scene_isotonic')}
    receipts, meta_refs = [], []
    conformal_covered = np.zeros(n, dtype=bool)
    conformal_size = np.zeros(n, dtype=int)
    for f, train, test in outer:
        companies = np.array(sorted(set(groups[train])))
        companies = np.random.default_rng(base.SEED+100+int(f)).permutation(companies)
        count = max(1, math.ceil(.2*len(companies)))
        threshold_companies, iso_companies = companies[:count], companies[count:2*count]
        threshold_rows = train[np.isin(groups[train], threshold_companies)]
        iso_rows = train[np.isin(groups[train], iso_companies)]
        proper = train[~np.isin(groups[train], companies[:2*count])]
        roles = [proper, iso_rows, threshold_rows, test]
        if any(set(groups[a]) & set(groups[b]) for i, a in enumerate(roles) for b in roles[i+1:]):
            raise ValueError('Confidence roles overlap')
        meta_pred = np.full((len(proper), 100), np.nan)
        splitter = GroupKFold(n_splits=3, shuffle=True, random_state=base.SEED)
        for k, (a, b) in enumerate(splitter.split(proper, groups=groups[proper])):
            meta_pred[b] = runner.score('R3', proper[a], proper[b], columns, f'confidence_meta_fold{f}_{k}')
        calibration_test = np.r_[iso_rows, threshold_rows, test]
        pred = runner.score('R3', proper, calibration_test, columns, f'confidence_common_fold{f}')
        iso_pred, limit_pred, test_pred = np.split(pred, [len(iso_rows), len(iso_rows)+len(threshold_rows)])
        common[test] = test_pred
        meta_top, _ = base.select_top(meta_pred)
        meta_target = y[proper, meta_top].astype(int)
        classifier = logistic(meta_target)
        classifier.fit(scene(meta_pred, X[proper]), meta_target)
        iso_top, _ = base.select_top(iso_pred)
        iso_target = y[iso_rows, iso_top].astype(int)
        iso_raw = probability(classifier, scene(iso_pred, X[iso_rows]))
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0., y_max=1.)
        iso.fit(iso_raw, iso_target)
        path = out/f'confidence_scene_fold{f}.pkl'
        with path.open('wb') as file:
            pickle.dump((classifier, iso), file, protocol=5)
        meta_refs.append(base.record(path))
        limit_top, limit_margin = base.select_top(limit_pred)
        test_top, test_margin = base.select_top(test_pred)
        limit_correct = y[threshold_rows, limit_top].astype(bool)
        test_scene = iso.predict(probability(classifier, scene(test_pred, X[test])))
        limit_scene = iso.predict(probability(classifier, scene(limit_pred, X[threshold_rows])))
        limits = {}
        for name, cal, value in [('margin', limit_margin, test_margin), ('scene_isotonic', limit_scene, test_scene)]:
            threshold, stats = base.threshold_choice(cal, limit_correct)
            choices[name][test] = value >= threshold if threshold is not None else False
            scores[name][test] = value
            limits[name] = dict(threshold=threshold, calibration=stats)
        # Candidate softmax plus an OUTSIDE class with zero mass. Missing truths
        # are nonconformity1, not discarded. Max per calibration company supports
        # group exchangeability (not independent-query exchangeability).
        pcal, ptest = softmax(limit_pred), softmax(test_pred)
        nonconformity = 1-(pcal*y[threshold_rows]).sum(axis=1)
        maxima = np.array([nonconformity[groups[threshold_rows] == g].max() for g in sorted(set(groups[threshold_rows]))])
        k = math.ceil((len(maxima)+1)*.98)
        q = float(np.sort(maxima)[k-1]) if k <= len(maxima) else math.inf
        candidate_set = 1-ptest <= q
        outside = 1 <= q
        size = candidate_set.sum(axis=1)+int(outside)
        choices['split_conformal'][test] = (size == 1) & ~np.full(len(test), outside)
        conformal_size[test] = size
        conformal_covered[test] = (candidate_set*y[test]).any(axis=1) | ((y[test].sum(axis=1) == 0) & outside)
        receipts.append(dict(fold=f, proper_fit_queries=[str(ids[i]) for i in proper],
            scene_fit_queries=[str(ids[i]) for i in proper], isotonic_queries=[str(ids[i]) for i in iso_rows],
            threshold_conformal_queries=[str(ids[i]) for i in threshold_rows], external_queries=[str(ids[i]) for i in test],
            proper_companies=len(set(groups[proper])), isotonic_companies=len(set(groups[iso_rows])),
            threshold_companies=len(set(groups[threshold_rows])), all_role_company_overlaps=0,
            scene_meta_targets='three-fold company-OOF predictions WITHIN proper fit only', limits=limits,
            conformal=dict(alpha=.02, calibration_companies=len(maxima), quantile_index=k,
                quantile=None if math.isinf(q) else q, infinite_quantile=math.isinf(q),
                missing_truths=int((y[threshold_rows].sum(axis=1) == 0).sum()),
                quantile_unit='max nonconformity per company', outside_class_in_sets=outside)))
        print(base.json.dumps(dict(stage='confidence', fold=f, limits=limits,
            conformal_quantile=None if math.isinf(q) else q, accepted={m: int(v[test].sum()) for m, v in choices.items()})), flush=True)
    top, _ = base.select_top(common)
    hits = views[np.arange(n), top]
    methods = {}
    for method, selected in choices.items():
        methods[method] = dict(totals=base.sur_metrics(hits, selected),
            by_fold={str(f): base.sur_metrics(hits[folds == f], selected[folds == f]) for f in (2, 3, 4)},
            selected_queries=[str(ids[i]) for i in np.flatnonzero(selected)])
    return dict(protocol='Common R3 trained on60% of outer-training companies;20% disjoint isotonic,20% disjoint threshold/conformal. Scene model fits3-fold company-OOF in proper60%. No calibration or test company in any ranking fit used on that role.',
        differs_from_full_R3=True, common_ranker_top1={v: base.metric(int(hits[:, j].sum()), n) for j, v in enumerate(base.VIEWS)},
        scene_features=SCENE_FEATURES, scene_classifier=dict(C=1., max_iter=1000, missing='train_median', scaling='train_standard', one_class_fallback='constant empirical prior'),
        conformal_definition='101 labels (100 candidates+OUTSIDE), nonconformity1-softmax(candidate), OUTSIDE=1; group-max split quantile ceil((m+1)*(1-alpha)); singleton candidate only SUR.',
        conformal_warning='98% marginal/group set coverage is NOT98% selective precision. Exchangeability is not certified on this consumed pilot and supervised frozen upstream.',
        conformal_set_coverage=base.metric(int(conformal_covered.sum()), n),
        conformal_set_size_histogram={str(k): int((conformal_size == k).sum()) for k in np.unique(conformal_size)},
        receipts=receipts, model_refs=meta_refs, methods=methods,
        per_query=[dict(query_id=str(ids[i]), fold=int(folds[i]), selected_native_rank=int(top[i])+1,
            hits={v: bool(hits[i, j]) for j, v in enumerate(base.VIEWS)},
            margin=float(scores['margin'][i]), scene_probability=float(scores['scene_isotonic'][i]),
            sur={m: bool(v[i]) for m, v in choices.items()}, conformal_set_size=int(conformal_size[i]),
            conformal_covered=bool(conformal_covered[i])) for i in range(n)]), common


def residuals(pred, X, y, outputs, ids, folds):
    top, _ = base.select_top(pred)
    f = {name: j for j, name in enumerate(base.FEATURES)}
    rows = []
    for i, qid in enumerate(ids):
        correct = np.flatnonzero(y[i])
        if not len(correct) or y[i, top[i]]:
            continue
        t, chosen = int(correct[0]), int(top[i])
        a, b = X[i, t], X[i, chosen]
        same = outputs[str(qid)][t]['siren'] == outputs[str(qid)][chosen]['siren']
        status = a[f['state_missing']] == b[f['state_missing']] == 0 and a[f['is_active']] != b[f['is_active']]
        seat = a[f['site_missing']] == b[f['site_missing']] == 0 and a[f['is_siege']] != b[f['is_siege']]
        homonym = a[f['name_jaro_max']] >= .95 and b[f['name_jaro_max']] >= .95
        near = (a[f['full_address_match']] == b[f['full_address_match']] == 1 or
            (a[f['geo_missing']] == b[f['geo_missing']] == 0 and
             max(a[f['geo_log_distance']], b[f['geo_log_distance']]) <= math.log1p(200)))
        family = 'ferme_vs_actif' if same and status else 'siege_vs_site' if same and seat else 'meme_siren_autre_site' if same else 'homonyme' if homonym else 'adresse_proche' if near else 'autre'
        rows.append((int(folds[i]), family))
    counts = {family: sum(r[1] == family for r in rows) for family in FAMILIES}
    return dict(total=len(rows), absent_truths=int((y.sum(axis=1) == 0).sum()),
        counts=counts, by_fold={str(f): {family: sum(r == (f, family) for r in rows) for family in FAMILIES} for f in (2, 3, 4)},
        priority=list(FAMILIES), diagnostic_only_not_training=True,
        definitions='Exclusive priority: same-SIREN known active/closed disagreement; then same-SIREN known siege disagreement; then same-SIREN other SIRET; then both official-name Jaro>=.95 vs CRM; then both full address concordant OR both admissible distances<=200m from CRM; else other. Proxies, NOT audited causes. No per-case output.')


def recommend(result):
    alg = result['algorithms']
    winners = [a for a in ('R1', 'R3', 'R4', 'R5') if
        alg[a]['totals']['top1']['exact']['hits'] > alg['R2']['totals']['top1']['exact']['hits'] and
        alg[a]['totals']['paired']['R2']['exact']['holm_p'] < .05]
    curve = result['learning_curve']['summary']
    if winners:
        best = max(winners, key=lambda a: alg[a]['totals']['top1']['exact']['hits'])
        return dict(choice='changer_algorithme', algorithm=best,
            reason='Gain exact vsR2 avec p de McNemar ajusté Holm<0,05 parmi les comparaisons pré-déclarées. À confirmer indépendamment, pas une promotion.')
    if curve[3]['mean_percent']-curve[2]['mean_percent'] >= 1 and curve[2]['mean_percent']-curve[1]['mean_percent'] >= 1:
        return dict(choice='plus_de_donnees', reason='Les deux derniers incréments gagnent chacun>=1 point moyen; pas de preuve que3000 suffiront ni de génération autorisée.')
    return dict(choice='meilleures_features', features=['preuve de nom propre du site distincte du nom UL',
        'adresse/geolocalisation datée et fiabilité de la source', 'compatibilité sémantique activité/usage CRM et données officielles'],
        reason='Pas de gain algorithmique confirmé après Holm et pas de croissance>=1 point sur chacun des deux derniers incréments. Priorité au signal distinctif, pas preuve de causalité ni exclusion d’un bénéfice de données supplémentaires.')


def mm(m):
    return base.md_metric(m)


def markdown(d):
    alg, curve = d['algorithms'], d['learning_curve']
    lines = ['# Étape2 — diagnostic données / algorithme / features', '',
        f'**{d["queries"]} requêtes ×100 candidats ×46 features V1 inchangés.** Recherche gelée `36c7a476` ; pilote consommé, pas certification production.',
        'Plan complet écrit avant fits dans le répertoire du run ; aucun réglage après résultats.', '',
        '##1. Courbe d’apprentissage R2', '',
        '| Part des entreprises train | Exact moyen / N | % moyen ± écart-type |', '|---:|---:|---:|']
    for r in curve['summary']:
        lines.append(f'| {100*r["fraction"]:.0f}% | {r["mean_hits"]:.2f}/{d["queries"]} | {r["mean_percent"]:.2f} ± {r["std_percent"]:.2f} |')
    lines += ['', '- Sous-échantillons imbriqués par entreprises entières, cinq graines fixes20261017–21 ; graine du modèle constante. Fractions d’entreprises, pas quotas exacts de requêtes : effectifs réels détaillés par fold/graine en JSON.',
        '- Écart-type échantillonnal entre graines, pas un IC ;100% réutilise le même fit déterministe, dispersion nulle attendue.',
        f'- Dernier incrément75→100% : **{curve["last_step_delta_mean_pp"]:+.2f} points**, positif pour {curve["last_step_positive_seeds"]}/5 graines. Montée observée : {curve["rising_observed_at_full"]}. Cela n’extrapole pas la performance à3000.', '',
        '##2. Comparaison des algorithmes', '',
        '- R1 : imputation médiane et standardisation fit uniquement sur train, logistique C=1, max_iter1000 (budget de convergence, pas HPO).',
        '- R2 : V1 pairwise200 arbres/profondeur3 ; R3 : seule l’objective devient rank:ndcg ; R4 : binary:logistic, mêmes paramètres hors réglages lambdarank inapplicables.',
        '- R5 : HistGradientBoostingClassifier par défaut, seule graine fixée ; early_stopping auto conserve son split interne par candidats, uniquement dans le train externe, pas un split entreprise interne.',
        '- Aucun poids de classe, recherche de paramètres, arrêt sur test, positif injecté ou requête retirée. Départage stable selon l’ordre V142.', '',
        '| Modèle | Exact | SIREN | Opérationnel | Site | Union | Gains/pertes exact vsR0 | p exact vsR0 / vsR2 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ALGORITHMS:
        a = alg[name]['totals']
        pair = a['paired']['R0']['exact']
        lines.append(f'| {name} | '+' | '.join(str(a['top1'][v]['hits']) for v in base.VIEWS)+
            f' | +{len(pair["gains"])}/−{len(pair["losses"])} | {pair["exact_two_sided_p"]:.4g} / {a["paired"]["R2"]["exact"]["exact_two_sided_p"]:.4g} |')
    lines += ['', '- Toutes valeurs ont le dénominateur intégral,19 absences restent erreurs. IC Wilson95, cinq vues, folds et query_id des paires en JSON.',
        '- McNemar exact bilatéral (discordances), exploratoire par requête : dépendances intra-entreprise/pilote consommé non corrigées. Holm sur les9 comparaisons exactes R1–5 vsR0 et R1/3/4/5 vsR2 pour la recommandation ; autres vues brutes descriptives.', '',
        '##3. Ablations dans R3', '',
        '| Retrait / variante | Top1 exact | Écart vsR3 |', '|---|---:|---:|']
    full = alg['R3']['totals']['top1']['exact']['hits']
    for name, a in d['ablations'].items():
        h = a['totals']['top1']['exact']['hits']
        lines.append(f'| {name} | {h}/{d["queries"]} | {h-full:+d} |')
    lines += ['', '- Les retraits nom/adresse enlèvent aussi interactions et deltas de scène dérivés ; retrait natif enlève aussi les deux ancres de scène top1 natif. Listes exactes dans le plan/JSON.',
        '- La variante native+scène conserve14 features existantes, dont deltas nom/voie et densité adresse : ce n’est PAS une variante « scores seuls ». Pool et départage restent natifs dans toutes les ablations.',
        '- Ablation ≠ importance causale : redondances, réentraînement et faible effectif peuvent changer le signe.', '',
        '##4. Confiance : marge / scène calibrée / conforme', '',
        '**Protocole de comparaison honnête, commun aux trois méthodes :** chaque train externe est séparé par entreprise en60% fit propre R3,20% isotonique,20% seuil/conforme. Le classifieur scène logistique C=1 est fit sur un OOF entreprise3-plis DANS les60%. Les20% isotonique ne servent pas au seuil. Aucun modèle R3 utilisé sur une calibration n’a vu les labels de ses entreprises.',
        'Ce R3 est donc entraîné sur moins de données que le R3 complet de la comparaison ; on ne transfère pas son seuil à un full-fit.',
        f'Top1 exact du socle commun : {mm(d["confidence"]["common_ranker_top1"]["exact"])}.', '',
        '| Méthode | SUR | Couverture avec Wilson95 | Précision exacte avec Wilson95 | Précision opérationnelle |', '|---|---:|---|---|---|']
    for name, r in d['confidence']['methods'].items():
        s = r['totals']
        lines.append(f'| {name} | {s["count"]} | {mm(s["rate"])} | {mm(s["precision"]["exact"])} | {mm(s["precision"]["operational"])} |')
    lines += ['', '- Marge et probabilité : seuil sur20% seuil seulement, couverture maximale à précision exacte empirique≥98%, aucune acceptation si impossible. Aucune borne Wilson/minimum d’effectif imposés par la recette.',
        '- Scène : marge, entropie et dispersion scores, accord natif, nombre de sites, nom/adresse, géographie et état du top1 ; uniquement les46 features et sorties du ranker, pas d’ID ni de vérité en entrée.',
        '- Conforme : split conformal α=2%, quantile fini corrigé sur maximum par entreprise de1−softmax(vérité).101e classe OUTSIDE à probabilité0 pour les vérités absentes ; celles-ci ne sont jamais retirées. SUR seulement pour un singleton candidat sans OUTSIDE.',
        f'- Couverture de l’ensemble conforme (OUTSIDE inclus) : {mm(d["confidence"]["conformal_set_coverage"])} ; tailles des ensembles : `{d["confidence"]["conformal_set_size_histogram"]}`.',
        '- **98% de couverture marginale d’un ensemble n’est pas98% de précision des SUR.** Une garantie conforme exige échangeabilité des entreprises, non certifiée ici ; une calibration avec absences peut imposer tous les101 labels et aucune décision.',
        '- Aucun essai de cascade décideur+risque n’est fait : ces mesures comparent marge seule à UNE confiance de scène calibrée. Elles ne démontrent pas un bénéfice d’ajouter un second étage appris ; seuils/poids ne sont pas promus.', '',
        '##5. Résiduels R3 : vérité présente mais non première', '',
        '| Famille exclusive | Effectif |', '|---|---:|']
    for family, count in d['residuals']['counts'].items():
        lines.append(f'| {family} | {count} |')
    lines += [f'| **Total** | **{d["residuals"]["total"]}** |', '',
        '- Diagnostic label-side uniquement, jamais des règles de ranking : priorité fermé/actif même SIREN, siège différent même SIREN, autre site même SIREN, homonyme approché (deux Jaro≥0,95), adresse proche approchée (deux concordances ou deux distances admissibles≤200m du CRM), autre.',
        '- Catégories exclusives et reproductibles, pas causes établies par audit humain ; ambiguïté sémantique et qualité du label ne sont pas résolues. Les absences restent au dénominateur global, hors ce tableau explicitement conditionnel.', '',
        '## Recommandation unique et FusionSet', '',
        f'**{d["recommendation"]["choice"]}** : {d["recommendation"]["reason"]}',
        ('Features à tester ultérieurement : '+', '.join(d['recommendation']['features'])+'.') if 'features' in d['recommendation'] else ('Algorithme candidat : '+d['recommendation'].get('algorithm', 'sans changement proposé')+'.'),
        '- Règle de décision écrite avant fits : gain algorithme confirmé vsR2 après Holm → changer ; sinon deux derniers incréments de courbe≥1 point chacun → données ; sinon features. C’est une priorité expérimentale, pas une identification causale certaine.',
        '- **FusionSet non testé.** Les compteurs/deltas de scène regardent déjà les100, sans attention apprise. Avec526 dossiers (~350 train externe), aucune preuve ne justifie maintenant un réseau d’attention original. Gemma pourrait apporter un signal sémantique absent des46, mais cet apport de features est confondu avec l’architecture proposée : l’isoler avant un modèle hybride, puis réévaluer l’attention avec plus de groupes indépendants. Aucun gain ni volume minimal promis.', '',
        '## Preuves / limites', '',
        f'- {d["actual_fit_count"]} fits candidats distincts (réemploi des100% et caches déterministes), reçus groupés/modèles sauvegardés ; durée{d["elapsed_seconds"]:.2f}s, pic RSS{d["peak_rss_bytes"]/1024**2:.1f}Mio, garde7,5Gio.',
        '- Matrice et listes vérifiées par SHA avant/après ; cinq vues identiques à la baseline. R2 complet vérifié requête par requête contre V1. Features et labels immuables, pas d’embedding/calcul retrieval, jeux fermés, VM, web, dépense ou push.',
        '- Validation OOF de ces seuls modèles aval : amont supervisé consommé/non full-OOF, routage par SIREN humain inchangé. Pas de précision nationale/production ni de garantie de SUR. Pas de déduction que « absence de significativité = égalité ».',
        '- Commande : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_diagnostic_v1.py --output-dir <dossier_CATNAT_neuf>` ; `--smoke` utilise36 dossiers réels, mêmes réglages, sans écraser les docs.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir or DEFAULT_RUN/('smoke' if args.smoke else 'full')
    if out.exists():
        raise ValueError('Require a NEW owned output directory; preserve prior runs')
    out.mkdir(parents=True)
    started, stop, resource = time.monotonic(), threading.Event(), dict(peak_rss_bytes=0)

    def monitor():
        process = psutil.Process()
        while not stop.is_set():
            resource['peak_rss_bytes'] = max(resource['peak_rss_bytes'], process.memory_info().rss)
            if resource['peak_rss_bytes'] >= int(7.5*1024**3):
                base.write(out/'resource_stop.json', resource)
                os._exit(75)
            stop.wait(.1)

    threading.Thread(target=monitor, daemon=True).start()
    X, y, views, groups, folds, ids, outputs, refs, outer, frozen, v1 = load(args.smoke)
    columns = list(range(46))
    ablation_columns = {f'without_{g}': [i for i, f in enumerate(base.FEATURES) if f not in set(names+DEPENDENCIES.get(g, []))] for g, names in GROUPS.items()}
    ablation_columns['native_plus_scene_only'] = list(range(32, 46))
    plan = dict(status='PREREGISTERED_BEFORE_FITS', source_refs=refs, frozen_commit=frozen, smoke=args.smoke,
        sampling_seeds=SEEDS, fractions=FRACTIONS, sampling_model='R2', model_seed=base.SEED,
        xgb_params=base.PARAMS, algorithms=list(ALGORITHMS), logistic_C=1., logistic_max_iter=1000,
        hgb='sklearn defaults; random_state20261017 only', groups=GROUPS, dependency_removals=DEPENDENCIES,
        ablation_features={k: [base.FEATURES[i] for i in v] for k, v in ablation_columns.items()},
        confidence='common R3 proper60%, iso20%, threshold/conformal20% companies; scene3-foldOOF only within proper; group max split conformal alpha.02 with OUTSIDE',
        confidence_scene_features=SCENE_FEATURES, residual_family_priority=FAMILIES,
        recommendation_rule='Holm-confirmed gain vsR2 => algorithm; else last2 learning increments>=1pp => data; else features')
    base.write(out/'plan.json', plan)
    runner = Runner(X, y, groups, ids, out/'models')
    curve = learning_curve(runner, outer, folds, ids, y, columns)
    predictions = {'R0': np.tile(-np.arange(100, dtype=float), (len(ids), 1))}
    for name in ALGORITHMS[1:]:
        pred = np.full((len(ids), 100), np.nan)
        for f, train, test in outer:
            pred[test] = runner.score(name, train, test, columns, f'comparison_{name}_fold{f}')
        predictions[name] = pred
        print(base.json.dumps(dict(stage='algorithm', name=name, exact=int(y[np.arange(len(ids)), base.select_top(pred)[0]].sum()))), flush=True)
    if not args.smoke:
        observed = (base.select_top(predictions['R2'])[0]+1).tolist()
        if observed != [r['selected_native_rank'] for r in v1['real']['per_query']]:
            raise ValueError('R2 does not exactly reproduce V1')
    comparisons = {name: views[np.arange(len(ids)), base.select_top(predictions[name])[0]] for name in ('R0', 'R2')}
    algorithms = {name: summarize(pred, views, folds, ids, comparisons)[0] for name, pred in predictions.items()}
    # Nine prespecified primary exact paired comparisons, adjusted together.
    tests = [(name, ref, algorithms[name]['totals']['paired'][ref]['exact']) for name in ALGORITHMS[1:] for ref in ('R0', 'R2') if name != ref]
    tests.sort(key=lambda r: r[2]['exact_two_sided_p'])
    previous = 0.
    for i, (_, _, r) in enumerate(tests):
        previous = max(previous, min(1., (len(tests)-i)*r['exact_two_sided_p']))
        r['holm_p'] = previous
    ablations = {}
    for name, cols in ablation_columns.items():
        pred = np.full((len(ids), 100), np.nan)
        for f, train, test in outer:
            pred[test] = runner.score('R3', train, test, cols, f'ablation_{name}_fold{f}')
        ablations[name], _ = summarize(pred, views, folds, ids, comparisons)
        ablations[name]['features'] = [base.FEATURES[i] for i in cols]
        predictions[name] = pred
        print(base.json.dumps(dict(stage='ablation', name=name, exact=ablations[name]['totals']['top1']['exact']['hits'])), flush=True)
    conf, conf_pred = confidence(runner, outer, X, y, views, groups, folds, ids, columns, out)
    predictions['confidence_common'] = conf_pred
    errors = residuals(predictions['R3'], X, y, outputs, ids, folds)
    if errors['total'] != int(y.any(axis=1).sum())-algorithms['R3']['totals']['top1']['exact']['hits']:
        raise ValueError('Residual denominator mismatch')
    np.savez_compressed(out/'predictions.npz', **predictions)
    for ref in refs.values():
        base.verified(ref)
    stop.set()
    result = dict(status='MEASURED_DEVELOPMENT_NOT_PRODUCTION', queries=len(ids), candidates=len(ids)*100,
        source_refs=refs, frozen_commit=frozen, plan=base.record(out/'plan.json'),
        predictions=base.record(out/'predictions.npz'), query_ids=ids.tolist(), folds=folds.tolist(),
        groups_by_fold={str(f): len(set(groups[folds == f])) for f in (2, 3, 4)}, features=base.FEATURES,
        learning_curve=curve, algorithms=algorithms, ablations=ablations, confidence=conf, residuals=errors,
        fit_receipts=runner.receipts, actual_fit_count=len(runner.receipts),
        recall100={v: base.metric(int(views[:, :, j].any(axis=1).sum()), len(ids)) for j, v in enumerate(base.VIEWS)},
        no_retrieval_executed=True, no_feature_recomputed=True, closed_sets_opened=False,
        source_labels_modified=False, production_promotion_allowed=False, smoke=args.smoke,
        versions=dict(python=sys.version, sklearn=sklearn.__version__, xgboost=xgb.__version__),
        elapsed_seconds=time.monotonic()-started, **resource)
    result['recommendation'] = recommend(result)
    base.write(out/'result.json', result)
    (out/'result.md').write_text(markdown(result))
    if not args.smoke:
        base.write(ROOT/'docs/step2_diagnostic_v1.json', result)
        (ROOT/'docs/step2_diagnostic_v1.md').write_text(markdown(result))
    print(base.json.dumps(dict(status=result['status'], queries=len(ids),
        top1={a: algorithms[a]['totals']['top1']['exact'] for a in ALGORITHMS},
        confidence={a: r['totals'] for a, r in conf['methods'].items()},
        residuals=errors['counts'], recommendation=result['recommendation'],
        elapsed_seconds=result['elapsed_seconds'], peak_rss_bytes=resource['peak_rss_bytes'])), flush=True)


if __name__ == '__main__':
    main()
