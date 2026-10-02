"""Small contrastively learned sparse query kernel, NOT a SPLADE reproduction.

Official documents have query-independent, field-typed nonnegative impacts.
The identical top-k log1p(ReLU) query activation is used in fit and serving.
This module trains on natural source contrasts only, not on a retrieval pool.
It has no national executor and cannot establish retrieval recall by itself.
"""
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from src.xgb_matcher.retrieval_impact_v68 import official_initialisms
from src.xgb_matcher.retrieval_initialisms_v67 import membership
from src.xgb_matcher.retrieval_joint_contrastive_v63 import basis, head_loss, unit_parts
from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_proof_v56 import BASE, FEATURES as NATIVE
from src.xgb_matcher.retrieval_sparse_query_v59 import words

FEATURES = (*NATIVE, 'official_initialism', 'initialism_at_place',
            'initialism_number_at_place', 'initialism_address_at_place')
PRIOR = np.r_[BASE, np.zeros(4)]


def impacts(document):
    """Typed words and initialisms; IDs and administrative keys are not terms.

    Normalization uses ALL terms in each view, never just the fitted vocab.
    Initialisms are compiled from individual aliases, not a merged word bag.
    Other views currently merge official aliases: provenance is NOT preserved.
    """
    views = [set().union(*(words(v) for f in (
        'legal_names', 'trade_names', 'site_names', 'historical_names', 'supporting_names')
        for v in document.get(f, []))),
        set().union(*(words(v) for f in ('addresses', 'historical_addresses', 'supporting_addresses')
                     for v in document.get(f, []))),
        set().union(*(official_initialisms(v) for f in ('legal_names', 'trade_names', 'site_names')
                     for v in document.get(f, [])))]
    return {prefix + ':' + term: 1. / np.sqrt(len(terms))
            for prefix, terms in zip(('N', 'A', 'I'), views) for term in terms}


def recipe(q, rows, documents, indices, forbidden, cap=1024, dimensions=16):
    """Fit vocabulary and basis from safe train entities ONLY (pos AND neg)."""
    blocked = set(forbidden); ids = set()
    for i in indices:
        row = rows[i]
        if row['positive_siren'] in blocked:
            raise ValueError('Protected positive in sparse recipe')
        ids.add(row['positive_siret'])
        ids.update(n['siret'] for n in row['negatives'] if n['siren'] not in blocked)
    counts = Counter(t for sid in sorted(ids) for t in documents[sid])
    vocab = sorted([t for t, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                    if 3 <= count <= .15 * len(ids)][:cap])
    if not vocab:
        raise ValueError('Empty safe sparse vocabulary')
    bases = [basis(q[indices, view], dimensions) for view in (0, 1)]
    z = latent(q, bases)
    lookup = {term: j for j, term in enumerate(vocab)}
    target = np.zeros((len(indices), len(vocab)))
    for j, i in enumerate(indices):
        for t, value in documents[rows[i]['positive_siret']].items():
            if t in lookup:
                target[j, lookup[t]] = value
    initial = np.linalg.solve(z[indices].T @ z[indices] + 10. * np.eye(z.shape[1]),
                              z[indices].T @ target)
    return vocab, bases, initial


def latent(q, bases):
    return np.column_stack([q[:, view] @ bases[view] for view in (0, 1)] + [np.ones(len(q))])


def activate(raw, topk=16):
    """Same deterministic hard sparsity in training and serving; no renorm."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all() or topk < 1:
        raise ValueError('Sparse activation drift')
    values = np.log1p(np.maximum(raw, 0.))
    # Vocabulary is lexically sorted. Stable sort provides deterministic ties.
    order = np.argsort(-values, axis=1, kind='stable')[:, :topk]
    keep = np.zeros_like(raw, dtype=bool)
    np.put_along_axis(keep, order, True, axis=1)
    keep &= raw > 0
    return np.where(keep, values, 0.), np.where(keep, 1. / (1. + np.maximum(raw, 0.)), 0.)


def proof(row, native, aliases):
    f = np.asarray(native, dtype=float)
    if f.shape != (len(NATIVE),) or not np.isfinite(f).all():
        raise ValueError('Native proof shape drift')
    a = membership(row['crm_name'], aliases)
    return np.r_[f, a, a*f[4], a*f[5], a*f[7]]


def design(rows, scenes, documents, aliases, indices, vocab, forbidden=()):
    blocked = set(forbidden); lookup = {t: j for j, t in enumerate(vocab)}
    rr, cc, vv, native, sizes, tags = [], [], [], [], [], []
    for i in indices:
        row, scene = rows[i], scenes[i]
        if any(row[k] != scene[k] for k in ('query_id', 'fold', 'positive_siren')):
            raise ValueError('Scene identity drift')
        if len(row['negatives']) != len(scene['negatives']):
            raise ValueError('Natural negative denominator drift')
        p = row['positive_siret']; size = 0
        pf = proof(row, scene['positive_features'], aliases[p])
        for n, nf in zip(row['negatives'], scene['negatives']):
            if n['siren'] != nf['siren']:
                raise ValueError('Negative order drift')
            if n['siren'] in blocked:
                continue
            sid = n['siret']; pair = len(native)
            native.append(pf - proof(row, nf['features'], aliases[sid])); size += 1
            for term in documents[p].keys() | documents[sid].keys():
                delta = documents[p].get(term, 0.) - documents[sid].get(term, 0.)
                if term in lookup and delta:
                    rr.append(pair); cc.append(lookup[term]); vv.append(delta)
            tags.append(dict(same_parent=n['siren'] == row['positive_siren']))
        if not size:
            raise ValueError('No safe negative; do not drop query')
        sizes.append(size)
    sizes = np.array(sizes, dtype=int)
    delta = csr_matrix((vv, (rr, cc)), shape=(len(native), len(vocab))).tocoo()
    return dict(native=np.array(native), delta=delta, sizes=sizes,
                owner=np.repeat(np.arange(len(sizes)), sizes), tags=tags)


def score(data, weights, activations, alpha=1.):
    coo = data['delta']; owner = data['owner']
    extra = np.bincount(coo.row, weights=coo.data*activations[owner[coo.row], coo.col],
                        minlength=len(owner))
    return data['native'] @ weights + alpha*extra


def objective(theta, z, data, initial, ridge, flops, topk):
    """Equal-query InfoNCE + sparse FLOPS + train-prior decoder ridge."""
    native = theta[:len(FEATURES)]; decoder = theta[len(FEATURES):].reshape(initial.shape)
    act, derivative = activate(z @ decoder, topk)
    margins = score(data, native, act)
    sizes = data['sizes']; starts = np.r_[0, np.cumsum(sizes)[:-1]]
    logits = -margins; maxima = np.maximum(0., np.maximum.reduceat(logits, starts))
    e = np.exp(logits - np.repeat(maxima, sizes))
    denom = np.exp(-maxima) + np.add.reduceat(e, starts)
    loss = np.mean(maxima + np.log(denom))
    probabilities = e / np.repeat(denom, sizes) / len(sizes)
    coo = data['delta']; owner = data['owner'][coo.row]; width = initial.shape[1]
    grad_act = np.bincount(owner*width + coo.col, weights=-coo.data*probabilities[coo.row],
                           minlength=len(z)*width).reshape(len(z), width)
    mean = act.mean(axis=0)
    grad_act += 2*flops*mean[None, :]/len(z)
    diff = decoder - initial; penalty = ridge/initial.shape[0]
    native_diff = native - PRIOR
    loss += .005*np.sum(native_diff**2) + penalty*np.sum(diff**2) + flops*np.sum(mean**2)
    grad = np.r_[-data['native'].T @ probabilities + .01*native_diff,
                 (z.T @ (grad_act*derivative) + 2*penalty*diff).ravel()]
    return float(loss), grad


def fit(z, data, initial, family, ridge, flops=.1, topk=16, maxiter=180):
    if family not in ('NATIVE23', 'RIDGE_SPARSE', 'CONTRASTIVE_SPARSE'):
        raise ValueError('Unknown sparse kernel family')
    if family == 'CONTRASTIVE_SPARSE':
        prior = np.r_[PRIOR, initial.ravel()]
        fun = lambda w: objective(w, z, data, initial, ridge, flops, topk)
        bounds = [(0., 64.)]*len(FEATURES) + [(-16., 16.)]*initial.size
    else:
        activation = activate(z @ initial, topk)[0]
        sparse = score(data, np.zeros(len(FEATURES)), activation)
        x = data['native'] if family == 'NATIVE23' else np.column_stack([data['native'], sparse])
        prior = PRIOR.copy() if family == 'NATIVE23' else np.r_[PRIOR, 1.]
        fun = lambda w: head_loss(w, x, data['sizes'], prior, np.full(len(prior), .005))
        bounds = [(0., 64.)]*len(prior)
    before = fun(prior)[0]
    result = minimize(fun, prior, jac=True, method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=maxiter, ftol=1e-8, gtol=1e-5))
    if not np.isfinite(result.x).all() or result.fun > before + 1e-8:
        raise ValueError('Sparse optimization drift')
    decoder = result.x[len(FEATURES):].reshape(initial.shape) if family == 'CONTRASTIVE_SPARSE' else initial
    alpha = 0. if family == 'NATIVE23' else (float(result.x[-1]) if family == 'RIDGE_SPARSE' else 1.)
    return (result.x[:len(FEATURES)], decoder, alpha), dict(
        converged=bool(result.success), message=str(result.message), iterations=int(result.nit),
        parameters=len(result.x), objective_before=before, objective_after=float(result.fun))


def summary(data, margins):
    sizes = data['sizes']; starts = np.r_[0, np.cumsum(sizes)[:-1]]
    counts = np.add.reduceat((margins <= 0).astype(int), starts)
    same = np.array([t['same_parent'] for t in data['tags']])
    return dict(queries=len(sizes), pairs=len(margins), head_error=float(np.mean(counts > 0)),
                mean_outrank_count=float(counts.mean()), hardest_margin=float(np.minimum.reduceat(margins, starts).mean()),
                mrr=float(np.mean(1./(1.+counts))), same_parent_pairs=int(same.sum()),
                same_parent_error=float(np.mean(margins[same] <= 0)) if same.any() else None)


def select(q, rows, scenes, raw_documents, aliases, registry, held, config):
    if q.shape[0] != len(rows) or q.shape[1] != 2 or len(rows) != len(scenes):
        raise ValueError('Query representation denominator drift')
    train, control, inner = positions(rows, held, registry)
    outer_blocked = set(registry[str(held)]); blocked = outer_blocked | inner
    documents = {sid: impacts(d) for sid, d in raw_documents.items()}
    def setup(indices, excluded):
        vocab, bases, initial = recipe(q, rows, documents, indices, excluded,
                                       config['vocabulary_cap'], config['latent_per_field'])
        z = latent(q, bases)
        data = design(rows, scenes, documents, aliases, indices, vocab, excluded)
        return vocab, bases, initial, z, data
    vocab, bases, initial, z, training = setup(train, blocked)
    validation = design(rows, scenes, documents, aliases, control, vocab, outer_blocked)
    candidates = []
    for family in ('NATIVE23', 'RIDGE_SPARSE', 'CONTRASTIVE_SPARSE'):
        for ridge in (config['ridges'] if family == 'CONTRASTIVE_SPARSE' else [config['ridges'][-1]]):
            fitted, report = fit(z[train], training, initial, family, ridge,
                                 config['flops'], config['query_terms'], config['maxiter'])
            w, decoder, alpha = fitted
            activation = activate(z[control] @ decoder, config['query_terms'])[0]
            candidates.append(dict(family=family, ridge=ridge, parameters=report['parameters'],
                                   converged=report['converged'], fit=report,
                                   inner=summary(validation, score(validation, w, activation, alpha))))
    viable = [c for c in candidates if c['converged']]
    if not viable:
        raise ValueError('No converged sparse candidate')
    chosen = min(viable, key=lambda c: (c['inner']['head_error'], c['inner']['mean_outrank_count'],
                                       c['parameters'], -c['ridge']))
    eligible = sorted(train + control)
    fv, fb, fi, fz, fd = setup(eligible, outer_blocked)
    fitted, final_fit = fit(fz[eligible], fd, fi, chosen['family'], chosen['ridge'],
                            config['flops'], config['query_terms'], config['maxiter'])
    if not final_fit['converged']:
        raise ValueError('Selected sparse refit not converged')
    reference, reference_fit = fit(fz[eligible], fd, fi, 'NATIVE23', config['ridges'][-1],
                                   config['flops'], config['query_terms'], config['maxiter'])
    if not reference_fit['converged']:
        raise ValueError('Reference native refit not converged')
    outer = [i for i, r in enumerate(rows) if r['fold'] == held]
    od = design(rows, scenes, documents, aliases, outer, fv)
    weights, decoder, alpha = fitted
    act = activate(fz[outer] @ decoder, config['query_terms'])[0]
    model = dict(family=chosen['family'], heldout_fold=held, weights=weights.tolist(),
                 alpha=alpha, decoder=decoder.tolist(), bases=[b.tolist() for b in fb],
                 vocabulary=fv, features=list(FEATURES), query_terms=config['query_terms'],
                 query_ids=[rows[i]['query_id'] for i in eligible],
                 training_folds=sorted({rows[i]['fold'] for i in eligible}), identifiers_as_features=False)
    return model, dict(heldout_fold=held, selected=chosen, candidates=candidates, final_fit=final_fit,
                       reference_fit=reference_fit, inner_entities=sorted(inner),
                       inner_training_queries=len(train), inner_validation_queries=len(control),
                       inner_vocabulary_terms=len(vocab), final_vocabulary_terms=len(fv),
                       outer_before=summary(od, score(od, reference[0], act, 0.)),
                       outer_after=summary(od, score(od, weights, act, alpha)),
                       outer_mean_nonzero_terms=float(np.mean(np.count_nonzero(act, axis=1))),
                       outer_controls_used_in_selection=False, recall_measured=False)


def query_impacts(model, query_vector):
    """Serialized query-only representation, without any label/ID input."""
    q = unit_parts(np.asarray(query_vector).reshape(1, -1))
    z = latent(q, [np.array(b) for b in model['bases']])
    act = activate(z @ np.array(model['decoder']), model['query_terms'])[0][0] * model['alpha']
    return {term: float(value) for term, value in zip(model['vocabulary'], act) if value > 0}
