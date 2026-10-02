"""Training-only identity/site kernel; no candidate pool or downstream scorer.

The learned metric can be written as two query vectors dotted with frozen
official document vectors plus nonnegative query-conditioned native proofs.
National indexed execution is NOT implemented/validated by this probe.
"""
import re

import numpy as np
from scipy.optimize import minimize

from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_native_v57 import fit as native_fit
from src.xgb_matcher.retrieval_proof_v56 import BASE, FEATURES, signal
from src.xgb_matcher.retrieval_joint_v46 import geo_parameters

CONTEXT = ('constant', 'short_name', 'long_name', 'number_present',
           'street_present', 'geo_available', 'geo_reliable', 'geo_weak')


def unit_parts(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] % 2 or not np.isfinite(x).all():
        raise ValueError('Structured vector drift')
    x = x.reshape(len(x), 2, -1)
    return x / np.maximum(np.linalg.norm(x, axis=2, keepdims=True), 1e-12)


def context(row, geo):
    words = re.findall(r'[A-Z0-9]+', signal(row['crm_name']))
    point, _, reliable = geo_parameters(geo)
    known = point is not None
    return np.array([1., 0 < len(words) <= 2, len(words) >= 4,
                     bool(signal(row['crm_number'])), bool(signal(row['crm_street'])),
                     known, known and reliable, known and not reliable], dtype=np.float64)


def validate(rows, scenes, q, docs, doc_ids, registry):
    if len(rows) != len(scenes) or len(rows) != len(q) or len(docs) != len(doc_ids):
        raise ValueError('Training denominator drift')
    if len(set(doc_ids)) != len(doc_ids) or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training identity')
    byid = dict(zip(doc_ids, range(len(doc_ids))))
    for row, scene in zip(rows, scenes):
        if any(row[k] != scene[k] for k in ('query_id', 'fold', 'positive_siren')):
            raise ValueError('Scene alignment drift')
        if row['fold'] not in (2, 3, 4) or row['positive_siren'] not in registry[str(row['fold'])]:
            raise ValueError('Protected fold membership drift')
        if len(row['negatives']) != len(scene['negatives']) or not row['negatives']:
            raise ValueError('Natural negative denominator drift')
        if row['positive_siret'] not in byid or not row['positive_siret'].startswith(row['positive_siren']):
            raise ValueError('Missing official positive')
        for n, s in zip(row['negatives'], scene['negatives']):
            if n['siren'] != s['siren'] or n['siret'] not in byid or not n['siret'].startswith(n['siren']):
                raise ValueError('Natural negative identity/order drift')
        f = np.array([scene['positive_features'], *[n['features'] for n in scene['negatives']]])
        if f.shape[1:] != (len(FEATURES),) or not np.isfinite(f).all():
            raise ValueError('Native/geographic proof drift')
    return byid


def basis(x, dimensions):
    """Training-only uncentered SVD; zero/unsupported directions stay absent."""
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    count = min(dimensions, int(np.count_nonzero(singular > singular[0] * 1e-10)))
    result = vt[:count].T.copy()
    # Deterministic sign convention, independent of IDs/labels.
    for col in range(count):
        if result[np.argmax(abs(result[:, col])), col] < 0:
            result[:, col] *= -1
    return result


def bases(rows, q, docs, byid, indices, forbidden):
    blocked = set(forbidden)
    doc_positions = set()
    for i in indices:
        row = rows[i]
        if row['positive_siren'] in blocked:
            raise ValueError('Protected positive in basis')
        doc_positions.add(byid[row['positive_siret']])
        doc_positions.update(byid[n['siret']] for n in row['negatives'] if n['siren'] not in blocked)
    ordered = sorted(doc_positions)
    return [(basis(q[indices, view], 8), basis(docs[ordered, view], 32)) for view in range(2)]


def design(rows, scenes, q, docs, byid, contexts, indices, forbidden, family, latent=None):
    if family not in ('HEAD19', 'JOINT_STATIC', 'JOINT_ADAPTED'):
        raise ValueError('Unknown joint kernel')
    blocked = set(forbidden); matrices = []; sizes = []; tags = []
    for i in indices:
        row, scene = rows[i], scenes[i]
        safe = [j for j, n in enumerate(row['negatives']) if n['siren'] not in blocked]
        if not safe:
            raise ValueError('No safe natural negative; do not drop query')
        p = docs[byid[row['positive_siret']]]
        n = docs[[byid[row['negatives'][j]['siret']] for j in safe]]
        proof = np.array(scene['positive_features']) - np.array([scene['negatives'][j]['features'] for j in safe])
        c = contexts[i] if family == 'JOINT_ADAPTED' else np.ones(1)
        parts = [np.einsum('c,nf->ncf', c, proof).reshape(len(safe), -1)]
        delta = p - n
        if family != 'HEAD19':
            parts.append(np.einsum('vf,nvf->nv', q[i], delta))
        if family == 'JOINT_ADAPTED':
            if latent is None:
                raise ValueError('Missing training-only bases')
            for view, (uq, ud) in enumerate(latent):
                parts.append(np.einsum('a,nb->nab', q[i, view] @ uq, delta[:, view] @ ud).reshape(len(safe), -1))
        matrices.append(np.column_stack(parts)); sizes.append(len(safe))
        tags.extend(dict(same_parent=row['negatives'][j]['siren'] == row['positive_siren'],
                         exact_address=bool(scene['negatives'][j]['features'][3]),
                         number_at_place=bool(scene['negatives'][j]['features'][5])) for j in safe)
    x = np.vstack(matrices)
    if not np.isfinite(x).all():
        raise ValueError('Joint feature finite drift')
    return x, np.array(sizes, dtype=int), tags


def head_loss(w, x, sizes, prior, penalty):
    """Equal-query log(1+sum exp(negative-positive)); all natural negatives."""
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    logits = -(x @ w)
    maxima = np.maximum(0., np.maximum.reduceat(logits, starts))
    exp_values = np.exp(logits - np.repeat(maxima, sizes))
    denominators = np.exp(-maxima) + np.add.reduceat(exp_values, starts)
    loss = np.mean(maxima + np.log(denominators))
    probabilities = exp_values / np.repeat(denominators, sizes) / len(sizes)
    delta = w - prior
    return float(loss + np.dot(penalty, delta * delta)), -x.T @ probabilities + 2 * penalty * delta


def fit_kernel(x, sizes, family, ridge, maxiter=180):
    native = len(FEATURES) * (len(CONTEXT) if family == 'JOINT_ADAPTED' else 1)
    prior = np.zeros(x.shape[1]); prior[:len(FEATURES)] = BASE
    penalty = np.full(x.shape[1], ridge); penalty[:native] = .005
    if family != 'HEAD19':
        penalty[native:native+2] = .005
    bounds = [(0., 64.)] * (native + (2 if family != 'HEAD19' else 0))
    bounds += [(-64., 64.)] * (len(prior) - len(bounds))
    result = minimize(head_loss, prior, args=(x, sizes, prior, penalty), jac=True,
                      method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=maxiter, ftol=1e-8, gtol=1e-5))
    if not np.isfinite(result.x).all() or result.fun > head_loss(prior, x, sizes, prior, penalty)[0] + 1e-8:
        raise ValueError('Joint optimization drift')
    return result.x, dict(objective_before=head_loss(prior, x, sizes, prior, penalty)[0],
                         objective_after=float(result.fun), optimizer_converged=bool(result.success),
                         optimizer_message=str(result.message), iterations=int(result.nit), parameters=len(prior))


def summarize(margins, sizes, tags):
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    hardest = np.minimum.reduceat(margins, starts)
    counts = np.add.reduceat((margins <= 0).astype(int), starts)
    result = dict(queries=len(sizes), pairs=len(margins), head_error=float(np.mean(hardest <= 0)),
                  hardest_margin=float(np.mean(hardest)), mean_outrank_count=float(np.mean(counts)),
                  mrr=float(np.mean(1. / (1. + counts))))
    for name in ('same_parent', 'exact_address', 'number_at_place'):
        mask = np.array([t[name] for t in tags], dtype=bool)
        result[name] = dict(pairs=int(mask.sum()), pair_error=float(np.mean(margins[mask] <= 0)) if mask.any() else None)
    return result


def baseline(rows, scenes, indices, held, registry):
    kept = [dict(s, positive_features=s['positive_features'][:11],
                 negatives=[dict(n, features=n['features'][:11]) for n in s['negatives']])
            for s in scenes]
    model = native_fit(kept, held, registry, .005)
    if set(model['query_ids']) != {rows[i]['query_id'] for i in indices}:
        raise ValueError('Baseline training exclusion/coverage drift')
    return np.r_[model['weights'], np.zeros(len(FEATURES)-11)], model


def select(rows, scenes, q, docs, doc_ids, contexts, registry, held, ridges, maxiter=180):
    byid = validate(rows, scenes, q, docs, doc_ids, registry)
    train, control, inner = positions(rows, held, registry)
    outer_blocked = set(registry[str(held)]); inner_blocked = outer_blocked | inner
    expanded = {k: list(v) for k, v in registry.items()}; expanded[str(held)] = sorted(inner_blocked)
    native, _ = baseline(rows, scenes, train, held, expanded)
    dx, ds, dt = design(rows, scenes, q, docs, byid, contexts, control, outer_blocked, 'HEAD19')
    candidates = [dict(family='BASELINE11', ridge=None, parameters=11, converged=True,
                       inner=summarize(dx @ native, ds, dt))]
    latent = bases(rows, q, docs, byid, train, inner_blocked)
    for family in ('HEAD19', 'JOINT_STATIC', 'JOINT_ADAPTED'):
        tx, ts, _ = design(rows, scenes, q, docs, byid, contexts, train, inner_blocked, family, latent)
        cx, cs, ct = design(rows, scenes, q, docs, byid, contexts, control, outer_blocked, family, latent)
        for ridge in (ridges if family == 'JOINT_ADAPTED' else [ridges[-1]]):
            weights, fitted = fit_kernel(tx, ts, family, ridge, maxiter)
            candidates.append(dict(family=family, ridge=ridge, parameters=len(weights),
                                   converged=fitted['optimizer_converged'], inner=summarize(cx @ weights, cs, ct), fit=fitted))
    selected = min([c for c in candidates if c['converged']], key=lambda c: (
        c['inner']['head_error'], c['inner']['mean_outrank_count'], c['parameters'], -(c['ridge'] or 0.)))
    eligible = sorted(train + control)
    before, bf = baseline(rows, scenes, eligible, held, registry)
    final_latent = bases(rows, q, docs, byid, eligible, outer_blocked) if selected['family'] == 'JOINT_ADAPTED' else None
    family = selected['family']
    if family == 'BASELINE11':
        weights, final_fit = before, bf
    else:
        fx, fs, _ = design(rows, scenes, q, docs, byid, contexts, eligible, outer_blocked, family, final_latent)
        weights, final_fit = fit_kernel(fx, fs, family, selected['ridge'], maxiter)
        if not final_fit['optimizer_converged']:
            raise ValueError('Final selected metric did not converge; no deployment')
    outer = [i for i, r in enumerate(rows) if r['fold'] == held]
    ox, os, ot = design(rows, scenes, q, docs, byid, contexts, outer, (), 'HEAD19')
    ax, ass, at = design(rows, scenes, q, docs, byid, contexts, outer, (),
                         'HEAD19' if family == 'BASELINE11' else family, final_latent)
    model = dict(family=family, weights=weights.tolist(), features=list(FEATURES), context_features=list(CONTEXT),
                 heldout_fold=held, training_folds=sorted({rows[i]['fold'] for i in eligible}),
                 query_ids=[rows[i]['query_id'] for i in eligible], training_queries=len(eligible),
                 latent=None if final_latent is None else [dict(query=u.tolist(), document=d.tolist()) for u, d in final_latent],
                 no_identifiers_as_features=True, no_candidate_generation=True)
    return model, dict(heldout_fold=held, selected=selected, candidates=candidates,
                       inner_training_queries=len(train), inner_validation_queries=len(control),
                       inner_entities=sorted(inner), final_fit=final_fit,
                       outer_before=summarize(ox @ before, os, ot), outer_after=summarize(ax @ weights, ass, at),
                       outer_controls_used_in_selection=False, recall_measured=False)


def query_kernel(model, query_vector, query_context):
    """Serving algebra only: native weights plus two transformed query vectors."""
    q = unit_parts(np.asarray(query_vector).reshape(1, -1))[0]
    weights = np.array(model['weights']); family = model['family']
    if family in ('BASELINE11', 'HEAD19'):
        return weights, np.zeros_like(q)
    count = len(CONTEXT) if family == 'JOINT_ADAPTED' else 1
    c = np.array(query_context) if count > 1 else np.ones(1)
    native = c @ weights[:count*len(FEATURES)].reshape(count, len(FEATURES))
    offset = count * len(FEATURES)
    semantic = q * weights[offset:offset+2, None]; offset += 2
    if family == 'JOINT_ADAPTED':
        for view, b in enumerate(model['latent']):
            uq, ud = np.array(b['query']), np.array(b['document'])
            length = uq.shape[1] * ud.shape[1]
            correction = weights[offset:offset+length].reshape(uq.shape[1], ud.shape[1]); offset += length
            semantic[view] += ud @ correction.T @ (q[view] @ uq)
    if offset != len(weights) or np.any(native < 0):
        raise ValueError('Query kernel serialization/parity drift')
    return native, semantic
