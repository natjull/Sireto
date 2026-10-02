"""Continuous <=16-term thresholded sparse kernel, unlike hard Top16 V69.

The (k+1)th log activation is subtracted before truncation at zero. Swapping
two terms at the support boundary no longer jumps document scores. Both
threshold value and its derivative participate in the contrastive fit.
This is a custom query decoder, not SPLADE/entmax or national validation.
"""
import numpy as np
from scipy.optimize import minimize

from src.xgb_matcher import retrieval_sparse_contrastive_v69 as v69
from src.xgb_matcher.retrieval_joint_contrastive_v63 import head_loss
from src.xgb_matcher.retrieval_nested_v53 import positions


def activation(raw, topk):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or not np.isfinite(raw).all() or topk < 1:
        raise ValueError('Continuous sparse shape drift')
    log = np.log1p(np.maximum(raw, 0.))
    if raw.shape[1] > topk:
        index = np.argsort(-log, axis=1, kind='stable')[:, topk]
        threshold = log[np.arange(len(raw)), index]
    else:
        index = np.zeros(len(raw), dtype=int); threshold = np.zeros(len(raw))
    values = np.maximum(log - threshold[:, None], 0.)
    return values, (values > 0, index, threshold, raw)


def backward(cache, upstream):
    active, index, threshold, raw = cache
    grad = np.asarray(upstream)*active
    # d tau / d log_{boundary}=1; at tau=0 this is a constant-zero branch.
    boundary_gradient = -grad.sum(axis=1)
    rows = np.flatnonzero(threshold > 0.)
    grad[rows, index[rows]] += boundary_gradient[rows]
    return grad*np.where(raw > 0., 1./(1.+np.maximum(raw, 0.)), 0.)


def objective(theta, z, data, initial, ridge, flops, topk):
    native = theta[:len(v69.FEATURES)]; decoder = theta[len(v69.FEATURES):].reshape(initial.shape)
    act, cache = activation(z @ decoder, topk)
    margins = v69.score(data, native, act)
    sizes = data['sizes']; starts = np.r_[0, np.cumsum(sizes)[:-1]]
    logits = -margins; maxima = np.maximum(0., np.maximum.reduceat(logits, starts))
    e = np.exp(logits - np.repeat(maxima, sizes)); denom = np.exp(-maxima)+np.add.reduceat(e, starts)
    probabilities = e/np.repeat(denom, sizes)/len(sizes)
    loss = np.mean(maxima+np.log(denom)); coo = data['delta']; owner = data['owner'][coo.row]
    width = initial.shape[1]
    grad_act = np.bincount(owner*width+coo.col, weights=-coo.data*probabilities[coo.row],
                           minlength=len(z)*width).reshape(len(z), width)
    mean = act.mean(axis=0); grad_act += 2*flops*mean[None, :]/len(z)
    diff = decoder-initial; penalty = ridge/initial.shape[0]; native_diff = native-v69.PRIOR
    loss += .005*np.sum(native_diff**2)+penalty*np.sum(diff**2)+flops*np.sum(mean**2)
    return float(loss), np.r_[-data['native'].T @ probabilities+.01*native_diff,
        (z.T @ backward(cache, grad_act)+2*penalty*diff).ravel()]


def fit(z, data, initial, family, ridge, flops=.1, topk=16, maxiter=180):
    if family not in ('NATIVE23', 'RIDGE_SPARSE', 'CONTRASTIVE_SPARSE'):
        raise ValueError('Unknown continuous sparse family')
    if family == 'CONTRASTIVE_SPARSE':
        prior = np.r_[v69.PRIOR, initial.ravel()]
        fun = lambda w: objective(w, z, data, initial, ridge, flops, topk)
        bounds = [(0., 64.)]*len(v69.FEATURES)+[(-16., 16.)]*initial.size
    else:
        act = activation(z @ initial, topk)[0]
        extra = v69.score(data, np.zeros(len(v69.FEATURES)), act)
        x = data['native'] if family == 'NATIVE23' else np.column_stack([data['native'], extra])
        prior = v69.PRIOR.copy() if family == 'NATIVE23' else np.r_[v69.PRIOR, 1.]
        fun = lambda w: head_loss(w, x, data['sizes'], prior, np.full(len(prior), .005))
        bounds = [(0., 64.)]*len(prior)
    before = fun(prior)[0]
    result = minimize(fun, prior, method='L-BFGS-B', jac=True, bounds=bounds,
                      options=dict(maxiter=maxiter, ftol=1e-8, gtol=1e-5))
    if not np.isfinite(result.x).all() or result.fun > before+1e-8:
        raise ValueError('Continuous sparse optimization drift')
    decoder = result.x[len(v69.FEATURES):].reshape(initial.shape) if family == 'CONTRASTIVE_SPARSE' else initial
    alpha = 0. if family == 'NATIVE23' else (float(result.x[-1]) if family == 'RIDGE_SPARSE' else 1.)
    return (result.x[:len(v69.FEATURES)], decoder, alpha), dict(
        converged=bool(result.success), message=str(result.message), iterations=int(result.nit),
        parameters=len(result.x), objective_before=before, objective_after=float(result.fun))


def select(q, rows, scenes, raw_documents, aliases, registry, held, config):
    train, control, inner = positions(rows, held, registry)
    outer_blocked = set(registry[str(held)]); blocked = outer_blocked | inner
    documents = {sid:v69.impacts(d) for sid, d in raw_documents.items()}
    def setup(indices, excluded):
        vocab, bases, initial = v69.recipe(q, rows, documents, indices, excluded,
            config['vocabulary_cap'], config['latent_per_field'])
        z = v69.latent(q, bases)
        return vocab, bases, initial, z, v69.design(rows, scenes, documents, aliases, indices, vocab, excluded)
    vocab, bases, initial, z, training = setup(train, blocked)
    validation = v69.design(rows, scenes, documents, aliases, control, vocab, outer_blocked)
    candidates = []
    for family in ('NATIVE23', 'RIDGE_SPARSE', 'CONTRASTIVE_SPARSE'):
        for ridge in (config['ridges'] if family == 'CONTRASTIVE_SPARSE' else [config['ridges'][-1]]):
            (w, decoder, alpha), report = fit(z[train], training, initial, family, ridge,
                config['flops'], config['query_terms'], config['maxiter'])
            act = activation(z[control] @ decoder, config['query_terms'])[0]
            candidates.append(dict(family=family, ridge=ridge, parameters=report['parameters'],
                converged=report['converged'], fit=report,
                inner=v69.summary(validation, v69.score(validation, w, act, alpha))))
    viable = [c for c in candidates if c['converged']]
    if not viable:
        raise ValueError('No converged continuous sparse candidate')
    chosen = min(viable, key=lambda c:(c['inner']['head_error'], c['inner']['mean_outrank_count'],
        c['parameters'], -c['ridge']))
    eligible = sorted(train+control); fv, fb, fi, fz, fd = setup(eligible, outer_blocked)
    fitted, report = fit(fz[eligible], fd, fi, chosen['family'], chosen['ridge'],
        config['flops'], config['query_terms'], config['maxiter'])
    if not report['converged']:
        raise ValueError('Selected continuous refit not converged: '+str(dict(selected=chosen, fit=report)))
    reference, reference_fit = fit(fz[eligible], fd, fi, 'NATIVE23', config['ridges'][-1],
        config['flops'], config['query_terms'], config['maxiter'])
    if not reference_fit['converged']:
        raise ValueError('Native reference not converged: '+str(reference_fit))
    outer = [i for i, r in enumerate(rows) if r['fold'] == held]
    od = v69.design(rows, scenes, documents, aliases, outer, fv)
    w, decoder, alpha = fitted; act = activation(fz[outer] @ decoder, config['query_terms'])[0]
    model = dict(family=chosen['family'], activation='SUBTRACT_KPLUS1_LOG_RELU', heldout_fold=held,
        weights=w.tolist(), alpha=alpha, decoder=decoder.tolist(), bases=[b.tolist() for b in fb],
        vocabulary=fv, features=list(v69.FEATURES), query_terms=config['query_terms'],
        query_ids=[rows[i]['query_id'] for i in eligible],
        training_folds=sorted({rows[i]['fold'] for i in eligible}), identifiers_as_features=False)
    return model, dict(heldout_fold=held, selected=chosen, candidates=candidates, final_fit=report,
        reference_fit=reference_fit, inner_entities=sorted(inner), inner_training_queries=len(train),
        inner_validation_queries=len(control), inner_vocabulary_terms=len(vocab), final_vocabulary_terms=len(fv),
        outer_before=v69.summary(od, v69.score(od, reference[0], act, 0.)),
        outer_after=v69.summary(od, v69.score(od, w, act, alpha)),
        outer_mean_nonzero_terms=float(np.mean(np.count_nonzero(act, axis=1))),
        outer_controls_used_in_selection=False, recall_measured=False)


def query_impacts(model, query_vector):
    q = v69.unit_parts(np.asarray(query_vector).reshape(1, -1))
    z = v69.latent(q, [np.array(b) for b in model['bases']])
    act = activation(z @ np.array(model['decoder']), model['query_terms'])[0][0]*model['alpha']
    return {t:float(v) for t, v in zip(model['vocabulary'], act) if v > 0}
