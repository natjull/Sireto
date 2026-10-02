"""V70's SAME objective/bounds, with a fixed ridge-based optimizer scaling.

Only optimizer coordinates change, not the query kernel, activation,
regularization, training data or evaluation. Scaling uses ridge/dimension
only; no labels/control metrics. Historic V70 source and evidence stay intact.
"""
import numpy as np
from scipy.optimize import minimize

from src.xgb_matcher import retrieval_sparse_contrastive_v69 as v69
from src.xgb_matcher import retrieval_continuous_sparse_v70 as v70
from src.xgb_matcher.retrieval_nested_v53 import positions


def coordinates(initial, ridge):
    if ridge <= 0 or initial.ndim != 2 or not np.isfinite(initial).all():
        raise ValueError('Invalid optimizer scaling')
    origin = np.r_[v69.PRIOR, initial.ravel()]
    scale = np.r_[np.ones(len(v69.FEATURES)),
                  np.full(initial.size, max(1., np.sqrt(initial.shape[0]/(2*ridge))))]
    lower = np.r_[np.zeros(len(v69.FEATURES)), np.full(initial.size, -16.)]
    upper = np.r_[np.full(len(v69.FEATURES), 64.), np.full(initial.size, 16.)]
    return origin, scale, (lower-origin)/scale, (upper-origin)/scale


def scaled_objective(eta, origin, scale, z, data, initial, ridge, flops, topk):
    value, physical_gradient = v70.objective(origin+scale*eta, z, data, initial, ridge, flops, topk)
    return value, physical_gradient*scale


def fit(z, data, initial, family, ridge, flops=.1, topk=16, maxiter=180):
    if family != 'CONTRASTIVE_SPARSE':
        # Native/ridge comparators are exactly the frozen V70 solver.
        return v70.fit(z, data, initial, family, ridge, flops, topk, maxiter)
    origin, scale, lower, upper = coordinates(initial, ridge)
    fun = lambda eta: scaled_objective(eta, origin, scale, z, data, initial, ridge, flops, topk)
    zero = np.zeros(len(origin)); before = fun(zero)[0]
    result = minimize(fun, zero, method='L-BFGS-B', jac=True, bounds=list(zip(lower, upper)),
                      options=dict(maxiter=maxiter, ftol=1e-8, gtol=1e-5))
    theta = origin+scale*result.x
    loss, gradient = v70.objective(theta, z, data, initial, ridge, flops, topk)
    physical_lower = origin+scale*lower; physical_upper = origin+scale*upper
    if not np.isfinite(theta).all() or loss > before+1e-8 or np.any(theta < physical_lower-1e-10) or np.any(theta > physical_upper+1e-10):
        raise ValueError('Conditioned optimizer physical objective/bounds drift')
    projected = theta-np.clip(theta-gradient, physical_lower, physical_upper)
    return (theta[:len(v69.FEATURES)], theta[len(v69.FEATURES):].reshape(initial.shape), 1.), dict(
        converged=bool(result.success), message=str(result.message), iterations=int(result.nit),
        evaluations=int(result.nfev), parameters=len(theta), objective_before=before, objective_after=float(loss),
        physical_projected_gradient_inf=float(np.max(abs(projected))),
        decoder_coordinate_scale=float(scale[-1]), native_coordinate_scale=1.,
        same_physical_objective_and_bounds=True, ftol=1e-8, gtol=1e-5)


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
            act = v70.activation(z[control] @ decoder, config['query_terms'])[0]
            candidates.append(dict(family=family, ridge=ridge, parameters=report['parameters'],
                converged=report['converged'], fit=report,
                inner=v69.summary(validation, v69.score(validation, w, act, alpha))))
    viable = [c for c in candidates if c['converged']]
    if not viable:
        raise ValueError('No converged conditioned candidate')
    chosen = min(viable, key=lambda c:(c['inner']['head_error'], c['inner']['mean_outrank_count'],
        c['parameters'], -c['ridge']))
    eligible = sorted(train+control); fv, fb, fi, fz, fd = setup(eligible, outer_blocked)
    fitted, report = fit(fz[eligible], fd, fi, chosen['family'], chosen['ridge'],
        config['flops'], config['query_terms'], config['maxiter'])
    if not report['converged']:
        raise ValueError('Selected conditioned refit not converged: '+str(dict(selected=chosen, fit=report)))
    reference, reference_fit = fit(fz[eligible], fd, fi, 'NATIVE23', config['ridges'][-1],
        config['flops'], config['query_terms'], config['maxiter'])
    if not reference_fit['converged']:
        raise ValueError('Native reference not converged: '+str(reference_fit))
    outer = [i for i, r in enumerate(rows) if r['fold'] == held]
    od = v69.design(rows, scenes, documents, aliases, outer, fv)
    w, decoder, alpha = fitted; act = v70.activation(fz[outer] @ decoder, config['query_terms'])[0]
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


query_impacts = v70.query_impacts
