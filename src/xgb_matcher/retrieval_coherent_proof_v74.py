"""One-observation RNE interaction bonus, TRAIN probe not retrieval recall.

Max is taken AFTER the fixed name/address/place/number interaction inside a
single official site observation. Missing RNE contributes zero, never a filter.
Existing native proofs (including XY/BAN) remain a separate comparator.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from scripts.audit_retrieval_v73_alias_counters import corrected_counts
from src.xgb_matcher.official_evidence import OfficialEvidence
from src.xgb_matcher.retrieval_initialisms_v67 import membership
from src.xgb_matcher.retrieval_joint_contrastive_v63 import head_loss
from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_observation_v72 import Observation, coverage, query, termset
from src.xgb_matcher.retrieval_owner_observation_v73 import compose
from src.xgb_matcher.retrieval_sparse_contrastive_v69 import FEATURES as NATIVE, PRIOR as NATIVE_PRIOR

FEATURES = (*NATIVE, 'rne_same_observation_name_address_place_number')
PRIOR = np.r_[NATIVE_PRIOR, 0.]
CRM_FIELDS = ('query_id', 'fold', 'crm_name', 'crm_number', 'crm_suffix',
              'crm_street', 'crm_insee', 'crm_postcode')


@dataclass(frozen=True)
class View:
    observation: Observation
    aliases: tuple[str, ...]


def views(raw_rows):
    """Only explicit LEGAL owner in the same RNE formality; no alias bags."""
    raw_rows = list(raw_rows)
    official = [OfficialEvidence.from_dict(r) for r in raw_rows]
    before = [Observation.from_row(r) for r in raw_rows]
    after, bindings, reported = compose(raw_rows)
    counts, fixed, duplicates = corrected_counts(before, after, bindings, reported)
    owners = {(o.source.value, o.source_record_id, o.siren): o for o in official
              if o.subject_kind.value == 'SIREN'}
    bound = {(b['site_evidence_id'], b['site_source_record_id']): b for b in fixed}
    output = defaultdict(list)
    for raw, observation in zip(official, after):
        if observation.subject_kind != 'SIRET':
            continue
        aliases = [n.normalized_value for n in raw.names]
        binding = bound.get((observation.evidence_id, observation.source_record_id))
        if binding:
            owner = owners[(observation.source, binding['owner_source_record_id'], observation.siren)]
            aliases += [n.normalized_value for n in owner.names if n.kind.value == 'LEGAL']
        aliases = tuple(dict.fromkeys(aliases))
        if set(termset(a) for a in aliases) != set(observation.names):
            raise ValueError('Raw single-alias / composed term support drift')
        output[observation.siret].append(View(observation, aliases))
    return dict(output), dict(counts=counts, duplicate_entries_collapsed=duplicates,
                             explicit_bindings=len(fixed))


def coherent(row, observations):
    # The whole teacher row is intentionally not passed to query(): no labels
    # or candidate IDs in the feature kernel, only this explicit CRM whitelist.
    q = query({k: row[k] for k in CRM_FIELDS if k in row})
    best = 0.
    for view in observations:
        o = view.observation
        if o.source != 'RNE' or o.subject_kind != 'SIRET':
            continue
        for alias in view.aliases:
            name = max(coverage(q['name'], termset(alias)), membership(row.get('crm_name', ''), [alias]))
            for address in o.addresses:
                addr = coverage(q['address'], address.terms)
                place = bool((q['insee'] and q['insee'] == address.insee) or
                             (q['postcode'] and q['postcode'] == address.postcode))
                number = bool(q['number'] and q['number'] == address.number and place)
                best = max(best, name * addr * (1. + place + number) / 3.)
    return float(best)


def design(rows, features, indices, blocked=()):
    forbidden = set(blocked); xs = []; sizes = []; query_ids = []
    for i in indices:
        row, values = rows[i], features[i]
        if values.shape != (1 + len(row['negatives']), len(FEATURES)) or not np.isfinite(values).all():
            raise ValueError('Whole natural scene feature denominator drift')
        safe = [j + 1 for j, n in enumerate(row['negatives']) if n['siren'] not in forbidden]
        if not safe:
            raise ValueError('No safe negatives; never drop query')
        xs.append(values[0] - values[safe]); sizes.append(len(safe)); query_ids.append(row['query_id'])
    return np.vstack(xs), np.asarray(sizes, dtype=int), query_ids


def fit(x, sizes, family):
    width = {'NATIVE23': len(NATIVE), 'NATIVE_PLUS_COHERENT24': len(FEATURES)}[family]
    prior = PRIOR[:width]; penalty = np.full(width, .005); selected = x[:, :width]
    before = head_loss(prior, selected, sizes, prior, penalty)[0]
    result = minimize(head_loss, prior, args=(selected, sizes, prior, penalty), jac=True,
        method='L-BFGS-B', bounds=[(0., 64.)] * width,
        options=dict(maxiter=180, ftol=1e-8, gtol=1e-5))
    if not np.isfinite(result.x).all() or result.fun > before + 1e-8:
        raise ValueError('Coherent proof optimization drift')
    return np.r_[result.x, np.zeros(len(FEATURES)-width)], dict(
        converged=bool(result.success), iterations=int(result.nit), message=str(result.message),
        parameters=width, objective_before=before, objective_after=float(result.fun))


def control(x, sizes, weights, query_ids):
    margins = x @ weights; starts = np.r_[0, np.cumsum(sizes)[:-1]]
    counts = np.add.reduceat((margins <= 0).astype(int), starts)
    worst = np.minimum.reduceat(margins, starts)
    return dict(queries=len(sizes), pairs=len(margins),
        queries_with_competitor_ahead_or_tied=int(np.count_nonzero(counts)),
        head_error=float(np.mean(counts > 0)), mean_outrank_count=float(counts.mean()),
        mrr=float(np.mean(1. / (1. + counts))),
        per_query=[dict(query_id=q, competitor_count=int(c), hardest_margin=float(m))
                   for q, c, m in zip(query_ids, counts, worst)])


def select(rows, features, registry, held):
    train, inner_control, inner = positions(rows, held, registry)
    outer_blocked = set(registry[str(held)]); blocked = outer_blocked | inner
    tx, ts, tq = design(rows, features, train, blocked)
    cx, cs, cq = design(rows, features, inner_control, outer_blocked)
    candidates = []
    for family in ('NATIVE23', 'NATIVE_PLUS_COHERENT24'):
        w, report = fit(tx, ts, family)
        candidates.append(dict(family=family, fit=report, inner=control(cx, cs, w, cq)))
    eligible_candidates = [c for c in candidates if c['fit']['converged']]
    if not eligible_candidates:
        raise ValueError('No converged inner candidate')
    selected = min(eligible_candidates, key=lambda c: (
        c['inner']['head_error'], c['inner']['mean_outrank_count'], c['fit']['parameters']))
    eligible = sorted(train + inner_control)
    fx, fs, fq = design(rows, features, eligible, outer_blocked)
    baseline, bf = fit(fx, fs, 'NATIVE23')
    weights, final = fit(fx, fs, selected['family'])
    if not bf['converged'] or not final['converged']:
        raise ValueError('Final selected/comparator fit did not converge')
    # Outer features are touched only AFTER all family selection and refits.
    outer = [i for i, r in enumerate(rows) if r['fold'] == held]
    ox, os, oq = design(rows, features, outer)
    before, after = control(ox, os, baseline, oq), control(ox, os, weights, oq)
    paired = dict(improved=sum(b['competitor_count'] > a['competitor_count']
                              for b, a in zip(before['per_query'], after['per_query'])),
                  worsened=sum(b['competitor_count'] < a['competitor_count']
                              for b, a in zip(before['per_query'], after['per_query'])))
    return dict(family=selected['family'], features=list(FEATURES), weights=weights.tolist(),
        query_ids=fq, training_queries=len(fq), heldout_fold=held, identifiers_as_features=False), dict(
        selected=selected['family'], candidates=candidates, baseline_refit=bf, selected_refit=final,
        inner_training_queries=len(train), inner_control_queries=len(inner_control),
        inner_entities=sorted(inner), outer_before=before, outer_after=after,
        paired_controls=paired, outer_controls_used_in_selection=False, recall_measured=False)
