"""Post-seal all-human metrics only; never a retrieval input or label repair."""
from collections import Counter, defaultdict
import math

SEGMENTS = ('fold', 'label_kind', 'qualification_v2', 'qualification_v3', 'data_origin',
            'ground_truth_state', 'unseen_siren', 'new_site_known_siren', 'mega_base_pool')


def identity(value, width):
    if not isinstance(value, str) or len(value) != width or not value.isascii() or not value.isdigit():
        raise ValueError('Original typed human or official identity required; no coercion/repair')
    return value


def keyed(rows, scope):
    result = {}
    for row in rows:
        qid, fold = row['query_id'], row['fold']
        if qid not in scope or type(fold) is not int or fold != scope[qid] or qid in result:
            raise ValueError('Exactly one original row for each safe generated query required')
        result[qid] = row
    if set(result) != set(scope): raise ValueError('Complete all-human denominator required')
    return result


def summary(rows):
    n = len(rows)
    if not n: raise ValueError('No empty denominator or filtered population')
    hits = sum(r['exact_hit'] for r in rows); operational = sum(r['operational_hit'] for r in rows)
    z = 1.959963984540054; p = hits/n; denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return dict(queries=n, exact_hits=hits, exact_recall=p, operational_hits=operational,
        operational_recall=operational/n, wilson95_lower=max(0., center-radius),
        wilson95_upper=min(1., center+radius),
        operational_only_hits=sum(r['operational_hit'] and not r['exact_hit'] for r in rows),
        active_successor_same_site=sum(r['active_successor_same_site'] for r in rows),
        multiple_active_equivalents=sum(r['multiple_active_equivalents'] for r in rows))


def evaluate(generated, controls, labels, official_side, acceptable_side, current_present):
    """All tables must be read by a separately authorized AFTER-SEALED worker.

    Operational targets are the immutable official sidecar, not inferred from
    hits or proximity. Current catalog membership is a separate exact lookup;
    historical absence is retained for audit, never substituted for it.
    """
    if (len(controls) != 128 or [r['position'] for r in controls] != list(range(128))
            or len({r['query_id'] for r in controls}) != 128
            or any(type(r['position']) is not int or not isinstance(r['query_id'], str) or not r['query_id']
                or type(r['fold']) is not int or r['fold'] not in (2, 3, 4) for r in controls)):
        raise ValueError('Entire original128 pilot controls on folds2/3/4 required')
    scope = {r['query_id']: r['fold'] for r in controls}
    outputs = keyed(generated, scope); truths = keyed(labels, scope); official = keyed(official_side, scope)
    if (set(current_present) != set(scope) or any(type(v) is not bool for v in current_present.values())):
        raise ValueError('Explicit current namespace presence for ALL human targets required')
    acceptable = defaultdict(dict)
    for row in acceptable_side:
        qid = row['query_id']; fold = row['fold']
        if qid not in scope or type(fold) is not int or fold != scope[qid]:
            raise ValueError('Foreign/contaminated operational sidecar row')
        siret = identity(row['siret'], 14)
        truth = identity(truths[qid]['human_ground_truth_siret'], 14)
        if siret in acceptable[qid] or siret[:9] != truth[:9]:
            raise ValueError('Unique same-SIREN official operational target required')
        if type(row['exact_truth']) is not bool or row['exact_truth'] != (siret == truth):
            raise ValueError('Official exact target flag must match unchanged human truth')
        if not row['exact_truth'] and row['same_site_evidence'] is not True:
            raise ValueError('A fuzzy/nearby address cannot create operational equivalence')
        acceptable[qid][siret] = row
    if set(acceptable) != set(scope): raise ValueError('Operational sidecar coverage must be complete')
    outcomes = []
    for control in controls:
        qid = control['query_id']; output, label, side = outputs[qid], truths[qid], official[qid]
        truth = identity(label['human_ground_truth_siret'], 14)
        siren = identity(label['human_ground_truth_siren'], 9)
        if (truth[:9] != siren or side['gt_siret'] != truth or side['gt_siren'] != siren
                or type(side['official_catalog_present']) is not bool
                or truth not in acceptable[qid] or not acceptable[qid][truth]['exact_truth']
                or type(output['position']) is not int or output['position'] != control['position']):
            raise ValueError('Unchanged human target and exact original query/official side required')
        candidates = output['candidates']
        found = {identity(c['siret'], 14) for c in candidates}
        if len(found) != len(candidates) or len(found) > 100:
            raise ValueError('Absolute100 unique initial output required')
        exact = truth in found; equivalent = found & set(acceptable[qid])
        if exact and not current_present[qid]: raise ValueError('Generated identity contradicts current namespace')
        active = {s for s, r in acceptable[qid].items()
                  if r.get('administrative_state') == 'A' and r.get('same_site_evidence') is True}
        outcome = {k: label.get(k) for k in SEGMENTS}
        outcome.update(query_id=qid, position=control['position'], exact_hit=exact,
            operational_hit=bool(equivalent), current_catalog_present=current_present[qid],
            historical_official_catalog_present=side['official_catalog_present'],
            exact_rank=next((i+1 for i, c in enumerate(candidates) if c['siret'] == truth), None),
            operational_rank=next((i+1 for i, c in enumerate(candidates) if c['siret'] in acceptable[qid]), None),
            active_successor_same_site=side.get('exact_state') == 'F' and bool((equivalent-{truth}) & active),
            multiple_active_equivalents=len(active) > 1,
            loss='HIT' if exact else 'NOT_RETRIEVED_AT100' if current_present[qid] else 'CURRENT_INDEX_CATALOG_ABSENT')
        outcomes.append(outcome)
    segments = {}
    for dimension in SEGMENTS:
        groups = defaultdict(list)
        for row in outcomes:
            value = row[dimension]
            groups['UNKNOWN' if value is None or value == '' else str(value)].append(row)
        segments[dimension] = {key: summary(rows) for key, rows in sorted(groups.items())}
    primary = summary(outcomes)
    return dict(status='MEASURED_V200_EXACT_INITIAL100_REUSED128_PILOT', primary=primary,
        historical_all_human=dict(primary), segments=segments, outcomes=outcomes,
        loss_counts=dict(Counter(r['loss'] for r in outcomes)), metric_family='EXACT_HUMAN_INITIAL_RETRIEVAL_AT100',
        decision='PIVOT_GENERATION' if primary['exact_recall'] < .992 else 'STOP',
        decision_reason='PILOT_ONLY_NO_FULL_OOF_GENERALIZATION_SLO_GATE',
        operational_scope='PREEXISTING_SEALED_OFFICIAL_SAME_SITE_TARGETS_NOT_CURRENT_CATALOG_EXPANSION',
        operational_current_catalog_completeness_measured=False, fine_loss_attribution_measured=False,
        full_oof_population=False, slo_validated=False, ready_for_fold1=False,
        fold0_opened=False, fold1_opened=False, downstream_models_changed=False)
