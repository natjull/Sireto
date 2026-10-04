"""Post-seal diagnostics ONLY. Never an index, alias list or serving feature."""
from collections import Counter
import math

from .retrieval_variants_v118 import normalized


def nominal_evidence(name, facts):
    words = set(normalized(name).split())
    eligible = [f for f in facts if f['lexical_eligible']]
    shared = [sorted(words & set(normalized(f['name_text']).split())) for f in eligible]
    return dict(query_words=sorted(words), official_names=[f['name_text'] for f in eligible],
                shared_words=sorted({w for part in shared for w in part}),
                literal_name_overlap=any(shared),
                whole_query_in_one_official_alias=bool(words) and any(
                    words <= set(normalized(f['name_text']).split()) for f in eligible),
                scope='SEALED_V103_RAW_OFFICIAL_SITE_AND_PARENT_FACTS_DIAGNOSTIC_ONLY')


def geometry(geo, xy):
    if not geo.get('spatial_crs_compatible', False):
        return dict(compatible=False, distance_m=None, inside_current_radius=None)
    sigma = geo['ban_uncertainty_m']
    if not isinstance(sigma, (float, int)) or not math.isfinite(sigma) or sigma < 5:
        raise ValueError('Actual compatible BAN uncertainty required')
    radius = max(300., 3.*sigma) if geo['ban_declared_weak'] else 300.
    if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
        return dict(compatible=True, distance_m=None, inside_current_radius=None, radius_m=radius)
    point = [geo['ban_x_l93'], geo['ban_y_l93']]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
        raise ValueError('Actual compatible BAN coordinates required')
    distance = math.hypot(xy[0]-point[0], xy[1]-point[1])
    return dict(compatible=True, reliable=not geo['ban_declared_weak'],
                distance_m=distance, radius_m=radius, inside_current_radius=distance <= radius+1e-6,
                operational_equivalence_inferred=False)


def summarize(rows):
    if len(rows) != 21 or len({r['query_id'] for r in rows}) != 21:
        raise ValueError('ALL21 original misses required; no selection within them')
    return dict(misses_audited=21, categories_nonexclusive=dict(
        no_literal_official_name_overlap=sum(not r['names']['literal_name_overlap'] for r in rows),
        partial_but_not_whole_official_alias=sum(r['names']['literal_name_overlap'] and not
            r['names']['whole_query_in_one_official_alias'] for r in rows),
        whole_official_alias_exists=sum(r['names']['whole_query_in_one_official_alias'] for r in rows),
        ban_incompatible=sum(not r['geometry']['compatible'] for r in rows),
        gt_inside_current_radius=sum(r['geometry']['inside_current_radius'] is True for r in rows),
        gt_outside_current_radius=sum(r['geometry']['inside_current_radius'] is False for r in rows),
        text_alone_top100_contains_gt=sum(r['text_counterfactual']['rank'] is not None for r in rows),
        operational_only=sum(r['original_operational_hit'] for r in rows)),
        scope='LABEL_SELECTED_LOSS_DIAGNOSTIC_NOT_NEW_RECALL_OR_UNBIASED_ABLATION',
        rows=rows, model_fit=False, outputs_promoted_to_features=False,
        generation_modified=False, ready_for_fold1=False, recall_measured=False,
        fold0_opened=False, fold1_opened=False, downstream_models_changed=False)
