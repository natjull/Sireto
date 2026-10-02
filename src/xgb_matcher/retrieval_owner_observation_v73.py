"""Explicit legal-owner proof within ONE RNE formality, no fuzzy name graph.

This binds legal parent names to an explicitly listed site of the SAME
official record. It does not assert a common validity period for closed sites
or transfer identity across BODACC asset transfers/successions/siblings.
"""
from collections import Counter
from dataclasses import replace

from src.xgb_matcher.retrieval_observation_v72 import Observation, termset


def compose(rows):
    rows = list(rows); observations = [Observation.from_row(r) for r in rows]
    owners = {}; counters = Counter()
    for row, o in zip(rows, observations):
        if o.source != 'RNE' or o.subject_kind != 'SIREN':
            continue
        key = (o.source, o.source_record_id, o.siren)
        if key in owners:
            raise ValueError('Ambiguous duplicate legal owner observation; no arbitrary choice')
        legal = tuple(termset(n['normalized_value']) for n in row.get('names', []) if n['kind'] == 'LEGAL')
        owners[key] = (o, tuple(n for n in legal if n))
    output = []; bindings = []
    for o in observations:
        if o.subject_kind != 'SIRET':
            output.append(o); continue
        counters['site_observations'] += 1
        counters['sites_named_before'] += bool(o.names)
        root, separator, suffix = o.source_record_id.rpartition(':site:')
        owner = owners.get((o.source, root, o.siren)) if separator and suffix == o.siret and root else None
        if o.source != 'RNE' or owner is None:
            counters['site_without_explicit_same_record_owner'] += 1
            output.append(o); counters['sites_named_after'] += bool(o.names); continue
        parent, legal = owner
        if parent.source_record_id+':site:'+o.siret != o.source_record_id:
            raise ValueError('Explicit RNE formality link drift')
        counters['sites_with_explicit_owner_binding'] += 1
        counters['bound_closed_sites'] += bool(not o.is_current)
        counters['bound_unknown_parent_valid_from'] += not bool(parent.valid_from)
        counters['bound_unknown_site_valid_from'] += not bool(o.valid_from)
        names = tuple(dict.fromkeys((*o.names, *legal)))
        added = len(names)-len(o.names); counters['added_legal_aliases'] += added
        counters['sites_receiving_legal_alias'] += bool(added)
        composed = replace(o, names=names)
        output.append(composed); counters['sites_named_after'] += bool(names)
        bindings.append(dict(site_evidence_id=o.evidence_id, owner_evidence_id=parent.evidence_id,
            site_source_record_id=o.source_record_id, owner_source_record_id=parent.source_record_id,
            source=o.source, siren=o.siren, siret=o.siret, legal_aliases_added=added,
            parent_valid_from=parent.valid_from, parent_valid_to=parent.valid_to,
            site_valid_from=o.valid_from, site_valid_to=o.valid_to,
            evidence_kind='LEGAL_OWNER_SAME_FORMALITY_NOT_TEMPORAL_EQUIVALENCE'))
    return output, bindings, dict(counters)
