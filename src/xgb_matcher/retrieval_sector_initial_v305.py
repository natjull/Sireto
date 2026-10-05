"""Official facility relationship hypotheses before one native initial100.

Base regional286 unchanged. No materialized fixed candidate pool, pair model,
borrowed exact-name feature or physical-site equivalence claim.
"""
import math
import time
from .retrieval_regional_initial_v286 import Evidence, compose as previous_compose
from .retrieval_structural_channels_v210 import identity
from .retrieval_proof_v56 import components


def compose(backend, q, weights, terms, alpha, parents, sites, related, rings, regions, facts):
    base = previous_compose(backend, q, weights, terms, alpha, parents, sites, related, rings, regions)
    if set(facts) != {'local', 'unknown', 'nonlocal'}:
        raise ValueError('All source locality categories must be preserved')
    all_sites = set().union(*facts.values())
    if any(not isinstance(s, str) or len(s) != 14 or not s.isascii() or not s.isdigit() for s in all_sites):
        raise ValueError('Published explicit SIRET joins only')
    endpoints = {s for s in all_sites - set(sites) if s[:9] not in parents}
    weak_parents = {s[:9] for s in all_sites} - set(parents)
    if not endpoints and not weak_parents:
        return base
    tv = backend._tantivy
    def maximum(parts):
        parts = [p for p in parts if p is not None]
        return tv.Query.disjunction_max_query(parts, tie_breaker=0.) if parts else None
    def scope(field, values):
        return tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema, field, sorted(values)), 0.) if values else None
    def both(*parts):
        return tv.Query.boolean_query([(tv.Occur.Must, p) for p in parts]) if all(p is not None for p in parts) else None
    parts = components(backend, q)
    support = maximum([parts[1], parts[4]])
    # Same uncertainty grades as original official owner/succession proofs.
    # Source locality can substantiate its published installation endpoint,
    # but it is never copied as exact-name/place features to sibling sites.
    direct = maximum([both(scope('siret', endpoints), support),
        tv.Query.const_score_query(scope('siret', endpoints & facts['local']), 1.) if endpoints & facts['local'] else None])
    sibling = both(scope('siren', weak_parents), support)
    if sibling is not None:
        # Do not stack a weak sibling relation onto an original strong SITE owner.
        if sites:
            sibling = tv.Query.boolean_query([(tv.Occur.Must, sibling), (tv.Occur.MustNot, scope('siret', sites))])
        sibling = tv.Query.boost_query(sibling, .5)
    relation = maximum([direct, sibling])
    if relation is None:
        return base
    leaf = tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'), 0.)
    return tv.Query.boolean_query([(tv.Occur.Should, base), (tv.Occur.Should, both(leaf, relation))])


def search(backend, evidence, q, geo, weights, terms, alpha, regions, facilities, max_seconds=20.):
    began = time.monotonic()
    old_guard = evidence.files.guard
    def guard():
        old_guard()
        if time.monotonic() - began > max_seconds:
            raise TimeoutError('STOP_SECTOR_INITIAL_QUERY_COST')
    evidence.files.guard = guard
    try:
        parents, sites, nt = evidence.name_owners(q.name)
        related, rt = evidence.related(parents, sites)
        facts, ft = facilities.facts(q)
        rings, gt = evidence.radial_proofs(geo)
        parsed = compose(backend, q, weights, terms, alpha, parents, sites, related, rings, regions, facts)
        guard()
        found = []
        for score, location in backend.searcher.search(parsed, limit=100, count=False).hits:
            guard()
            doc = backend.searcher.doc(location).to_dict()
            row = dict(siret=backend._first(doc, 'siret'), siren=backend._first(doc, 'siren'), source_score=float(score), channel='joint_proof')
            identity(row)
            if not math.isfinite(score) or score < 0:
                raise ValueError('Finite native initial score required')
            found.append(row)
        found.sort(key=lambda r: (-r['source_score'], r['siret']))
        if len(found) > 100 or len({r['siret'] for r in found}) != len(found):
            raise ValueError('Single native unique initial100 cap required')
        for rank, row in enumerate(found, 1):
            row['retrieval_rank'] = rank
        guard()
        return found, dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
            source_admitted={'joint_proof': len(found)}, fixed_reservoir=None, score_after_fixed_pool=False,
            initial_cap=100, exact_global_fused_score_top100=True, names=nt, relations=rt, geometry=gt,
            facilities=ft, source_hypothesis='V305_OFFICIAL_DATA_ES_INSTALLATION_ENDPOINT_AND_WEAK_PARENT_RELATIONS',
            facility_endpoint_relation_grade=1., facility_parent_relation_grade=.5,
            source_coordinates_used=False, source_owner_name_used=False,
            official_physical_site_identity_certified=False, sibling_alias_identity_claimed=False,
            same_parent_not_operational_equivalence=True,
            native_leaf_sites=43896818, full_appended_leaf_universe=False, downstream_models_changed=False,
            label_lookup=False, identifiers_as_features=False, recall_measured=False, new_fit=False,
            production_slo_validated=False, seconds=time.monotonic()-began, actual217_fit_parity_claimed=False,
            aliases_materialized_as_site_candidate_pool=False,
            tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS')
    finally:
        evidence.files.guard = old_guard
