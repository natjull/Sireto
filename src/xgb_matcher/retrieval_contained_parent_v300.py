"""Weak official multiword-alias containment -> same-parent site hypothesis.

No name tokens are copied to siblings. All owners of all eligible whole
official aliases are retained; the new half-weight relation needs local or
address corroboration before the single native global collector100.
"""
import math
import time
import numpy as np
from .retrieval_regional_initial_v286 import Evidence as PreviousEvidence, compose as previous_compose, PARTICLES, LEGAL
from .retrieval_structural_channels_v210 import identity
from .retrieval_variants_v118 import normalized
from .retrieval_proof_v56 import components

RELATION_WEIGHT = .5


def fragments(text):
    words = normalized(text).split()
    return sorted({' '.join(words[lo:hi]) for lo in range(len(words))
        for hi in range(lo + 2, len(words) + 1)
        if len(set(words[lo:hi]) - (PARTICLES | LEGAL)) >= 2})


class Evidence(PreviousEvidence):
    def contained_parents(self, text):
        spans = fragments(text)
        selected = np.empty(0, dtype=np.uint32)
        for span in spans:
            self.files.guard()
            # Exact WHOLE official alias, not a conjunction crossing aliases.
            selected = np.union1d(selected, self.lookup(3, span)).astype(np.uint32)
        parents, facts, legal_facts, site_facts = set(), 0, 0, 0
        for bucket in self.items:
            self.files.guard()
            lo, hi = bucket['name_start'], bucket['name_start'] + bucket['names']
            ids = selected[(selected >= lo) & (selected < hi)] - np.uint32(lo)
            if not len(ids):
                continue
            arrays = {}
            try:
                specs = dict(name_fact_offsets=(np.uint64, bucket['names'] + 1),
                    owner_key=(np.uint64, bucket['facts']), parent_key=(np.uint32, bucket['facts']),
                    subject_code=(np.uint8, bucket['facts']), scope_code=(np.uint8, bucket['facts']))
                for key, (dtype, size) in specs.items():
                    arrays[key] = self.files.mapped(bucket['artifacts'][key + '.npy'], dtype, (size,))
                for alias in ids:
                    self.files.guard()
                    begin, end = map(int, arrays['name_fact_offsets'][int(alias):int(alias) + 2])
                    if not 0 <= begin < end <= bucket['facts']:
                        raise ValueError('Complete nonempty official name-owner interval required')
                    for row in range(begin, end):
                        if (row - begin) % 1024 == 0:
                            self.files.guard()
                        owner, parent, subject, scope = (int(arrays[k][row]) for k in
                            ('owner_key', 'parent_key', 'subject_code', 'scope_code'))
                        if subject == 1 and scope in (1, 2) and owner == parent and 0 <= parent < 10**9:
                            legal_facts += 1
                        elif subject == 2 and scope == 3 and 0 <= owner < 10**14 and owner // 100000 == parent:
                            site_facts += 1
                        else:
                            raise ValueError('Official contained alias owner/scope conflict')
                        parents.add(f'{parent:09d}')
                        facts += 1
            finally:
                for array in arrays.values():
                    array._mmap.close()
                self.files.check_epochs()
        return parents, dict(eligible_spans=len(spans), aliases=len(selected), owner_facts=facts,
            legal_owner_facts=legal_facts, site_owner_facts=site_facts, parents=len(parents),
            owner_cap=None, exact_whole_official_alias=True, weak_same_parent_hypothesis=True,
            sibling_alias_identity_claimed=False, singleton_alias_expansion=False)


def compose(backend, q, weights, terms, alpha, parents, sites, related, rings, regions, contained):
    base = previous_compose(backend, q, weights, terms, alpha, parents, sites, related, rings, regions)
    # Full-query legal owners already provide a stronger same-parent proof.
    # Endpoint owners still only bind their original endpoint in the base.
    weak = set(contained) - set(parents)
    if not weak:
        return base
    tv = backend._tantivy
    parts = components(backend, q)
    support = [p for p in (parts[1], parts[4]) if p is not None]
    if not support:
        return base
    support = tv.Query.disjunction_max_query(support, tie_breaker=0.)
    scope = tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema, 'siren', sorted(weak)), 0.)
    leaf = tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'), 0.)
    proof = tv.Query.boost_query(tv.Query.boolean_query([(tv.Occur.Must, p) for p in (leaf, scope, support)]), RELATION_WEIGHT)
    return tv.Query.boolean_query([(tv.Occur.Should, base), (tv.Occur.Should, proof)])


def search(backend, evidence, q, geo, weights, terms, alpha, regions, max_seconds=20.):
    began = time.monotonic()
    old_guard = evidence.files.guard
    def guard():
        old_guard()
        if time.monotonic() - began > max_seconds:
            raise TimeoutError('STOP_CONTAINED_PARENT_QUERY_COST')
    evidence.files.guard = guard
    try:
        parents, sites, nt = evidence.name_owners(q.name)
        related, rt = evidence.related(parents, sites)
        contained, ct = evidence.contained_parents(q.name)
        rings, gt = evidence.radial_proofs(geo)
        parsed = compose(backend, q, weights, terms, alpha, parents, sites, related, rings, regions, contained)
        guard()
        found = []
        for score, location in backend.searcher.search(parsed, limit=100, count=False).hits:
            guard()
            doc = backend.searcher.doc(location).to_dict()
            row = dict(siret=backend._first(doc, 'siret'), siren=backend._first(doc, 'siren'),
                source_score=float(score), channel='joint_proof')
            identity(row)
            if not math.isfinite(score) or score < 0:
                raise ValueError('Finite native joint score required')
            found.append(row)
        found.sort(key=lambda r: (-r['source_score'], r['siret']))
        if len(found) > 100 or len({r['siret'] for r in found}) != len(found):
            raise ValueError('Native direct100 distinct leaf cap required')
        for i, row in enumerate(found, 1):
            row['retrieval_rank'] = i
        guard()
        return found, dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
            source_admitted={'joint_proof': len(found)}, fixed_reservoir=None, score_after_fixed_pool=False,
            initial_cap=100, exact_global_fused_score_top100=True, names=nt, relations=rt, geometry=gt,
            contained_parent=ct, contained_relation_weight=RELATION_WEIGHT,
            strong_parent_deduplicated=len(set(contained) & set(parents)),
            source_hypothesis='V300_WEAK_MULTIPLE_WORD_WHOLE_OFFICIAL_ALIAS_PARENT_CONTAINMENT',
            native_leaf_sites=43896818, full_appended_leaf_universe=False, downstream_models_changed=False,
            label_lookup=False, identifiers_as_features=False, recall_measured=False,
            production_slo_validated=False, seconds=time.monotonic()-began,
            actual217_fit_parity_claimed=False, new_fit=False,
            aliases_materialized_as_site_candidate_pool=False,
            current_and_historical_official_aliases=True, singleton_alias_expansion=False,
            sibling_alias_identity_claimed=False, same_parent_not_operational_equivalence=True,
            tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS')
    finally:
        evidence.files.guard = old_guard
