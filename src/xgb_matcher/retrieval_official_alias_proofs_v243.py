"""Whole official historical aliases contained in query text, never CRM aliases.

Role-preserving, uncapped aliases enter the native lexical proof components.
The V229 head is fixed; this is a NEW score/source hypothesis, not a refit.
"""
import math
import time

import numpy as np

from .retrieval_corroborated_initial_v217 import Evidence as OriginalEvidence
from .retrieval_corroborated_initial_v217 import alias_fragments, compose as original_compose, PARTICLES, LEGAL
from .retrieval_proof_v56 import components
from .retrieval_structural_channels_v210 import identity
from .retrieval_variants_v118 import normalized


def fragment_grades(text):
    value = normalized(text)
    content = sum(w not in PARTICLES | LEGAL for w in value.split())
    # Whole-query exact/all-word OWNER path remains; this separate lexical
    # source admits ONLY an entire official alias equal to a query span.
    return {s: (1. if s == value or len(s.split()) > 1 else 1./max(1, content))
            for s in alias_fragments(value)}


def grouped(bindings):
    """Highest proof per actual owner. Never sum repeated observations/aliases."""
    result = {}
    for (scope, owner), grade in bindings.items():
        if scope not in ('parent', 'site') or not math.isfinite(grade) or not 0 < grade <= 1:
            raise ValueError('Explicit official owner role and bounded evidence required')
        groups = result.setdefault(float(grade), (set(), set()))
        groups[scope == 'site'].add(owner)
    return [(parents, sites, grade) for grade, (parents, sites) in sorted(result.items(), reverse=True)]


class Evidence(OriginalEvidence):
    def fragment_owners(self, text):
        selected = {}
        fragments = fragment_grades(text)
        for fragment, grade in fragments.items():
            self.files.guard()
            # Channel3 = WHOLE normalized official name, not token containment
            # inside an alias. Includes historical names/sigles already in134.
            for ordinal, alias in enumerate(self.lookup(3, fragment)):
                if ordinal % 1024 == 0:
                    self.files.guard()
                a = int(alias)
                selected[a] = max(selected.get(a, 0.), grade)
        ids = np.asarray(sorted(selected), dtype=np.uint32)
        bindings, observations = {}, 0
        for bucket in self.items:
            self.files.guard()
            lo, hi = bucket['name_start'], bucket['name_start']+bucket['names']
            local = ids[(ids >= lo) & (ids < hi)]-np.uint32(lo)
            if not len(local):
                continue
            arrays = {}
            try:
                specs = dict(name_fact_offsets=(np.uint64, bucket['names']+1),
                             owner_key=(np.uint64, bucket['facts']), parent_key=(np.uint32, bucket['facts']),
                             subject_code=(np.uint8, bucket['facts']), scope_code=(np.uint8, bucket['facts']))
                for key, (dtype, size) in specs.items():
                    arrays[key] = self.files.mapped(bucket['artifacts'][key+'.npy'], dtype, (size,))
                for alias in local:
                    self.files.guard()
                    start, end = map(int, arrays['name_fact_offsets'][int(alias):int(alias)+2])
                    if not 0 <= start < end <= bucket['facts']:
                        raise ValueError('Complete official alias-owner interval required')
                    grade = selected[int(alias)+lo]
                    for row in range(start, end):
                        if (row-start) % 1024 == 0:
                            self.files.guard()
                        owner, parent, subject, scope = (int(arrays[k][row]) for k in
                            ('owner_key', 'parent_key', 'subject_code', 'scope_code'))
                        if subject == 1 and scope in (1, 2) and owner == parent and 0 <= parent < 10**9:
                            key = ('parent', f'{parent:09d}')
                        elif subject == 2 and scope == 3 and 0 <= owner < 10**14 and owner//100000 == parent:
                            key = ('site', f'{owner:014d}')
                        else:
                            raise ValueError('Official owner/scope conflict, no sibling promotion')
                        bindings[key] = max(bindings.get(key, 0.), grade)
                        observations += 1
            finally:
                for array in arrays.values():
                    array._mmap.close()
                self.files.check_epochs()
        return grouped(bindings), dict(fragments=len(fragments), aliases=len(ids), owner_facts=observations,
            parent_owners=sum(k[0] == 'parent' for k in bindings),
            endpoint_owners=sum(k[0] == 'site' for k in bindings), owner_cap=None,
            whole_official_alias_exact_only=True, crm_alias_dictionary=False,
            historical_and_current_official_names=True, max_not_sum_repeated_owner_evidence=True)


def compose(backend, q, weights, terms, alpha, owner_groups, relation_groups, rings, alias_groups):
    """MAX native and WHOLE official alias proof per component, not bonuses.

    Name evidence from any period is explicit; it does NOT certify a current
    denomination. Whole aliases stay inside their own legal/site role.
    """
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(11,) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Fixed safe nonnegative native11 weights')
    tv = backend._tantivy
    def maximum(items):
        items=[p for p in items if p is not None]
        return tv.Query.disjunction_max_query(items,tie_breaker=0.) if items else None
    def all_of(*items):
        return tv.Query.boolean_query([(tv.Occur.Must,p) for p in items]) if all(p is not None for p in items) else None
    def const(p,value=0.):
        return tv.Query.const_score_query(p,float(value)) if p is not None else None
    def scoped(parents,sites,grade):
        if not math.isfinite(grade) or not 0<grade<=1:
            raise ValueError('Bounded whole-alias proof required')
        scope=maximum([tv.Query.term_set_query(backend.index.schema,f,sorted(values))
            for f,values in (('siren',parents),('siret',sites)) if values])
        return const(scope,grade)
    parts=components(backend,q)
    support=maximum([parts[1],parts[4]])
    owner=maximum([tv.Query.boost_query(all_of(const(scoped(p,s,grade)),support),float(grade))
        for p,s,grade in owner_groups if (p or s) and support is not None])
    related=maximum([tv.Query.boost_query(all_of(const(scoped(set(),s,grade)),support),float(grade))
        for s,grade in relation_groups if s and support is not None])
    # Fixed217 sparse + XY only; owner/related handled once by MAX below.
    fixed=original_compose(backend,q,np.zeros(11),terms,alpha,set(),set(),set(),rings)
    name=maximum([scoped(p,s,grade) for p,s,grade in alias_groups])
    gate=const(parts[4])
    alias_parts={0:name,2:name,6:all_of(name,gate),
                 8:all_of(name,parts[1],gate),10:all_of(name,const(parts[3]),gate)}
    scored=[fixed,*[p for p in (owner,related) if p is not None]]
    for index,w in enumerate(weights):
        if not w:continue
        unit=np.zeros(11);unit[index]=1.
        native=original_compose(backend,q,unit,[],0.,set(),set(),set(),[])
        proof=maximum([native,alias_parts.get(index)])
        if proof is not None:
            scored.append(tv.Query.boost_query(proof,float(w)))
    leaf=const(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'))
    return tv.Query.boolean_query([(tv.Occur.Must,leaf),
        (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in scored]))])


def search(backend, evidence, q, geo, weights, terms, alpha, max_seconds=20.):
    began = time.monotonic(); old_guard = evidence.files.guard
    def guard():
        old_guard()
        if time.monotonic()-began > max_seconds:
            raise TimeoutError('STOP_OFFICIAL_FRAGMENT_QUERY_COST')
    evidence.files.guard = guard
    try:
        parents, sites, nt = evidence.name_owners(q.name)
        groups, ft = evidence.fragment_owners(q.name)
        owner_groups = [(parents, sites, 1.), *groups]
        relations, relation_trace = [], []
        for p, s, grade in owner_groups:
            neighbours, rt = evidence.related(p, s)
            relations.append((neighbours, .5*grade)); relation_trace.append(rt)
        rings, gt = evidence.radial_proofs(geo)
        parsed = compose(backend, q, weights, terms, alpha, owner_groups, relations, rings, groups)
        guard(); found = []
        for score, location in backend.searcher.search(parsed, limit=100, count=False).hits:
            guard(); doc = backend.searcher.doc(location).to_dict()
            row = dict(siret=backend._first(doc, 'siret'), siren=backend._first(doc, 'siren'),
                       source_score=float(score), channel='joint_proof')
            identity(row)
            if not math.isfinite(score) or score < 0:
                raise ValueError('Finite nonnegative native score required')
            found.append(row)
        found.sort(key=lambda r: (-r['source_score'], r['siret']))
        if len(found) > 100 or len({r['siret'] for r in found}) != len(found):
            raise ValueError('Direct100 distinct leaf cap required')
        for i, row in enumerate(found, 1):
            row['retrieval_rank'] = i
        guard()
        return found, dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
            source_admitted={'joint_proof': len(found)}, fixed_reservoir=None, score_after_fixed_pool=False,
            initial_cap=100, exact_global_fused_score_top100=True, names=nt, official_fragments=ft,
            relations=relation_trace, geometry=gt, native_leaf_sites=43896818,
            full_appended_leaf_universe=False, downstream_models_changed=False,
            label_lookup=False, identifiers_as_features=False, recall_measured=False,
            production_slo_validated=False, seconds=time.monotonic()-began,
            bm25_term_frequency_score=False, frozen_oof11_sparse8_head=True,
            official_aliases_used_in_name_components=[0,2,6,8,10],
            official_any_period_name_proof_not_current_name_assertion=True,
            new_native_feature_semantics=True, actual217_fit_parity_claimed=False,
            initial_place_training='V228_OOF_TRAIN64_NATIONAL_CONTRASTS', query_time_training=False,
            source_access_hypothesis='V243_WHOLE_OFFICIAL_ALIAS_LEXICAL_PROOFS',
            tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS',
            aliases_materialized_as_site_candidate_pool=False)
    finally:
        evidence.files.guard = old_guard
