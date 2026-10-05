"""One native global top100 over corroborated bounded proofs, not four quotas.

Frozen OOF11/sparse query head; no new fit. Official aliases contained in the
CRM may substantiate name evidence without requiring the entire CRM as alias.
Spatial rings are SOFT proof predicates: outside/missing XY stays reachable.
"""
import math
import json
from pathlib import Path
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from .retrieval_proof_v56 import components
from .retrieval_sparse_query_v59 import compiled_component
from .retrieval_structural_channels_v210 import OfficialEvidence, identity
from .retrieval_joint_v46 import geo_parameters
from .retrieval_variants_v118 import normalized
from .retrieval_nominal_abbreviation_v256 import component as abbreviation_component
from .retrieval_public_role_v280 import ROLES,scores as role_scores
from scripts import retrieval_v280_public_role_training as trainer
from .retrieval_postscore_audit_v187 import check_epochs

PARTICLES = frozenset('A AU AUX AVEC D DE DES DU EN ET L LA LE LES OU POUR SUR UN UNE'.split())
LEGAL = frozenset('SA SAS SASU SARL EURL SCI SNC SOCIETE ETS STE'.split())
EXACT_FIELDS = ('legal_names_exact', 'trade_names_exact', 'site_names_exact')
RADII = (25., 100., 300., 500., 1000., 2000.)


def alias_fragments(name):
    """ALL contiguous spans, not a CRM alias dictionary or a top-k span list.

    A multiword span needs two nonparticle/nonlegal words. Single long tokens
    remain eligible for true one-word names; short acronyms are not broadened
    by this exact-containment predicate. Original full alias is always tried.
    """
    text = normalized(name); words = text.split(); result = {text} if text else set()
    for lo in range(len(words)):
        for hi in range(lo+1, len(words)+1):
            span = words[lo:hi]; content = [w for w in span if w not in PARTICLES | LEGAL]
            if len(content) >= 2 or (len(span) == 1 and content and len(content[0]) >= 4):
                result.add(' '.join(span))
    return sorted(result)


class Evidence(OfficialEvidence):
    def radial_proofs(self, geo, chunk=131072):
        point, sigma, reliable = geo_parameters(geo)
        if point is None:
            return [], dict(compatible=False, sites_scanned=0, hard_geographic_filter=False)
        radii = sorted(set((*RADII, max(RADII[-1], 3.*sigma))))
        xy_path = self.files.file(self.xy_ref)
        keys = self.files.mapped(self.keys_ref, np.uint64, (self.sites,))
        groups = [set() for _ in radii]
        try:
            for lo in range(0, self.sites, chunk):
                self.files.guard()
                a = np.load(xy_path, mmap_mode='r', allow_pickle=False)
                try:
                    if a.dtype != np.float64 or a.shape != (self.sites, 2) or a.flags.writeable:
                        raise ValueError('Whole readonly native Lambert93 coordinates required')
                    xy = a[lo:min(lo+chunk, self.sites)].copy()
                finally: a._mmap.close()
                d2 = np.sum((xy-point)**2, axis=1)
                valid = np.isfinite(xy).all(axis=1) & (d2 <= radii[-1]**2+1e-6)
                local = np.flatnonzero(valid)
                bands = np.searchsorted(np.asarray(radii)**2+1e-6, d2[local])
                values = keys[local+lo]
                for i in np.unique(bands):
                    groups[int(i)].update(f'{int(s):014d}' for s in values[bands == i])
            factor = 1. if reliable else .5
            proofs = [(sites, factor/(1.+radius/500.)) for sites, radius in zip(groups, radii) if sites]
            return proofs, dict(compatible=True, reliable=bool(reliable), radii_m=radii,
                band_sites=[len(s) for s in groups], sites_scanned=self.sites,
                metric='SOFT_STEPWISE_500M_PROXIMITY_WITH_WEAK_BAN_HALF_CONFIDENCE',
                candidate_or_nearest_k_cut=None, hard_geographic_filter=False,
                outside_radius_reachable_in_global_native_query=True,
                appended_coordinates_unknown=True)
        finally:
            keys._mmap.close(); self.files.check_epochs()


def public_component(backend,parts,parents,grades):
    if set(parents)!=set(ROLES):raise ValueError('Whole official municipal/EPCI scope required')
    values=np.asarray(grades,dtype=float)
    if values.shape!=(2,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Bounded query-only two-role relevance required')
    if parts[4] is None:return None
    tv=backend._tantivy;proofs=[]
    for role,grade in zip(ROLES,values):
        ids=parents[role]
        if any(not isinstance(p,str) or len(p)!=9 or not p.isascii() or not p.isdigit() for p in ids):
            raise ValueError('Official IDs only categorical joins')
        if ids and grade>0:
            predicate=tv.Query.boolean_query([
                (tv.Occur.Must,tv.Query.term_set_query(backend.index.schema,'siren',sorted(ids))),
                (tv.Occur.Must,tv.Query.const_score_query(parts[4],0.))])
            proofs.append(tv.Query.const_score_query(predicate,float(grade)))
    return tv.Query.disjunction_max_query(proofs,tie_breaker=0.) if proofs else None


class RoleModels:
    def __init__(self,files,reference,old,queries,heads):
        root=files.file(reference).parent;m,r,models=trainer.sealed(root)
        for path in ('launch.json','phase/manifest.json','phase/supervision.json','phase/preflight.json'):
            files.file(trainer.record(root/path))
        files.file(m['models']);files.file(m['report'])
        for ref in json.loads(files.file(m['launch']).read_text())['implementation'].values():files.file(ref)
        self.training_epochs=r['source_epochs']
        provenance=r['provenance'];bundle=json.loads(files.file(old['bundle']).read_text())
        encoded=json.loads(files.file(old['expansions']).read_text())
        if (old['truth_opened'] is not False or encoded['truth_opened'] is not False
                or bundle['encoder_prefix']!='nom'
                or bundle['model_refs']!=provenance['model_refs'] or bundle['model_path']!=provenance['model_path']
                or bundle['max_length']!=provenance['max_length']
                or [(e['query_id'],e['fold']) for e in encoded['rows']]!=[(q['query_id'],q['fold']) for q in queries]):
            raise ValueError('Same original complete query-only NOM encoder required')
        for ref in bundle['model_refs'].values():files.file(ref)
        vectors=np.load(files.file(encoded['vectors']),allow_pickle=False)
        if vectors.shape!=(128,384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-6):
            raise ValueError('Whole query-only128 normalized NOM vectors required')
        self.grades={}
        for f,model in models.items():
            if model['query_ids']!=heads[f]['query_ids'] or set(model['query_ids'])&{q['query_id'] for q in queries}:
                raise ValueError('Original OOF safe heads disjoint from serving queries required')
            values=role_scores(vectors,np.load(files.file(model['parameter']),allow_pickle=False))
            for i,q in enumerate(queries):
                if q['fold']==int(f):self.grades[q['query_id']]=values[i]
        if len(self.grades)!=128:raise ValueError('Every original query role required')
        self.files=files;self.close()
    def close(self):
        # Statistical source authority only; NEVER open supervised TRAIN scenes.
        check_epochs(self.training_epochs);self.files.check_epochs()


def compose(backend, q, weights, terms, alpha, parents, sites, related, rings, role_parents, role_grades):
    """Native postings score global leaves BEFORE collector100. No pair reranker."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (11,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Frozen nonnegative OOF11 weights required')
    if not math.isfinite(alpha) or alpha < 0: raise ValueError('Frozen sparse coefficient required')
    tv = backend._tantivy
    def maximum(parts):
        parts = [p for p in parts if p is not None]
        return tv.Query.disjunction_max_query(parts, tie_breaker=0.) if parts else None
    def all_of(*parts):
        return tv.Query.boolean_query([(tv.Occur.Must, p) for p in parts]) if all(p is not None for p in parts) else None
    def const(p, value):
        return tv.Query.const_score_query(p, float(value)) if p is not None else None
    def scope(parents, sites):
        return maximum([tv.Query.term_set_query(backend.index.schema, f, sorted(values))
            for f, values in (('siren', parents), ('siret', sites)) if values])
    parts = components(backend, q)  # Same fixed lexical primitives.
    abbreviation = abbreviation_component(backend, q, parts)
    public = public_component(backend, parts, role_parents, role_grades)
    if abbreviation is not None or public is not None:
        parts[0] = maximum([parts[0], abbreviation, public])
        parts[6] = all_of(parts[0], const(parts[4], 0.))
        parts[8] = all_of(parts[0], parts[1], const(parts[4], 0.))
    contained = maximum([const(tv.Query.term_set_query(backend.index.schema, f, alias_fragments(q.name)), 1.)
        for f in EXACT_FIELDS]) if alias_fragments(q.name) else None
    parts[2] = maximum([parts[2], contained])  # Same bounded name proof; never double-count full alias.
    place_gate = const(parts[4], 0.)
    parts[10] = const(all_of(parts[2], parts[3], place_gate), 1.)
    scored = [tv.Query.boost_query(p, float(w)) for p, w in zip(parts, weights) if p is not None and w > 0]
    semantic = compiled_component(backend, q, terms)
    if semantic is not None and alpha:
        scored.append(tv.Query.boost_query(semantic, float(alpha)))
    # CRM may name both legal parent and physical site. They are separate
    # current fields of THIS leaf, not tokens borrowed from unrelated aliases
    # or sibling sites. This is corroboration, not an invented exact alias.
    content = [w.lower() for w in normalized(q.name).split() if w not in PARTICLES | LEGAL]
    if len(set(content)) >= 2 and place_gate is not None:
        token_views = {f: [tv.Query.term_query(backend.index.schema, f, w, 'basic')
                          for w in sorted(set(content))] for f in ('legal_names', 'site_names')}
        full = all_of(*[maximum([token_views[f][i] for f in token_views]) for i in range(len(set(content)))])
        coherent = const(all_of(full, maximum(token_views['legal_names']),
                               maximum(token_views['site_names']), place_gate), 1.)
        if coherent is not None and weights[6] > 0:
            scored.append(tv.Query.boost_query(coherent, float(weights[6])))
    # Official owners bind native leaves; IDs only join evidence, never similarity.
    support = maximum([parts[1], parts[4]])
    for predicate, grade in ((scope(parents, sites), 1.), (scope(set(), related), .5)):
        corroboration = all_of(const(predicate, 0.), support)
        if corroboration is not None:
            scored.append(tv.Query.boost_query(corroboration, grade))
    if rings:
        spatial = maximum([const(scope(set(), ids), value) for ids, value in rings])
        scored.append(spatial)
    if not scored: return tv.Query.empty_query()
    leaf = const(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'), 0.)
    return tv.Query.boolean_query([(tv.Occur.Must, leaf),
        (tv.Occur.Must, tv.Query.boolean_query([(tv.Occur.Should, p) for p in scored]))])


def search(backend, evidence, q, geo, weights, terms, alpha, municipal, role_grades, max_seconds=20.):
    began = time.monotonic(); old_guard = evidence.files.guard
    def guard():
        old_guard()
        if time.monotonic()-began > max_seconds: raise TimeoutError('STOP_CORROBORATED_QUERY_COST')
    evidence.files.guard = guard
    try:
        parents, sites, nt = evidence.name_owners(q.name)
        related, rt = evidence.related(parents, sites)
        rings, gt = evidence.radial_proofs(geo)
        role_parents = municipal.parents(q)
        parsed = compose(backend, q, weights, terms, alpha, parents, sites, related, rings, role_parents, role_grades)
        guard(); found = []
        for score, location in backend.searcher.search(parsed, limit=100, count=False).hits:
            guard(); doc = backend.searcher.doc(location).to_dict()
            row = dict(siret=backend._first(doc, 'siret'), siren=backend._first(doc, 'siren'),
                       source_score=float(score), channel='joint_proof')
            identity(row)
            if not math.isfinite(score) or score < 0: raise ValueError('Finite native joint score required')
            found.append(row)
        found.sort(key=lambda r: (-r['source_score'], r['siret']))
        if len(found) > 100 or len({r['siret'] for r in found}) != len(found):
            raise ValueError('Native direct100 distinct leaf cap required')
        for i, r in enumerate(found, 1): r['retrieval_rank'] = i
        guard()
        trace = dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
            source_admitted={'joint_proof': len(found)}, fixed_reservoir=None, score_after_fixed_pool=False,
            initial_cap=100, exact_global_fused_score_top100=True, names=nt, relations=rt, geometry=gt,
            native_leaf_sites=43896818, full_appended_leaf_universe=False, downstream_models_changed=False,
            label_lookup=False, identifiers_as_features=False, recall_measured=False,
            production_slo_validated=False, seconds=time.monotonic()-began,
            bm25_term_frequency_score=False, frozen_oof11_sparse8_head=True,
            new_fit=False, exact_alias_containment_not_crm_alias_dictionary=True,
            source_hypothesis='V280_QUERY_PUBLIC_ROLE_AND_V279_ADMINISTRATIVE_PARENT_JOIN',
            public_role_grades=list(map(float,role_grades)),
            public_parent_counts={r:len(role_parents[r]) for r in ROLES},
            public_role_calibrated_probability=False, official_administration_is_not_certified_equipment_owner=True,
            public_is_global_hard_filter=False, training_scenes_reopened=False,
            name_feature_semantics_changed=True, actual217_fit_parity_claimed=False,
            abbreviation_not_exact_string_identity=True,
            abbreviation_conditioned_on_place_no_global_filter=True,
            current_official_whole_alias_only=True, initial_place_training='V228_OOF_TRAIN64_NATIONAL_CONTRASTS',
            tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS',
            aliases_materialized_as_site_candidate_pool=False)
        return found, trace
    finally: evidence.files.guard = old_guard
