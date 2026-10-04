"""One native global top100 over corroborated bounded proofs, not four quotas.

Frozen OOF11/sparse query head; no new fit. Official aliases contained in the
CRM may substantiate name evidence without requiring the entire CRM as alias.
Spatial rings are SOFT proof predicates: outside/missing XY stays reachable.
"""
import math
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from .retrieval_proof_v56 import components
from .retrieval_sparse_query_v59 import compiled_component
from .retrieval_structural_channels_v210 import OfficialEvidence, identity
from .retrieval_joint_v46 import geo_parameters
from .retrieval_variants_v118 import normalized
from .retrieval_crossfield_proofs_v250 import compose,FEATURES

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


def search(backend, evidence, q, geo, weights, terms, alpha, extra_weights, max_seconds=20.):
    began = time.monotonic(); old_guard = evidence.files.guard
    def guard():
        old_guard()
        if time.monotonic()-began > max_seconds: raise TimeoutError('STOP_CORROBORATED_QUERY_COST')
    evidence.files.guard = guard
    try:
        parents, sites, nt = evidence.name_owners(q.name)
        related, rt = evidence.related(parents, sites)
        rings, gt = evidence.radial_proofs(geo)
        parsed = compose(backend, q, weights, terms, alpha, parents, sites, related, rings, extra_weights)
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
            new_fit=True, query_time_training=False, exact_alias_containment_not_crm_alias_dictionary=True,
            extra_proof_features=list(FEATURES), original11_head_fixed_v228=True,
            initial_training='V252_OOF_TRAIN64_NEW3_PROOFS_FIXED228',
            cross_name_address_is_not_legal_identity_or_alias=True,
            tie_scope='CANONICAL_ONLY_AMONG_RETURNED_NATIVE_SCORE_EQUALS',
            aliases_materialized_as_site_candidate_pool=False)
        return found, trace
    finally: evidence.files.guard = old_guard
