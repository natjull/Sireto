"""Direct multi-channel initial100 experiment, not a pool then a reranker.

New semantics: lazy round-robin of independent native text, complete official
name-owner, one-hop succession and geometric-neighbourhood searches. This is
NOT exact top100 under V114, RRF, or a claim of recall/production readiness.
The native leaf universe remains V38; appended identities are not indexed.
"""
from collections import Counter, deque
import json
import math
from pathlib import Path
import time

import duckdb
import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from .retrieval_joint_v46 import geo_parameters
from .retrieval_name_pages_v151 import fingerprint
from .retrieval_national_names_v138 import PackedTerms
from .retrieval_proof_bridge_v137 import PROTECTED
from .retrieval_variants_v118 import normalized
from .retrieval_v20_workbench import verify


def identity(row):
    s, p = row['siret'], row['siren']
    if (not isinstance(s, str) or not isinstance(p, str) or len(s) != 14 or len(p) != 9
            or not s.isascii() or not p.isascii() or not s.isdigit() or not p.isdigit() or s[:9] != p):
        raise ValueError('Explicit official endpoint and parent required')
    return s, p


def interleave(streams, guard, limit=100):
    """Consume until100 DISTINCT leaves or exhaustion; no per-source rank cut.

    There is no cross-channel score comparison or score-completion callback.
    Duplicate consumption continues within that channel's turn. Empty channels
    relinquish their turns. The policy is fixed BEFORE seeing query labels.
    """
    if type(limit) is not int or not 1 <= limit <= 100 or not isinstance(streams, dict):
        raise ValueError('Direct initial cap1..100 and named streams required')
    active = deque(streams)
    seen, source_seen, output = {}, {k: set() for k in streams}, []
    consumed, admitted = Counter(), Counter()
    while active and len(output) < limit:
        guard()
        name = active.popleft()
        while True:
            guard()
            row = streams[name].next()
            if row is None:
                break
            s, p = identity(row)
            if s in source_seen[name]:
                raise ValueError('Duplicate/unstable pagination inside a source')
            source_seen[name].add(s)
            consumed[name] += 1
            value = row['score']
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative native source score required')
            if s in seen:
                if seen[s] != p:
                    raise ValueError('Official membership conflict across sources')
                continue
            seen[s] = p
            admitted[name] += 1
            output.append(dict(siret=s, siren=p, retrieval_rank=len(output)+1,
                               channel=name, source_score=float(value)))
            active.append(name)
            break
    guard()
    return output, dict(policy='LAZY_DISTINCT_ROUND_ROBIN', source_consumed=dict(consumed),
                        source_admitted=dict(admitted), fixed_reservoir=None,
                        score_after_fixed_pool=False, initial_cap=limit,
                        exact_global_fused_score_top100=False)


class NativeCursor:
    """Physical pages are not a candidate cutoff. Native score never changes."""
    def __init__(self, backend, parsed, guard, page_size=16):
        self.backend, self.parsed, self.guard = backend, parsed, guard
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError('Positive bounded physical page size required')
        self.page_size, self.offset, self.buffer = page_size, 0, deque()
        self.exhausted = parsed is None
        self.previous = math.inf

    def next(self):
        self.guard()
        if not self.buffer and not self.exhausted:
            hits = self.backend.searcher.search(self.parsed, limit=self.page_size,
                                                offset=self.offset, count=False).hits
            self.guard()
            self.offset += len(hits)
            self.exhausted = len(hits) < self.page_size
            for score, location in hits:
                doc = self.backend.searcher.doc(location).to_dict()
                row = dict(siret=self.backend._first(doc, 'siret'),
                           siren=self.backend._first(doc, 'siren'), score=float(score))
                identity(row)
                if not math.isfinite(score) or score < 0 or score > self.previous:
                    raise ValueError('Native source order/score drift')
                self.previous = score
                self.buffer.append(row)
        return self.buffer.popleft() if self.buffer else None


class SourceFiles:
    """Original full hashes AND pre-read epochs for every file actually used.

    Data stay readonly. Hash verification is not replaced by metadata. This
    experiment does not re-certify all unused files of predecessor compilers.
    """
    def __init__(self, guard):
        self.guard, self.entries = guard, {}

    def file(self, reference):
        self.guard()
        path = str(Path(reference['path']).resolve())
        if not Path(reference['path']).is_absolute():
            raise ValueError('Absolute pinned official reference required')
        if path in self.entries:
            entry = self.entries[path]
            if entry['reference']['sha256'] != reference['sha256'] or list(fingerprint(path)) != entry['fingerprint']:
                raise ValueError('Contradictory hash or changed source epoch')
        else:
            before = list(fingerprint(path))
            verify(reference)
            if list(fingerprint(path)) != before:
                raise ValueError('Source changed while hashing before read')
            if 'size_bytes' in reference and reference['size_bytes'] != before[2]:
                raise ValueError('Original pinned size required')
            self.entries[path] = dict(reference=reference.copy(), fingerprint=before)
        self.guard()
        return Path(path)

    def manifest(self, reference, status):
        value = json.loads(self.file(reference).read_text())
        if value.get('status') != status or any(value.get(k) is not False for k in PROTECTED):
            raise ValueError('Protected complete official source manifest required')
        launch = json.loads(self.file(value['launch']).read_text())
        for ref in launch['implementation'].values():
            self.file(ref)
        return value

    def mapped(self, reference, dtype, shape):
        path = self.file(reference)
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        if not isinstance(a, np.memmap) or a.flags.writeable or a.dtype != np.dtype(dtype) or a.shape != shape:
            if isinstance(a, np.memmap):
                a._mmap.close()
            raise ValueError('Whole readonly typed source array required')
        return a

    def check_epochs(self):
        for path, entry in self.entries.items():
            if list(fingerprint(path)) != entry['fingerprint']:
                raise ValueError('Original source epoch drift')

    def verify_after(self):
        self.check_epochs()
        for entry in self.entries.values():
            self.guard()
            verify(entry['reference'])
        self.check_epochs()
        self.guard()


class OfficialEvidence:
    """Exact/all-word WHOLE variants -> their actual owners, no alias fanout cap.

    A legal name binds all sites of its legal person through the native siren
    predicate. An endpoint name binds ONLY that endpoint, never its siblings.
    Related names are hypotheses in a separate channel, not identity merges.
    """
    def __init__(self, cfg, files):
        self.files = files
        self.names = files.manifest(cfg['names'], 'SEALED_V134_COMPLETE_ALL_OFFICIAL_NAME_CHANNEL_ACCESS')
        self.joins = files.manifest(cfg['joins'], 'SEALED_V149_ALL_OFFICIAL_NAME_PARENT_APPEND_JOINS')
        stage = json.loads(files.file(self.names['source_stage']).read_text())
        owners = files.manifest(self.joins['sources']['owners'], 'SEALED_V132_COMPLETE_ALL_OFFICIAL_NAME_OWNER_CSR')
        if owners['names'] != self.names['names'] or stage['source'] != owners['source']:
            raise ValueError('Same entire official name and owner namespace required')
        self.items = [x['source_bucket'] for x in self.joins['buckets']]
        if (self.items != owners['buckets'] or len(self.items) != 64 or sum(x['names'] for x in self.items) != self.names['names']
                or any(x['bucket'] != i or x['name_start'] != sum(b['names'] for b in self.items[:i])
                       for i, x in enumerate(self.items))):
            raise ValueError('Whole contiguous64 official owner buckets required')
        launch = json.loads(files.file(self.names['launch']).read_text())
        self.buckets = launch['config']['term_buckets']
        if duckdb.__version__ != launch['config']['duckdb_version'] or np.__version__ != launch['config']['numpy_version']:
            raise ValueError('Pinned official hash and numerical libraries required')
        self.access = {(x['channel'], x['bucket']): x for x in self.names['access']}
        if len(self.access) != 4*self.buckets:
            raise ValueError('Complete four-channel access inventory required')
        self.c = duckdb.connect()
        self.c.execute('SET threads=1')
        self.c.execute("SET memory_limit='32MiB'")
        bridge = json.loads(files.file(self.joins['sources']['bridge']).read_text())
        self.graph = files.manifest(bridge['sources']['graph'], 'SEALED_V133_COMPLETE_ALL_RAW_OFFICIAL_SUCCESSION')
        self.structured = files.manifest(cfg['structured'], 'SEALED_V139_ALL_CURRENT_STRUCTURED_SITE_ATTRIBUTES')
        catalog = json.loads(files.file(self.structured['sources']['catalog']).read_text())
        self.xy_ref = self.structured['reused_arrays']['site_xy']
        self.keys_ref = catalog['input_arrays']['endpoint_keys']
        self.sites = self.structured['stats']['sites']

    def close(self):
        self.c.close()

    def lookup(self, channel, token):
        self.files.guard()
        bucket = int(self.c.execute('SELECT hash(?) % ?', [token, self.buckets]).fetchone()[0])
        item = self.access[channel, bucket]
        specs = dict(post_ids=(np.uint32, item['postings']),
                     post_offsets=(np.uint64, item['terms']+1),
                     token_offsets=(np.uint64, item['terms']+1), token_data=(np.uint8, None))
        arrays = {}
        try:
            for name, (dtype, size) in specs.items():
                # Header size is not assumed for token_data; derive its shape
                # from the actual readonly header and validate CSR endpoints.
                if name == 'token_data':
                    a = np.load(self.files.file(item['artifacts'][name+'.npy']), mmap_mode='r', allow_pickle=False)
                    if a.dtype != np.uint8 or a.ndim != 1 or a.flags.writeable:
                        a._mmap.close()
                        raise ValueError('Original ASCII token byte vector required')
                    arrays[name] = a
                else:
                    arrays[name] = self.files.mapped(item['artifacts'][name+'.npy'], dtype, (size,))
            if (arrays['post_offsets'][0] != 0 or arrays['post_offsets'][-1] != len(arrays['post_ids'])
                    or arrays['token_offsets'][0] != 0 or arrays['token_offsets'][-1] != len(arrays['token_data'])):
                raise ValueError('Full original token/posting CSR required')
            result = PackedTerms(arrays).lookup(token).copy()
            if result.dtype != np.uint32 or np.any(result >= self.names['names']) or np.any(result[1:] <= result[:-1]):
                raise ValueError('All unique sorted official alias positions required')
            return result
        finally:
            for a in arrays.values():
                a._mmap.close()
            self.files.check_epochs()

    def name_owners(self, text):
        value = normalized(text)
        if not value:
            return set(), set(), dict(aliases=0, observations=0)
        exact = self.lookup(3, value)
        words = sorted(set(value.split()))
        # Full conjunctive alias semantics is a new retrieval channel, not a
        # truncation of the old graded coverage score. Other channels retain
        # typo/partial-name access in the unchanged native full index.
        postings = [self.lookup(0, word) for word in words]
        postings.sort(key=len)
        both = postings[0]
        for part in postings[1:]:
            self.files.guard()
            both = np.intersect1d(both, part, assume_unique=True)
        selected = np.union1d(exact, both).astype(np.uint32)
        parents, sites, observations = set(), set(), 0
        for b in self.items:
            self.files.guard()
            lo, hi = b['name_start'], b['name_start']+b['names']
            ids = selected[(selected >= lo) & (selected < hi)]-np.uint32(lo)
            if not len(ids):
                continue
            arrays = {}
            try:
                specs = dict(name_fact_offsets=(np.uint64, b['names']+1), owner_key=(np.uint64, b['facts']),
                             parent_key=(np.uint32, b['facts']), subject_code=(np.uint8, b['facts']),
                             scope_code=(np.uint8, b['facts']))
                for k, (d, n) in specs.items():
                    arrays[k] = self.files.mapped(b['artifacts'][k+'.npy'], d, (n,))
                for alias in ids:
                    self.files.guard()
                    begin, end = map(int, arrays['name_fact_offsets'][int(alias):int(alias)+2])
                    if not 0 <= begin < end <= b['facts']:
                        raise ValueError('Whole nonempty official alias-owner interval required')
                    for row in range(begin, end):
                        if (row-begin) % 1024 == 0:
                            self.files.guard()
                        owner, parent, subject, scope = (int(arrays[k][row]) for k in ('owner_key', 'parent_key', 'subject_code', 'scope_code'))
                        if subject == 1 and scope in (1, 2) and owner == parent and 0 <= parent < 10**9:
                            parents.add(f'{parent:09d}')
                        elif subject == 2 and scope == 3 and 0 <= owner < 10**14 and owner//100000 == parent:
                            sites.add(f'{owner:014d}')
                        else:
                            raise ValueError('Official name owner/scope conflict')
                        observations += 1
            finally:
                for a in arrays.values():
                    a._mmap.close()
        return parents, sites, dict(aliases=len(selected), exact_aliases=len(exact), owner_facts=observations,
                                    parent_owners=len(parents), endpoint_owners=len(sites), owner_cap=None)

    def related(self, parents, sites):
        if not parents and not sites:
            return set(), dict(anchors=0, observations=0)
        g, arrays = self.graph, {}
        st = g['stats']
        specs = dict(endpoint_keys=(np.uint64, st['endpoints']), endpoint_offsets=(np.uint64, st['endpoints']+1),
                     endpoint_incidences=(np.uint32, st['oriented_incidences']),
                     parent_keys=(np.uint32, st['parents']), parent_endpoint_offsets=(np.uint64, st['parents']+1),
                     predecessor_keys=(np.uint64, st['observations']), successor_keys=(np.uint64, st['observations']))
        try:
            for k, (d, n) in specs.items():
                arrays[k] = self.files.mapped(g['artifacts'][k+'.npy'], d, (n,))
            anchors = set()
            for parent in sorted(parents):
                self.files.guard()
                p = int(np.searchsorted(arrays['parent_keys'], np.uint32(int(parent))))
                if p < st['parents'] and arrays['parent_keys'][p] == int(parent):
                    lo, hi = map(int, arrays['parent_endpoint_offsets'][p:p+2])
                    anchors.update(range(lo, hi))
            for siret in sorted(sites):
                p = int(np.searchsorted(arrays['endpoint_keys'], np.uint64(int(siret))))
                if p < st['endpoints'] and arrays['endpoint_keys'][p] == int(siret):
                    anchors.add(p)
            result, observations = set(), 0
            for anchor in sorted(anchors):
                self.files.guard()
                source = int(arrays['endpoint_keys'][anchor])
                lo, hi = map(int, arrays['endpoint_offsets'][anchor:anchor+2])
                for incidence in arrays['endpoint_incidences'][lo:hi]:
                    observations += 1
                    ordinal = int(incidence) % st['observations']
                    pred, succ = (int(arrays[k][ordinal]) for k in ('predecessor_keys', 'successor_keys'))
                    if (pred if incidence < st['observations'] else succ) != source:
                        raise ValueError('Original relation incidence/endpoint drift')
                    if pred != succ:
                        result.add(f'{succ if incidence < st["observations"] else pred:014d}')
            return result, dict(anchors=len(anchors), observations=observations, related_sites=len(result),
                                hops=1, endpoint_cap=None, identity_merge=False, continuity_filtered=False)
        finally:
            for a in arrays.values():
                a._mmap.close()

    def nearby(self, geo, chunk=131072):
        point, sigma, reliable = geo_parameters(geo)
        if point is None:
            return set(), dict(compatible=False, sites_scanned=0)
        radius = 300. if reliable else max(300., 3.*sigma)
        xy_path = self.files.file(self.xy_ref)
        keys = self.files.mapped(self.keys_ref, np.uint64, (self.sites,))
        result = set()
        try:
            # Independent spatial predicate; no nearest100 cut BEFORE text.
            # Close only our readonly coordinate map after each full chunk.
            for lo in range(0, self.sites, chunk):
                self.files.guard()
                a = np.load(xy_path, mmap_mode='r', allow_pickle=False)
                try:
                    if a.dtype != np.float64 or a.shape != (self.sites, 2) or a.flags.writeable:
                        raise ValueError('Original complete Lambert93 source coordinates required')
                    xy = a[lo:min(lo+chunk, self.sites)].copy()
                finally:
                    a._mmap.close()
                valid = np.isfinite(xy).all(axis=1)
                distance2 = np.sum((xy-point)**2, axis=1)
                ids = np.flatnonzero(valid & (distance2 <= radius*radius+1e-6))+lo
                result.update(f'{int(s):014d}' for s in keys[ids])
            return result, dict(compatible=True, reliable=bool(reliable), radius_m=radius,
                                sites_scanned=self.sites, joined_sites=len(result), truncated=False,
                                appended_coordinates_unknown=True)
        finally:
            keys._mmap.close()
            self.files.check_epochs()


def views(backend, fields, text):
    parts = [v38.mean_query(backend, f, v38.tokens(normalized(text))) for f in fields]
    parts = [p for p in parts if p is not None]
    return backend._tantivy.Query.disjunction_max_query(parts, tie_breaker=0.) if parts else None


def scoped_query(backend, q, parents, sites, text_fields, text):
    if not parents and not sites:
        return None
    tv = backend._tantivy
    predicates = [tv.Query.term_set_query(backend.index.schema, f, sorted(values))
                  for f, values in (('siren', parents), ('siret', sites)) if values]
    scope = tv.Query.disjunction_max_query(predicates, tie_breaker=0.)
    doc = tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'), 0.)
    # The constant is official name/graph/geometry evidence, not ID similarity.
    parts = [(tv.Occur.Must, doc), (tv.Occur.Must, tv.Query.const_score_query(scope, 1.))]
    lexical = views(backend, text_fields, text)
    if lexical is not None:
        parts.append((tv.Occur.Should, lexical))
    place = [tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, f, value, 'basic'), 1.)
             for f, value in (('insee', q.insee), ('postcode', q.postcode)) if value]
    if place:
        parts.append((tv.Occur.Should, tv.Query.disjunction_max_query(place, tie_breaker=0.)))
    return tv.Query.boolean_query(parts)


def search(backend, evidence, q, geo, max_seconds=20.):
    if isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('Positive unchanged query20 budget required')
    began = time.monotonic()
    old_guard = evidence.files.guard
    def guard():
        old_guard()
        if time.monotonic()-began > max_seconds:
            raise TimeoutError('STOP_STRUCTURAL_CHANNEL_QUERY_COST')
    evidence.files.guard = guard
    try:
        parents, sites, nt = evidence.name_owners(q.name)
        related, rt = evidence.related(parents, sites)
        nearby, gt = evidence.nearby(geo)
        queries = dict(text=v38.query_for(backend, q, 'primary'),
                       official_name=scoped_query(backend, q, parents, sites, v38.ADDRESSES, q.address),
                       succession=scoped_query(backend, q, set(), related, v38.ADDRESSES, q.address),
                       geo_text=scoped_query(backend, q, set(), nearby, v38.NAMES, q.name))
        streams = {k: NativeCursor(backend, parsed, guard) for k, parsed in queries.items() if parsed is not None}
        output, trace = interleave(streams, guard)
        trace.update(seconds=time.monotonic()-began, names=nt, relations=rt, geometry=gt,
                     native_leaf_sites=43896818, full_appended_leaf_universe=False,
                     current_and_history_official_name_dictionary=True, name_or_relation_cut=None,
                     downstream_models_changed=False, label_lookup=False, identifiers_as_features=False,
                     production_slo_validated=False, recall_measured=False)
        return output, trace
    finally:
        evidence.files.guard = old_guard
