"""Exact V146 scores with dictionary equality instead of per-fact strings.

Unmeasured nationally. No changed feature, model, proof, site universe or
candidate pool. The original V146 oracle stays immutable and independent.
"""
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from .retrieval_certified_address_facts_v181 import CertifiedBoundFacts
from .retrieval_coupled_blocks_v121 import point, SCALES, weights
from .retrieval_initial_collector_v146 import bounded_vector, collect
from .retrieval_mapped_blocks_v125 import matrix
from .retrieval_site_attributes_v139 import encode_code, MISSING
from .retrieval_variants_v118 import normalized


@dataclass(frozen=True)
class CodedAddressBatch:
    site_positions: np.ndarray
    coverage: np.ndarray
    alias_exact: np.ndarray
    current_exact: np.ndarray
    postcode_codes: np.ndarray
    insee_codes: np.ndarray
    number_matches: np.ndarray


def number_lookup(book, query_number):
    """Validate each original dictionary entry ONCE; never normalize it again."""
    if (not isinstance(query_number, str) or not book or book[0] != ''
            or len(book) != len(set(book)) or any(not isinstance(s, str) or not s.isascii()
            or s != s.strip() or '  ' in s
            or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ' for c in s) for s in book)):
        raise ValueError('Original once-normalized proof number dictionary required')
    number = normalized(query_number)
    values = np.fromiter((bool(number) and s == number for s in book), dtype=np.bool_, count=len(book))
    values.flags.writeable = False
    return values


def coded_batches(reader, ids, snapshot, query_number, chunk):
    """Same V147 slices and proof scope; only string gather becomes bool gather."""
    if (ids.dtype != np.uint32 or ids.ndim != 1 or not len(ids)
            or ids[0] < reader.stats['start'] or ids[-1] >= reader.stats['stop']
            or not np.array_equal(ids, np.arange(int(ids[0]), int(ids[0])+len(ids), dtype=np.uint32))
            or type(chunk) is not int or chunk <= 0 or snapshot.aliases != reader.stats['aliases']
            or snapshot.coverage.flags.writeable or snapshot.exact.flags.writeable):
        raise ValueError('Same whole proof snapshot and exact physical interval required')
    matches = number_lookup(reader.book, query_number)
    if len(matches) != reader.stats['number_strings']:
        raise ValueError('Whole original number dictionary required')
    offsets = reader.a['site_offsets']
    lo = int(offsets[int(ids[0])-reader.stats['start']])
    hi = int(offsets[int(ids[-1])-reader.stats['start']+1])
    if not 0 <= lo <= hi <= reader.stats['facts']:
        raise ValueError('Original physical CSR extent required')
    for start in range(lo, hi, chunk):
        sl = slice(start, min(start+chunk, hi))
        alias = reader.a['alias_position'][sl]
        codes = reader.a['number_codes'][sl]
        if np.any(alias >= snapshot.aliases) or np.any(codes >= len(matches)):
            raise ValueError('Every exact original alias and number code required')
        yield CodedAddressBatch(reader.a['site_position'][sl], snapshot.coverage[alias], snapshot.exact[alias],
            reader.a['current_exact'][sl], reader.a['postcode_code'][sl], reader.a['insee_code'][sl], matches[codes])


class CodedBoundFacts(CertifiedBoundFacts):
    """Same complete V158 traversal and V181 hash-before-publication finish."""
    def batches(self, ids, query):
        source = self.source
        if (query is not self.query or not source.valid or ids.dtype != np.uint32
                or ids.ndim != 1 or not len(ids) or ids[0] != self.cursor or ids[-1] >= self.sites
                or not np.array_equal(ids, np.arange(self.cursor, self.cursor+len(ids), dtype=np.uint32))):
            raise ValueError('Same query and complete ordered site traversal required')
        index = bisect_right(source.starts, self.cursor)-1
        cursor, stop = self.cursor, int(ids[-1])+1
        while cursor < stop:
            stats = source.intervals[index]
            hi = min(stop, stats['stop'])
            reader = source._open(index)
            try:
                for batch in coded_batches(reader, ids[cursor-self.cursor:hi-self.cursor], self.snapshot,
                                           query.number, self.fact_chunk):
                    self.consumed += len(batch.site_positions)
                    yield batch
                source._check_part(source.checked[index])
            finally:
                if source.active is reader:
                    reader.close()
                    source.active = None
            cursor = hi
            index += 1
        self.cursor = stop


def bind(source, snapshot, query, fact_chunk=65536):
    source.bind(snapshot, query, fact_chunk)  # Preserve V181/V158 namespace/lease validation.
    return CodedBoundFacts(source, snapshot, query, fact_chunk)


def score_sites(ids, q, value, name_coverage, name_exact, attributes, address_batches, activity=None):
    """V146 arithmetic and float reduction order, with identical boolean NUM."""
    w = weights(value)
    n = len(ids)
    if (ids.dtype != np.uint32 or ids.ndim != 1 or not n or np.any(ids >= attributes.sites)
            or not np.array_equal(ids, np.arange(int(ids[0]), int(ids[0])+n, dtype=np.uint32))):
        raise ValueError('Complete contiguous physical site chunk required')
    bounded_vector(name_coverage, n, np.float64, 'whole-name')
    bounded_vector(name_exact, n, np.float64, 'exact-name')
    if np.any(name_exact > name_coverage):
        raise ValueError('Exact name cannot exceed its whole proof coverage')
    if not isinstance(q.reliable_geo, bool):
        raise ValueError('Explicit BAN reliability required')
    activity = {} if activity is None else activity
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 0 or not np.isfinite(v) or not 0 <= v <= 1
           for k, v in activity.items()):
        raise ValueError('Explicit activity grades required')
    place, num, hq, classes = attributes.flags(ids, q.postcode, q.insee, q.number)
    for a, key in ((place, 'place'), (num, 'number'), (hq, 'headquarters')):
        bounded_vector(a, n, np.float64, key)
    if classes.dtype != np.int32 or classes.shape != (n,) or np.any(classes < -1):
        raise ValueError('Explicit current activity or UNKNOWN required')
    xy = attributes.coordinates(ids)
    if (xy.dtype != np.float64 or xy.shape != (n, 2) or np.isinf(xy).any()
            or np.any(np.isfinite(xy[:, 0]) != np.isfinite(xy[:, 1]))):
        raise ValueError('Original same-CRS XY pair or UNKNOWN required')
    query_xy = point(q.xy)
    geometry = np.zeros((n, 4))
    known = np.isfinite(xy[:, 0])
    if query_xy is not None and np.any(known):
        if np.any(np.abs(xy[known]) > 1e9):
            raise ValueError('Projected source XY range required')
        distance = np.sqrt(np.sum((xy[known]-query_xy)**2, axis=1))
        geometry[known] = 1./(1.+distance[:, None]/SCALES)
    grades = np.array([activity.get(int(k), 0.) for k in classes], dtype=np.float64)
    zero = np.zeros(n)
    f = matrix(name_coverage, zero, name_exact, zero, place, num, hq, geometry, q.reliable_geo, grades)
    scores = np.sum(f*w, axis=1)
    count = batch_count = 0
    cp, ins = encode_code(q.postcode), encode_code(q.insee)
    for b in address_batches:
        if not isinstance(b, CodedAddressBatch):
            raise ValueError('Explicit source-coded coherent address batch required')
        size = len(b.site_positions)
        if (b.site_positions.dtype != np.uint32 or b.site_positions.shape != (size,)
                or np.any(b.site_positions < ids[0]) or np.any(b.site_positions > ids[-1])):
            raise ValueError('Every address explicitly bound to this site chunk required')
        bounded_vector(b.coverage, size, np.float64, 'whole-address')
        for a in (b.alias_exact, b.current_exact, b.number_matches):
            if a.dtype != np.bool_ or a.shape != (size,):
                raise ValueError('Exact original address and number flags required')
        if np.any(b.alias_exact & (b.coverage != 1.)):
            raise ValueError('Exact address requires whole alias coverage')
        for a in (b.postcode_codes, b.insee_codes):
            if a.dtype != np.uint32 or a.shape != (size,) or np.any((a != MISSING) & (a >= 36**5)):
                raise ValueError('Exact proof-owned public codes required')
        local = (b.site_positions-ids[0]).astype(np.int64)
        p = np.zeros(size)
        if cp != MISSING:
            p = np.maximum(p, b.postcode_codes == cp)
        if ins != MISSING:
            p = np.maximum(p, b.insee_codes == ins)
        nm = b.number_matches.astype(np.float64)
        ae = (b.alias_exact & b.current_exact).astype(np.float64)
        f = matrix(name_coverage[local], b.coverage, name_exact[local], ae, p, nm, hq[local],
                   geometry[local], q.reliable_geo, grades[local])
        np.maximum.at(scores, local, np.sum(f*w, axis=1))
        count += size
        batch_count += 1
    if not np.isfinite(scores).all():
        raise ValueError('Finite complete leaf scores required')
    return scores, dict(sites=n, address_facts=count, address_batches=batch_count,
        coherent_whole_address_score=True, proof_or_owner_cap=None,
        relation_address_or_attribute_inheritance=False, closed_or_unknown_sites_filtered=False)


def search(q, value, catalog, names, attributes, addresses, activity=None, chunk=65536, max_seconds=20.):
    if (names.sites != catalog.sites or attributes.sites != catalog.sites or addresses.sites != catalog.sites
            or type(chunk) is not int or chunk <= 0 or not isinstance(addresses, CodedBoundFacts)):
        raise ValueError('Whole identical source namespace and coded complete binding required')
    w = weights(value)
    totals = dict(address_facts=0, address_batches=0, score_chunks=0)
    def scored():
        for start in range(0, catalog.sites, chunk):
            ids = np.arange(start, min(start+chunk, catalog.sites), dtype=np.uint32)
            n, ne = names.gather(ids)
            scores, stats = score_sites(ids, q, w, n, ne, attributes, addresses.batches(ids, q), activity)
            for key in ('address_facts', 'address_batches'):
                totals[key] += stats[key]
            totals['score_chunks'] += 1
            yield ids, scores
    rows, stats = collect(catalog, scored(), limit=100, max_seconds=max_seconds)
    addresses.finish()  # Complete traversal and full immutable-query hash before any return.
    return rows, dict(stats, **totals, original_number_dictionary_equality=True,
                     float_score_precision_and_reduction_order_unchanged=True)
