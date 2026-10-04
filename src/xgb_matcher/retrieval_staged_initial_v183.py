"""Exact physical staging of initial100, never pool selection and rescoring.

National caller MUST pin all sources/query/recipe, supervise every phase,
validate actual exits and source hashes before publication. This module is
not itself a national source, recall, latency or resource certificate.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .retrieval_coded_leaf_v182 import CodedBoundFacts, bind, score_sites
from .retrieval_coupled_blocks_v121 import Query, weights
from .retrieval_initial_collector_v146 import bounded_vector, collect
from .retrieval_name_pages_v151 import fingerprint
from .retrieval_paged_addresses_v178 import ReadonlyDiskArray, SnapshotVector
from .retrieval_paged_names_v170 import DiskVector, quantum
from .retrieval_v20_workbench import verify


def recipe(query, value, references, fold):
    if not isinstance(query, Query) or type(fold) is not int or fold not in (2, 3, 4):
        raise ValueError('Typed open OOF query and frozen initial weights required')
    w = weights(value)
    return hashlib.sha256(json.dumps(dict(query=asdict(query), weights=w.tolist(), sources=references, fold=fold),
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def reopen_vector(reference, size, dtype):
    """Readonly lease of a previously sealed file; never truncate or write it."""
    quantum(size)
    path = verify(reference)
    epoch = fingerprint(path)
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if not isinstance(a, np.memmap) or a.flags.writeable or a.dtype != np.dtype(dtype) or a.shape != (size,):
            raise ValueError('Exact original whole sealed proof vector required')
    finally:
        if isinstance(a, np.memmap):
            a._mmap.close()
    if fingerprint(path) != epoch:
        raise ValueError('Sealed query vector drift during readonly open')
    vector = SnapshotVector.__new__(SnapshotVector)
    vector.path, vector.size, vector.dtype = path, size, np.dtype(dtype)
    vector.valid, vector.epoch, vector.live_maps = True, epoch, 0
    return vector


class FlatNames:
    """Same gathered V170 whole-site evidence, immutable file-backed storage."""
    def __init__(self, cov, exact, guard, chunk):
        quantum(chunk)
        if (not isinstance(cov, DiskVector) or not isinstance(exact, DiskVector)
                or cov.dtype != np.float64 or exact.dtype != np.bool_ or len(cov) != len(exact)):
            raise ValueError('Whole float64/bool name evidence required')
        self.sites = len(cov)
        self.coverage, self.exact = ReadonlyDiskArray(cov), ReadonlyDiskArray(exact)
        for lo in range(0, self.sites, chunk):
            guard()
            hi = min(lo+chunk, self.sites)
            values, flags = self.coverage[lo:hi], self.exact[lo:hi]
            bounded_vector(values, hi-lo, np.float64, 'whole-name')
            if np.any(flags & (values != 1.)):
                raise ValueError('Direct exact-name flag requires original whole coverage1')
        guard()

    def gather(self, ids):
        if ids.dtype != np.uint32 or ids.ndim != 1 or np.any(ids >= self.sites):
            raise ValueError('Actual full source-bound site positions required')
        return self.coverage[ids], self.exact[ids].astype(np.float64)

    def verify_after(self):
        for a in (self.coverage, self.exact):
            a.assert_live()
            verify(a.reference)
            a.assert_live()

    def close(self):
        for a in (self.coverage, self.exact):
            if a._vector.valid:
                a._vector.close()  # Never delete a retained sealed stage file.


def flatten_names(snapshot, directory, guard, chunk=65536):
    """EVERY site once, no cap, no change to parent/site/one-hop scope."""
    quantum(chunk)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cov = DiskVector(directory / 'name_coverage.npy', snapshot.sites, np.float64)
    exact = DiskVector(directory / 'name_exact.npy', snapshot.sites, np.bool_)
    digest = hashlib.sha256()
    try:
        for lo in range(0, snapshot.sites, chunk):
            guard()
            ids = np.arange(lo, min(lo+chunk, snapshot.sites), dtype=np.uint32)
            values, flags = snapshot.gather(ids)
            bounded_vector(values, len(ids), np.float64, 'whole-name')
            if (flags.dtype != np.float64 or flags.shape != ids.shape
                    or np.any((flags != 0.) & (flags != 1.)) or np.any(flags > values)):
                raise ValueError('Original direct whole-name exact flags required')
            digest.update(values.tobytes())
            digest.update(flags.tobytes())
            cov.assign(lo, values)
            exact.assign(lo, flags.astype(np.bool_))
        result = FlatNames(cov, exact, guard, chunk)
        result.verify_after()
        return result, dict(sites=snapshot.sites, every_original_site_gathered_once=True,
                           original_float64_gather_digest=digest.hexdigest(), candidates_emitted=False)
    except BaseException:
        for a in (cov, exact):
            if a.valid:
                a.close()  # Keep failed own files; no partial proof publication.
        raise


def spans(source, partitions):
    """Physical groups of whole original CSR intervals, not search gates."""
    if type(partitions) is not int or not 1 <= partitions <= len(source.intervals):
        raise ValueError('Positive whole physical interval partition count required')
    groups = np.array_split(np.arange(len(source.intervals)), partitions)
    rows = []
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        rows.append(dict(start=source.intervals[first]['start'], stop=source.intervals[last]['stop'],
                         facts=sum(st['facts'] for st in source.intervals[first:last+1])))
    if rows[0]['start'] != 0 or rows[-1]['stop'] != source.sites or sum(r['facts'] for r in rows) != source.facts:
        raise ValueError('Every original site/fact in exactly one physical partition required')
    return rows


class PartitionFacts(CodedBoundFacts):
    def __init__(self, source, snapshot, query, span, fact_chunk):
        bind(source, snapshot, query, fact_chunk)
        super().__init__(source, snapshot, query, fact_chunk)
        starts = {st['start']: i for i, st in enumerate(source.intervals)}
        stops = {st['stop']: i for i, st in enumerate(source.intervals)}
        if (set(span) != {'start', 'stop', 'facts'} or any(type(v) is not int for v in span.values())
                or span['start'] not in starts or span['stop'] not in stops or span['start'] >= span['stop']
                or span['facts'] != sum(st['facts'] for st in source.intervals[starts[span['start']]:stops[span['stop']]+1])):
            raise ValueError('Exact whole original physical CSR interval group required')
        self.span = span.copy()
        self.cursor = span['start']

    def batches(self, ids, query):
        if ids.ndim != 1 or not len(ids) or ids[-1] >= self.span['stop']:
            raise ValueError('No missing/overlapping physical partition site allowed')
        yield from super().batches(ids, query)

    def finish(self):
        self.source.assert_live()
        if self.cursor != self.span['stop'] or self.consumed != self.span['facts']:
            raise ValueError('Every original fact/site in this physical partition required')
        self.snapshot.verify_after()


class PartitionKeys:
    def __init__(self, keys, start):
        self.source, self.start = keys, start

    def reconstruct(self, ids, grain):
        if ids.dtype != np.uint32:
            raise ValueError('Exact physical positions required')
        return self.source.reconstruct(ids+np.uint32(self.start), grain)


def partition_search(q, value, catalog, names, attributes, addresses, signature,
                     activity=None, chunk=65536, max_seconds=20.):
    """Score ALL leaves in a physical shard; same score goes to final merge."""
    quantum(chunk)
    if (not isinstance(addresses, PartitionFacts) or q is not addresses.query
            or any(s.sites != catalog.sites for s in (names, attributes, addresses))
            or not isinstance(signature, str) or len(signature) != 64):
        raise ValueError('Same entire namespace/query and source-bound recipe required')
    start, stop = addresses.span['start'], addresses.span['stop']
    local_catalog = SimpleNamespace(sites=stop-start, keys=PartitionKeys(catalog.keys, start))
    w = weights(value)
    totals = dict(address_facts=0, address_batches=0, score_chunks=0)
    def scored():
        for lo in range(start, stop, chunk):
            ids = np.arange(lo, min(lo+chunk, stop), dtype=np.uint32)
            nc, ne = names.gather(ids)
            scores, stats = score_sites(ids, q, w, nc, ne, attributes, addresses.batches(ids, q), activity)
            for k in ('address_facts', 'address_batches'):
                totals[k] += stats[k]
            totals['score_chunks'] += 1
            yield ids-np.uint32(start), scores
    rows, stats = collect(local_catalog, scored(), limit=100, max_seconds=max_seconds)
    addresses.finish()
    names.verify_after()
    if totals['address_facts'] != addresses.span['facts']:
        raise ValueError('Every coherent original address proof scored required')
    return dict(span=addresses.span, signature=signature, candidates=rows, stats=dict(stats, **totals),
                partition_only_not_initial_output=True, same_scores_never_rescored=True)


def merge_partitions(catalog, total_facts, parts):
    """Exact top100 reduction after EVERY site, not reranking a fixed pool."""
    if type(total_facts) is not int or total_facts < 0 or not parts:
        raise ValueError('All physical partitions and original fact denominator required')
    seen = facts = 0
    signature = parts[0]['signature']
    candidates = []
    identities = set()
    for part in parts:
        span, stats, rows = part['span'], part['stats'], part['candidates']
        if (part['signature'] != signature or not isinstance(signature, str) or len(signature) != 64
                or part['partition_only_not_initial_output'] is not True or part['same_scores_never_rescored'] is not True
                or span['start'] != seen or not seen < span['stop'] <= catalog.sites
                or stats['sites_completed'] != span['stop']-seen or stats['address_facts'] != span['facts']
                or stats['every_site_scored_once'] is not True or stats['fixed_reservoir'] is not None
                or stats['score_after_fixed_pool'] is not False or stats['initial_cap'] != 100 or len(rows) > 100):
            raise ValueError('Complete once-only identical-score physical partitions required')
        if any(set(r) != {'siret', 'siren', 'score'} or not isinstance(r['siret'], str)
               or len(r['siret']) != 14 or not r['siret'].isascii() or not r['siret'].isdigit()
               or r['siren'] != r['siret'][:9] or isinstance(r['score'], bool)
               or not np.isfinite(r['score']) or r['score'] <= 0 for r in rows):
            raise ValueError('Unmodified positive finite official leaf score required')
        keys = np.array([int(r['siret']) for r in rows], dtype=np.uint64)
        positions = catalog.keys.site_positions(keys)
        if np.any(positions < seen) or np.any(positions >= span['stop']):
            raise ValueError('Every partition collector identity belongs to its exact physical sites')
        if rows != sorted(rows, key=lambda r: (-r['score'], r['siret'])):
            raise ValueError('Same global score/identity tie order required')
        for row in rows:
            if row['siret'] in identities:
                raise ValueError('No duplicate site across physical partitions permitted')
            identities.add(row['siret'])
        candidates.extend(rows)
        seen = span['stop']
        facts += span['facts']
    if seen != catalog.sites or facts != total_facts:
        raise ValueError('Every original national site and address fact before initial100 publication')
    return sorted(candidates, key=lambda r: (-r['score'], r['siret']))[:100], dict(
        sites_completed=seen, address_facts=facts, initial_cap=100, signature=signature,
        physical_partitions=len(parts), every_site_scored_once=True, fixed_reservoir=None,
        score_after_fixed_pool=False, same_original_score_never_rescored=True,
        national_source_coverage_certified_here=False, recall_measured=False, slo_validated=False)
