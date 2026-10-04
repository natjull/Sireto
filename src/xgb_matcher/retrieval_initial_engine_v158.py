"""Source-bound address integration for the direct V146 initial100 collector.

Complete V154/V155 authorities are mandatory. No candidate reservoir, fit,
label lookup or downstream model. This is an UNMEASURED integration: neither
its tests nor opening its sources certify national recall, latency or RSS.
"""
from bisect import bisect_right
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from .retrieval_address_reverse_v147 import AliasSnapshot, ReversePart
from .retrieval_coupled_blocks_v121 import Query, point, weights
from .retrieval_initial_collector_v146 import search
from .retrieval_v20_workbench import verify


def fingerprint(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


class CompleteAddressFacts:
    """All explicit address facts, with one physical interval mapped at a time.

    Full original source hashes and V147 array audits at open and full hashes
    again at verify_after. Between these audits every borrowed file must keep
    the same inode/size/mtime; no repeated multi-GB rehash per query chunk.
    The V147 batches method is unchanged, including coherent proof attributes.
    """
    def __init__(self, reference):
        from scripts.compile_retrieval_v155_address_access import config, store, source_epochs
        self.valid = False
        self.active = None
        self.reference = reference
        self.metadata, dictionary = store(reference, config())
        self.sites = self.metadata['sites']
        self.aliases = self.metadata['aliases']
        self.facts = self.metadata['facts']
        self.catalog_reference = self.metadata['catalog']
        self.dictionary_reference = self.metadata['dictionary']
        self.intervals = self.metadata['intervals']
        self.starts = [s['start'] for s in self.intervals]
        self.checked = []
        # Actual V147 readers validate every interval before any query can run.
        # Only array specs/epochs survive, not all books/maps/pages in memory.
        for stats in self.intervals:
            reader = ReversePart(stats)
            try:
                refs = [*stats['artifacts'].values(), stats['numbers'], stats['ordered_source'], *stats['sources']]
                self.checked.append(dict(
                    epochs={r['path']: fingerprint(r['path']) for r in refs},
                    specs={k: (a.dtype, a.shape) for k, a in reader.a.items()}))
            finally:
                reader.close()
        self.epochs = source_epochs(reference, self.metadata, dictionary)
        self.valid = True

    def assert_live(self):
        if not self.valid:
            raise ValueError('Closed complete address source')
        for item in self.epochs:
            if list(fingerprint(item['reference']['path'])) != item['fingerprint']:
                self.valid = False
                raise ValueError('Complete address source epoch drift')

    def _open(self, index):
        if not self.valid or self.active is not None:
            raise ValueError('Single live address interval required')
        checked = self.checked[index]
        self._check_part(checked)
        reader = ReversePart.__new__(ReversePart)
        reader.stats = self.intervals[index]
        reader.a = {}
        try:
            for name, ref in reader.stats['artifacts'].items():
                key = name.removesuffix('.npy')
                a = reader.a[key] = np.load(ref['path'], mmap_mode='r', allow_pickle=False)
                if not isinstance(a, np.memmap) or a.flags.writeable or (a.dtype, a.shape) != checked['specs'][key]:
                    raise ValueError('Exact unchanged readonly address arrays required')
            reader.book = pq.ParquetFile(reader.stats['numbers']['path']).read()['number_text'].to_pylist()
            self._check_part(checked)
            self.active = reader
            return reader
        except BaseException:
            reader.close()
            raise

    def _check_part(self, checked):
        for path, epoch in checked['epochs'].items():
            if fingerprint(path) != epoch:
                self.valid = False
                raise ValueError('Original address interval epoch drift')

    def bind(self, snapshot, query, fact_chunk=65536):
        self.assert_live()
        if (not isinstance(snapshot, AliasSnapshot) or snapshot.aliases != self.aliases
                or not isinstance(query, Query) or not isinstance(fact_chunk, int)
                or isinstance(fact_chunk, bool) or fact_chunk <= 0):
            raise ValueError('Whole identical address snapshot and query required')
        return BoundAddressFacts(self, snapshot, query, fact_chunk)

    def verify_after(self):
        from scripts.compile_retrieval_v155_address_access import config, store
        self.assert_live()
        for item in self.epochs:
            verify(item['reference'])
        metadata, _ = store(self.reference, config())
        if metadata != self.metadata:
            raise ValueError('Same complete original address authority required')
        self.assert_live()

    def close(self):
        if self.active is not None:
            self.active.close()
            self.active = None
        self.valid = False


class BoundAddressFacts:
    """A single immutable query binding; site chunks may cross CSR partitions."""
    def __init__(self, source, snapshot, query, fact_chunk):
        self.source = source
        self.snapshot = snapshot
        self.query = query
        self.fact_chunk = fact_chunk
        self.sites = source.sites
        self.consumed = 0
        self.cursor = 0

    def batches(self, ids, query):
        source = self.source
        if (query is not self.query or not source.valid or ids.dtype != np.uint32
                or ids.ndim != 1 or not len(ids) or ids[0] != self.cursor or ids[-1] >= self.sites
                or not np.array_equal(ids, np.arange(self.cursor, self.cursor + len(ids), dtype=np.uint32))):
            raise ValueError('Same query and complete ordered site traversal required')
        index = bisect_right(source.starts, self.cursor) - 1
        cursor = self.cursor
        stop = int(ids[-1]) + 1
        while cursor < stop:
            stats = source.intervals[index]
            hi = min(stop, stats['stop'])
            reader = source._open(index)
            try:
                part_ids = ids[cursor-self.cursor:hi-self.cursor]
                for batch in reader.batches(part_ids, self.snapshot, self.fact_chunk):
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

    def finish(self):
        self.source.assert_live()
        if self.cursor != self.sites or self.consumed != self.source.facts:
            raise ValueError('Every original site/address fact required before initial100 publication')


def initial100(query, value, names_source, name_index, address_index,
               address_facts, attributes, activity=None, chunk=65536, max_seconds=20.):
    """One total deadline, official proof snapshots then exact whole-universe100.

    Source factories/launches must bind the frozen OOF model and query-only
    inputs separately. This function accepts no GT, old candidates or ranker.
    No partial result escapes a source drift, incomplete traversal or timeout.
    """
    if (not isinstance(query, Query) or any(not isinstance(getattr(query, k), str)
            for k in ('name', 'address', 'postcode', 'insee', 'number'))
            or not isinstance(query.reliable_geo, bool) or isinstance(max_seconds, bool)
            or not np.isfinite(max_seconds) or max_seconds <= 0
            or not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0):
        raise ValueError('Typed CRM query and positive total deadline required')
    point(query.xy)
    value = weights(value)
    catalog = names_source.catalog
    if (catalog.sites != attributes.sites or catalog.sites != address_facts.sites
            or address_index.source != address_facts.reference
            or address_facts.catalog_reference != names_source.metadata['sources']['catalog']):
        raise ValueError('Same complete source-bound site/catalog/address namespace required')
    began = time.monotonic()

    def remaining():
        seconds = max_seconds - (time.monotonic() - began)
        if seconds <= 0:
            raise TimeoutError('STOP_COMPLETE_INITIAL100_TOTAL_COST')
        return seconds

    address_facts.assert_live()
    names, name_stats = names_source.query(name_index, query.name, max_seconds=remaining(), chunk=chunk)
    snapshot, address_stats = address_index.snapshot(query.address, max_seconds=remaining(), chunk=chunk)
    bound = address_facts.bind(snapshot, query, fact_chunk=chunk)
    try:
        rows, stats = search(query, value, catalog, names, attributes, bound, activity,
                             limit=100, chunk=chunk, max_seconds=remaining())
        bound.finish()
        remaining()
    finally:
        # A failed/abandoned generator must not retain an mmap into the corpus.
        if address_facts.active is not None:
            address_facts.active.close()
            address_facts.active = None
    return rows, dict(stats, name_query=name_stats, address_query=address_stats,
                     total_query_seconds=time.monotonic()-began,
                     all_original_address_facts_consumed=True, recall_measured=False,
                     slo_validated=False, ready_for_fold1=False)


class OfficialInitial100:
    """Explicit full-source factory, NOT a measured resource/recall certificate.

    No fallback to a smaller catalog or a partial V154/V155 root. The caller
    MUST run any national probe under the approved external memory/pressure
    supervisor; the old dense name stage is known to exceed a 1GiB worker.
    Do not launch this factory unguarded, or imply this integration solved RSS.
    """
    def __init__(self, *, names, joins, structured, addresses):
        from .retrieval_address_access_v155 import NationalAddresses
        from .retrieval_compact_coverage_v156 import CompactNamePages
        from .retrieval_rebound_names_v150 import ReboundNames
        from .retrieval_site_attributes_v139 import StructuredSites
        from .retrieval_initial_collector_v146 import AppendedAttributes
        self.valid = False
        self.owned = []
        self.structured = None
        self.references = [names, joins, structured, addresses]
        try:
            for ref in self.references:
                verify(ref)
            self.name_index = CompactNamePages(names)
            self.owned.append(self.name_index)
            self.names_source = ReboundNames(joins)
            self.owned.append(self.names_source)
            if (self.name_index.source != self.names_source.source
                    or self.name_index.names != self.names_source.names):
                raise ValueError('Whole original shared name/site/address authorities required')
            self.address_index = NationalAddresses(addresses)
            self.owned.append(self.address_index)
            self.address_facts = CompleteAddressFacts(self.address_index.source)
            self.owned.append(self.address_facts)
            self.structured = StructuredSites(structured)
            self.attributes = AppendedAttributes(self.structured, self.names_source.keys)
            if (self.name_index.source != self.names_source.source
                    or self.name_index.names != self.names_source.names
                    or self.address_facts.catalog_reference != self.names_source.metadata['sources']['catalog']
                    or self.address_facts.sites != self.names_source.sites
                    or self.address_index.aliases != self.address_facts.aliases):
                raise ValueError('Whole original shared name/site/address authorities required')
            # Only explicit official authority fields, never recursively follow
            # arbitrary JSON or activity training-proof/CRM references.
            self.references.extend([self.name_index.metadata['launch'],
                                    self.name_index.metadata['source_stage'], self.name_index.source])
            self.references.extend(self.names_source.refs)
            for item in self.name_index.metadata['access']:
                self.references.extend(item['artifacts'].values())
            m = self.structured.metadata
            self.references.extend([m['launch'], m['number_dictionary'], m['activity_catalog'],
                                    *m['artifacts'].values(), *m['reused_arrays'].values(), *m['sources'].values()])
            cat = json.loads(verify(m['activity_catalog']).read_text())
            self.references.append(cat['official_labels'])
            for ref in (self.name_index.metadata['launch'], m['launch']):
                self.references.extend(json.loads(verify(ref).read_text())['implementation'].values())
            self.epochs = {r['path']: fingerprint(verify(r)) for r in self.references}
            self.valid = True
        except BaseException:
            self.close()
            raise

    def _live(self):
        if not self.valid:
            raise ValueError('Closed official initial100 integration')
        for path, epoch in self.epochs.items():
            if fingerprint(path) != epoch:
                self.valid = False
                raise ValueError('Official initial100 source epoch drift')

    def query(self, query, value, activity=None, chunk=65536, max_seconds=20.):
        if isinstance(max_seconds, bool) or not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Positive total official query deadline required')
        began = time.monotonic()
        self._live()
        remaining = max_seconds-(time.monotonic()-began)
        if remaining <= 0:
            raise TimeoutError('STOP_COMPLETE_INITIAL100_TOTAL_COST')
        rows, stats = initial100(query, value, self.names_source, self.name_index,
                                 self.address_index, self.address_facts, self.attributes,
                                 activity, chunk, remaining)
        self._live()
        elapsed = time.monotonic()-began
        if elapsed > max_seconds:
            raise TimeoutError('STOP_COMPLETE_INITIAL100_TOTAL_COST')
        return rows, dict(stats, proof_collection_seconds=stats['total_query_seconds'],
                          total_query_seconds=elapsed, epoch_guards_included_in_total_time=True)

    def verify_after(self):
        self._live()
        self.names_source.verify_after()
        self.address_index.verify_after()
        self.address_facts.verify_after()
        for ref in self.references:
            verify(ref)
        self._live()

    def close(self):
        self.valid = False
        for reader in reversed(self.owned):
            reader.close()
        self.owned = []
        if self.structured is not None:
            for a in (*self.structured.a.values(), *self.structured.reused.values()):
                a._mmap.close()
            self.structured = None

    def __enter__(self):
        self._live()
        return self

    def __exit__(self, *unused):
        self.close()
