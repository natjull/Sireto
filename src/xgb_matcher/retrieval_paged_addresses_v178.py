"""Complete V177 address access and immutable file-backed V147 snapshots.

Same V170/V138 whole-alias arithmetic, no candidate pool or proof fanout.
Unmeasured nationally: a source-bound reader is not a recall/SLO certificate.
"""
import json
from contextlib import contextmanager
from pathlib import Path
import time
from types import SimpleNamespace

import duckdb
import numpy as np

from .retrieval_address_authority_v168 import parent_exit
from .retrieval_address_reverse_v147 import AliasSnapshot
from .retrieval_memory_policy_v153 import validate_report
from .retrieval_name_pages_v151 import fingerprint
from .retrieval_national_names_v138 import NationalNames, PackedTerms
from .retrieval_paged_names_v170 import DiskVector, coverage, quantum
from .retrieval_proof_bridge_v137 import PROTECTED
from .retrieval_v20_workbench import record, verify


class SnapshotVector(DiskVector):
    """Transfer ownership once; even the original producer cannot mutate it.

mtime alone cannot detect successive writes to an already dirty mmap page.
Therefore immutable snapshots must revoke the writable producer's lease.
"""
    @classmethod
    def transfer(cls, vector):
        if not isinstance(vector, DiskVector) or not vector.valid or vector.live_maps:
            raise ValueError('Complete unmapped proof vector ownership required')
        sealed = cls.__new__(cls)
        sealed.__dict__ = vector.__dict__.copy()
        vector.close()
        return sealed

    @contextmanager
    def mapped(self, write=False):
        if write:
            raise ValueError('Sealed address query vector is readonly')
        with super().mapped(False) as values:
            yield values


class ReadonlyDiskArray:
    """Small copied gathers; no borrowed maps and no implicit whole heap array."""
    def __init__(self, vector):
        if not isinstance(vector, DiskVector) or not vector.valid or vector.live_maps:
            raise ValueError('Live singly owned complete proof vector required')
        self.reference = record(vector.path)
        self._vector = SnapshotVector.transfer(vector)
        self._epoch = self._vector.epoch
        self.dtype = vector.dtype
        self.shape = (len(vector),)
        self.ndim = 1

    @property
    def flags(self):
        return SimpleNamespace(writeable=False)

    def __len__(self):
        return self.shape[0]

    def assert_live(self):
        if (not self._vector.valid or self._vector.epoch != self._epoch
                or fingerprint(self._vector.path) != self._epoch):
            raise ValueError('Immutable whole-address query snapshot drift')

    def __getitem__(self, ids):
        self.assert_live()
        if isinstance(ids, np.ndarray):
            return self._vector.take(ids)
        if isinstance(ids, slice):
            start, stop, step = ids.indices(len(self))
            if step != 1:
                raise ValueError('Explicit contiguous proof interval or typed gather required')
            return self._vector.read(start, max(start, stop))
        if isinstance(ids, (int, np.integer)) and not isinstance(ids, (bool, np.bool_)):
            pos = int(ids)
            if not 0 <= pos < len(self):
                raise ValueError('Actual whole-address alias position required')
            return self._vector.read(pos, pos + 1)[0]
        raise ValueError('Explicit actual whole-address alias positions required')

    def __array__(self, *args, **kwargs):
        raise ValueError('Do not materialize a whole proof vector implicitly')


class PagedAliasSnapshot(AliasSnapshot):
    """V147 interface, all aliases validated in chunks, readonly copied gathers.

No parent initializer: it would touch/materialize entire national vectors.
The same finite/range/exact=>coverage1 invariants are checked exhaustively.
"""
    def __init__(self, cov, exact, guard, chunk):
        quantum(chunk)
        if (not isinstance(cov, DiskVector) or not isinstance(exact, DiskVector)
                or cov.dtype != np.float64 or exact.dtype != np.bool_ or len(cov) != len(exact)):
            raise ValueError('Exact complete address vectors required')
        self.aliases = len(cov)
        self.coverage = ReadonlyDiskArray(cov)
        self.exact = ReadonlyDiskArray(exact)
        for lo in range(0, self.aliases, chunk):
            guard()
            hi = min(self.aliases, lo + chunk)
            values = self.coverage[lo:hi]
            flags = self.exact[lo:hi]
            if (not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                    or np.any(flags & (values != 1.))):
                raise ValueError('Every coherent whole-address alias proof required')
        guard()

    def assert_live(self):
        self.coverage.assert_live()
        self.exact.assert_live()

    def verify_after(self):
        self.assert_live()
        verify(self.coverage.reference)
        verify(self.exact.reference)
        self.assert_live()

    def close(self, remove=False):
        if remove and (self.coverage._vector.valid or self.exact._vector.valid):
            self.verify_after()
        for value in (self.coverage, self.exact):
            if value._vector.valid:
                if remove:
                    value.assert_live()
                value._vector.close(remove=remove)


class PagedAddresses:
    """Only a complete supervised V177 seal; all 256 original packs required."""
    def __init__(self, reference):
        from scripts import compile_retrieval_v177_packed_addresses as backend
        self.valid = False
        self.in_coverage = False
        self.access = {}
        self.files = {}
        self.references = []
        self.c = None
        self.backend = backend
        try:
            path = verify(reference)
            root = path.parent
            if path.resolve() != root / 'manifest.json':
                raise ValueError('Exact complete supervised address seal required')
            l, p, e = backend.load(root)
            supervision = parent_exit(root / 'seal')
            m = json.loads(path.read_text())
            validate_report(m['seal_resources'])
            if (m['status'] != backend.STATUS or m['launch'] != record(root / 'launch.json')
                    or any(m[k] != p[k] for k in ('staging', 'source_store', 'dictionary', 'catalog',
                                                  'aliases', 'sites', 'facts', 'postings'))
                    or m['complete_retrieval_index'] or m['ready_for_fold1']
                    or any(m.get(k) is not False for k in PROTECTED)):
                raise ValueError('Same protected entire official address source required')
            self.aliases = self.names = p['aliases']
            self.source = p['source_store']
            self.buckets = l['config']['term_buckets']
            expected = []
            phase_refs = []
            authority_refs = [reference, supervision, record(root / 'seal/preflight.json')]
            posts = terms = size = 0
            for ch in range(4):
                for b in range(self.buckets):
                    phase = backend.phase_record(root, ch, b, context=(l, p, e))
                    s = phase['stats']
                    dest = backend.directory(root, ch, b)
                    ref = record(dest / 'manifest.json')
                    phase_refs.append(ref)
                    authority_refs.extend([ref, parent_exit(dest), phase['preflight']])
                    expected.append(dict(channel=ch, bucket=b, postings=s['postings'], terms=s['terms'],
                                         artifacts=phase['artifacts']))
                    arrays = {}
                    access = PackedTerms.__new__(PackedTerms)
                    access.a = arrays
                    access.terms = 0
                    self.access[ch, b] = access
                    specs = dict(post_ids=(np.uint32, s['postings']), post_offsets=(np.uint64, s['terms']+1),
                                 token_offsets=(np.uint64, s['terms']+1), token_data=(np.uint8, s['token_bytes']))
                    for name, (dtype, n) in specs.items():
                        ref = phase['artifacts'][name+'.npy']  # Already fully hashed by phase_record.
                        epoch = fingerprint(ref['path'])
                        a = np.load(ref['path'], mmap_mode='r', allow_pickle=False)
                        arrays[name] = a
                        if (not isinstance(a, np.memmap) or a.flags.writeable or a.dtype != dtype
                                or a.shape != (n,) or fingerprint(ref['path']) != epoch):
                            raise ValueError('Every unchanged unsigned readonly whole pack required')
                        self.files.setdefault((ch, b), {})[name] = dict(
                            path=Path(ref['path']), fingerprint=epoch, dtype=a.dtype, shape=a.shape)
                        self.references.append(ref)
                    if (arrays['post_offsets'][0] != 0 or arrays['post_offsets'][-1] != s['postings']
                            or arrays['token_offsets'][0] != 0 or arrays['token_offsets'][-1] != s['token_bytes']):
                        raise ValueError('Same whole original token/posting CSR required')
                    self.access[ch, b].terms = s['terms']
                    posts += s['postings']
                    terms += s['terms']
                    size += phase['output_bytes']
            if (expected != m['access'] or phase_refs != m['phase_authorities']
                    or (posts, terms, size) != (m['postings'], m['terms'], m['output_bytes'])):
                raise ValueError('Every original supervised complete pack once required')
            self.launch, self.epochs = l, e
            self.authorities = {r['path']: r for r in authority_refs}
            self.metadata = m
            self.c = duckdb.connect()
            self.c.execute('SET threads=1')
            self.c.execute("SET memory_limit='32MiB'")
            self.valid = True
            self.check_live()
        except BaseException:
            self.close()
            raise

    def check_live(self):
        if not self.valid:
            raise ValueError('Closed or failed complete address reader')
        try:
            self.backend.check_live(self.launch, None, self.epochs)
            for ref in self.authorities.values():
                verify(ref)
            for files in self.files.values():
                for spec in files.values():
                    if fingerprint(spec['path']) != spec['fingerprint']:
                        raise ValueError('Every complete original address pack epoch must remain unchanged')
        except BaseException:
            self.valid = False
            raise

    def posting(self, channel, token):
        if not self.valid:
            raise ValueError('Closed or failed complete address reader')
        values = NationalNames.posting(self, channel, token)
        return values if self.in_coverage else values.copy()

    def snapshot(self, text, directory, max_seconds=20., chunk=524288):
        quantum(chunk)
        if (not self.valid or self.in_coverage or not isinstance(text, str)
                or isinstance(max_seconds, bool) or not np.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError('Single live typed complete address query required')
        began = time.monotonic()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        def guard():
            if time.monotonic() - began > max_seconds:
                raise TimeoutError('STOP_COMPLETE_ADDRESS_QUERY_COST')
        cov = exact = snapshot = None
        self.in_coverage = True
        try:
            self.check_live()
            guard()
            cov, exact, stats = coverage(self, text, directory, guard, chunk)
            snapshot = PagedAliasSnapshot(cov, exact, guard, chunk)
            self.check_live()
            guard()
            return snapshot, dict(stats, total_query_seconds=time.monotonic()-began,
                                  scope='WHOLE_ADDRESS_ALIAS_ONLY', every_original_alias_checked=True,
                                  query_snapshot_file_backed=True, site_or_parent_proof_propagated=False,
                                  candidates_emitted=False, recall_measured=False, slo_validated=False,
                                  ready_for_fold1=False)
        except BaseException:
            self.valid = False
            if snapshot is not None:
                snapshot.close()
            for value in (cov, exact):
                if value is not None and value.valid:
                    value.close()  # Failed query files remain, no partial snapshot escapes.
            raise
        finally:
            self.in_coverage = False

    def verify_after(self):
        try:
            self.check_live()
            for ref in self.references:
                verify(ref)
            self.check_live()
        except BaseException:
            self.valid = False
            raise

    def close(self):
        self.valid = False
        if self.c is not None:
            self.c.close()
            self.c = None
        for access in self.access.values():
            for a in access.a.values():
                a._mmap.close()
            access.a = {}
