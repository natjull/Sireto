"""Bit-exact frozen one-hop; every endpoint/observation, not capped sources."""
from pathlib import Path

import numpy as np
import pytest

from src.xgb_matcher import retrieval_bounded_name_relations_v201 as bounded
from src.xgb_matcher import retrieval_rebound_names_v150 as scalar
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from test_retrieval_paged_names_v170 import TokenIndex
from test_retrieval_rebound_names_v150 import setup


@pytest.mark.parametrize('chunk', [1, 3, 131072, 524288])
def test_all_parent_site_and_relation_proofs_equal_independent_scalar150(tmp_path, monkeypatch, chunk):
    ref, unused = setup(tmp_path, monkeypatch)
    class Index(TokenIndex):
        def __init__(self, source):
            super().__init__(['ALPHA BETA', 'ALPHA', 'BETA', 'BETA ALPHA', '', 'ALPHA BETA']); self.source = source
        def coverage(self, text, **kwargs): return NationalNames.coverage(self, text, **kwargs)
    old = scalar.ReboundNames(ref); reader = bounded.BoundedRelationNames(ref)
    try:
        expected, trace = old.query(Index(old.source), 'ALPHA BETA', chunk=chunk)
        got, stats = reader.query(Index(reader.source), 'ALPHA BETA', tmp_path / 'new', chunk=chunk)
        for x, y in zip(got.vectors, (expected.snapshot.parent_cov, expected.snapshot.parent_exact,
                expected.snapshot.previous.site_cov, expected.snapshot.previous.site_exact, expected.snapshot.previous.relation_cov)):
            np.testing.assert_array_equal(x.read(0, len(x)), y)
        ids = np.array([4, 0, 2, 1, 3, 0], dtype=np.uint32)
        for x, y in zip(got.gather(ids), expected.gather(ids)): np.testing.assert_array_equal(x, y)
        for key in ('references_processed', 'original_parent_proofs', 'appended_parent_proofs', 'direct_site_proofs',
                    'related_endpoint_proofs', 'appended_sites_with_name_proofs', 'appended_sites_with_direct_exact'):
            assert stats[key] == trace[key]
        assert stats['relation_physical_chunk'] == min(chunk, bounded.MAX_RELATION_CHUNK)
        assert stats['no_alias_or_owner_or_member_cap'] and not stats['site_alias_lifted_to_parent']
        reader.verify_after(); got.close(remove=True)
    finally: reader.close(); old.close()


def test_relation_chunk_bound_is_physical_and_complete_across_last_partial_block(tmp_path):
    n = bounded.MAX_RELATION_CHUNK+7
    graph = dict(graph_site_positions=np.arange(n, dtype=np.uint32),
        graph_parent_positions=np.arange(n, dtype=np.uint32)%3,
        predecessor_graph_positions=np.arange(n, dtype=np.uint32),
        successor_graph_positions=np.roll(np.arange(n, dtype=np.uint32), -1),
        continuity_codes=(np.arange(n, dtype=np.uint32)%3).astype(np.uint8))
    graph['successor_graph_positions'][0] = 0  # self edge excluded in both algorithms
    pc = bounded.DiskVector(tmp_path / 'parent.npy', 3, np.float64)
    sc = bounded.DiskVector(tmp_path / 'site.npy', n, np.float64)
    parent = np.array([.1, .3, .7], dtype=np.float64)
    site = np.zeros(n, dtype=np.float64); site[[0, 1, n-1]] = [1., .6, .9]
    pc.assign(0, parent); sc.assign(0, site)
    calls = []; seen = {k: set() for k in graph}
    def read(key, ids):
        calls.append((key, len(ids))); seen[key].update(map(int, ids)); return graph[key][ids].copy()
    out = None
    try:
        out = bounded.one_hop(pc, sc, read, n, n, tmp_path, lambda: None, 524288)
        snapshot = np.maximum(site, parent[graph['graph_parent_positions']])
        pred = graph['predecessor_graph_positions']; succ = graph['successor_graph_positions']; mask = pred != succ
        grade = np.where(graph['continuity_codes'][mask] == 2, .5, .25)
        expected = np.zeros(n, dtype=np.float64)
        np.maximum.at(expected, pred[mask], snapshot[succ[mask]]*grade)
        np.maximum.at(expected, succ[mask], snapshot[pred[mask]]*grade)
        np.testing.assert_array_equal(out.read(0, n), expected)
        assert all(size <= bounded.MAX_RELATION_CHUNK for key, size in calls)
        assert all(values == set(range(n)) for key, values in seen.items())
        assert not (tmp_path / 'endpoint_snapshot.npy').exists()
    finally:
        if out is not None: out.close()
        pc.close(); sc.close()


def test_relation_failure_returns_no_partial_vectors_and_retains_own_evidence(tmp_path):
    pc = bounded.DiskVector(tmp_path / 'parent.npy', 1, np.float64)
    sc = bounded.DiskVector(tmp_path / 'site.npy', 1, np.float64)
    def fail(): raise TimeoutError('EXPLICIT_TEST_STOP')
    try:
        with pytest.raises(TimeoutError): bounded.one_hop(pc, sc, lambda k, i: i, 1, 1, tmp_path, fail, 524288)
        assert (tmp_path / 'endpoint_snapshot.npy').exists() and (tmp_path / 'relation_coverage.npy').exists()
    finally: pc.close(); sc.close()
