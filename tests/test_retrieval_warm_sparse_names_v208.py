"""Complete bit-exact source/kernel fixtures only; no real label reads."""
import os
import numpy as np
import pytest

from src.xgb_matcher import retrieval_warm_sparse_names_v208 as fast
from src.xgb_matcher import retrieval_rebound_names_v150 as scalar
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector
from src.xgb_matcher.retrieval_grouped_proof_updates_v204 import GroupedProofVector
from test_retrieval_paged_names_v170 import TokenIndex
from test_retrieval_rebound_names_v150 import setup


@pytest.mark.parametrize('chunk', [1, 3, 131072, 524288])
@pytest.mark.parametrize('text', ['ALPHA BETA', 'ALPHA', 'MISSING WORD', ''])
@pytest.mark.parametrize('ordering', ['natural_density', 'forced_sparse_order'])
def test_all_vectors_gather_counts_and_native_maximum_equal150(tmp_path, monkeypatch, chunk, text, ordering):
    if ordering == 'forced_sparse_order':
        monkeypatch.setattr(fast, 'MIN_ORDER_ROWS', 1)
        monkeypatch.setattr(fast, 'SPARSE_SPAN_PER_REFERENCE', 0)
    ref, unused = setup(tmp_path, monkeypatch)
    class Index(TokenIndex):
        def __init__(self, source):
            super().__init__(['ALPHA BETA', 'ALPHA', 'BETA', 'BETA ALPHA', '', 'ALPHA BETA']); self.source = source
        def coverage(self, name, **kwargs): return NationalNames.coverage(self, name, **kwargs)
    def forbidden(*args): raise AssertionError('Rejected grouped maximum called')
    monkeypatch.setattr(GroupedProofVector, 'maximum', forbidden)
    old = scalar.ReboundNames(ref); reader = fast.WarmOrderedNames(ref)
    try:
        expected, trace = old.query(Index(old.source), text, chunk=chunk)
        events = []
        got, stats = reader.query(Index(reader.source), text, tmp_path / 'new', chunk=chunk, progress=events.append)
        for x, y in zip(got.vectors, (expected.snapshot.parent_cov, expected.snapshot.parent_exact,
                expected.snapshot.previous.site_cov, expected.snapshot.previous.site_exact, expected.snapshot.previous.relation_cov)):
            assert x.read(0, len(x)).tobytes() == y.tobytes()
        ids = np.array([4, 0, 2, 1, 3, 0], dtype=np.uint32)
        for x, y in zip(got.gather(ids), expected.gather(ids)): assert x.tobytes() == y.tobytes()
        for key in ('references_processed', 'original_parent_proofs', 'appended_parent_proofs', 'direct_site_proofs',
                    'related_endpoint_proofs', 'appended_sites_with_name_proofs', 'appended_sites_with_direct_exact'):
            assert stats[key] == trace[key]
        profile = stats['update_profile']
        maximum_refs = sum(r['maximum_references'] for r in profile.values())
        flags_refs = sum(r['flags_references'] for r in profile.values())
        assert maximum_refs == flags_refs <= stats['references_processed']
        assert all(r['maximum_seconds'] >= 0 and r['flags_seconds'] >= 0 for r in profile.values())
        for event in events:
            if event['stage'] == 'OWNER_BUCKET_COMPLETE': assert set(event['update_profile']) == set(profile)
        reader.verify_after(); got.close(remove=True)
    finally: reader.close(); old.close()


def test_false_flags_validated_and_original_signed_zero_maximum_retained(tmp_path):
    old = DiskVector(tmp_path / 'old.npy', 3, np.float64)
    new = fast.WarmOrderedVector(tmp_path / 'new.npy', 3, np.float64)
    flags = fast.WarmOrderedVector(tmp_path / 'flags.npy', 3, np.bool_)
    ids = np.array([0, 0, 0], dtype=np.uint32); values = np.array([-0., 0., -0.], dtype=np.float64)
    try:
        old.maximum(ids, values); new.maximum(ids, values)
        assert old.read(0, 3).tobytes() == new.read(0, 3).tobytes()
        assert new.update_profile['maximum_calls'] == 1 and new.update_profile['maximum_references'] == 3
        flags.flags(ids, np.array([False, True, False], dtype=np.bool_))
        assert flags.read(0, 3).tolist() == [True, False, False]
        assert flags.update_profile['true_flag_references'] == 1
        with pytest.raises(ValueError): flags.flags(np.array([3], dtype=np.uint32), np.array([False], dtype=np.bool_))
        assert flags.update_profile['flags_calls'] == 1
    finally: old.close(); new.close(); flags.close()


@pytest.mark.parametrize('chunk', [1, 3, 131072])
@pytest.mark.parametrize('dtype', [np.uint32, np.int64])
def test_stable_sparse_order_preserves_duplicate_signed_zero_sequence_across_calls(tmp_path, monkeypatch, chunk, dtype):
    monkeypatch.setattr(fast, 'MAX_UPDATE_CHUNK', chunk)
    monkeypatch.setattr(fast, 'MIN_ORDER_ROWS', 1)
    monkeypatch.setattr(fast, 'SPARSE_SPAN_PER_REFERENCE', 0)
    old = DiskVector(tmp_path / 'old.npy', 13, np.float64)
    new = fast.WarmOrderedVector(tmp_path / 'new.npy', 13, np.float64)
    try:
        assert old.read(0, 13).tobytes() == new.read(0, 13).tobytes()
        ids = np.array([12, 0, 3, 12, 0, 3, 12, 0, 3], dtype=dtype)
        for values in ([0., -0., -0., -0., 0., 0., 0., -0., -0.], [-0., 0., 0., 0., -0., -0., -0., 0., 0.],
                       [.5, .1, .3, .1, .9, .1, .2, .3, .2]):
            scores = np.array(values, dtype=np.float64)
            old.maximum(ids, scores); new.maximum(ids, scores)
            assert old.read(0, 13).tobytes() == new.read(0, 13).tobytes()
        assert new.update_profile['zero_initialized_positions'] == 13
        assert new.update_profile['ordered_references'] == 27
    finally: old.close(); new.close()


def test_initialization_is_bounded_and_charged_to_original_guard(tmp_path):
    calls = []
    def guard():
        calls.append(True)
        if len(calls) == 2: raise TimeoutError('STOP_ORIGINAL_GLOBAL_QUERY_GUARD')
    with pytest.raises(TimeoutError):
        fast.WarmOrderedVector(tmp_path / 'own.npy', fast.MAX_UPDATE_CHUNK+7, np.float64, guard=guard)
    assert len(calls) == 2
    a = np.load(tmp_path / 'own.npy', mmap_mode='r', allow_pickle=False)
    try: assert a.shape == (fast.MAX_UPDATE_CHUNK+7,) and a.dtype == np.float64 and not np.any(a)
    finally: a._mmap.close()


@pytest.mark.parametrize('bad', ['negative', 'over1', 'nan', 'shape', 'dtype', 'id', 'closed', 'changed_file'])
def test_all_original_maximum_guards_survive_physical_optimization(tmp_path, bad):
    v = fast.WarmOrderedVector(tmp_path / 'own.npy', 3, np.float64)
    ids = np.array([0], dtype=np.uint32); values = np.array([.5], dtype=np.float64)
    if bad == 'negative': values[0] = -.1
    elif bad == 'over1': values[0] = 1.1
    elif bad == 'nan': values[0] = np.nan
    elif bad == 'shape': values = np.array([.5, .5], dtype=np.float64)
    elif bad == 'dtype': values = values.astype(np.float32)
    elif bad == 'id': ids[0] = 3
    elif bad == 'closed': v.close()
    else:
        a = np.load(v.path, mmap_mode='r+', allow_pickle=False); a[0] = .25; a._mmap.close()
        st = v.path.stat()
        os.utime(v.path, ns=(st.st_atime_ns, st.st_mtime_ns+1000000000))  # Explicit fixture epoch drift.
    try:
        with pytest.raises(ValueError): v.maximum(ids, values)
    finally: v.close()


def test_same_epoch_payload_change_still_requires_full_hash_not_stat_trust(tmp_path):
    from src.xgb_matcher.retrieval_v20_workbench import record, verify
    v = fast.WarmOrderedVector(tmp_path / 'own.npy', 3, np.float64)
    ref = record(v.path); st = v.path.stat()
    try:
        a = np.load(v.path, mmap_mode='r+', allow_pickle=False); a[0] = .25; a._mmap.close()
        os.utime(v.path, ns=(st.st_atime_ns, st.st_mtime_ns))
        with pytest.raises(ValueError): verify(ref)  # Content digest mandatory, even unchanged size/mtime.
    finally: v.close()
