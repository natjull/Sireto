"""Complete bit-exact source/kernel fixtures only; no real label reads."""
import numpy as np
import pytest

from src.xgb_matcher import retrieval_flags_only_names_v205 as fast
from src.xgb_matcher import retrieval_rebound_names_v150 as scalar
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector
from src.xgb_matcher.retrieval_grouped_proof_updates_v204 import GroupedProofVector
from test_retrieval_paged_names_v170 import TokenIndex
from test_retrieval_rebound_names_v150 import setup


@pytest.mark.parametrize('chunk', [1, 3, 131072, 524288])
@pytest.mark.parametrize('text', ['ALPHA BETA', 'ALPHA', 'MISSING WORD', ''])
def test_all_vectors_gather_counts_and_native_maximum_equal150(tmp_path, monkeypatch, chunk, text):
    ref, unused = setup(tmp_path, monkeypatch)
    class Index(TokenIndex):
        def __init__(self, source):
            super().__init__(['ALPHA BETA', 'ALPHA', 'BETA', 'BETA ALPHA', '', 'ALPHA BETA']); self.source = source
        def coverage(self, name, **kwargs): return NationalNames.coverage(self, name, **kwargs)
    def forbidden(*args): raise AssertionError('Rejected grouped maximum called')
    monkeypatch.setattr(GroupedProofVector, 'maximum', forbidden)
    old = scalar.ReboundNames(ref); reader = fast.FlagsOnlyNames(ref)
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
    new = fast.FlagsOnlyVector(tmp_path / 'new.npy', 3, np.float64)
    flags = fast.FlagsOnlyVector(tmp_path / 'flags.npy', 3, np.bool_)
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
