"""Bit-exact tiny native proofs, no national cost or label/recall data."""
from pathlib import Path

import numpy as np
import pytest

from src.xgb_matcher import retrieval_ram_name_counters_v198 as ram
from src.xgb_matcher import retrieval_paged_names_v170 as paged
from src.xgb_matcher import retrieval_rebound_names_v150 as original
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from test_retrieval_paged_names_v170 import TokenIndex
from test_retrieval_rebound_names_v150 import setup


@pytest.mark.parametrize('chunk', [1, 3, 131072, 524288])
@pytest.mark.parametrize('text', ['ALPHA BETA', 'RUE JEAN JAURES', 'A', '', 'A & B SOCIETE',
                                 ' '.join('TOKEN'+str(i) for i in range(300))])
def test_full_token_namespace_float64_and_bool_bit_exact_to_independent138(tmp_path, chunk, text):
    texts = ['ALPHA BETA', 'BETA ALPHA', 'ALPHA', 'RUE JEAN JAURES', 'RUE JEAN JAURES', 'A', '', 'TOKEN1 TOKEN2']
    index = TokenIndex(texts); native = TokenIndex(texts)
    a, b, stats = ram.coverage(index, text, tmp_path, lambda: None, chunk)
    x, y, old = NationalNames.coverage(native, text, chunk=chunk)
    try:
        np.testing.assert_array_equal(a.read(0, len(texts)), x); np.testing.assert_array_equal(b.read(0, len(texts)), y)
        assert stats['channels'] == old['channels'] and stats['positive_aliases'] == old['positive_aliases']
        assert stats['exact_aliases'] == old['exact_aliases'] and index.calls == native.calls
        assert stats['counter_backing'] == 'RAM_INTEGRAL_ONLY_FULL_NAMESPACE'
        assert stats['counter_ram_bytes'] <= ram.MAX_COUNTER_RAM_BYTES and stats['postings_or_alias_cap'] is None
        assert not (tmp_path / 'token_counts.npy').exists()
    finally: a.close(); b.close()


def test_larger_counter_uses_old_exact_disk_without_token_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(ram, 'MAX_COUNTER_RAM_BYTES', 1)
    a, b, stats = ram.coverage(TokenIndex(['ALPHA BETA', 'BETA']), 'ALPHA BETA', tmp_path / 'fallback', lambda: None, 1)
    c, d, old = paged.coverage(TokenIndex(['ALPHA BETA', 'BETA']), 'ALPHA BETA', tmp_path / 'old', lambda: None, 1)
    try:
        np.testing.assert_array_equal(a.read(0, 2), c.read(0, 2)); np.testing.assert_array_equal(b.read(0, 2), d.read(0, 2))
        assert stats['counter_backing'] == 'ORIGINAL_DISK_EXACT_FALLBACK' and stats['counter_ram_bytes'] == 0
        assert stats['channels'] == old['channels']
    finally:
        for x in (a, b, c, d): x.close()


@pytest.mark.parametrize('values', [np.array([0, 0], dtype=np.uint32), np.array([2], dtype=np.uint32),
    np.array([0], dtype=np.int64), np.array([1, 0], dtype=np.uint32), np.array([[0]], dtype=np.uint32)])
def test_corrupt_postings_never_get_approximated(tmp_path, values):
    class Bad(TokenIndex):
        def posting(self, *args): return values
    with pytest.raises(ValueError, match='sorted unique'):
        ram.coverage(Bad(['ALPHA', 'BETA']), 'ALPHA', tmp_path, lambda: None, 1)


def test_too_small_counter_dtype_cannot_saturate(tmp_path, monkeypatch):
    monkeypatch.setattr(ram, 'counter_dtype', lambda n: np.dtype(np.uint8))
    with pytest.raises(ValueError, match='FULL query bound'):
        ram.coverage(TokenIndex(['A']), ' '.join('TOKEN'+str(i) for i in range(300)), tmp_path, lambda: None, 1)


def test_timeout_returns_no_partial_proof_and_preserves_owned_evidence(tmp_path):
    def fail(): raise TimeoutError('STOP_PAGED_NAME_QUERY_COST')
    with pytest.raises(TimeoutError): ram.coverage(TokenIndex(['ALPHA']), 'ALPHA', tmp_path, fail, 1)
    assert (tmp_path / 'alias_coverage.npy').exists() and (tmp_path / 'alias_exact.npy').exists()


@pytest.mark.parametrize('chunk', [1, 3, 524288])
def test_full_parent_site_patches_and_onehop_equal_independent150(tmp_path, monkeypatch, chunk):
    ref, unused = setup(tmp_path, monkeypatch)
    texts = ['ALPHA BETA', 'ALPHA', 'BETA', 'BETA ALPHA', '', 'ALPHA BETA']
    class Index(TokenIndex):
        def __init__(self, source): super().__init__(texts); self.source = source
        def coverage(self, text, **kwargs): return NationalNames.coverage(self, text, **kwargs)
    scalar = original.ReboundNames(ref); reader = ram.RamCountersNames(ref)
    try:
        expected, old = scalar.query(Index(scalar.source), 'ALPHA BETA', chunk=chunk)
        progress = []
        actual, stats = reader.query(Index(reader.source), 'ALPHA BETA', tmp_path / 'query', chunk=chunk, progress=progress.append)
        ids = np.array([4, 0, 2, 1, 3, 0], dtype=np.uint32)
        for x, y in zip(actual.gather(ids), expected.gather(ids)): np.testing.assert_array_equal(x, y)
        for x, y in zip(actual.vectors, (expected.snapshot.parent_cov, expected.snapshot.parent_exact,
            expected.snapshot.previous.site_cov, expected.snapshot.previous.site_exact, expected.snapshot.previous.relation_cov)):
            np.testing.assert_array_equal(x.read(0, len(x)), y)
        for k in ('references_processed', 'original_parent_proofs', 'appended_parent_proofs', 'appended_parent_exact',
                  'direct_site_proofs', 'related_endpoint_proofs', 'appended_sites_with_name_proofs', 'appended_sites_with_direct_exact'):
            assert stats[k] == old[k]
        assert progress[0]['stage'] == 'LEXICAL_COMPLETE' and progress[-1]['stage'] == 'ONE_HOP_COMPLETE'
        assert stats['lexical']['counter_backing'] == 'RAM_INTEGRAL_ONLY_FULL_NAMESPACE'
        reader.verify_after(); actual.close(remove=True)
    finally: reader.close(); scalar.close()
