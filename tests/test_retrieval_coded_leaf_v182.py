"""Independent immutable V146/V147 oracle, no national quality claim."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.xgb_matcher import retrieval_coded_leaf_v182 as workbench
from src.xgb_matcher.retrieval_certified_address_facts_v181 import CertifiedAddressFacts
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_initial_collector_v146 import AddressBatch, score_sites, search
from src.xgb_matcher.retrieval_site_attributes_v139 import encode_code
from src.xgb_matcher.retrieval_variants_v118 import normalized
from test_retrieval_certified_address_facts_v181 import fixture, snapshot
from test_retrieval_initial_collector_v146 import Attributes, catalog


@pytest.mark.parametrize('raw', ['', '7', ' 7 bis ', '１２', '１２３４５６７８９', '①②③④⑤⑥⑦⑧⑨', '123456789', '0', '[ND]'])
def test_number_dictionary_exact_scalar_and_never_renormalized(raw):
    book = ['', '7', '7 BIS', '12', '123456789', '0']
    match = workbench.number_lookup(book, raw)
    number = normalized(raw)
    expected = np.array([bool(number) and text == number for text in book])
    np.testing.assert_array_equal(match, expected)
    assert not match.flags.writeable
    if raw == '①②③④⑤⑥⑦⑧⑨':
        assert match[4]  # Normalizing the canonical dictionary again would lose it.


@pytest.mark.parametrize('book', [[], ['7'], ['', '7', '7'], ['', ' 7'], ['', '7  BIS'], ['', 'é'], ['', '[ND]']])
def test_dictionary_corruption_refused_not_repaired(book):
    with pytest.raises(ValueError):
        workbench.number_lookup(book, '7')


@pytest.mark.parametrize('raw_number', ['', '7 BIS', '１２３４５６７８９', '①②③④⑤⑥⑦⑧⑨', '123456789'])
@pytest.mark.parametrize('reliable', [False, True])
def test_8192_coherent_facts_all30_features_bitexact146(raw_number, reliable):
    rng = np.random.default_rng(20261004)
    n, facts = 257, 8192
    ids = np.arange(n, dtype=np.uint32)
    book = ['', '7', '7 BIS', '123456789', '8', '0']
    codes = rng.integers(0, len(book), facts, dtype=np.uint32)
    nc = rng.integers(0, 21, n).astype(np.float64)/20
    ne = np.where(nc == 1., 1., 0.)
    cover = rng.integers(0, 21, facts).astype(np.float64)/20
    exact = (cover == 1.) & (rng.integers(0, 2, facts) == 1)
    current = rng.integers(0, 2, facts).astype(np.bool_)
    positions = rng.integers(0, n, facts, dtype=np.uint32)
    cp_book = np.array([encode_code(v) for v in ('', '75001', '97110', '69001')], dtype=np.uint32)
    ins_book = np.array([encode_code(v) for v in ('', '2A004', '75056', '97101')], dtype=np.uint32)
    cp = cp_book[rng.integers(0, len(cp_book), facts)]
    ins = ins_book[rng.integers(0, len(ins_book), facts)]
    numbers = tuple(book[int(k)] for k in codes)
    scalar = AddressBatch(positions, cover, exact, current, cp, ins, numbers)
    coded = workbench.CodedAddressBatch(positions, cover, exact, current, cp, ins,
                                      workbench.number_lookup(book, raw_number)[codes])
    flags = tuple(rng.integers(0, 2, n).astype(np.float64) for _ in range(3))
    classes = rng.integers(-1, 7, n, dtype=np.int32)
    xy = rng.normal(0., 2000., (n, 2))
    xy[::5] = np.nan
    class Evidence:
        sites = n
        def flags(self, values, *unused):
            return tuple(v[values].copy() for v in flags)+(classes[values].copy(),)
        def coordinates(self, values):
            return xy[values].copy()
    q = Query(postcode='75001', insee='2A004', number=raw_number, xy=(12., 25.), reliable_geo=reliable)
    activity = {k: k/10 for k in range(7)}
    for w in (np.arange(1, 31, dtype=np.float64)/30, rng.uniform(.001, 64., 30)):
        a, sa = score_sites(ids, q, w, nc, ne, Evidence(), [scalar], activity)
        b, sb = workbench.score_sites(ids, q, w, nc, ne, Evidence(), [coded], activity)
        np.testing.assert_array_equal(a, b)
        assert sa == sb
        # Every input fact, even with zero lexical coverage, remains scored.
        assert sb['address_facts'] == facts and sb['proof_or_owner_cap'] is None


def test_real_tiny_original181_fulltraversal_and_initial100_equal146(tmp_path, monkeypatch):
    index, m, _ = fixture(tmp_path, monkeypatch)
    source = CertifiedAddressFacts(index)
    snap = snapshot(tmp_path)
    q = Query(postcode='75001', number='7')
    c = catalog(np.arange(100001, 100001+source.sites, dtype=np.uint64)[::-1])
    class Names:
        sites = source.sites
        def gather(self, ids):
            return (ids.astype(np.float64)/source.sites, np.zeros(len(ids)))
    try:
        for chunk in (1, 3, source.sites):
            original = source.bind(snap, q, 1)
            expected, before = search(q, np.ones(30), c, Names(), Attributes(source.sites), original, chunk=chunk)
            original.finish()
            coded = workbench.bind(source, snap, q, 1)
            got, stats = workbench.search(q, np.ones(30), c, Names(), Attributes(source.sites), coded, chunk=chunk)
            assert got == expected
            assert stats['address_facts'] == before['address_facts'] == source.facts
            assert stats['sites_completed'] == source.sites and coded.cursor == source.sites
            assert stats['fixed_reservoir'] is None and not stats['score_after_fixed_pool']
            assert stats['float_score_precision_and_reduction_order_unchanged']
        source.verify_after()
    finally:
        source.close()
        snap.close(remove=True)


@pytest.mark.parametrize('change', ['float_number', 'wrong_size', 'out_of_site', 'franken_exact', 'invalid_cp', 'untyped'])
def test_corrupt_coded_proof_never_silently_ignored(change):
    n = 3
    base = workbench.CodedAddressBatch(np.array([0], dtype=np.uint32), np.array([1.]),
        np.array([True]), np.array([False]), np.array([encode_code('')], dtype=np.uint32),
        np.array([encode_code('')], dtype=np.uint32), np.array([False]))
    if change == 'float_number':
        base = replace(base, number_matches=np.array([.5]))
    elif change == 'wrong_size':
        base = replace(base, number_matches=np.array([False, True]))
    elif change == 'out_of_site':
        base = replace(base, site_positions=np.array([3], dtype=np.uint32))
    elif change == 'franken_exact':
        base = replace(base, coverage=np.array([.5]))
    elif change == 'invalid_cp':
        base = replace(base, postcode_codes=np.array([36**5], dtype=np.uint32))
    else:
        base = SimpleNamespace(**base.__dict__)
    with pytest.raises(ValueError):
        workbench.score_sites(np.arange(n, dtype=np.uint32), Query(), np.ones(30),
                             np.zeros(n), np.zeros(n), Attributes(n), [base])


def test_snapshot_hash_drift_prevents_even_completed_search_from_returning(tmp_path, monkeypatch):
    index, _, _ = fixture(tmp_path, monkeypatch)
    source = CertifiedAddressFacts(index)
    snap = snapshot(tmp_path)
    q = Query()
    bound = workbench.bind(source, snap, q, 1)
    c = catalog(np.arange(100001, 100001+source.sites, dtype=np.uint64))
    class Names:
        sites = source.sites
        def gather(self, ids):
            return np.ones(len(ids)), np.zeros(len(ids))
    monkeypatch.setattr(snap, 'verify_after', lambda: (_ for _ in ()).throw(ValueError('FINAL_QUERY_HASH_DRIFT')))
    try:
        with pytest.raises(ValueError, match='FINAL_QUERY_HASH'):
            workbench.search(q, np.ones(30), c, Names(), Attributes(source.sites), bound)
        assert bound.cursor == source.sites and bound.consumed == source.facts
    finally:
        source.close()
        snap.close()
