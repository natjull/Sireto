"""Full tiny fixtures and exact global top100 oracle, not national sources."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.xgb_matcher import retrieval_staged_initial_v183 as workbench
from src.xgb_matcher.retrieval_certified_address_facts_v181 import CertifiedAddressFacts
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_initial_collector_v146 import collect, search
from src.xgb_matcher.retrieval_v20_workbench import record
from test_retrieval_certified_address_facts_v181 import fixture, snapshot
from test_retrieval_initial_collector_v146 import Attributes, catalog, chunks


def keys_catalog(values):
    c = catalog(values)
    inverse = {int(k): i for i, k in enumerate(values)}
    c.keys.site_positions = lambda keys: np.array([inverse.get(int(k), 2**32-1) for k in keys], dtype=np.uint32)
    return c


class Names:
    def __init__(self, n):
        self.sites = n
    def gather(self, ids):
        return ids.astype(np.float64)/max(1, self.sites-1), (ids == self.sites-1).astype(np.float64)


def test_flat_name_storage_reopen_gather_same_float_evidence(tmp_path):
    full = Names(101)
    names, stats = workbench.flatten_names(full, tmp_path / 'flat', lambda: None, 17)
    try:
        assert stats['sites'] == 101 and stats['every_original_site_gathered_once']
        for ids in (np.arange(101, dtype=np.uint32), np.array([100, 0, 57, 57], dtype=np.uint32)):
            for a, b in zip(names.gather(ids), full.gather(ids)):
                np.testing.assert_array_equal(a, b)
                assert a.flags.owndata
        refs = names.coverage.reference, names.exact.reference
        for a in (names.coverage, names.exact):
            with pytest.raises(ValueError, match='readonly'):
                with a._vector.mapped(write=True):
                    pass
    finally:
        names.close()
    restored = workbench.FlatNames(workbench.reopen_vector(refs[0], 101, np.float64),
                                   workbench.reopen_vector(refs[1], 101, np.bool_), lambda: None, 13)
    try:
        ids = np.arange(101, dtype=np.uint32)
        for a, b in zip(restored.gather(ids), full.gather(ids)):
            np.testing.assert_array_equal(a, b)
        restored.verify_after()
    finally:
        restored.close()
    assert all(__import__('pathlib').Path(ref['path']).is_file() for ref in refs)


def test_flat_vector_typedshape_and_nonexact_proof_rejected(tmp_path):
    names, _ = workbench.flatten_names(Names(5), tmp_path / 'flat', lambda: None, 2)
    ref = names.coverage.reference
    names.close()
    for size, dtype in ((4, np.float64), (5, np.float32)):
        with pytest.raises(ValueError, match='Exact original'):
            workbench.reopen_vector(ref, size, dtype)
    class Corrupt(Names):
        def gather(self, ids):
            return np.full(len(ids), .5), np.ones(len(ids))
    with pytest.raises(ValueError, match='exact flags'):
        workbench.flatten_names(Corrupt(5), tmp_path / 'bad', lambda: None, 2)


def test_all_three_original_intervals_partitioned_scores_equal_unsplit146(tmp_path, monkeypatch):
    index, _, _ = fixture(tmp_path, monkeypatch)
    source = CertifiedAddressFacts(index)
    address = snapshot(tmp_path)
    c = keys_catalog(np.arange(100001, 100001+source.sites, dtype=np.uint64)[::-1])
    names, _ = workbench.flatten_names(Names(source.sites), tmp_path / 'names', lambda: None, 3)
    q = Query(postcode='75001', number='7')
    w = np.linspace(.01, 3., 30)
    signature = workbench.recipe(q, w, {'catalog': 'fixture_only'}, 3)
    try:
        baseline = source.bind(address, q, 1)
        expected, _ = search(q, w, c, names, Attributes(source.sites), baseline, chunk=2)
        baseline.finish()
        for n in (1, 2, 3):
            spans = workbench.spans(source, n)
            parts = []
            for span in spans:
                bound = workbench.PartitionFacts(source, address, q, span, 1)
                part = workbench.partition_search(q, w, c, names, Attributes(source.sites), bound, signature, chunk=2)
                assert bound.consumed == span['facts'] and bound.cursor == span['stop']
                parts.append(part)
            actual, trace = workbench.merge_partitions(c, source.facts, parts)
            assert actual == expected and trace['sites_completed'] == source.sites
            assert trace['address_facts'] == source.facts
            assert trace['fixed_reservoir'] is None and not trace['score_after_fixed_pool']
        source.verify_after()
    finally:
        source.close()
        address.close(remove=True)
        names.close()


def physical_parts(c, scores, sizes):
    start, parts = 0, []
    for size in sizes:
        shard = SimpleNamespace(sites=size, keys=workbench.PartitionKeys(c.keys, start))
        rows, stats = collect(shard, chunks(scores[start:start+size], 29))
        stats['address_facts'] = size*2
        parts.append(dict(span=dict(start=start, stop=start+size, facts=size*2), signature='a'*64,
                          candidates=rows, stats=stats, partition_only_not_initial_output=True,
                          same_scores_never_rescored=True))
        start += size
    return parts


def test_physical_shard_top100_reduction_equals_all_5007_scores_sort_and_ties():
    rng = np.random.default_rng(183)
    values = rng.permutation(np.arange(100001, 105008, dtype=np.uint64))
    scores = rng.integers(0, 5, len(values)).astype(np.float64)
    c = keys_catalog(values)
    expected, _ = collect(c, chunks(scores, 83))
    for sizes in ([5007], [1000, 2000, 2007], [97]*51+[60]):
        parts = physical_parts(c, scores, sizes)
        actual, trace = workbench.merge_partitions(c, 10014, parts)
        assert actual == expected and trace['every_site_scored_once']
        assert trace['sites_completed'] == 5007 and len(actual) == 100
        assert trace['same_original_score_never_rescored']


@pytest.mark.parametrize('change', ['missing_tail', 'overlap', 'recipe', 'missing_fact', 'wrong_identity',
                                   'rerank', 'unsorted', 'cap101', 'not_completed'])
def test_no_partial_initial100_on_partition_gap_different_score_recipe_or_corruption(change):
    c = keys_catalog(np.arange(100001, 100010, dtype=np.uint64))
    parts = physical_parts(c, np.arange(1., 10.), [3, 3, 3])
    parts = deepcopy(parts)
    if change == 'missing_tail':
        parts.pop()
    elif change == 'overlap':
        parts[1]['span']['start'] -= 1
    elif change == 'recipe':
        parts[1]['signature'] = 'b'*64
    elif change == 'missing_fact':
        parts[1]['stats']['address_facts'] -= 1
    elif change == 'wrong_identity':
        parts[0]['candidates'] = deepcopy(parts[1]['candidates'])
    elif change == 'rerank':
        parts[1]['same_scores_never_rescored'] = False
    elif change == 'unsorted':
        parts[0]['candidates'].reverse()
    elif change == 'cap101':
        parts[0]['stats']['initial_cap'] = 101
    else:
        parts[0]['stats']['every_site_scored_once'] = False
    with pytest.raises(ValueError):
        workbench.merge_partitions(c, 18, parts)


def test_partition_finish_requires_every_fact_and_query_hash(tmp_path, monkeypatch):
    index, _, _ = fixture(tmp_path, monkeypatch)
    source = CertifiedAddressFacts(index)
    address = snapshot(tmp_path)
    q = Query()
    span = workbench.spans(source, 1)[0]
    bound = workbench.PartitionFacts(source, address, q, span, 1)
    try:
        list(bound.batches(np.arange(5, dtype=np.uint32), q))
        with pytest.raises(ValueError, match='Every original'):
            bound.finish()
        list(bound.batches(np.arange(5, source.sites, dtype=np.uint32), q))
        monkeypatch.setattr(address, 'verify_after', lambda: (_ for _ in ()).throw(ValueError('REAL_QUERY_HASH')))
        with pytest.raises(ValueError, match='REAL_QUERY_HASH'):
            bound.finish()
    finally:
        source.close()
        address.close()


def test_recipe_changes_on_query_source_weight_and_fold_not_identity_similarity():
    q = Query(name='ETABLISSEMENT TEST', postcode='75001')
    w = np.ones(30)
    original = workbench.recipe(q, w, {'source': 'official_a'}, 2)
    assert original != workbench.recipe(q, w, {'source': 'official_b'}, 2)
    assert original != workbench.recipe(q, w*2, {'source': 'official_a'}, 2)
    assert original != workbench.recipe(q, w, {'source': 'official_a'}, 3)
    assert original != workbench.recipe(Query(name=q.name, postcode='69001'), w, {'source': 'official_a'}, 2)
    for fold in (0, 1, True):
        with pytest.raises(ValueError):
            workbench.recipe(q, w, {}, fold)
