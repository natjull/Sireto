"""Integration fixtures, not national measurements or source certificates."""
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.compile_retrieval_v155_address_access as access
import src.xgb_matcher.retrieval_initial_engine_v158 as engine
from src.xgb_matcher.retrieval_address_access_v155 import NationalAddresses
from src.xgb_matcher.retrieval_address_reverse_v147 import ReversePart
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_initial_collector_v146 import score_sites, collect
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_compile_retrieval_v155_address_access import setup, finish
from test_retrieval_initial_collector_v146 import Attributes, catalog, chunks


@pytest.fixture
def sources(tmp_path, monkeypatch):
    source, output = setup(tmp_path, monkeypatch)
    access.init(output, record(source/'manifest.json'))
    finish(output)
    facts = engine.CompleteAddressFacts(record(source/'manifest.json'))
    lexical = NationalAddresses(record(output/'manifest.json'))
    try:
        yield facts, lexical
    finally:
        lexical.close()
        facts.close()


def name_source(facts):
    c = catalog(np.arange(100001, 100001+facts.sites, dtype=np.uint64)[::-1])
    cov = np.linspace(.1, .9, facts.sites)
    class Names:
        sites = facts.sites
        def gather(self, ids):
            return cov[ids], np.zeros(len(ids))
    class Source:
        catalog = c
        metadata = {'sources': {'catalog': facts.catalog_reference}}
        def query(self, index, text, max_seconds, chunk):
            return Names(), {'seconds': 0., 'unit_fixture_not_source_certificate': True}
    return Source(), cov


def test_whole_collector_with_all_source_facts_matches_scalar_and_cross_interval_chunks(sources):
    facts, lexical = sources
    names, nc = name_source(facts)
    query = Query(name='NOM', address='RUE TEST', postcode='75001', number='7')
    weights = np.ones(30)
    attributes = Attributes(facts.sites)
    snapshot, _ = lexical.snapshot(query.address)
    batches = []
    for interval in facts.intervals:
        reader = ReversePart(interval)
        try:
            ids = np.arange(interval['start'], interval['stop'], dtype=np.uint32)
            for batch in reader.batches(ids, snapshot, chunk=1):
                # Borrowed site/proof arrays must be detached before map close.
                batches.append(type(batch)(*(a.copy() for a in (
                    batch.site_positions, batch.coverage, batch.alias_exact, batch.current_exact,
                    batch.postcode_codes, batch.insee_codes)), batch.number_texts))
        finally:
            reader.close()
    ids = np.arange(facts.sites, dtype=np.uint32)
    scores, _ = score_sites(ids, query, weights, nc, np.zeros(facts.sites), attributes, batches)
    expected, _ = collect(names.catalog, chunks(scores, 2))
    for chunk in (1, 3, facts.sites):
        rows, stats = engine.initial100(query, weights, names, None, lexical, facts,
                                       attributes, chunk=chunk)
        assert rows == expected
        assert stats['address_facts'] == facts.facts == 2
        assert stats['sites_completed'] == facts.sites == 11
        assert stats['all_original_address_facts_consumed']
        assert stats['fixed_reservoir'] is None and not stats['score_after_fixed_pool']
        assert not stats['recall_measured'] and not stats['slo_validated'] and not stats['ready_for_fold1']
        assert facts.active is None
    facts.verify_after()


def test_binding_is_single_query_and_full_traversal_is_mandatory(sources):
    facts, lexical = sources
    q = Query(address='RUE TEST')
    snapshot, _ = lexical.snapshot(q.address)
    bound = facts.bind(snapshot, q, 1)
    with pytest.raises(ValueError, match='Same query'):
        list(bound.batches(np.arange(2, dtype=np.uint32), Query(address=q.address)))
    with pytest.raises(ValueError, match='Same query'):
        list(bound.batches(np.array([1], dtype=np.uint32), q))
    list(bound.batches(np.arange(5, dtype=np.uint32), q))
    with pytest.raises(ValueError, match='Every original'):
        bound.finish()
    list(bound.batches(np.arange(5, facts.sites, dtype=np.uint32), q))
    bound.finish()
    with pytest.raises(ValueError, match='Same query'):
        list(bound.batches(np.array([0], dtype=np.uint32), q))


def test_epoch_drift_after_binding_stops_and_releases_active_map(sources):
    facts, lexical = sources
    q = Query(address='RUE TEST')
    snapshot, _ = lexical.snapshot(q.address)
    bound = facts.bind(snapshot, q, 1)
    gen = bound.batches(np.arange(facts.sites, dtype=np.uint32), q)
    next(gen)
    assert facts.active is not None
    index = facts.starts.index(facts.active.stats['start'])
    path = next(iter(facts.checked[index]['epochs']))
    # Test corruption only inside pytest's temporary fixture, never real data.
    with open(path, 'ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError, match='epoch drift'):
        list(gen)
    assert facts.active is None and not facts.valid


def test_generator_close_releases_only_own_interval(sources):
    facts, lexical = sources
    q = Query(address='RUE TEST')
    snapshot, _ = lexical.snapshot(q.address)
    bound = facts.bind(snapshot, q, 1)
    gen = bound.batches(np.arange(facts.sites, dtype=np.uint32), q)
    next(gen)
    assert facts.active is not None
    gen.close()
    assert facts.active is None and facts.valid
    with pytest.raises(ValueError, match='Every original'):
        bound.finish()


def test_namespace_mismatch_stops_before_name_query(sources, monkeypatch):
    facts, lexical = sources
    names, _ = name_source(facts)
    monkeypatch.setattr(names, 'query', lambda *a, **kw: pytest.fail('No query with source drift'))
    names.metadata['sources']['catalog'] = {'path': 'other', 'sha256': '0'*64}
    with pytest.raises(ValueError, match='source-bound'):
        engine.initial100(Query(), np.ones(30), names, None, lexical, facts, Attributes(facts.sites))


def test_one_total_deadline_not_three_independent_allowances(sources, monkeypatch):
    facts, lexical = sources
    names, _ = name_source(facts)
    clock = [0.]
    monkeypatch.setattr(engine.time, 'monotonic', lambda: clock[0])
    def query(*a, **kwargs):
        assert kwargs['max_seconds'] == 2.
        clock[0] = 2.1
        return SimpleNamespace(sites=facts.sites), {}
    monkeypatch.setattr(names, 'query', query)
    monkeypatch.setattr(lexical, 'snapshot', lambda *a, **kw: pytest.fail('No address query after total timeout'))
    with pytest.raises(TimeoutError, match='TOTAL_COST'):
        engine.initial100(Query(), np.ones(30), names, None, lexical, facts,
                          Attributes(facts.sites), max_seconds=2.)
    assert facts.active is None


@pytest.mark.parametrize('kwargs', [{'chunk': True}, {'chunk': 0}, {'max_seconds': True}, {'max_seconds': 0}])
def test_invalid_query_budget_fails_before_source_queries(sources, kwargs):
    facts, lexical = sources
    names, _ = name_source(facts)
    with pytest.raises(ValueError, match='Typed CRM'):
        engine.initial100(Query(), np.ones(30), names, None, lexical, facts, Attributes(facts.sites), **kwargs)


def test_closed_source_never_reuses_previous_query_snapshot(sources):
    facts, lexical = sources
    q = Query()
    snapshot, _ = lexical.snapshot(q.address)
    facts.close()
    facts.close()
    with pytest.raises(ValueError, match='Closed'):
        facts.bind(snapshot, q)


def test_factory_missing_full_access_fails_before_any_reader(tmp_path, monkeypatch):
    import src.xgb_matcher.retrieval_compact_coverage_v156 as compact
    monkeypatch.setattr(compact, 'CompactNamePages', lambda *a: pytest.fail('No reader without all authorities'))
    ref = {'path': str(tmp_path/'unfinished.json'), 'sha256': '0'*64}
    with pytest.raises((ValueError, FileNotFoundError)):
        engine.OfficialInitial100(names=ref, joins=ref, structured=ref, addresses=ref)


def test_factory_mismatched_name_proofs_close_readers_without_address_load(tmp_path, monkeypatch):
    import src.xgb_matcher.retrieval_compact_coverage_v156 as compact
    import src.xgb_matcher.retrieval_rebound_names_v150 as rebound
    import src.xgb_matcher.retrieval_address_access_v155 as lexical
    path = tmp_path/'fixture_authority.json'
    write_json(path, {'unit_fixture_only': True})
    ref = record(path)
    closed = []
    class Index:
        names = 3
        source = {'path': 'other', 'sha256': '1'*64}
        def close(self): closed.append('index')
    class Names:
        names = 3
        source = ref
        def close(self): closed.append('names')
    monkeypatch.setattr(compact, 'CompactNamePages', lambda *a: Index())
    monkeypatch.setattr(rebound, 'ReboundNames', lambda *a: Names())
    monkeypatch.setattr(lexical, 'NationalAddresses', lambda *a: pytest.fail('No address load with mismatched names'))
    with pytest.raises(ValueError, match='shared name/site/address'):
        engine.OfficialInitial100(names=ref, joins=ref, structured=ref, addresses=ref)
    assert closed == ['names', 'index']
