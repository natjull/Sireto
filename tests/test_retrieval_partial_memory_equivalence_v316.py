"""Tiny exact-score/lifetime fixtures, not national RSS or retry permission."""
from types import SimpleNamespace
import weakref
import numpy as np
from src.xgb_matcher import retrieval_partial_legal_parent_v313 as old
from src.xgb_matcher import retrieval_partial_legal_parent_v316 as new
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_structural_channels_v210 import owner_fixture, native_backend


def test_complete_pair_postings_equivalent_not_retained_all_words(tmp_path):
    source = owner_fixture(tmp_path)
    observations, references = [], []
    class Words:
        def lookup(self, channel, text):
            # At most the preceding single operand coexists during a pair.
            assert sum(r() is not None for r in references) <= 1
            result = source.lookup(channel, text)
            references.append(weakref.ref(result))
            observations.append((channel, text))
            return result
    e = new.Evidence.__new__(new.Evidence)
    e.files, e.items, e.lookup = source.files, source.items, Words().lookup
    previous = old.Evidence.__new__(old.Evidence)
    previous.files, previous.items, previous.lookup = source.files, source.items, source.lookup
    assert e.partial_parents('ALPHA BETA ALPHA PROJECT') == previous.partial_parents('ALPHA BETA ALPHA PROJECT')
    assert all(r() is None for r in references)
    assert observations == [(0, w) for p in new.content_pairs('ALPHA BETA ALPHA PROJECT') for w in p]


def test_compiled_native_scores_and_initial_outputs_exactly_same(tmp_path):
    backend = native_backend(tmp_path)
    q = RetrievalQuery(name='PROJECT ALPHA BETA', insee='75056')
    guard = lambda: None
    def evidence():
        return SimpleNamespace(files=SimpleNamespace(guard=guard), name_owners=lambda t: (set(), set(), {}),
            related=lambda p, s: (set(), {}), radial_proofs=lambda g: ([({'10000000000001'}, .5)], {}),
            partial_parents=lambda t: ({'100000000'}, {}))
    for weights in (np.zeros(11), np.ones(11)):
        a, at = old.search(backend, evidence(), q, {}, weights, [], 0., set())
        b, bt = new.search(backend, evidence(), q, {}, weights, [], 0., set())
        assert a == b
        assert at['strong_parent_deduplicated'] == bt['strong_parent_deduplicated']
        assert not bt['score_after_fixed_pool'] and bt['initial_cap'] == 100


def test_proof_python_sets_released_before_actual_native_collector(tmp_path):
    backend = native_backend(tmp_path)
    refs = []
    def tracked(values):
        s = set(values)
        refs.append(weakref.ref(s))
        return s
    guard = lambda: None
    e = SimpleNamespace(files=SimpleNamespace(guard=guard),
        name_owners=lambda t: (tracked(['100000000']), tracked(['10000000100001']), {}),
        related=lambda p, s: (tracked(['10000000000002']), {}),
        radial_proofs=lambda g: ([(tracked(['10000000000001']), .5)], {}),
        partial_parents=lambda t: (tracked(['100000000', '100000001']), {}))
    original = backend.searcher
    calls = []
    def collect(query, **kw):
        assert all(r() is None for r in refs)
        calls.append(kw)
        return original.search(query, **kw)
    backend.searcher = SimpleNamespace(search=collect, doc=original.doc)
    rows, trace = new.search(backend, e, RetrievalQuery(name='ALPHA BETA', insee='75056'), {}, np.ones(11), [], 0., set())
    assert rows and calls == [dict(limit=100, count=False)] and trace['strong_parent_deduplicated'] == 1
