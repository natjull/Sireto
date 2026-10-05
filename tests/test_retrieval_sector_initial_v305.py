"""Tiny native engine fixtures; no national retrieval conclusion."""
from types import SimpleNamespace
import numpy as np
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_sector_initial_v305 import compose, search
from src.xgb_matcher.retrieval_regional_initial_v286 import compose as previous
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_structural_channels_v210 import native_backend

IDS = ['10000000000001', '10000000000002', '10000000100001']


def facts(local=(), unknown=(), outside=()):
    return {'local': set(local), 'unknown': set(unknown), 'nonlocal': set(outside)}


def test_source_local_endpoint_and_native_local_sibling_not_exact_name(tmp_path):
    b = native_backend(tmp_path)
    q = RetrievalQuery(name='ALPHA BETA', insee='75056')
    args = (b, q, np.zeros(11), [], 0., set(), set(), set(), [], set())
    assert np.allclose(points(b, compose(*args, facts(local=[IDS[1]])), IDS), [.5, 1., 0.])
    # Closed original first endpoint is retained; source-local proves a
    # relation to its explicitly published endpoint, not same physical site.
    assert np.allclose(points(b, compose(*args, facts(local=[IDS[0]])), IDS), [1., 0., 0.])


def test_unknown_nonlocal_sources_not_global_exclusion_and_support_needed(tmp_path):
    b = native_backend(tmp_path)
    q = RetrievalQuery(name='ALPHA BETA', insee='75056')
    args = (b, q, np.zeros(11), [], 0., set(), set(), set(), [], set())
    assert np.allclose(points(b, compose(*args, facts(outside=[IDS[0]])), IDS), [1., 0., 0.])
    assert np.allclose(points(b, compose(*args, facts(unknown=[IDS[1]])), IDS), [.5, 0., 0.])
    q = RetrievalQuery(name='ALPHA BETA')
    args = (b, q, np.zeros(11), [], 0., set(), set(), set(), [], set())
    assert np.allclose(points(b, compose(*args, facts(unknown=[IDS[0]])), IDS), [0., 0., 0.])


def test_no_source_facts_original_metric_and_strong_owners_not_stacked(tmp_path):
    b = native_backend(tmp_path)
    q = RetrievalQuery(name='ALPHA BETA', insee='75056')
    args = (b, q, np.ones(11), [], 0., {'100000000'}, set(), set(), [], set())
    baseline = points(b, previous(*args), IDS)
    assert np.allclose(points(b, compose(*args, facts()), IDS), baseline)
    assert np.allclose(points(b, compose(*args, facts(local=IDS[:2])), IDS), baseline)
    args = (b, q, np.ones(11), [], 0., set(), {IDS[0]}, set(), [], set())
    assert np.allclose(points(b, compose(*args, facts(local=[IDS[0]])), IDS), points(b, previous(*args), IDS))


def test_one_global_native_collector_no_previous_top100_pool(tmp_path):
    b = native_backend(tmp_path)
    original, calls = b.searcher, []
    def called(p, **kw):
        calls.append(kw)
        return original.search(p, **kw)
    b.searcher = SimpleNamespace(search=called, doc=original.doc)
    guard = lambda: None
    e = SimpleNamespace(files=SimpleNamespace(guard=guard), name_owners=lambda q: (set(), set(), {}),
        related=lambda p, s: (set(), {}), radial_proofs=lambda g: ([], {}))
    f = SimpleNamespace(facts=lambda q: (facts(local=[IDS[0]]), {}))
    rows, t = search(b, e, RetrievalQuery(name='ALPHA BETA'), {}, np.zeros(11), [], 0., set(), f)
    assert [r['siret'] for r in rows] == [IDS[0]]
    assert calls == [dict(limit=100, count=False)] and e.files.guard is guard
    assert not t['score_after_fixed_pool'] and t['fixed_reservoir'] is None
    assert not t['source_coordinates_used'] and not t['source_owner_name_used']
