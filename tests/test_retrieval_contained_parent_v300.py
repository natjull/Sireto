"""Synthetic source/engine fixtures only; never national business recall."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_contained_parent_v300 import Evidence, fragments, compose, search
from src.xgb_matcher.retrieval_regional_initial_v286 import compose as previous
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_structural_channels_v210 import owner_fixture, native_backend, array


def test_all_contiguous_multiword_spans_no_singleton_or_identifier_feature():
    assert fragments('CODE ALPHA BETA') == ['ALPHA BETA', 'CODE ALPHA', 'CODE ALPHA BETA']
    for text in ('THEATRE', 'LA SOCIETE ALPHA', 'ALPHA ALPHA', '[ND]', '123456789 12345678900001'):
        assert fragments(text) == []
    assert 'ALPHA ET BETA' in fragments('PROJECT ALPHA ET BETA SUFFIX')


def contained_fixture(tmp_path):
    old = owner_fixture(tmp_path)
    e = Evidence.__new__(Evidence)
    e.files, e.items = old.files, old.items
    calls = []
    def lookup(channel, text):
        calls.append((channel, text))
        return np.asarray([0, 1] if (channel, text) == (3, 'ALPHA BETA') else [], dtype=np.uint32)
    e.lookup = lookup
    return e, calls


def test_complete_exact_alias_fragment_legal_and_site_parent_joins(tmp_path):
    e, calls = contained_fixture(tmp_path)
    parents, t = e.contained_parents('PROJECT ALPHA BETA')
    assert parents == {'100000000', '100000001'}
    assert all(channel == 3 for channel, text in calls)
    assert set(text for channel, text in calls) == set(fragments('PROJECT ALPHA BETA'))
    assert t['aliases'] == 2 and t['owner_facts'] == 2 and t['owner_cap'] is None
    assert t['site_owner_facts'] == t['legal_owner_facts'] == 1
    assert not t['sibling_alias_identity_claimed']
    e.files.verify_after()


def test_no_whole_alias_hit_does_not_cross_words_or_owners(tmp_path):
    e, calls = contained_fixture(tmp_path)
    parents, t = e.contained_parents('ALPHA GAMMA')
    assert not parents and t['aliases'] == t['owner_facts'] == 0


def test_guard_failure_has_no_partial_owner_return(tmp_path):
    e, calls = contained_fixture(tmp_path)
    def fail():
        raise TimeoutError('Source guard')
    e.files.guard = fail
    with pytest.raises(TimeoutError):
        e.contained_parents('ALPHA BETA')


def test_bad_official_owner_scope_is_not_silently_lifted(tmp_path):
    e, calls = contained_fixture(tmp_path)
    e.items[0]['artifacts']['parent_key.npy'] = array(tmp_path, 'bad_parents', [100000000, 100000099, 100000002], np.uint32)
    with pytest.raises(ValueError, match='scope conflict'):
        e.contained_parents('ALPHA BETA')


def test_weak_proof_native_leaf_scope_keeps_closed_site_not_exact_name(tmp_path):
    b = native_backend(tmp_path)
    q = RetrievalQuery(name='PROJECT ALPHA BETA', insee='75056')
    ids = ['10000000000001', '10000000000002', '10000000100001']
    w = np.zeros(11)
    args = (b, q, w, [], 0., set(), set(), set(), [], set())
    values = points(b, compose(*args, {'100000000'}), ids)
    assert np.allclose(values, [.5, 0., 0.])
    # Unrelated local leaf remains reachable through the original place path.
    w[4] = 1.
    assert np.allclose(points(b, compose(*args, {'100000000'}), ids), [1.5, 0., 1.])


def test_no_contained_parent_or_no_support_preserves_old_metric(tmp_path):
    b = native_backend(tmp_path)
    ids = ['10000000000001', '10000000000002', '10000000100001']
    q = RetrievalQuery(name='ALPHA BETA')
    args = (b, q, np.ones(11), [], 0., set(), set(), set(), [], set())
    baseline = points(b, previous(*args), ids)
    assert np.allclose(points(b, compose(*args, set()), ids), baseline)
    assert np.allclose(points(b, compose(*args, {'100000000'}), ids), baseline)


def test_existing_strong_parent_not_double_counted_and_site_not_equivalence(tmp_path):
    b = native_backend(tmp_path)
    q = RetrievalQuery(name='ALPHA BETA', insee='75056')
    ids = ['10000000000001', '10000000000002', '10000000100001']
    args = (b, q, np.zeros(11), [], 0., {'100000000'}, set(), set(), [], set())
    assert np.allclose(points(b, compose(*args, {'100000000'}), ids), points(b, previous(*args), ids))


def test_one_native_collector_no_previous_output_pool_and_guard_restored(tmp_path):
    b = native_backend(tmp_path)
    original = b.searcher
    calls, texts = [], []
    def called(p, **kw):
        calls.append(kw)
        return original.search(p, **kw)
    b.searcher = SimpleNamespace(search=called, doc=original.doc)
    guard = lambda: None
    def owners(text):
        texts.append(text)
        return set(), set(), {}
    e = SimpleNamespace(files=SimpleNamespace(guard=guard), name_owners=owners,
        related=lambda p, s: (set(), {}), radial_proofs=lambda g: ([], {}),
        contained_parents=lambda text: ({'100000000'}, {}))
    q = RetrievalQuery(name='PROJECT ALPHA BETA', insee='75056')
    rows, t = search(b, e, q, {}, np.zeros(11), [], 0., set())
    assert calls == [dict(limit=100, count=False)] and texts == [q.name]
    assert [r['siret'] for r in rows] == ['10000000000001']
    assert t['fixed_reservoir'] is None and not t['score_after_fixed_pool']
    assert not t['label_lookup'] and not t['sibling_alias_identity_claimed']
    assert e.files.guard is guard
