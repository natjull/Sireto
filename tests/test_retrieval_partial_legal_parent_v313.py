"""Synthetic source/engine fixtures only; never national business recall."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_partial_legal_parent_v313 import Evidence, content_pairs, compose, search
from src.xgb_matcher.retrieval_regional_initial_v286 import compose as previous
from src.xgb_matcher.retrieval_training_score_v231 import points
from test_retrieval_structural_channels_v210 import owner_fixture, native_backend, array


def test_adjacent_distinct_pairs_equal_union_of_all_content_spans():
    from src.xgb_matcher.retrieval_regional_initial_v286 import PARTICLES, LEGAL
    from src.xgb_matcher.retrieval_variants_v118 import normalized
    texts = ('CODE ALPHA BETA', 'ALPHA ET ALPHA BETA PROJECT', 'LA SOCIETE ALPHA',
        'THEATRE', 'ALPHA ALPHA', '[ND]', '123456789 12345678900001', 'ALPHA BETA GAMMA ALPHA')
    vocab = ('ALPHA', 'BETA', 'GAMMA', 'CODE', 'PROJECT')
    import itertools
    aliases = [set(x) for k in range(len(vocab) + 1) for x in itertools.combinations(vocab, k)]
    for text in texts:
        words = [w for w in normalized(text).split() if w not in PARTICLES | LEGAL]
        spans = [set(words[lo:hi]) for lo in range(len(words)) for hi in range(lo + 1, len(words) + 1)
            if len(set(words[lo:hi])) >= 2]
        for alias in aliases:
            assert any(s <= alias for s in spans) == any(set(p) <= alias for p in content_pairs(text))
    assert content_pairs('PROJECT ALPHA ET BETA') == [('ALPHA', 'BETA'), ('ALPHA', 'PROJECT')]
    assert not content_pairs('ALPHA ALPHA') and not content_pairs('[ND]')


def contained_fixture(tmp_path):
    old = owner_fixture(tmp_path)
    e = Evidence.__new__(Evidence)
    e.files, e.items = old.files, old.items
    calls = []
    def lookup(channel, text):
        calls.append((channel, text))
        return np.asarray([0, 1] if channel == 0 and text in ('ALPHA', 'BETA') else [], dtype=np.uint32)
    e.lookup = lookup
    return e, calls


def test_complete_exact_alias_fragment_legal_and_site_parent_joins(tmp_path):
    e, calls = contained_fixture(tmp_path)
    parents, t = e.partial_parents('PROJECT ALPHA BETA')
    assert parents == {'100000000'}
    assert all(channel == 0 for channel, text in calls)
    assert set(text for channel, text in calls) == {'PROJECT', 'ALPHA', 'BETA'}
    assert t['aliases'] == 2 and t['legal_owner_facts'] == 1 and t['owner_cap'] is None
    assert t['site_owner_facts_ignored'] == t['legal_owner_facts'] == 1
    assert not t['sibling_alias_identity_claimed']
    e.files.verify_after()


def test_no_whole_alias_hit_does_not_cross_words_or_owners(tmp_path):
    e, calls = contained_fixture(tmp_path)
    parents, t = e.partial_parents('ALPHA GAMMA')
    assert not parents and t['aliases'] == t['legal_owner_facts'] == 0


def test_guard_failure_has_no_partial_owner_return(tmp_path):
    e, calls = contained_fixture(tmp_path)
    def fail():
        raise TimeoutError('Source guard')
    e.files.guard = fail
    with pytest.raises(TimeoutError):
        e.partial_parents('ALPHA BETA')


def test_bad_official_owner_scope_is_not_silently_lifted(tmp_path):
    e, calls = contained_fixture(tmp_path)
    e.items[0]['artifacts']['parent_key.npy'] = array(tmp_path, 'bad_parents', [100000000, 100000099, 100000002], np.uint32)
    with pytest.raises(ValueError, match='scope conflict'):
        e.partial_parents('ALPHA BETA')


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
        partial_parents=lambda text: ({'100000000'}, {}))
    q = RetrievalQuery(name='PROJECT ALPHA BETA', insee='75056')
    rows, t = search(b, e, q, {}, np.zeros(11), [], 0., set())
    assert calls == [dict(limit=100, count=False)] and texts == [q.name]
    assert [r['siret'] for r in rows] == ['10000000000001']
    assert t['fixed_reservoir'] is None and not t['score_after_fixed_pool']
    assert not t['label_lookup'] and not t['sibling_alias_identity_claimed']
    assert e.files.guard is guard

def test_all_legal_parent_facts_not_capped_to100(tmp_path):
    old = owner_fixture(tmp_path)
    e = Evidence.__new__(Evidence)
    e.files = old.files
    count = 151
    parents = np.arange(100000000, 100000000 + count, dtype=np.uint32)
    e.items = [dict(name_start=0, names=1, facts=count, artifacts={
        k + '.npy': array(tmp_path, 'many_' + k, values, dtype) for k, values, dtype in (
            ('name_fact_offsets', [0, count], np.uint64), ('owner_key', parents, np.uint64),
            ('parent_key', parents, np.uint32), ('subject_code', np.ones(count), np.uint8),
            ('scope_code', np.ones(count), np.uint8))})]
    e.lookup = lambda channel, word: np.asarray([0], dtype=np.uint32)
    selected, trace = e.partial_parents('ALPHA BETA')
    assert len(selected) == trace['legal_owner_facts'] == 151 and trace['owner_cap'] is None
    assert not trace['partial_site_name_lifted_to_parent']
    e.files.verify_after()
