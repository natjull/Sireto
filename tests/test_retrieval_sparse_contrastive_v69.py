import numpy as np
import pytest
from scipy.sparse import coo_matrix

from src.xgb_matcher import retrieval_sparse_contrastive_v69 as m
from src.xgb_matcher.retrieval_impact_v68 import ImpactIndex


def data():
    delta = coo_matrix(np.array([[.5, -.2, 0., .1], [-.1, .4, -.3, 0.],
                                [.2, 0., -.4, .3], [-.4, .3, .1, 0.]]))
    rng = np.random.default_rng(3)
    return dict(delta=delta, native=rng.normal(size=(4, len(m.FEATURES))),
                sizes=np.array([2, 2]), owner=np.array([0, 0, 1, 1]),
                tags=[dict(same_parent=True)]*4)


def test_piecewise_gradient_includes_flops_decoder_and_native():
    d = data(); z = np.array([[1., .3, 1.], [.2, 1., 1.]])
    initial = np.array([[.1, .05, .2, .1], [.07, .15, .1, .3], [.2, .1, .05, .03]])
    theta = np.r_[m.PRIOR+.15, (initial+.01).ravel()]
    loss, gradient = m.objective(theta, z, d, initial, .1, .1, 3)
    assert np.isfinite(loss)
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy(); plus[j] += 1e-6; minus[j] -= 1e-6
        difference = (m.objective(plus, z, d, initial, .1, .1, 3)[0] -
                      m.objective(minus, z, d, initial, .1, .1, 3)[0])/2e-6
        assert difference == pytest.approx(gradient[j], abs=2e-8)


def test_activation_sparse_nonnegative_stable_ties_and_zeros():
    values, derivative = m.activate([[2., 2., -1., 0., 4.]], 2)
    assert values[0].tolist() == pytest.approx([np.log(3.), 0., 0., 0., np.log(5.)])
    assert derivative[0].tolist() == pytest.approx([1/3, 0., 0., 0., 1/5])
    assert np.count_nonzero(values) <= 2
    with pytest.raises(ValueError):
        m.activate([[np.nan]], 1)


def test_document_normalization_is_not_fold_vocabulary_dependent():
    document = dict(legal_names=['ALPHA BETA'], addresses=['RUE GAMMA'],
                    siret=['12345678900001'], siren=['123456789'], insee=['75056'])
    impact = m.impacts(document)
    assert impact['N:alpha'] == pytest.approx(1/np.sqrt(2))
    assert impact['N:beta'] == pytest.approx(1/np.sqrt(2))
    assert 'A:gamma' in impact and 'I:AB' in impact
    assert not any('123456789' in t or '75056' in t for t in impact)
    # No invented sigle from words in separate aliases.
    assert 'I:AB' not in m.impacts(dict(legal_names=['ALPHA', 'BETA']))


def test_recipe_excludes_protected_negative_terms_and_basis():
    rng = np.random.default_rng(4); q = rng.normal(size=(18, 2, 4))
    rows = [dict(positive_siren='SAFE', positive_siret=f'p{i}',
                 negatives=[dict(siren='SAFE', siret=f'n{i}'), dict(siren='BLOCK', siret='bad')])
            for i in range(18)]
    docs = {f'p{i}': {f'N:token{i%4}':1.} for i in range(18)}
    docs.update({f'n{i}': {f'A:token{i%4}':1.} for i in range(18)})
    docs['bad'] = {'N:secret':1.}
    a = m.recipe(q, rows, docs, list(range(16)), {'BLOCK'}, dimensions=2)
    changed = q.copy(); changed[16:] = 1e10
    docs['bad'] = {'N:entirelydifferent':1.}
    b = m.recipe(changed, rows, docs, list(range(16)), {'BLOCK'}, dimensions=2)
    assert a[0] == b[0] and not any('secret' in t for t in a[0])
    for left, right in zip(a[1], b[1]):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a[2], b[2])
    blocked = [dict(r, positive_siren='BLOCK') for r in rows]
    with pytest.raises(ValueError, match='Protected positive'):
        m.recipe(q, blocked, docs, list(range(16)), {'BLOCK'})


def test_sparse_pair_score_matches_dense_and_serialized_serving():
    d = data(); rng = np.random.default_rng(5); z = rng.normal(size=(2, 3))
    decoder = rng.normal(size=(3, 4)); act = m.activate(z @ decoder, 2)[0]
    expected = d['native'] @ m.PRIOR + (d['delta'].toarray()*act[d['owner']]).sum(axis=1)
    np.testing.assert_allclose(m.score(d, m.PRIOR, act), expected, atol=1e-12)
    q = np.array([1., 0., 0., 1.])
    model = dict(bases=[[[1.], [0.]], [[0.], [1.]]], decoder=decoder.tolist(),
                 query_terms=2, vocabulary=['A:a', 'I:B', 'N:c', 'N:d'], alpha=2.)
    actual = m.query_impacts(model, q)
    manual = m.activate(np.array([[1., 1., 1.]]) @ decoder, 2)[0][0]*2.
    assert actual == {t: float(v) for t, v in zip(model['vocabulary'], manual) if v > 0}


def test_learned_weights_execute_direct100_over_entire_fixture():
    rng = np.random.default_rng(7)
    docs = {f'{i:014d}': {f'N:{j}': float(v) for j, v in enumerate(rng.random(9))}
            for i in range(231)}
    index = ImpactIndex.build(docs, block_size=16)
    act = m.activate(rng.normal(size=(1, 9)), 3)[0][0]
    query = {f'N:{j}':float(v) for j, v in enumerate(act) if v}
    indexed, _ = index.search(query, 100)
    # Sparse executor parity uses scores for ALL231, not a truth-filtered set.
    exhaustive = index.exhaustive(query, 100)
    assert len(indexed) == 100
    assert indexed == exhaustive


def test_joint_fit_improves_regularized_objective_without_nan():
    d = data(); z = np.array([[1., .3, 1.], [.2, 1., 1.]])
    initial = np.array([[.1, .05, .2, .1], [.07, .15, .1, .3], [.2, .1, .05, .03]])
    fitted, report = m.fit(z, d, initial, 'CONTRASTIVE_SPARSE', .1, topk=4, maxiter=180)
    assert report['objective_after'] < report['objective_before']
    assert report['converged']
    assert np.isfinite(fitted[1]).all() and np.all(fitted[0] >= 0)


@pytest.mark.parametrize('continuous', [False, True])
def test_outer_changes_cannot_select_family_or_change_fitted_model(monkeypatch, continuous):
    import copy
    from src.xgb_matcher import retrieval_continuous_sparse_v70
    kernel = retrieval_continuous_sparse_v70 if continuous else m
    rng = np.random.default_rng(19); rows = []; scenes = []; raw = {}; aliases = {}
    registry = {str(f): [] for f in (2, 3, 4)}
    q = rng.normal(size=(180, 2, 4))
    for i in range(180):
        parent = f'{i+100:09d}'; sid = parent+'00001'; nid = '999999999'+f'{i+1:05d}'
        fold = (2, 3, 4)[i%3]; registry[str(fold)].append(parent)
        rows.append(dict(query_id=str(i), fold=fold, positive_siren=parent, positive_siret=sid,
                         crm_name='ALPHA', negatives=[dict(siren='999999999', siret=nid)]))
        scenes.append(dict(query_id=str(i), fold=fold, positive_siren=parent,
                           positive_features=[1.]*len(m.NATIVE),
                           negatives=[dict(siren='999999999', features=[0.]*len(m.NATIVE))]))
        raw[sid] = dict(legal_names=[f'ALPHA TOKEN{chr(65+i%20)}'])
        raw[nid] = dict(legal_names=[f'BETA TOKEN{chr(65+(i+1)%20)}'])
        aliases[sid] = ['ALPHA']; aliases[nid] = ['BETA']
    # Deterministic solver isolates selection/basis/vocabulary causality;
    # numerical solver correctness is checked separately above.
    def fixed(z, data, initial, family, ridge, *args):
        return (m.PRIOR.copy(), initial.copy(), 0. if family == 'NATIVE23' else 1.), dict(
            converged=True, parameters=len(m.FEATURES)+(0 if family == 'NATIVE23' else initial.size))
    monkeypatch.setattr(kernel, 'fit', fixed)
    cfg = dict(vocabulary_cap=128, latent_per_field=2, ridges=[.1], flops=.1, query_terms=4, maxiter=1)
    before, report_before = kernel.select(q, rows, scenes, raw, aliases, registry, 2, cfg)
    changed_scenes = copy.deepcopy(scenes); changed_raw = copy.deepcopy(raw); changed_alias = copy.deepcopy(aliases)
    for i, row in enumerate(rows):
        if row['fold'] == 2:
            q[i] *= -100.
            changed_scenes[i]['positive_features'] = [100.]*len(m.NATIVE)
            changed_raw[row['positive_siret']] = dict(legal_names=['OUTER SECRET ONLY'])
            changed_alias[row['positive_siret']] = ['OUTER SECRET ONLY']
    after, report_after = kernel.select(q, rows, changed_scenes, changed_raw, changed_alias, registry, 2, cfg)
    assert before == after
    assert report_before['candidates'] == report_after['candidates']
    assert report_before['selected'] == report_after['selected']
    assert report_before['outer_before'] != report_after['outer_before']


def test_alias_manifest_verifies_referenced_launch_not_manifest(tmp_path, monkeypatch):
    import json
    from scripts import preflight_retrieval_v69_sparse_contrastive as probe
    from src.xgb_matcher.retrieval_v20_workbench import record
    launch = tmp_path/'launch.json'; launch.write_text(json.dumps(dict(implementation={'sentinel':'checked'})))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(launch=record(launch), native_current_alias_parity=True, recall_measured=False)))
    seen = []
    monkeypatch.setattr(probe.v38, 'verify_launch', lambda item: seen.append(item))
    result = probe.validated_alias_source(record(manifest))
    assert result['native_current_alias_parity']
    assert seen == [dict(implementation={'sentinel':'checked'})]
