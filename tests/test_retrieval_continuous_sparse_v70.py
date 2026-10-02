import numpy as np
import pytest

from src.xgb_matcher import retrieval_continuous_sparse_v70 as m
from src.xgb_matcher import retrieval_sparse_contrastive_v69 as old
from tests.test_retrieval_sparse_contrastive_v69 import data


def test_continuous_support_exchange_unlike_hard_topk():
    eps = 1e-8
    left = np.array([[2.+eps, 2.-eps]])
    right = left[:, ::-1]
    # One document contains only first term: hard cutoff jumps ~log(3).
    assert abs(old.activate(left, 1)[0][0, 0]-old.activate(right, 1)[0][0, 0]) > 1.
    assert abs(m.activation(left, 1)[0][0, 0]-m.activation(right, 1)[0][0, 0]) < 1e-7


def test_boundary_gradient_and_flops_match_finite_differences():
    d = data(); z = np.array([[1., .3, 1.], [.2, 1., 1.]])
    initial = np.array([[.1, .05, .2, .1], [.07, .15, .1, .3], [.2, .1, .05, .03]])
    theta = np.r_[old.PRIOR+.15, (initial+.01).ravel()]
    loss, grad = m.objective(theta, z, d, initial, .1, .1, 2)
    assert np.isfinite(loss)
    for i in range(len(theta)):
        plus, minus = theta.copy(), theta.copy(); plus[i] += 1e-6; minus[i] -= 1e-6
        finite = (m.objective(plus, z, d, initial, .1, .1, 2)[0]-
                  m.objective(minus, z, d, initial, .1, .1, 2)[0])/2e-6
        assert finite == pytest.approx(grad[i], abs=2e-8)


def test_cap_zero_threshold_and_negative_raw_values():
    rng = np.random.default_rng(11); raw = rng.normal(size=(14, 79))
    values, cache = m.activation(raw, 16)
    assert np.all(np.count_nonzero(values, axis=1) <= 16)
    assert np.isfinite(values).all() and np.all(values >= 0)
    assert np.isfinite(m.backward(cache, np.ones_like(values))).all()
    actual, _ = m.activation([[-1., 0., 2.]], 16)
    np.testing.assert_allclose(actual, [[0., 0., np.log(3.)]])
    assert not m.activation(np.ones((1, 17)), 16)[0].any()


def test_serialized_query_uses_the_same_thresholded_activation():
    decoder = np.array([[.1, .05, .2, .1], [.07, .15, .1, .3], [.2, .1, .05, .03]])
    model = dict(bases=[[[1.], [0.]], [[0.], [1.]]], decoder=decoder.tolist(),
                 query_terms=2, vocabulary=['A:a', 'I:B', 'N:c', 'N:d'], alpha=2.)
    actual = m.query_impacts(model, np.array([1., 0., 0., 1.]))
    weights = m.activation(np.array([[1., 1., 1.]]) @ decoder, 2)[0][0]*2
    assert actual == {t:float(v) for t, v in zip(model['vocabulary'], weights) if v > 0}


def test_continuous_fit_converges_and_improves_objective():
    d = data(); z = np.array([[1., .3, 1.], [.2, 1., 1.]])
    initial = np.array([[.1, .05, .2, .1], [.07, .15, .1, .3], [.2, .1, .05, .03]])
    _, report = m.fit(z, d, initial, 'CONTRASTIVE_SPARSE', .1, topk=2)
    assert report['converged']
    assert report['objective_after'] < report['objective_before']
