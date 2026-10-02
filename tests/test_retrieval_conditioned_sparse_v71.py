import numpy as np
import pytest

from src.xgb_matcher import retrieval_conditioned_sparse_v71 as m
from src.xgb_matcher import retrieval_continuous_sparse_v70 as original
from tests.test_retrieval_sparse_contrastive_v69 import data


def setup():
    z = np.array([[1., .3, 1.], [.2, 1., 1.]])
    initial = np.array([[.1, .05, .2, .1], [.07, .15, .1, .3], [.2, .1, .05, .03]])
    return z, initial


def test_optimizer_coordinates_preserve_objective_and_physical_bounds():
    z, initial = setup(); d = data()
    origin, scale, lower, upper = m.coordinates(initial, .1)
    assert np.all(scale >= 1.)
    np.testing.assert_allclose((origin+scale*lower)[:23], 0., atol=1e-14)
    np.testing.assert_allclose((origin+scale*upper)[:23], 64., atol=1e-14)
    np.testing.assert_allclose((origin+scale*lower)[23:], -16., atol=1e-14)
    np.testing.assert_allclose((origin+scale*upper)[23:], 16., atol=1e-14)
    eta = np.ones(len(origin))*.0001
    value, gradient = m.scaled_objective(eta, origin, scale, z, d, initial, .1, .1, 2)
    physical, physical_grad = original.objective(origin+scale*eta, z, d, initial, .1, .1, 2)
    assert value == physical
    np.testing.assert_array_equal(gradient, physical_grad*scale)
    for j in range(len(eta)):
        plus, minus = eta.copy(), eta.copy(); plus[j] += 1e-6; minus[j] -= 1e-6
        fd = (m.scaled_objective(plus, origin, scale, z, d, initial, .1, .1, 2)[0]-
              m.scaled_objective(minus, origin, scale, z, d, initial, .1, .1, 2)[0])/2e-6
        assert fd == pytest.approx(gradient[j], abs=1e-7)


def test_comparators_are_identical_to_frozen_solver():
    z, initial = setup(); d = data()
    for family in ('NATIVE23', 'RIDGE_SPARSE'):
        a, ra = m.fit(z, d, initial, family, .1, topk=2)
        b, rb = original.fit(z, d, initial, family, .1, topk=2)
        for av, bv in zip(a, b):
            np.testing.assert_array_equal(av, bv)
        assert ra == rb


def test_conditioned_fit_keeps_kernel_and_improves_same_loss():
    z, initial = setup(); d = data()
    fitted, report = m.fit(z, d, initial, 'CONTRASTIVE_SPARSE', .1, topk=2)
    assert report['converged']
    assert report['same_physical_objective_and_bounds']
    assert report['objective_after'] < report['objective_before']
    theta = np.r_[fitted[0], fitted[1].ravel()]
    assert original.objective(theta, z, d, initial, .1, .1, 2)[0] == report['objective_after']
    assert m.query_impacts is original.query_impacts
