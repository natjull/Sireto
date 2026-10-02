import numpy as np
import pytest

from src.xgb_matcher.retrieval_learned_v41 import (
    TrainingMembership, exact_top100_reference, fit_query_transform,
    official_text, oof_training_positions, transform_queries, unit_vectors,
)


def members():
    return [TrainingMembership('a', 2, '100000001', '100000002'),
            TrainingMembership('b', 3, '100000003', '100000004'),
            TrainingMembership('c', 4, '100000005', '100000006'),
            TrainingMembership('d', 3, '100000001', '100000007'),
            TrainingMembership('e', 4, '100000008', '100000001')]


def test_oof_excludes_fold_and_heldout_entities_on_both_sides():
    assert oof_training_positions(members(), 2, {'100000001'}).tolist() == [1, 2]


@pytest.mark.parametrize('fold', [0, 1, 5])
def test_forbidden_folds_fail_closed(fold):
    with pytest.raises(ValueError):
        oof_training_positions(members(), fold, set())
    with pytest.raises(ValueError):
        oof_training_positions([TrainingMembership('x', fold, '100000001', '100000002')], 2, set())


def test_heldout_vectors_have_no_effect_on_fit():
    rng = np.random.default_rng(41)
    arrays = [rng.normal(size=(5, 8)) for _ in range(3)]
    weight, report = fit_query_transform(*arrays, members(), 2, {'100000001'})
    mutated = [a.copy() for a in arrays]
    for a in mutated:
        a[[0, 3, 4]] = rng.normal(size=(3, 8)) * 500
    other, _ = fit_query_transform(*mutated, members(), 2, {'100000001'})
    np.testing.assert_array_equal(weight, other)
    assert report['fitted_rows'] == 2 and report['training_folds'] == [3, 4]
    assert transform_queries(arrays[0], weight).shape == (5, 8)


def test_first_stage_reference_searches_full_corpus_and_caps_output():
    rng = np.random.default_rng(12)
    docs = unit_vectors(rng.normal(size=(500, 12)))
    query = unit_vectors(rng.normal(size=(1, 12)))[0]
    ids = [f'{10000000000000+i:014d}' for i in range(500)]
    found = exact_top100_reference(query, docs, ids)
    expected = sorted(zip(ids, (unit_vectors(docs) @ unit_vectors(query[None])[0]).astype(np.float64)),
                      key=lambda pair: (-pair[1], pair[0]))[:100]
    assert found == expected and len(found) == 100
    with pytest.raises(ValueError):
        exact_top100_reference(query, docs, ids, 101)
    with pytest.raises(ValueError):
        exact_top100_reference(query, docs, [ids[0]]*500)


def test_text_schema_has_no_identity_or_cross_site_propagation():
    assert official_text(['ALPHA', 'ALPHA'], '12 RUE BLEUE') == 'noms: ALPHA adresse: 12 RUE BLEUE'
    with pytest.raises(ValueError):
        official_text(['ALPHA 12345678900001'], '')
    with pytest.raises(ValueError):
        official_text('ALPHA', '')


def test_invalid_embeddings_and_alignment_stop():
    for values in (np.zeros((2, 3)), np.array([[np.nan, 1.]]), np.ones(3)):
        with pytest.raises(ValueError):
            unit_vectors(values)
    arrays = [np.ones((5, 3)) for _ in range(3)]
    with pytest.raises(ValueError):
        fit_query_transform(*arrays, members(), 2, {r.positive_siren for r in members()})
    with pytest.raises(ValueError):
        fit_query_transform(*arrays, members()[:2], 2, set())
