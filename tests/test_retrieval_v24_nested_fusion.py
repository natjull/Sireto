from copy import deepcopy
import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_v24_nested_fusion import (
    inner_jobs, check_split, optimum, calibrate, parent_probabilities, make_master, config_at,
)


def truth_fixture():
    return pd.DataFrame([dict(query_id=f'q{f}', fold=f, gt_siren=f'P{f}') for f in (2, 3, 4)])


def probability_fixture(outer=2):
    return pd.DataFrame([dict(query_id=f'q{f}', fold=f, siren=s,
                             base_probability=b, learned_probability=l)
                         for f in (2, 3, 4) if f != outer
                         for s, b, l in [(f'P{f}', .2, .7), ('negative', .8, .3)]])


def test_inner_lineage_excludes_outer_in_fit_and_calibration():
    for outer in (2, 3, 4):
        jobs = inner_jobs(outer)
        assert {target for _, target in jobs} == {2, 3, 4}-{outer}
        for train, target in jobs:
            assert train != target and outer not in (train, target)
    with pytest.raises(ValueError):
        inner_jobs(0)


def test_split_rejects_wrong_training_and_target_query_metadata():
    split = dict(train_folds=[3, 4], target_folds=[2], trained_query_ids=['q3', 'q4'], target_query_ids=['q2'])
    check_split(split, [3, 4], [2], truth_fixture())
    for key, value in [('trained_query_ids', ['q2']), ('target_query_ids', ['q1']), ('train_folds', [2, 3])]:
        altered = {**split, key: value}
        with pytest.raises(ValueError, match='contamination'):
            check_split(altered, [3, 4], [2], truth_fixture())


def test_optimum_symmetry_and_global_convex_minimum():
    assert optimum([.2, .3], [.2, .3])['learned_weight'] == .5
    b, l = np.array([.1, .9, .2]), np.array([.7, .1, .5])
    fit = optimum(b, l); a = fit['learned_weight']
    assert 0 < a < 1 and abs(fit['derivative']) < 1e-12
    assert abs(optimum(l, b)['learned_weight'] - (1-a)) < 1e-12
    for point in [1e-5, .1, .3, .5, .9, .99999]:  # solver unit check, not a recipe sweep
        objective = -np.log(b+point*(l-b)).mean()-.01*(np.log(point)+np.log1p(-point))
        assert fit['regularized_objective'] <= objective+1e-14


@pytest.mark.parametrize('base,learned', [([], []), ([0], [.1]), ([np.nan], [.1]), ([1.1], [.5]), ([.1], [.1, .2])])
def test_invalid_probabilities_fail(base, learned):
    with pytest.raises(ValueError):
        optimum(base, learned)


def test_target_truth_perturbation_and_order_do_not_change_calibration():
    t = truth_fixture(); p = probability_fixture()
    before = calibrate(p, t, 2)
    t.loc[t.fold.eq(2), 'gt_siren'] = 'COMPLETELY_DIFFERENT'
    after = calibrate(p.iloc[::-1], t.iloc[::-1], 2)
    assert before == after
    assert before['fitted_query_ids'] == ['q3', 'q4']
    assert before['learned_weight'] > .5


def test_missing_natural_positive_is_not_injected():
    t = truth_fixture(); t.loc[t.fold.eq(3), 'gt_siren'] = 'MISSING'
    fit = calibrate(probability_fixture(), t, 2)
    assert fit['queries'] == 2 and fit['skipped_absent_parent'] == 1 and fit['fitted_query_ids'] == ['q4']


def test_outer_predictions_and_duplicate_parents_rejected():
    p = probability_fixture(); t = truth_fixture()
    with pytest.raises(ValueError):
        calibrate(pd.concat([p, p.iloc[:1]]), t, 2)
    with pytest.raises(ValueError):
        calibrate(probability_fixture(3), t, 2)


def test_rank_probabilities_ties_normalization_order_and_missing_scores():
    rows = [dict(query_id='q', fold=2, siren=str(i), parent_rank=r, parent_score=s)
            for i, r, s in [(0, 1, .1), (1, 1, .2), (2, 2, .2)]]
    c = duckdb.connect()
    c.register('base_parents', pa.Table.from_pylist(rows))
    scores = pa.Table.from_pylist(rows).select(['query_id', 'fold', 'siren', 'parent_score'])
    a = parent_probabilities(c, scores)
    b = parent_probabilities(c, scores.take([2, 1, 0]))
    pd.testing.assert_frame_equal(a, b)
    assert abs(a.base_probability.sum()-1) < 1e-14 and abs(a.learned_probability.sum()-1) < 1e-14
    assert a.iloc[0].base_probability == a.iloc[1].base_probability
    assert a.iloc[1].learned_probability == a.iloc[2].learned_probability
    with pytest.raises(ValueError, match='Missing'):
        parent_probabilities(c, scores.slice(0, 2))
    c.close()


def master_fixture():
    return dict(
        pool=[dict(query_id='q', fold=2, siren='P', siret=str(i), rescue_score=.3, direct_rrf=.1) for i in range(2100)],
        site_features=[dict(query_id='q', fold=2, siren='P', siret=str(i), direct_rrf=.1,
                            local_site=True, joint_site_rank=1) for i in range(2100)],
        parent_scores=[dict(query_id='q', fold=2, siren='P', parent_score=5.)],
        fold_weights=[dict(fold=f, learned_weight=.25) for f in (2, 3, 4)])


def run_master(rows):
    c = duckdb.connect()
    for name, value in rows.items():
        c.register(name, pa.Table.from_pylist(value))
    make_master(c)
    result = c.execute('SELECT siret,rescue_score,candidate_rank FROM master ORDER BY candidate_rank').fetchall()
    c.close()
    return result


def test_master_cap_no_direct_double_count_and_order_invariance():
    r = master_fixture(); a = run_master(r)
    assert len(a) == 2000 and len({x[0] for x in a}) == 2000 and a[-1][2] == 2000
    assert abs(a[0][1] - (.1 + .75*.2 + .25/62)) < 1e-14
    for values in r.values():
        values.reverse()
    assert run_master(r) == a


def test_outside_geography_keeps_direct_score_only():
    r = master_fixture()
    for p, s in zip(r['pool'], r['site_features']):
        p['rescue_score'] = .1; s['local_site'] = False
    assert all(row[1] == .1 for row in run_master(r))


def test_master_invalid_weights_and_duplicate_sites_fail():
    r = master_fixture(); r['fold_weights'][0]['learned_weight'] = float('nan')
    with pytest.raises(ValueError, match='weights'):
        run_master(r)
    r = master_fixture(); r['site_features'].append(deepcopy(r['site_features'][0]))
    with pytest.raises(ValueError, match='population'):
        run_master(r)


def test_frozen_config_rejects_new_recipe_parameters(tmp_path):
    c = config_at('config/retrieval_v24_nested_fusion.json')
    c['log_barrier'] = .1
    p = tmp_path/'bad.json'; p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='drift'):
        config_at(p)
