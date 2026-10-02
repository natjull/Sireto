from copy import deepcopy

import duckdb
import numpy as np
import pytest

from src.xgb_matcher.retrieval_coherent_proof_v74 import (
    FEATURES, coherent, design, select, views,
)
from tests.test_retrieval_owner_observation_v73 import parent, site


def address(text, number='10', insee='75056', postcode='75001'):
    return dict(raw_value=text, normalized_value=text, number=number, number_suffix='',
                insee=insee, postcode=postcode)


def query():
    return dict(crm_name='ALPHA', crm_number='10', crm_street='RUE BETA',
                crm_insee='75056', crm_postcode='75001', fold=2)


def test_one_observation_support_and_no_cross_record_product():
    a, b = site('A'), site('B')
    a['names'] = [dict(raw_value='ALPHA', normalized_value='ALPHA', kind='SIGN')]
    a['addresses'] = [address('20 AVENUE GAMMA', number='20', insee='99999', postcode='99999')]
    b['names'] = [dict(raw_value='OTHER', normalized_value='OTHER', kind='SIGN')]
    b['addresses'] = [address('10 RUE BETA')]
    bound, _ = views([a, b]); sid = a['siren']+'00001'
    assert coherent(query(), bound[sid]) == 0.
    a['addresses'] = b['addresses']
    bound, _ = views([a, b])
    assert coherent(query(), bound[sid]) == 1.


def test_no_cross_alias_or_address_alternative_and_no_bodacc_identity_guess():
    s = site(); s['names'] = [dict(raw_value=n, normalized_value=n, kind='SIGN') for n in ('ALPHA', 'BETA')]
    s['addresses'] = [address('10 RUE'), address('BETA', number='20', insee='99999', postcode='99999')]
    bound, _ = views([s]); q = query(); q['crm_name'] = 'ALPHA BETA'
    assert coherent(q, bound[s['siren']+'00001']) == pytest.approx(1/3)
    s['source'] = 'BODACC'
    bound, _ = views([s])
    assert coherent(q, bound[s['siren']+'00001']) == 0.


def test_owner_alias_single_formality_initialism_and_missing_evidence():
    p, s = parent(), site(); p['names'][0]['normalized_value'] = 'ALPHA BETA CONSEIL'
    s['addresses'] = [address('10 RUE BETA')]
    bound, counts = views([p, s]); q = query(); q['crm_name'] = 'ABC'
    assert coherent(q, bound[s['siren']+'00001']) == 1.
    assert counts['counts']['added_legal_aliases'] == 1
    assert coherent(q, []) == 0.
    p['source_record_id'] = 'ANOTHER'
    bound, _ = views([p, s])
    assert coherent(q, bound[s['siren']+'00001']) == 0.


def test_identifiers_state_dates_do_not_supply_similarity_or_filter():
    p, s = parent(), site(); s['addresses'] = [address('10 RUE BETA')]
    bound, _ = views([p, s]); q = query()
    value = coherent(q, bound[s['siren']+'00001'])
    q['positive_siret'] = '99999999999999'; q['positive_siren'] = '999999999'
    q['crm_name'] += ' 12345678900001'
    assert coherent(q, bound[s['siren']+'00001']) == value == 1.
    s['is_current'] = True; s['valid_to'] = None
    bound, _ = views([p, s])
    assert coherent(q, bound[s['siren']+'00001']) == 1.


def training():
    rows = []; registry = {str(f): [] for f in (2, 3, 4)}; values = []
    for fold in (2, 3, 4):
        for j in range(60):
            p = str(fold*100000000+j)
            registry[str(fold)].append(p)
            rows.append(dict(query_id=f'{fold}-{j}', positive_siren=p, fold=fold,
                             negatives=[dict(siren='900000001')]))
            v = np.zeros((2, len(FEATURES)))
            v[0, 0] = .1; v[1, 1] = .1
            v[0, -1] = j % 2
            values.append(v)
    return rows, values, registry


def test_outer_features_cannot_choose_family_or_weights():
    rows, values, registry = training()
    model, report = select(rows, values, registry, 2)
    changed = deepcopy(values)
    for i, row in enumerate(rows):
        if row['fold'] == 2:
            changed[i] = changed[i][::-1].copy() * 100
    other, other_report = select(rows, changed, registry, 2)
    assert model == other
    assert report['selected'] == other_report['selected']
    assert report['candidates'] == other_report['candidates']
    assert report['outer_after'] != other_report['outer_after']
    assert all(q.startswith(('3-', '4-')) for q in model['query_ids'])


def test_missing_bonus_preserves_scenes_and_all_safe_negatives():
    rows, values, _ = training()
    values[0][:, -1] = 0.
    x, sizes, qids = design(rows, values, range(len(rows)))
    assert len(sizes) == len(rows) == len(qids) and len(x) == len(rows)
    with pytest.raises(ValueError, match='No safe'):
        design(rows, values, [0], ['900000001'])
    with pytest.raises(ValueError, match='denominator'):
        design(rows, [np.zeros((1, len(FEATURES)))], [0])


def test_duckdb_setting_parameter_supported_without_global_settings():
    c = duckdb.connect(':memory:')
    c.execute('SET temp_directory=?', ['/private/tmp/sireto-v74-tests'])
    assert c.execute("SELECT current_setting('temp_directory')").fetchone()[0].endswith('sireto-v74-tests')
    c.close()
