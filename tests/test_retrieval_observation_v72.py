import numpy as np
import pytest

from src.xgb_matcher import retrieval_observation_v72 as m


def observation(siret, name='ALPHA', address='10 RUE BETA', evidence_id='e', source='RNE', start='2010-01-01'):
    return m.Observation.from_row(dict(source=source, source_record_id=evidence_id,
        subject_kind='SIRET', siren=siret[:9], siret=siret, evidence_id=evidence_id,
        valid_from=start, valid_to=None, observed_at=None, is_current=False,
        names=[dict(raw_value=name, normalized_value=name, kind='TRADE')],
        addresses=[dict(raw_value=address, normalized_value=address, number='10',
                        number_suffix='', insee='75056', postcode='75001')]))


def test_name_address_cannot_cross_observations_or_dates():
    old = observation('12345678900001', 'ALPHA', 'ANCIENNE', 'old')
    new = observation('12345678900001', 'BETA', 'NOUVELLE', 'new', 'BODACC', '2020-01-01')
    row = dict(crm_name='ALPHA', crm_street='NOUVELLE')
    q = m.query(row)
    assert max(m.score(old, q), m.score(new, q)) == 1.
    assert old.valid_from != new.valid_from and old.source != new.source
    # A bag-of-all-names + bag-of-all-addresses would invent a score3.
    assert m.ObservationIndex([old, new]).search(row)[0] == [('12345678900001', 1.)]


def test_words_cannot_mix_between_name_aliases():
    o = observation('12345678900001', 'ALPHA')
    o = m.Observation(**{**o.__dict__, 'names':(m.termset('ALPHA'), m.termset('BETA'))})
    assert m.score(o, m.query(dict(crm_name='ALPHA BETA'))) == .5


def test_no_free_parent_to_site_or_sibling_inheritance():
    p = m.Observation.from_row(dict(source='BODACC', source_record_id='parent', subject_kind='SIREN',
        siren='123456789', siret='', names=[dict(raw_value='ALPHA', normalized_value='ALPHA', kind='LEGAL')],
        addresses=[]))
    site = observation('12345678900001', 'BETA', 'GAMMA')
    other = observation('12345678900002', 'ALPHA', 'DELTA', 'sibling')
    index = m.ObservationIndex([p, site, other])
    result, trace = index.search(dict(crm_name='ALPHA'))
    assert result == [('12345678900002', 1.)]
    assert trace['parent_only_not_projected'] == 1


def test_group_top_updates_evicts_and_keeps_deterministic_ties():
    h = m.GroupTop(2)
    h.offer('12345678900003', 1.); h.offer('12345678900002', 1.); h.offer('12345678900001', 1.)
    assert h.result() == [('12345678900001', 1.), ('12345678900002', 1.)]
    h.offer('12345678900003', 3.)
    for i in range(200):
        h.offer('12345678900001', 2.+i)
    assert h.result()[0] == ('12345678900001', 201.)
    assert h.peak_live == 2 and len(h.heap) <= 4
    with pytest.raises(ValueError):
        m.GroupTop(101)


def test_direct100_distinct_sites_matches_all_observation_oracle():
    rng = np.random.default_rng(71); rows = []
    # Repeated observations must not crowd out distinct sites.
    for i in range(145):
        for j in range(3):
            rows.append(observation(f'123456789{i+1:05d}', 'ALPHA '+str(rng.integers(1, 9)),
                                    '10 RUE BETA '+str(rng.integers(1, 9)), f'{i}-{j}'))
    index = m.ObservationIndex(rows)
    for row in (dict(crm_name='ALPHA 4', crm_street='RUE BETA 3'),
                dict(crm_name='', crm_street='RUE BETA'), dict(crm_name='NOHIT')):
        actual, trace = index.search(row)
        assert actual == index.exhaustive(row)
        assert len(actual) <= 100 and len({k for k, _ in actual}) == len(actual)
        assert trace['peak_live_siret_groups'] <= 100
    assert len(index.search(dict(crm_street='RUE BETA'))[0]) == 100


def test_query_guard_and_identifier_masking_and_far_sites_remain():
    with pytest.raises(ValueError):
        m.query(dict(crm_name='ALPHA', positive_siret='12345678900001'))
    with pytest.raises(ValueError):
        m.query(dict(crm_name='ALPHA', fold=1))
    assert '123456789' not in m.termset('ALPHA 123456789 12345678900001')
    row = dict(crm_name='ALPHA', crm_insee='00000', crm_postcode='00000')
    result, _ = m.ObservationIndex([observation('12345678900001')]).search(row)
    assert result == [('12345678900001', 1.)]
