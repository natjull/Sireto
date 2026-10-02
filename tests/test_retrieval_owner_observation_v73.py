import pytest

from src.xgb_matcher.retrieval_owner_observation_v73 import compose
from src.xgb_matcher.retrieval_observation_v72 import ObservationIndex


def parent(record='FORM', siren='123456789', source='RNE'):
    return dict(source=source, source_record_id=record, subject_kind='SIREN', siren=siren,
        evidence_id='p:'+record, names=[dict(kind='LEGAL', raw_value='ALPHA', normalized_value='ALPHA'),
        dict(kind='TRADE', raw_value='OTHER', normalized_value='OTHER')], addresses=[])


def site(record='FORM', siren='123456789', number='00001', source='RNE'):
    siret = siren+number
    return dict(source=source, source_record_id=record+':site:'+siret, subject_kind='SIRET',
        siren=siren, siret=siret, evidence_id='s:'+record+number, names=[],
        valid_to='2020-01-01', is_current=False,
        addresses=[dict(raw_value='RUE BETA', normalized_value='RUE BETA', number='', number_suffix='',
            insee='75056', postcode='75001')])


def test_legal_name_and_site_address_bind_only_with_explicit_record_owner():
    out, bindings, counts = compose([parent(), site()])
    assert counts['sites_named_before'] == 0 and counts['sites_named_after'] == 1
    assert counts['bound_closed_sites'] == 1 and counts['bound_unknown_parent_valid_from'] == 1
    assert bindings[0]['evidence_kind'].endswith('NOT_TEMPORAL_EQUIVALENCE')
    assert ObservationIndex(out).search(dict(crm_name='ALPHA', crm_street='RUE BETA'))[0] == [('12345678900001', 3.)]
    assert not ObservationIndex(out).search(dict(crm_name='OTHER'))[0]


@pytest.mark.parametrize('p,s', [
    (parent('OLD'), site('NEW')), (parent(), site(siren='999999999')),
    (parent(source='BODACC'), site()), (parent(), site(source='BODACC')),
])
def test_no_free_same_siren_cross_record_source_or_sibling_inheritance(p, s):
    out, bindings, counts = compose([p, s])
    assert not bindings and counts['sites_named_after'] == 0
    assert not ObservationIndex(out).search(dict(crm_name='ALPHA'))[0]


def test_suffix_mismatch_and_duplicate_parent_stop_not_guess():
    s = site(); s['source_record_id'] = 'FORM:site:12345678900002'
    out, bindings, _ = compose([parent(), s])
    assert not bindings and not ObservationIndex(out).search(dict(crm_name='ALPHA'))[0]
    with pytest.raises(ValueError, match='Ambiguous duplicate'):
        compose([parent(), parent(), site()])
