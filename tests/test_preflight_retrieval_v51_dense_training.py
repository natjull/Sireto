import pytest

from scripts.preflight_retrieval_v51_dense_training import query_text,safe_negatives


def row():
    return dict(fold=3,positive_siren='123456789',positive_siret='12345678900001',acceptable=['12345678900001','12345678900002'],
        natural_negatives=[dict(siren='123456789',siret='12345678900002'),dict(siren='987654321',siret='98765432100001'),
            dict(siren='555555555',siret='55555555500001')],
        crm_name='ALPHA',crm_number='12',crm_suffix='BIS',crm_street='RUE BLEUE',crm_postcode='75001')


def test_fold_entity_and_operational_exclusions_before_mining():
    registry={'2':['987654321'],'3':['123456789'],'4':['444444444']}
    assert safe_negatives(row(),2,registry)==[dict(siren='555555555',siret='55555555500001')]
    assert safe_negatives(row(),3,registry)==[]
    changed=row(); changed['fold']=4; changed['positive_siren']='987654321'
    assert safe_negatives(changed,2,registry)==[]


def test_query_encoder_has_no_label_or_identifier_argument():
    text=query_text(row())
    assert 'ALPHA' in text and 'RUE BLEUE' in text and '75001' in text
    assert '123456789' not in text and '987654321' not in text


def test_identifier_shaped_query_feature_fails_closed():
    r=row(); r['crm_name']='ALPHA 12345678900001'
    with pytest.raises(ValueError,match='Identifier'): query_text(r)
