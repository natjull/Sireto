import pytest
from scripts.audit_retrieval_v103_national_name_catalog import attach


def fixture():
    q=[dict(query_id=str(i),fold=2+i%3,crm_name='ABC') for i in range(128)]
    p=[dict(query_id=str(i),fold=2+i%3,human_siret='00123456700001',human_siren='001234567',
        exact_hit=bool(i%2),branch_zero_or_positive_reason='DIAGNOSTIC') for i in range(128)]
    f=[dict(subject_code=1,owner_key=1234567,official_parent=1234567,scope_code=1,name_text='ABC'),
       dict(subject_code=1,owner_key=1234567,official_parent=1234567,scope_code=0,name_text=None),
       dict(subject_code=2,owner_key=123456700001,official_parent=1234567,scope_code=3,name_text='DEF'),
       dict(subject_code=2,owner_key=123456700002,official_parent=1234567,scope_code=3,name_text='ABC')]
    return q,p,f


def test_all128_owners_no_sibling_alias_and_context_retained():
    q,p,f=fixture(); rows=attach(q,p,f)
    assert len(rows)==128 and all(len(r['facts'])==3 for r in rows)
    assert rows[0]['facts'][0]['name_text']=='DEF'
    assert rows[0]['facts'][1]['shared_literal_words']==['ABC']
    assert rows[0]['facts'][2]['name_text'] is None and not rows[0]['facts'][2]['lexical_eligible']


def test_partial_contaminated_fold_and_owner_conflict_fail_closed():
    q,p,f=fixture()
    with pytest.raises(ValueError,match='denominator'): attach(q[:-1],p,f)
    q[0]['fold']=0
    with pytest.raises(ValueError,match='denominator'): attach(q,p,f)
    q,p,f=fixture(); f[0]['official_parent']=999999999
    with pytest.raises(ValueError,match='membership'): attach(q,p,f)
