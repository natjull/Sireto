"""Synthetic metadata fixtures only, never entity matching evidence."""
from copy import deepcopy
import pytest
from scripts.retrieval_v227_national_training_negatives import project, FIELDS, valid


def source():
    return [dict(query_id=f'T{i}', fold=2+i%3, crm_name='EXEMPLE', crm_number='', crm_suffix='',
                 crm_street='', crm_postcode='', crm_insee='', positive_siret='SECRET',
                 positive_siren='SECRET', negatives=['SECRET']) for i in range(512)]


def test_projection_strips_every_target_and_preserves_preexisting_order():
    r = project(source(), [dict(query_id='PILOT')])
    assert len(r) == 64 and r[-1]['query_id'] == 'T63'
    assert all(set(q) == set(FIELDS) for q in r)
    s = source(); s[0]['positive_siret']='OTHER'
    assert project(s, []) == r


@pytest.mark.parametrize('bad', ['partial', 'overlap', 'fold0', 'typed'])
def test_unsafe_projection_is_rejected(bad):
    s=source(); controls=[]
    if bad=='partial':s.pop()
    elif bad=='overlap':controls=[dict(query_id='T500')]
    elif bad=='fold0':s[0]['fold']=0
    else:s[0]['crm_name']=None
    with pytest.raises(ValueError):project(s,controls)


def test_complete_natural_feature_references_required():
    q=project(source(),[])[:1]
    c=dict(siret='12345678900001',siren='123456789')
    r=[dict(query_id=q[0]['query_id'],fold=q[0]['fold'],seconds=1.,
            channels=dict(name=[c],joint=[c]),features={c['siret']:[0.]*11})]
    valid(r,q)
    r[0]['features']['98765432100001']=[0.]*11
    with pytest.raises(ValueError):valid(r,q)
