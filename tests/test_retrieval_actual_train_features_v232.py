"""Synthetic authorization/join fixtures, not national evidence."""
from copy import deepcopy
import pytest
from scripts.retrieval_v232_actual_train_features import references,valid


def fixture():
    raw=dict(query_id='TRAIN',fold=2)
    s=dict(**raw,positive_siret='12345678900001',positive_siren='123456789',
           negatives=[dict(siret='98765432100001',siren='987654321')])
    mined=dict(**raw,channels=dict(name=[dict(siret='98765432100001',siren='987654321')],
                                  joint=[dict(siret='12345678900001',siren='123456789')]))
    return raw,s,mined


def test_supervised_point_union_not_product_candidate_emission():
    assert references(*fixture())=={'12345678900001':'123456789','98765432100001':'987654321'}


@pytest.mark.parametrize('bad',['query','fold','parent','identity'])
def test_misaligned_or_contradictory_point_namespace_stops(bad):
    raw,s,m=fixture()
    if bad=='query':m['query_id']='PILOT'
    elif bad=='fold':m['fold']=1
    elif bad=='parent':s['positive_siren']='987654321'
    else:s['positive_siret']='INVENTED'
    with pytest.raises(ValueError):references(raw,s,m)


def test_invalid_matrix_or_partial_training_never_admitted():
    q,s,m=fixture();refs=references(q,s,m)
    r=dict(**q,seconds=1.,positive_siret=s['positive_siret'],positive_siren=s['positive_siren'],
           references=refs,features=[[0.]*11 for _ in refs],
           fixed_scores={str(f):[0.]*len(refs) for f in (2,3,4)},
           maximum_native_parity_error={str(f):0. for f in (2,3,4)})
    valid([r],[q])
    with pytest.raises(ValueError):valid([],[q])
    r['features'][0][0]=float('nan')
    with pytest.raises(ValueError):valid([r],[q])
