import numpy as np
import pytest

from scripts.retrieval_v78_safe_sparse_initial import expansion_rows,validate_models
from scripts.compare_retrieval_v78_baselines import candidate_set


def test_encoder_uses_only_correct_fold_vocabulary():
    vectors=np.array([[1.,0.],[1.,0.],[1.,0.]])
    q=[dict(query_id=str(f),fold=f,crm_name='UNKNOWN') for f in (2,3,4)]
    vocab={str(f):[word] for f,word in zip((2,3,4),('commune','college','hopital'))}
    projections={str(f):np.array([[1.],[0.]]) for f in (2,3,4)}
    rows=expansion_rows(vectors,q,vocab,projections,8)
    assert [r['terms'] for r in rows]==[[('commune',1.)],[('college',1.)],[('hopital',1.)]]


def test_empty_name_remains_in_encoder_population():
    rows=expansion_rows(np.array([[1.,0.]]),[dict(query_id='blank',fold=2,crm_name='')],
        {'2':['commune']},{'2':np.array([[1.],[0.]])},8)
    assert len(rows)==1 and rows[0]['terms']==[]


@pytest.mark.parametrize('fold',[0,1,5])
def test_closed_fold_not_encoded(fold):
    with pytest.raises(ValueError,match='Closed'):
        expansion_rows(np.array([[1.,0.]]),[dict(query_id='bad',fold=fold,crm_name='ALPHA')],{}, {},8)


def test_model_bundle_cannot_omit_or_add_fold():
    with pytest.raises(ValueError,match='fold'):
        validate_models({'1':{}})


def test_comparison_cannot_relax_cap_or_uniqueness():
    with pytest.raises(ValueError,match='cap'):
        candidate_set([dict(siret=str(j)) for j in range(101)])
    with pytest.raises(ValueError,match='duplicate'):
        candidate_set([dict(siret='12345678900001')]*2)
