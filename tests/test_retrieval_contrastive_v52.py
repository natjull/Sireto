import numpy as np
import pytest

from src.xgb_matcher.retrieval_contrastive_v52 import surface,combine,fit
from src.xgb_matcher.retrieval_learned_v41 import transform_queries


def test_controller_import_and_frozen_config():
    import json
    from scripts.preflight_retrieval_v52_contrastive_training import CONFIG
    config=json.loads(CONFIG.read_text())
    assert config['first_batch']==8 and config['training_queries']==512
    assert config['scope']=='TRAINING_ONLY_NO_RETRIEVAL_RECALL_NO_INDEX_BUILD'


def fixture():
    rng=np.random.default_rng(52); q=rng.normal(size=(60,8)); p=q@np.eye(8)[::-1]
    memberships=[]; negatives=[]; registry={str(f):[] for f in (2,3,4)}
    for i in range(60):
        fold=2+i//20; parent=f'{100000000+i}'; registry[str(fold)].append(parent)
        memberships.append(dict(query_id=str(i),fold=fold,positive_siren=parent,negative_sirens=['100000000','999999999']))
        negatives.append(rng.normal(size=(2,8)))
    return q,p,negatives,memberships,registry


def test_fit_excludes_complete_heldout_entities_positive_and_negative():
    q,p,n,m,r=fixture(); w,report=fit(q,p,n,m,2,r,maxiter=50)
    assert report['training_queries']==40 and report['training_pairs']==40
    assert report['safe_source_negative_pairs']==40 and report['training_folds']==[3,4]
    assert not set(report['query_ids'])&{str(i) for i in range(20)}
    assert report['objective_after']<report['objective_before'] and np.isfinite(w).all()
    assert report['document_transform']=='IDENTITY' and not report['identifiers_as_features']


def test_initial_index_score_is_single_dot_product_no_pairwise_features():
    q,p,n,m,r=fixture(); w,_=fit(q,p,n,m,2,r,maxiter=30)
    query=transform_queries(q[:1],w)[0]
    docs=p/np.linalg.norm(p,axis=1,keepdims=True)
    native=docs@query
    raw=docs@(q[0]/np.linalg.norm(q[0])@w)
    assert np.array_equal(np.argsort(native),np.argsort(raw))


@pytest.mark.parametrize('change',['empty_registry','bad_fold','negative_alignment'])
def test_invalid_protected_scope_fails_closed(change):
    q,p,n,m,r=fixture()
    if change=='empty_registry': r['2']=[]
    if change=='bad_fold': m[0]['fold']=0
    if change=='negative_alignment': m[20]['negative_sirens']=[]
    with pytest.raises(ValueError): fit(q,p,n,m,2,r)


def test_structured_fields_remain_separate_and_unit_length():
    combined=combine(np.eye(3),np.eye(3)[::-1])
    assert combined.shape==(3,6) and np.allclose(np.linalg.norm(combined,axis=1),1.)
    assert np.array_equal(combined[:,:3],np.eye(3)/np.sqrt(2.))


def test_identifier_masks_are_label_blind_and_do_not_drop_surface():
    text,count=surface(['ALPHA 123456789','12 RUE BLEUE 12345678900001','75001'],'nom')
    assert count==2 and '123456789' not in text and 'ALPHA' in text and '75001' in text
    assert '12 RUE BLEUE' in text


def test_invalid_view_container_rejected():
    with pytest.raises(ValueError): surface('ALPHA','nom')
