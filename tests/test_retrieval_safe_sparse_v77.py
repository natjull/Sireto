import copy

import numpy as np
import pytest

from src.xgb_matcher.retrieval_safe_sparse_v77 import allowed_documents,fit_view,select,target_matrix


def fixture():
    rng=np.random.default_rng(77)
    registry={str(f):[str(100000000+f*100+i) for i in range(50)] for f in (2,3,4)}
    rows=[]; scenes=[]; docs={}
    for f in (2,3,4):
        for j,p in enumerate(registry[str(f)]):
            sid=p+'00001'; neg='900000000'+f'{j+1:05d}'
            rows.append(dict(query_id=f'q{f}_{j}',fold=f,positive_siren=p,positive_siret=sid,
                crm_name='MAIRIE',negatives=[dict(siren='900000000',siret=neg)]))
            docs[sid]={'legal_names':['COMMUNE COLLEGE']}; docs[neg]={'legal_names':['BAZAR HOPITAL']}
            scenes.append(dict(query_id=f'q{f}_{j}',fold=f,positive_siren=p,positive_features=np.ones(19).tolist(),
                negatives=[dict(siren='900000000',features=np.zeros(19).tolist())]))
    config=dict(vocabulary_cap=32,minimum_document_frequency=1,maximum_document_fraction=1.,
        minimum_vocabulary_words=1,ridges=[1.],alphas=[.5],native_ridge=.005)
    return rng.normal(size=(150,6)),rows,scenes,docs,registry,config


def test_vocab_excludes_protected_parents_even_as_negative():
    _,rows,_,docs,registry,config=fixture()
    blocked=set(registry['2']); r=copy.deepcopy(rows[50]); r['negatives'].append(dict(siren=rows[0]['positive_siren'],siret=rows[0]['positive_siret']))
    safe=allowed_documents([r],docs,[0],blocked)
    assert rows[0]['positive_siret'] not in safe
    assert set(safe)=={r['positive_siret'],r['negatives'][0]['siret']}
    with pytest.raises(ValueError,match='Protected'):
        allowed_documents(rows,docs,[0],blocked)


def test_targets_are_never_constructed_outside_fit():
    _,rows,_,docs,_,_=fixture(); rows=copy.deepcopy(rows)
    rows[0]['positive_siret']='not_present'
    y=target_matrix(rows,docs,['commune'],list(range(50,100)))
    assert np.all(y[:50]==0) and np.all(y[100:]==0) and np.all(y[50:100]>0)


def test_outer_document_mutation_cannot_change_vocab_selection_projection():
    q,rows,scenes,docs,registry,config=fixture()
    p,v,r=select(q,rows,scenes,docs,registry,2,config)
    qq=q.copy(); qq[:50]*=100
    changed=copy.deepcopy(docs)
    for row in rows[:50]:
        changed[row['positive_siret']]={'legal_names':['WORDONLYOUTER HIDDEN SECRET']*20}
    ss=copy.deepcopy(scenes)
    for s in ss[:50]:
        s['positive_features']=[999.]*19
    pp,vv,rr=select(qq,rows,ss,changed,registry,2,config)
    assert v==vv and r['inner_vocabulary']==rr['inner_vocabulary']
    assert r['selected']==rr['selected'] and np.array_equal(p,pp)
    assert r['native']['weights']==rr['native']['weights']
    assert 'wordonlyouter' not in v


def test_inner_vocabulary_uses_only_permitted_fit_documents():
    _,rows,_,docs,registry,config=fixture()
    vocab,_,scope=fit_view(rows,docs,[50,51],registry['2'],config)
    changed=copy.deepcopy(docs)
    changed[rows[52]['positive_siret']]={'legal_names':['OTHER VALIDATION ONLY']}
    assert fit_view(rows,changed,[50,51],registry['2'],config)[0]==vocab
    assert scope['target_queries']==2 and scope['protected_document_overlap']==0


def test_only_id_consistent_official_membership_can_enter_vocab():
    _,rows,_,docs,_,_=fixture(); r=copy.deepcopy(rows[50]); r['positive_siren']='999999999'
    with pytest.raises(ValueError,match='membership'):
        allowed_documents([r],docs,[0],[])
