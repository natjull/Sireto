import copy
import numpy as np
import pytest
from src.xgb_matcher.retrieval_initialisms_v65 import membership,patterns,acronym_components,query,feature_vector,select
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from tests.test_retrieval_proof_v56 import backend


def test_generic_initialisms_prefixes_particles_and_site_numbers():
    assert membership('ABC24',['SAS ALPHA BETA COMMERCE ANNEXE'])==1
    assert membership('ABC',['ALPHA DE BETA COMMERCE'])==1
    assert membership('AC',['ALPHA EN COMMERCE'])==0  # EN must remain a word.
    assert membership('AEC',['L ALPHA EN COMMERCE'])==1
    assert membership('ABC',['ALPHA BRAVO','COMMERCE'])==0
    assert membership('ABC',['BRAVO ALPHA COMMERCE'])==0


def test_empty_institutional_or_identifiers_do_not_create_alias_queries():
    assert patterns('MAIRIE ECOLE SARL 123456789 12345678900001')==[]
    assert patterns('')==[]
    assert patterns('ABC 123456789')==patterns('ABC 987654321')
    assert len(patterns('ABC DEF GHI'))==2


def test_native_full_alias_regex_parity_no_cross_alias_or_global_place_filter(tmp_path):
    b=backend(tmp_path,[dict(legal_names='SAS ALPHA BETA COMMERCE ANNEXE',addresses='12 RUE BLEUE'),
        dict(legal_names=['ALPHA BRAVO','COMMERCE'],addresses='12 RUE BLEUE'),
        dict(site_names='ALPHA BETA COMMERCE',insee='99999',postcode='99999',addresses='ZZZ')])
    q=RetrievalQuery(name='ABC24',address='12 RUE BLEUE',number='12',insee='75056')
    scores=[]
    for part in acronym_components(b,q):
        scores.append({b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s)
            for s,loc in b.searcher.search(part,limit=10,count=False).hits})
    assert set(scores[0])=={'10000000000001','10000000200001'}
    assert scores[1]==scores[2]==scores[3]=={'10000000000001':1.}
    w=np.zeros(16); w[11]=1
    assert len(b.searcher.search(query(b,q,w,[]),limit=100,count=False).hits)==2


def test_all_native_initialism_scores_match_offline_features(tmp_path):
    from src.xgb_matcher.retrieval_proof_v56 import point_features
    b=backend(tmp_path,[dict(legal_names='ALPHA BETA COMMERCE',addresses='12 RUE BLEUE'),
        dict(legal_names='OTHER',addresses='13 RUE VERTE')])
    q=RetrievalQuery(name='ABC',address='12 RUE BLEUE',number='12',insee='75056')
    ids,native=point_features(b,q,['10000000000001','10000000100001']); w=np.arange(1,17)*.1
    expected=[feature_vector(n,float(i==0),0.)@w for i,n in enumerate(native)]
    actual={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s)
        for s,loc in b.searcher.search(query(b,q,w,[]),limit=10,count=False).hits}
    np.testing.assert_allclose([actual[s] for s in ids],expected,atol=1e-5,rtol=0)


def test_outer_query_alias_target_mutation_cannot_change_selection_or_weights():
    from tests.test_retrieval_joint_contrastive_v63 import fixture
    rows,scenes,q,docs,ids,c,registry=fixture(); q=q[:,0]
    words={sid:dict(legal_names=['ALPHA BETA']) for sid in ids}
    aliases={sid:['ALPHA BETA'] for sid in ids}
    a,va,ma,ra=select(q,rows,scenes,words,aliases,registry,2)
    r2=copy.deepcopy(rows); s2=copy.deepcopy(scenes); q2=q.copy(); al2=copy.deepcopy(aliases); words2=copy.deepcopy(words)
    for i,r in enumerate(rows):
        if r['fold']==2:
            r2[i]['crm_name']='AB'; q2[i]*=-1
            s2[i]['positive_features']=[0.]*19; al2[r['positive_siret']]=['BRAVO DELTA']
            words2[r['positive_siret']]=dict(legal_names=['EXTERNAL WORD MUTATION'])
    b,vb,mb,rb=select(q2,r2,s2,words2,al2,registry,2)
    np.testing.assert_array_equal(a,b); assert va==vb and ma==mb and ra['selected']==rb['selected']
    assert ra['outer_after']!=rb['outer_after']
