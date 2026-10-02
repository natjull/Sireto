import numpy as np
from tests.test_retrieval_proof_v56 import backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_sparse_query_v59 import words,vocabulary,expand,solve,semantic_score,compiled_component,field_bags,select

def test_only_official_words_and_no_identifiers_are_expanded():
    assert words('CA 123456789 12345678900001 COMMUNE DE LA VILLE')==frozenset(('ca','commune','ville'))
    assert not expand([1.,0.],np.array([[1.,2.],[0.,0.]]),['commune','ville'],'')
    terms=expand([1.,0.],np.array([[1.,2.],[0.,0.]]),['commune','ville'],'VILLE')
    assert terms==[('commune',1.)]

def test_native_weighted_membership_matches_offline_same_field_and_place(tmp_path):
    fields={'legal_names':['COMMUNE DE VILLE'],'site_names':['COLLEGE']}
    b=backend(tmp_path,[fields,dict(fields,insee='99999',postcode='99999')])
    terms=[('commune',.4),('college',.6)]
    assert semantic_score(fields,terms,1.)==.6
    assert semantic_score(field_bags(fields),terms,1.)==.6
    q=RetrievalQuery(name='MAIRIE',insee='75056')
    hits=b.searcher.search(compiled_component(b,q,terms),limit=2,count=False).hits
    assert len(hits)==1 and np.isclose(hits[0][0],.6)
    assert compiled_component(b,RetrievalQuery(name='MAIRIE'),terms) is None

def test_regression_uses_only_training_positions_not_heldout_targets():
    rng=np.random.default_rng(59); q=rng.normal(size=(24,6)); y=rng.normal(size=(24,8)); indices=list(range(16))
    fitted=solve(q,y,indices,1.); q[16:]*=1000; y[16:]*=1000
    assert np.array_equal(fitted,solve(q,y,indices,1.))

def test_nested_selection_ignores_external_fold_queries_and_labels():
    rng=np.random.default_rng(60); registry={str(f):[str(100000000+f*100+i) for i in range(50)] for f in (2,3,4)}
    rows=[]; scenes=[]; docs={}; vocab=['commune','college','bazar','hopital']; q=rng.normal(size=(150,6)); y=rng.random((150,4))
    for f in (2,3,4):
        for i,parent in enumerate(registry[str(f)]):
            truth=parent+'00001'; neg='900000000'+f'{i+1:05d}'
            rows.append(dict(query_id=f'q{f}_{i}',fold=f,positive_siren=parent,positive_siret=truth,crm_name='MAIRIE',
                negatives=[dict(siren='900000000',siret=neg)]))
            docs[truth]={'legal_names':['COMMUNE']}; docs[neg]={'legal_names':['BAZAR']}
            positive=np.ones(19); negative=np.zeros(19)
            scenes.append(dict(query_id=f'q{f}_{i}',fold=f,positive_siren=parent,positive_features=positive.tolist(),
                negatives=[dict(siren='900000000',features=negative.tolist())]))
    p,r=select(q,y,vocab,rows,scenes,docs,registry,2,[1.],[.5])
    q[:50]*=100; y[:50]*=100
    pp,rr=select(q,y,vocab,rows,scenes,docs,registry,2,[1.],[.5])
    assert r['selected']==rr['selected'] and np.array_equal(p,pp)
    assert set(r['query_ids']).isdisjoint({row['query_id'] for row in rows[:50]})
