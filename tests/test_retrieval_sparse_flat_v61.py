import numpy as np
from tests.test_retrieval_proof_v56 import backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_sparse_native_v60 import query as old
from src.xgb_matcher.retrieval_sparse_flat_v61 import query as new

def results(b,parsed):
    return {b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s) for s,loc in b.searcher.search(parsed,limit=1000,count=False).hits}

def test_flattened_metric_has_same_membership_and_native_scores(tmp_path):
    rng=np.random.default_rng(61); rows=[]
    for i in range(320):
        rows.append(dict(legal_names='ALPHA COMMUNE' if i%2 else 'BETA COLLEGE',
            site_names='ECOLE' if i%3 else 'MAIRIE',addresses='12 RUE BLEUE' if i%5 else '13 RUE VERTE',
            insee='75056' if i%7 else '99999',postcode='75001' if i%7 else '99999',is_siege='1' if i%11 else '0'))
    b=backend(tmp_path,rows); q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',number='12',insee='75056',postcode='75001')
    for alpha in (0.,.5,2.):
        w=rng.uniform(0,3,11); terms=[('commune',.17),('college',.83)]
        before=results(b,old(b,q,w,terms,alpha)); after=results(b,new(b,q,w,terms,alpha))
        assert before.keys()==after.keys()
        assert all(np.isclose(before[s],after[s],atol=1e-5,rtol=0.) for s in before)

def test_no_text_can_still_match_learned_site_component_without_changing_score(tmp_path):
    b=backend(tmp_path,[dict(legal_names='COMMUNE')]); q=RetrievalQuery(name='',insee='75056',number='12')
    w=np.zeros(11); w[5]=2.
    assert results(b,old(b,q,w,[],0.))==results(b,new(b,q,w,[],0.))
